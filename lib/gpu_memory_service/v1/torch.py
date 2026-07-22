# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The complete Torch-specific surface for snapshot-only V1."""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import TYPE_CHECKING

from .client import AllocationPools, Manager
from .errors import GMSError
from .protocol import AccessClass, Mapping

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def discover_parameter_mappings(
    model: object, mappings: "Iterable[Mapping]"
) -> tuple[Mapping, ...]:
    """Resolve complete unique CUDA Parameter storages to allocator mappings."""
    unique: dict[tuple[str, int, int, int], Mapping] = {}
    for record in mappings:
        key = (
            record.allocation.allocation_id,
            record.base,
            record.allocation.aligned_size,
            record.reservation_size,
        )
        previous = unique.setdefault(key, record)
        if previous != record:
            raise GMSError("mapping identity has inconsistent allocation metadata")
    records = tuple(unique.values())
    storages: dict[int, object] = {}
    for parameter in model.parameters():  # type: ignore[attr-defined]
        if parameter.device.type != "cuda":
            continue
        storage = parameter.untyped_storage()
        if int(storage.nbytes()) != 0:
            storages.setdefault(int(storage._cdata), storage)

    found: dict[tuple[str, int, int, int], Mapping] = {}
    by_id: dict[str, tuple[int, int, int]] = {}
    by_base: dict[int, str] = {}
    for storage in storages.values():
        start = int(storage.data_ptr())  # type: ignore[attr-defined]
        size = int(storage.nbytes())  # type: ignore[attr-defined]
        end = start + size
        if end <= start:
            raise GMSError("invalid parameter storage range")
        owners = [
            record for record in records if record.base <= start and end <= record.end
        ]
        if len(owners) != 1:
            overlaps = [
                record for record in records if start < record.end and record.base < end
            ]
            reason = (
                "crosses an allocation boundary" if overlaps else "is not V1-backed"
            )
            raise GMSError(f"parameter storage {reason}")
        owner = owners[0]
        if owner.allocation.access is not AccessClass.PARAMETER_RO:
            raise GMSError("parameter storage is not in the parameter domain")
        key = (
            owner.allocation.allocation_id,
            owner.base,
            owner.allocation.aligned_size,
            owner.reservation_size,
        )
        shape = key[1:]
        if by_id.setdefault(key[0], shape) != shape:
            raise GMSError("allocation id has inconsistent base or size")
        if by_base.setdefault(owner.base, key[0]) != key[0]:
            raise GMSError("mapping base names different allocations")
        found[key] = owner

    result = tuple(sorted(found.values(), key=lambda record: record.base))
    parameter_records = {
        (
            record.allocation.allocation_id,
            record.base,
            record.allocation.aligned_size,
            record.reservation_size,
        )
        for record in records
        if record.allocation.access is AccessClass.PARAMETER_RO
    }
    if set(found) != parameter_records:
        raise GMSError("live parameter-domain allocation has no registered Parameter")
    return result


def validate_buffer_mappings(model: object, mappings: "Iterable[Mapping]") -> None:
    """Require every live CUDA named buffer storage to avoid parameter backing."""
    parameters = tuple(
        record
        for record in mappings
        if record.allocation.access is AccessClass.PARAMETER_RO
    )
    storages: dict[int, tuple[str, int, int]] = {}
    for name, buffer in model.named_buffers(  # type: ignore[attr-defined]
        remove_duplicate=False
    ):
        if buffer.device.type != "cuda":
            continue
        storage = buffer.untyped_storage()
        size = int(storage.nbytes())
        if size == 0:
            continue
        start = int(storage.data_ptr())
        end = start + size
        if end <= start:
            raise GMSError(f"invalid CUDA buffer storage range for {name!r}")
        identity = int(storage._cdata)
        previous = storages.setdefault(identity, (name, start, end))
        if previous[1:] != (start, end):
            raise GMSError("buffer storage identity has inconsistent range")

    for name, start, end in storages.values():
        if any(start < record.end and record.base < end for record in parameters):
            raise GMSError(
                f"CUDA buffer storage for {name!r} overlaps parameter backing"
            )


class TorchPools:
    """Pinned Torch 2.11 parameter/private MemPools."""

    def __init__(self, manager: Manager):
        if not isinstance(manager, Manager):
            raise TypeError("TorchPools requires a capture-capable V1 Manager")

        import torch
        from gpu_memory_service.client.torch.extensions import _allocator_ext

        self._torch = torch
        self._manager = manager
        self._allocator = AllocationPools(manager)
        self.device = manager.device
        if _allocator_ext is None:
            raise RuntimeError("GPU Memory Service allocator extension is not built")
        _allocator_ext.init_module(self._allocator.malloc, self._allocator.free)
        pluggable = torch.cuda.CUDAPluggableAllocator(
            _allocator_ext.__file__, "my_malloc", "my_free"
        )
        native = pluggable.allocator()
        with torch.cuda.device(self.device):
            self.parameter: torch.cuda.MemPool | None = torch.cuda.MemPool(
                allocator=native
            )
            self.private: torch.cuda.MemPool | None = torch.cuda.MemPool(
                allocator=native
            )

    @contextmanager
    def parameter_pool(self) -> "Iterator[None]":
        parameter = self.parameter
        if parameter is None:
            raise GMSError("parameter pool has been destroyed")
        with self._allocator.parameter_pool():
            with self._torch.cuda.device(self.device):
                with self._torch.cuda.use_mem_pool(parameter, device=self.device):
                    yield

    @contextmanager
    def private_pool(self) -> "Iterator[None]":
        private = self.private
        if private is None:
            raise GMSError("private pool has been destroyed")
        with self._allocator.private_pool():
            with self._torch.cuda.device(self.device):
                with self._torch.cuda.use_mem_pool(private, device=self.device):
                    yield

    def collect_and_destroy(self) -> None:
        """Evict inactive segments without invalidating live storage mappings."""
        gc.collect()
        with self._torch.cuda.device(self.device):
            self._torch.cuda.empty_cache()
        parameter, private = self.parameter, self.private
        self.parameter = None
        self.private = None
        # Torch 2.11 MemPool destruction calls releasePool followed by exact
        # CUDACachingAllocator::emptyCache(pool.id), releasing only inactive
        # segments. Live Parameter/private storages retain their segments and
        # data pointers; the CUDA integration test checks this exact contract.
        del parameter, private
        gc.collect()
        if self._allocator.failure is not None:
            raise GMSError(
                "allocator free callback failed"
            ) from self._allocator.failure

    def prepare_capture(self, model: object) -> tuple[Mapping, ...]:
        """Run the one supported capture preparation sequence."""
        try:
            self.collect_and_destroy()
            validate_buffer_mappings(model, self._manager.mappings)
            parameters = discover_parameter_mappings(model, self._manager.mappings)
        except Exception as cause:
            failures: list[Exception] = []
            self._manager._abandon_capture(failures, pinned=False)
            detail = failures[0] if failures else cause
            raise self._manager._latch("capture preparation failed", detail) from cause
        self._manager.seal()
        self._manager.sleep()
        return parameters

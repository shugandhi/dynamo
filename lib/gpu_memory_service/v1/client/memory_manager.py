# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-resident mappings and fail-stop lifecycle for GMS V1."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import threading
from uuid import uuid4

from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.vmm import VMMDevice

from ..errors import FatalGMSError, GMSError
from .rpc import AllocationClient


class AccessClass(str, Enum):
    PARAMETER_RO = "parameter_ro"
    PRIVATE_RW = "private_rw"


@dataclass(frozen=True)
class LocalMapping:
    """One canonical allocator segment whose VA is preserved by CRIU."""

    allocation_id: str
    requested_size: int
    aligned_size: int
    access: AccessClass
    base: int
    reservation_size: int

    def __post_init__(self) -> None:
        if not self.allocation_id:
            raise ValueError("allocation ID must not be empty")
        if (
            self.requested_size <= 0
            or self.aligned_size < self.requested_size
            or self.base <= 0
            or self.reservation_size < self.aligned_size
        ):
            raise ValueError("invalid local allocation mapping")

    @property
    def end(self) -> int:
        return self.base + self.aligned_size

    def with_allocation_id(self, allocation_id: str) -> "LocalMapping":
        if not allocation_id:
            raise ValueError("allocation ID must not be empty")
        return replace(self, allocation_id=allocation_id)


class _Lifecycle(Enum):
    AWAKE = "awake"
    PREPARING = "preparing"
    ASLEEP = "asleep"
    WAKING = "waking"
    RETIRED = "retired"
    FATAL = "fatal"


class SnapshotMemoryManager:
    """Own local mapping topology while the sidecar owns physical backing."""

    def __init__(
        self,
        service: AllocationClient,
        vmm: VMMDevice,
        device: int,
    ):
        self.service = service
        self.vmm = vmm
        self.device = device
        self._mappings: dict[int, LocalMapping] = {}
        self._imports: dict[int, int] = {}
        self._unmapped_imports: set[int] = set()
        self._released_reservations: set[int] = set()
        self._server_allocations: set[str] = set()
        self._lifecycle_lock = threading.RLock()
        self._lifecycle = _Lifecycle.AWAKE
        self._fatal: FatalGMSError | None = None
        self.vmm.ensure_initialized()
        self._granularity = int(self.vmm.get_allocation_granularity(device))
        if self._granularity <= 0:
            raise ValueError("allocation granularity must be positive")
        self._server_nonce, self._gpu_uuid = self.service.hello()
        if self._gpu_identity() != self._gpu_uuid:
            raise self._latch("GMS V1 sidecar is on another physical GPU")

    @property
    def mappings(self) -> tuple[LocalMapping, ...]:
        with self._lifecycle_lock:
            return tuple(self._mappings[base] for base in sorted(self._mappings))

    def allocate_parameter(self, size: int) -> int:
        return self._allocate(size, AccessClass.PARAMETER_RO)

    def allocate_private(self, size: int) -> int:
        return self._allocate(size, AccessClass.PRIVATE_RW)

    def _allocate(self, size: int, access: AccessClass) -> int:
        with self._lifecycle_lock:
            self._check_awake()
            if size <= 0:
                raise ValueError("allocation size must be positive")
            aligned_size = self._align(size)
            allocation_id = f"allocation-{uuid4()}"
            server_owned = reserved = False
            base = 0
            mapping: LocalMapping | None = None
            try:
                server_owned = True
                # A transport failure may happen after commit, so retain the
                # caller-generated ID before issuing the mutation.
                self._server_allocations.add(allocation_id)
                self.service.allocate(allocation_id, aligned_size)
                self._select_device()
                base = int(self.vmm.address_reserve(aligned_size, self._granularity))
                self._released_reservations.discard(base)
                reserved = True
                mapping = LocalMapping(
                    allocation_id,
                    size,
                    aligned_size,
                    access,
                    base,
                    aligned_size,
                )
                handle = self._install(
                    mapping, self.service.export(allocation_id), GrantedLockType.RW
                )
            except Exception as cause:
                failures: list[Exception] = []
                dropped = True
                if mapping is not None and base in self._imports:
                    dropped = self._drop_import(mapping, failures)
                if reserved and dropped and mapping is not None:
                    self._free_reservation(mapping, failures)
                if server_owned and dropped:
                    self._free_backing(allocation_id, failures)
                if failures or isinstance(cause, FatalGMSError):
                    if mapping is not None and (
                        base in self._imports or base not in self._released_reservations
                    ):
                        self._mappings[base] = mapping
                    detail = failures[0] if failures else cause
                    raise self._latch(
                        f"allocation failed ({cause}) and cleanup lost ownership",
                        detail,
                    ) from cause
                self._released_reservations.discard(base)
                raise
            self._mappings[base] = mapping
            self._imports[base] = handle
            return base

    def free_from_allocator(self, base: int, size: int) -> None:
        """Release one exact segment for the allocator's void-return callback."""
        with self._lifecycle_lock:
            self._check_allocator_free()
            mapping = self._mappings.get(base)
            if mapping is None or mapping.requested_size != size:
                raise self._latch("allocator free does not match an exact mapping")
            if base not in self._imports:
                raise self._latch("allocator freed a mapping without an import")
            self._select_device()
            failures: list[Exception] = []
            dropped = self._drop_import(mapping, failures)
            if dropped:
                self._free_backing(mapping.allocation_id, failures)
                self._free_reservation(mapping, failures)
            if failures:
                raise self._latch("allocator free cleanup failed", failures[0])
            del self._mappings[base]
            self._released_reservations.discard(base)

    def begin_snapshot_preparation(self) -> None:
        """Quiesce new allocations before Torch destroys its pools."""
        with self._lifecycle_lock:
            self._check_awake()
            self._lifecycle = _Lifecycle.PREPARING

    def sleep(self) -> None:
        """Make all parameters RO, drop imports, and free private backing."""
        with self._lifecycle_lock:
            self._check()
            if self._lifecycle is _Lifecycle.AWAKE:
                self._lifecycle = _Lifecycle.PREPARING
            if self._lifecycle is not _Lifecycle.PREPARING:
                raise GMSError("snapshot memory manager is not preparing")
            parameters = tuple(
                mapping
                for mapping in self.mappings
                if mapping.access is AccessClass.PARAMETER_RO
            )
            if not parameters:
                raise GMSError("snapshot has no parameter allocations")
            if any(
                mapping.base not in self._imports
                or mapping.reservation_size != mapping.aligned_size
                for mapping in self.mappings
            ):
                raise self._latch("local mapping preflight failed")

            self._select_device()
            try:
                self.vmm.synchronize()
            except Exception as cause:
                failures: list[Exception] = []
                self._abandon(failures)
                raise self._latch(
                    "snapshot synchronization failed",
                    failures[0] if failures else cause,
                ) from cause

            changed: list[LocalMapping] = []
            try:
                for mapping in parameters:
                    self.vmm.set_access(
                        mapping.base,
                        mapping.aligned_size,
                        self.device,
                        GrantedLockType.RO,
                    )
                    changed.append(mapping)
            except Exception as cause:
                for mapping in reversed(changed):
                    try:
                        self.vmm.set_access(
                            mapping.base,
                            mapping.aligned_size,
                            self.device,
                            GrantedLockType.RW,
                        )
                    except Exception:
                        pass
                failures = []
                self._abandon(failures)
                raise self._latch(
                    "whole-set parameter read-only transition failed",
                    failures[0] if failures else cause,
                ) from cause

            failures = []
            for mapping in reversed(self.mappings):
                if mapping.base not in self._imports:
                    failures.append(
                        GMSError(f"mapping 0x{mapping.base:x} has no imported handle")
                    )
                    dropped = False
                else:
                    dropped = self._drop_import(mapping, failures)
                if dropped and mapping.access is AccessClass.PRIVATE_RW:
                    self._free_backing(mapping.allocation_id, failures)
            if failures:
                self._abandon(failures)
                raise self._latch("snapshot sleep cleanup failed", failures[0])
            self._lifecycle = _Lifecycle.ASLEEP

    def wake(self) -> None:
        """Restore parameters and install fresh private backing at exact VAs."""
        with self._lifecycle_lock:
            self._check()
            if self._lifecycle is not _Lifecycle.ASLEEP or self._imports:
                raise GMSError("snapshot memory manager is not fully asleep")
            self._lifecycle = _Lifecycle.WAKING
            self._verify_identity()

            attempted: dict[int, LocalMapping] = {}
            fresh: list[str] = []
            updated = dict(self._mappings)
            try:
                self._select_device()
                for original in self.mappings:
                    mapping = original
                    if original.access is AccessClass.PRIVATE_RW:
                        allocation_id = f"allocation-{uuid4()}"
                        mapping = original.with_allocation_id(allocation_id)
                        fresh.append(allocation_id)
                    updated[mapping.base] = mapping
                    attempted[mapping.base] = mapping
                    if original.access is AccessClass.PRIVATE_RW:
                        self._server_allocations.add(allocation_id)
                        self.service.allocate(allocation_id, mapping.aligned_size)
                    protection = (
                        GrantedLockType.RO
                        if mapping.access is AccessClass.PARAMETER_RO
                        else GrantedLockType.RW
                    )
                    handle = self._install(
                        mapping,
                        self.service.export(mapping.allocation_id),
                        protection,
                    )
                    self._imports[mapping.base] = handle
            except Exception as cause:
                # Retain the exact fresh IDs even if rollback cannot complete.
                self._mappings = updated
                failures: list[Exception] = []
                safe_to_free = set(fresh)
                for mapping in reversed(tuple(attempted.values())):
                    if mapping.base in self._imports:
                        if not self._drop_import(mapping, failures):
                            safe_to_free.discard(mapping.allocation_id)
                for allocation_id in reversed(fresh):
                    if allocation_id not in safe_to_free:
                        continue
                    self._free_backing(allocation_id, failures)
                self._abandon(failures)
                raise self._latch(
                    f"snapshot wake failed: {cause}",
                    failures[0] if failures else cause,
                ) from cause
            self._mappings = updated
            self._lifecycle = _Lifecycle.AWAKE

    def retire(self) -> None:
        """Release every local import, physical allocation, and VA reservation."""
        with self._lifecycle_lock:
            if self._lifecycle is _Lifecycle.RETIRED:
                return
            self._check()
            if self._lifecycle in (_Lifecycle.PREPARING, _Lifecycle.WAKING):
                raise GMSError("snapshot lifecycle transition is in progress")
            self._select_device()
            failures: list[Exception] = []
            if self._imports:
                self._cleanup(failures, self.vmm.synchronize)
            for mapping in reversed(self.mappings):
                if mapping.base in self._imports:
                    dropped = self._drop_import(mapping, failures)
                else:
                    dropped = True
                if dropped:
                    self._free_backing(mapping.allocation_id, failures)
                    self._free_reservation(mapping, failures)
            if failures:
                raise self._latch("snapshot retirement failed", failures[0])
            self._mappings.clear()
            self._released_reservations.clear()
            self._lifecycle = _Lifecycle.RETIRED

    def abort_snapshot(self, cause: Exception) -> None:
        """Fail-stop capture preparation after cleaning independent ownership."""
        with self._lifecycle_lock:
            failures: list[Exception] = []
            self._cleanup(failures, self._select_device)
            self._abandon(failures)
            raise self._latch(
                "snapshot preparation failed", failures[0] if failures else cause
            ) from cause

    def _verify_identity(self) -> None:
        try:
            identity = self.service.hello()
            local_gpu = self._gpu_identity()
        except Exception as cause:
            raise self._latch("cannot verify GMS V1 sidecar identity", cause) from cause
        if identity != (self._server_nonce, self._gpu_uuid):
            raise self._latch("GMS V1 sidecar incarnation or physical GPU changed")
        if local_gpu != self._gpu_uuid:
            raise self._latch("restored process is on another physical GPU")

    def _gpu_identity(self) -> str:
        import torch

        return str(torch.cuda.get_device_properties(self.device).uuid)

    def _install(
        self,
        mapping: LocalMapping,
        fd: int,
        protection: GrantedLockType,
    ) -> int:
        handle = 0
        mapped = False
        try:
            # VMMDevice consumes the FD on both success and failure.
            handle = int(self.vmm.import_shareable_handle_close_fd(fd))
            self.vmm.map(mapping.base, mapping.aligned_size, handle)
            mapped = True
            self.vmm.set_access(
                mapping.base,
                mapping.aligned_size,
                self.device,
                protection,
            )
            return handle
        except Exception:
            failures: list[Exception] = []
            if mapped:
                try:
                    self.vmm.unmap(mapping.base, mapping.aligned_size)
                except Exception as exc:
                    failures.append(exc)
                else:
                    mapped = False
            if handle and not mapped:
                try:
                    self.vmm.release(handle)
                except Exception as exc:
                    failures.append(exc)
            if failures and handle:
                self._imports[mapping.base] = handle
                if not mapped:
                    self._unmapped_imports.add(mapping.base)
            raise

    def _drop_import(self, mapping: LocalMapping, failures: list[Exception]) -> bool:
        base = mapping.base
        handle = self._imports[base]
        if base not in self._unmapped_imports:
            try:
                self.vmm.unmap(base, mapping.aligned_size)
            except Exception as exc:
                failures.append(exc)
                return False
            self._unmapped_imports.add(base)
        try:
            self.vmm.release(handle)
        except Exception as exc:
            failures.append(exc)
            return False
        del self._imports[base]
        self._unmapped_imports.discard(base)
        return True

    def _abandon(self, failures: list[Exception]) -> None:
        for mapping in reversed(self.mappings):
            if mapping.base in self._imports:
                dropped = self._drop_import(mapping, failures)
            else:
                dropped = True
            if dropped:
                self._free_backing(mapping.allocation_id, failures)
                self._free_reservation(mapping, failures)

    def _select_device(self) -> None:
        self.vmm.runtime_set_device(self.device)

    def _free_backing(self, allocation_id: str, failures: list[Exception]) -> None:
        if allocation_id not in self._server_allocations:
            return
        try:
            self.service.free(allocation_id)
        except Exception as exc:
            failures.append(exc)
        else:
            self._server_allocations.remove(allocation_id)

    def _free_reservation(
        self, mapping: LocalMapping, failures: list[Exception]
    ) -> None:
        if mapping.base in self._released_reservations:
            return
        try:
            self.vmm.address_free(mapping.base, mapping.reservation_size)
        except Exception as exc:
            failures.append(exc)
        else:
            self._released_reservations.add(mapping.base)

    def _align(self, size: int) -> int:
        return (size + self._granularity - 1) // self._granularity * self._granularity

    def _check_awake(self) -> None:
        self._check()
        if self._lifecycle is not _Lifecycle.AWAKE:
            raise GMSError("snapshot memory manager is not awake")

    def _check_allocator_free(self) -> None:
        self._check()
        if self._lifecycle not in (_Lifecycle.AWAKE, _Lifecycle.PREPARING):
            raise GMSError("allocator free is not allowed in this lifecycle state")

    def _check(self) -> None:
        if self._fatal is not None:
            raise self._fatal
        if self._lifecycle is _Lifecycle.RETIRED:
            raise GMSError("snapshot memory manager is retired")

    def _latch(self, message: str, cause: Exception | None = None) -> FatalGMSError:
        if self._fatal is None:
            suffix = f": {cause}" if cause is not None else ""
            self._fatal = FatalGMSError(message + suffix)
            self._lifecycle = _Lifecycle.FATAL
        return self._fatal

    @staticmethod
    def _cleanup(failures: list[Exception], operation, *args: object) -> None:
        try:
            operation(*args)
        except Exception as exc:
            failures.append(exc)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two Torch MemPools routed to the client-owned V1 access domains."""

from __future__ import annotations

import gc
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from ...errors import FatalGMSError, GMSError
from ..memory_manager import AccessClass, SnapshotMemoryManager

if TYPE_CHECKING:
    from collections.abc import Iterator


_active_access: ContextVar[AccessClass | None] = ContextVar(
    "gms_v1_access", default=None
)


class _AllocatorCallbacks:
    def __init__(self, manager: SnapshotMemoryManager):
        self.manager = manager
        self.failure: Exception | None = None
        self._failure_lock = threading.Lock()

    def _record_failure(self, failure: Exception) -> None:
        with self._failure_lock:
            self.failure = self.failure or failure

    @contextmanager
    def scope(self, access: AccessClass) -> "Iterator[None]":
        token = _active_access.set(access)
        try:
            yield
        finally:
            _active_access.reset(token)

    def malloc(self, size: int, device: int, _stream: int) -> int:
        try:
            if device != self.manager.device:
                raise GMSError(
                    f"allocator callback device {device} != {self.manager.device}"
                )
            access = _active_access.get()
            if access is AccessClass.PARAMETER_RO:
                return self.manager.allocate_parameter(size)
            if access is AccessClass.PRIVATE_RW:
                return self.manager.allocate_private(size)
            raise GMSError("allocation occurred outside a V1 pool scope")
        except Exception as exc:
            self._record_failure(exc)
            raise

    def free(self, base: int, size: int, device: int, _stream: int) -> None:
        try:
            if device != self.manager.device:
                raise GMSError(
                    f"allocator callback device {device} != {self.manager.device}"
                )
            self.manager.free_from_allocator(base, size)
        except Exception as exc:
            # CUDAPluggableAllocator's free ABI returns void.
            self._record_failure(exc)


class SnapshotTorchPools:
    """Pinned Torch 2.11 parameter and private allocation pools."""

    def __init__(self, manager: SnapshotMemoryManager):
        import torch
        from gpu_memory_service.client.torch.extensions import _allocator_ext

        self._torch = torch
        self._manager = manager
        self._allocator = _AllocatorCallbacks(manager)
        self._condition = threading.Condition()
        self._active_scopes = 0
        self._preparing = False
        self.device = manager.device
        if _allocator_ext is None:
            raise RuntimeError("GPU Memory Service allocator extension is not built")
        _allocator_ext.init_module_strict(self._allocator.malloc, self._allocator.free)
        # MemPools and live storages hold only a non-owning native pointer.
        # The process runtime retains this wrapper after pool teardown.
        self._pluggable_allocator = torch.cuda.CUDAPluggableAllocator(
            _allocator_ext.__file__, "my_malloc", "my_free"
        )
        native = self._pluggable_allocator.allocator()
        with torch.cuda.device(self.device):
            self.parameter: torch.cuda.MemPool | None = torch.cuda.MemPool(
                allocator=native
            )
            self.private: torch.cuda.MemPool | None = torch.cuda.MemPool(
                allocator=native
            )

    @contextmanager
    def parameter_pool(self) -> "Iterator[None]":
        with self._pool_scope("parameter", AccessClass.PARAMETER_RO):
            yield

    @contextmanager
    def private_pool(self) -> "Iterator[None]":
        with self._pool_scope("private", AccessClass.PRIVATE_RW):
            yield

    @contextmanager
    def _pool_scope(self, name: str, access: AccessClass) -> "Iterator[None]":
        with self._condition:
            if self._preparing:
                raise GMSError("snapshot preparation has started")
            pool = getattr(self, name)
            if pool is None:
                raise GMSError(f"{name} pool has been destroyed")
            self._active_scopes += 1
        try:
            with self._allocator.scope(access):
                with self._torch.cuda.device(self.device):
                    with self._torch.cuda.use_mem_pool(pool, device=self.device):
                        yield
        finally:
            with self._condition:
                self._active_scopes -= 1
                self._condition.notify_all()

    def _collect_and_destroy(self) -> None:
        gc.collect()
        with self._torch.cuda.device(self.device):
            self._torch.cuda.empty_cache()
        parameter, private = self.parameter, self.private
        self.parameter = None
        self.private = None
        # Torch releases inactive segments while live storages retain their
        # allocator segments and data pointers.
        del parameter, private
        gc.collect()
        with self._allocator._failure_lock:
            failure = self._allocator.failure
        if failure is not None:
            raise GMSError("allocator free callback failed") from failure

    def prepare_snapshot(self) -> None:
        """Destroy both pools, then atomically prepare all recorded mappings."""
        with self._condition:
            if self._preparing:
                raise GMSError("snapshot preparation has already started")
            self._preparing = True
        try:
            self._manager.begin_snapshot_preparation()
        except FatalGMSError as cause:
            self._manager.abort_snapshot(cause)
        except Exception:
            with self._condition:
                self._preparing = False
                self._condition.notify_all()
            raise
        with self._condition:
            while self._active_scopes:
                self._condition.wait()
        try:
            self._collect_and_destroy()
            self._manager.sleep()
        except Exception as cause:
            self._manager.abort_snapshot(cause)

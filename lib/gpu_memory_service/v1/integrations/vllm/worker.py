# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership-based GMS V1 worker for vLLM's normal model loader.

Select explicitly with::

    --worker-cls gpu_memory_service.v1.integrations.vllm.worker.GMSV1Worker
"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager

from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.common.vmm import get_vmm
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.workspace import current_workspace_manager

from ...client.memory_manager import SnapshotMemoryManager
from ...client.rpc import AllocationClient
from ...client.torch import SnapshotTorchPools
from ...errors import GMSError
from .backend import BACKEND_NAME
from .patches import install_vllm_integration
from .runtime import VllmSnapshotRuntime, install_runtime

logger = logging.getLogger(__name__)


def _vllm_model_memory_bytes(
    pools: SnapshotTorchPools,
    measured_active_bytes: int,
) -> int:
    """Include parameter cache that cannot be reused for vLLM's KV cache.

    Parameter-pool cached space cannot be reused by the private/KV pool, so
    add its inactive segment bytes to vLLM's normal active-byte delta. This
    preserves normal accounting for allocations outside the two V1 pools.
    Cached private-pool space is left available because KV can reuse it.

    Torch 2.11 includes both MemPools in device-wide allocator statistics.
    Adding only the parameter pool's inactive segment bytes accounts for cache
    unavailable to KV without charging private cache that KV can reuse.
    """
    if pools.parameter is None or pools.private is None:
        raise GMSError("V1 pools have been destroyed")
    parameter_segments = pools.parameter.snapshot(include_traces=False)
    parameter_cache = sum(
        int(segment["total_size"]) - int(segment["allocated_size"])
        for segment in parameter_segments
    )
    return measured_active_bytes + parameter_cache


class GMSV1Worker(Worker):
    """Route vLLM's native weight and KV allocation scopes into GMS V1."""

    def init_device(self) -> None:
        model_config = self.vllm_config.model_config
        if not model_config.enable_sleep_mode:
            raise RuntimeError("GMS V1 requires vLLM sleep mode")
        model_config.sleep_mode_backend = BACKEND_NAME

        super().init_device()

        device = self.device.index
        if device is None:
            raise RuntimeError("GMS V1 requires an indexed CUDA device")
        client = AllocationClient(get_socket_path(device, "snapshot-v1"))
        try:
            manager = SnapshotMemoryManager(client, get_vmm(), device)
            pools = SnapshotTorchPools(manager)
            install_vllm_integration(current_workspace_manager(), pools)
            runtime = install_runtime(manager, pools)
        except BaseException:
            client.close()
            raise
        self._gms_v1_runtime: VllmSnapshotRuntime = runtime

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager[None]:
        if tag == "weights":
            return self._gms_v1_runtime.pools.parameter_pool()
        if tag == "kv_cache":
            return self._gms_v1_runtime.pools.private_pool()
        raise ValueError(f"unsupported GMS V1 vLLM allocation tag: {tag}")

    def sleep(self, level: int = 1) -> None:
        if level != 1:
            raise ValueError("GMS V1 supports only whole-engine level 1 suspend")
        try:
            super().sleep(level)
        except Exception as cause:
            logger.exception("GMS V1 suspend failed; terminating the worker process")
            raise SystemExit(1) from cause

    def wake_up(self, tags: list[str] | None = None) -> None:
        if tags is not None:
            raise ValueError("GMS V1 does not support partial-tag resume")
        try:
            super().wake_up(tags)
        except Exception as cause:
            logger.exception("GMS V1 resume failed; terminating the worker process")
            raise SystemExit(1) from cause

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        self.model_runner.model_memory_usage = _vllm_model_memory_bytes(
            self._gms_v1_runtime.pools,
            int(self.model_runner.model_memory_usage),
        )

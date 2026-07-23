# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route vLLM active-DBO workspace growth into the GMS V1 private pool.

V1 only supports mutable allocations that are created during initialization or
warmup, before snapshot preparation destroys the private pool. The single
validated vLLM mutable path is ``WorkspaceManager._ensure_workspace_size``:
growth enters the private scope, a no-growth call bypasses the (possibly
destroyed) pool, and growth after pool destruction fails stop. Kernels that
allocate scratch lazily inside ``forward()`` (e.g. Marlin MoE / Humming) are
out of scope; see ``lib/gpu_memory_service/v1/README.md``.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

from ...client.torch import SnapshotTorchPools


def install_vllm_integration(
    workspace_manager: Any,
    pools: SnapshotTorchPools,
) -> None:
    """Route vLLM workspace growth into PRIVATE_RW before model construction."""
    from vllm.v1.worker.workspace import dbo_current_ubatch_id

    original = workspace_manager._ensure_workspace_size

    def ensure_workspace_size(self: Any, required_bytes: int) -> Any:
        ubatch_id = dbo_current_ubatch_id()
        current = self._current_workspaces[ubatch_id]
        if self._workspace_size_bytes(current) >= required_bytes:
            return original(required_bytes)
        with pools.private_pool():
            return original(required_bytes)

    workspace_manager._ensure_workspace_size = MethodType(
        ensure_workspace_size, workspace_manager
    )

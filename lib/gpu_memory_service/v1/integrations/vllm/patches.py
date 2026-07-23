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

import inspect
from types import MethodType
from typing import Any

from ...client.torch import SnapshotTorchPools
from ...errors import GMSError

_OWNER_ATTR = "_dynamo_gms_v1_owner"


def install_vllm_integration(
    workspace_manager: object,
    pools: SnapshotTorchPools,
) -> None:
    """Route vLLM workspace growth into PRIVATE_RW; run before model construction.

    Reinstalling for the same manager and pools is idempotent. A second owner
    or a replaced hook is rejected.
    """
    if not isinstance(pools, SnapshotTorchPools):
        raise TypeError("pools must be SnapshotTorchPools")

    from vllm.v1.worker.workspace import (
        WorkspaceManager,
        dbo_current_ubatch_id,
    )

    if not isinstance(workspace_manager, WorkspaceManager):
        raise TypeError("workspace_manager must be a vLLM WorkspaceManager")

    hook = WorkspaceManager._ensure_workspace_size
    if not inspect.isfunction(hook) or tuple(inspect.signature(hook).parameters) != (
        "self",
        "required_bytes",
    ):
        raise GMSError("vLLM WorkspaceManager._ensure_workspace_size hook was replaced")

    owner = vars(workspace_manager).get(_OWNER_ATTR)
    if owner is not None:
        if (
            owner[0] is not pools
            or workspace_manager._ensure_workspace_size is not owner[1]
        ):
            raise GMSError("vLLM workspace hook already has another owner")
        return

    original = workspace_manager._ensure_workspace_size
    if (
        getattr(original, "__self__", None) is not workspace_manager
        or getattr(original, "__func__", None) is not hook
    ):
        raise GMSError("vLLM workspace allocation hook was replaced")

    def ensure_workspace_size(self: Any, required_bytes: int) -> Any:
        ubatch_id = dbo_current_ubatch_id()
        current = self._current_workspaces[ubatch_id]
        if self._workspace_size_bytes(current) >= required_bytes:
            return original(required_bytes)
        with pools.private_pool():
            return original(required_bytes)

    installed = MethodType(ensure_workspace_size, workspace_manager)
    setattr(workspace_manager, "_ensure_workspace_size", installed)
    setattr(workspace_manager, _OWNER_ATTR, (pools, installed))

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned vLLM 0.25.1 bindings to explicit V1 allocation scopes."""

from __future__ import annotations

from .scopes import AllocationScope, install_scoped_call


def install_workspace_routing(
    workspace_manager: object, private_scope: AllocationScope
) -> None:
    """Route this vLLM manager's workspace growth through V1 PRIVATE_RW.

    Installation is deliberately instance-local. Reinstalling the same owner
    is a no-op; another owner or a replaced wrapper is a fatal configuration
    error.
    """
    from vllm.v1.worker import workspace as workspace_module

    if not isinstance(workspace_manager, workspace_module.WorkspaceManager):
        raise TypeError("workspace_manager must be a vLLM 0.25.1 WorkspaceManager")
    supported_hook = workspace_module.WorkspaceManager._ensure_workspace_size

    def grows(self, required_bytes):
        ubatch_id = workspace_module.dbo_current_ubatch_id()
        current = self._current_workspaces[ubatch_id]
        return self._workspace_size_bytes(current) < required_bytes

    install_scoped_call(
        workspace_manager,
        "_ensure_workspace_size",
        supported_hook,
        private_scope,
        grows,
        owner=install_workspace_routing,
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from contextlib import contextmanager
from functools import wraps
from types import MethodType, ModuleType

import pytest
from gpu_memory_service.v1.client import AllocationPools
from gpu_memory_service.v1.errors import GMSError
from gpu_memory_service.v1.vllm import install_workspace_routing

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class Pools:
    def __init__(self, events, active):
        self.events = events
        self.active = active
        self.destroyed = False

    @contextmanager
    def private_pool(self):
        if self.destroyed:
            raise GMSError("private pool has been destroyed")
        self.events.append("private-enter")
        self.active.append("private")
        try:
            yield
        finally:
            self.active.pop()
            self.events.append("private-exit")


@pytest.fixture
def workspace(monkeypatch):
    events = []
    active = []
    active_ubatch = [0]

    class Tensor:
        def __init__(self, size):
            self.size = size

    class WorkspaceManager:
        def __init__(self):
            self._current_workspaces = [None, None]
            self.active_ubatch = active_ubatch

        @staticmethod
        def _workspace_size_bytes(value):
            return 0 if value is None else value.size

        def _ensure_workspace_size(self, required_bytes):
            events.append(("original", required_bytes, tuple(active)))
            ubatch_id = active_ubatch[0]
            current = self._current_workspaces[ubatch_id]
            if self._workspace_size_bytes(current) < required_bytes:
                allocation = getattr(self, "allocation", None)
                if allocation is not None:
                    allocation(required_bytes)
                self._current_workspaces[ubatch_id] = Tensor(required_bytes)
            return self._current_workspaces[ubatch_id]

    workspace_module_name = "vllm.v1.worker.workspace"
    WorkspaceManager._ensure_workspace_size.__module__ = workspace_module_name
    WorkspaceManager._ensure_workspace_size.__qualname__ = (
        "WorkspaceManager._ensure_workspace_size"
    )
    workspace_module = ModuleType(workspace_module_name)
    workspace_module.WorkspaceManager = WorkspaceManager
    workspace_module.dbo_current_ubatch_id = lambda: active_ubatch[0]
    worker_module = ModuleType("vllm.v1.worker")
    worker_module.workspace = workspace_module
    v1_module = ModuleType("vllm.v1")
    v1_module.worker = worker_module
    vllm_module = ModuleType("vllm")
    vllm_module.v1 = v1_module
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", worker_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.workspace", workspace_module)
    return WorkspaceManager(), events, active


def test_workspace_growth_overrides_outer_parameter_scope(workspace) -> None:
    manager, events, active = workspace
    allocations = []

    class AllocationManager:
        device = 0
        _mappings = {}

        def allocate_parameter(self, size):
            allocations.append(("parameter", size))
            return 1

        def allocate_private(self, size):
            allocations.append(("private", size))
            return 2

        def free(self, base):
            raise AssertionError(base)

    pools = AllocationPools(AllocationManager())

    @contextmanager
    def private_scope():
        active.append("private")
        try:
            with pools.private_pool():
                yield
        finally:
            active.pop()

    manager.allocation = lambda required_bytes: pools.malloc(required_bytes, 0, 0)
    install_workspace_routing(manager, private_scope)

    active.append("parameter")
    with pools.parameter_pool():
        result = manager._ensure_workspace_size(1024)
    active.pop()

    assert result.size == 1024
    assert allocations == [("private", 1024)]
    assert events == [("original", 1024, ("parameter", "private"))]


def test_workspace_no_growth_does_not_enter_destroyed_pool(workspace) -> None:
    manager, events, _active = workspace
    manager._current_workspaces[0] = type("Tensor", (), {"size": 1024})()
    pools = Pools(events, _active)
    install_workspace_routing(manager, pools.private_pool)
    pools.destroyed = True

    result = manager._ensure_workspace_size(512)

    assert result.size == 1024
    assert events == [("original", 512, ())]


def test_workspace_growth_uses_only_active_dbo_ubatch(workspace) -> None:
    manager, events, active = workspace
    pools = Pools(events, active)
    manager._current_workspaces = [
        type("Tensor", (), {"size": 2048})(),
        type("Tensor", (), {"size": 128})(),
    ]
    install_workspace_routing(manager, pools.private_pool)

    manager.active_ubatch[0] = 1
    grown = manager._ensure_workspace_size(1024)
    manager.active_ubatch[0] = 0
    unchanged = manager._ensure_workspace_size(1024)

    assert grown.size == 1024
    assert unchanged.size == 2048
    assert events == [
        "private-enter",
        ("original", 1024, ("private",)),
        "private-exit",
        ("original", 1024, ()),
    ]


def test_workspace_routing_is_idempotent_and_rejects_conflicts(workspace) -> None:
    manager, events, _active = workspace
    pools = Pools(events, _active)
    install_workspace_routing(manager, pools.private_pool)
    installed = manager._ensure_workspace_size

    install_workspace_routing(manager, pools.private_pool)
    assert manager._ensure_workspace_size is installed

    with pytest.raises(GMSError, match="another allocation scope owner"):
        install_workspace_routing(manager, Pools(events, _active).private_pool)


def test_workspace_growth_after_pool_destruction_fails_stop(workspace) -> None:
    manager, events, _active = workspace
    pools = Pools(events, _active)
    install_workspace_routing(manager, pools.private_pool)
    pools.destroyed = True

    with pytest.raises(GMSError, match="private pool has been destroyed"):
        manager._ensure_workspace_size(1024)

    assert manager._current_workspaces == [None, None]
    assert events == []


def test_workspace_routing_allows_inherited_unmodified_hook(workspace) -> None:
    manager, events, active = workspace

    class InheritedWorkspaceManager(manager.__class__):
        pass

    inherited = InheritedWorkspaceManager()
    install_workspace_routing(inherited, Pools(events, active).private_pool)

    assert inherited._ensure_workspace_size(64).size == 64


def test_workspace_routing_rejects_subclass_override(workspace) -> None:
    manager, events, active = workspace

    class OverriddenWorkspaceManager(manager.__class__):
        def _ensure_workspace_size(self, required_bytes):
            return required_bytes

    with pytest.raises(GMSError, match="not the supported allocation hook"):
        install_workspace_routing(
            OverriddenWorkspaceManager(), Pools(events, active).private_pool
        )


def test_workspace_routing_rejects_wraps_wrapper(workspace) -> None:
    manager, events, active = workspace
    base_hook = manager.__class__._ensure_workspace_size

    @wraps(base_hook)
    def wrapped(self, required_bytes):
        return base_hook(self, required_bytes)

    manager._ensure_workspace_size = MethodType(wrapped, manager)
    with pytest.raises(GMSError, match="not the supported allocation hook"):
        install_workspace_routing(manager, Pools(events, active).private_pool)

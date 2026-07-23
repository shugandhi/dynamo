# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
]


@pytest.fixture(scope="module")
def vllm_modules():
    pytest.importorskip("vllm.device_allocator.sleep_mode_backend")
    pytest.importorskip("vllm.v1.worker.gpu_worker")
    pytest.importorskip("vllm.v1.worker.workspace")
    backend = importlib.import_module("gpu_memory_service.v1.integrations.vllm.backend")
    worker = importlib.import_module("gpu_memory_service.v1.integrations.vllm.worker")
    return backend, worker


def test_worker_routes_native_vllm_pools_and_installs_workspace_hook(
    vllm_modules,
    monkeypatch,
) -> None:
    backend, worker_module = vllm_modules
    events = []
    workspace = object()
    client = SimpleNamespace(close=lambda: events.append("client_close"))
    manager = object()
    parameter_context = nullcontext()
    private_context = nullcontext()
    pools = SimpleNamespace(
        parameter_pool=lambda: parameter_context,
        private_pool=lambda: private_context,
    )
    runtime = SimpleNamespace(manager=manager, pools=pools)

    def upstream_init(instance) -> None:
        events.append("upstream_init")
        instance.device = SimpleNamespace(index=3)

    def socket_path(device, tag):
        events.append(("socket", device, tag))
        return "/run/gms/snapshot-v1.sock"

    monkeypatch.setattr(worker_module.Worker, "init_device", upstream_init)
    monkeypatch.setattr(worker_module, "current_workspace_manager", lambda: workspace)
    monkeypatch.setattr(worker_module, "get_socket_path", socket_path)
    monkeypatch.setattr(
        worker_module,
        "AllocationClient",
        lambda path: events.append(("client", path)) or client,
    )
    monkeypatch.setattr(worker_module, "get_vmm", lambda: "vmm")
    monkeypatch.setattr(
        worker_module,
        "SnapshotMemoryManager",
        lambda received_client, vmm, device: (
            events.append(("manager", received_client, vmm, device)) or manager
        ),
    )
    monkeypatch.setattr(worker_module, "SnapshotTorchPools", lambda received: pools)
    monkeypatch.setattr(
        worker_module,
        "install_vllm_integration",
        lambda received_workspace, received_pools: events.append(
            ("workspace_hook", received_workspace, received_pools)
        ),
    )
    monkeypatch.setattr(
        worker_module,
        "install_runtime",
        lambda received_manager, received_pools: (
            events.append(("runtime", received_manager, received_pools)) or runtime
        ),
    )

    worker = object.__new__(worker_module.GMSV1Worker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_sleep_mode=True,
            sleep_mode_backend="cumem",
        )
    )
    worker.init_device()

    assert worker.vllm_config.model_config.sleep_mode_backend == backend.BACKEND_NAME
    assert worker._maybe_get_memory_pool_context("weights") is parameter_context
    assert worker._maybe_get_memory_pool_context("kv_cache") is private_context
    assert events == [
        "upstream_init",
        ("socket", 3, "snapshot-v1"),
        ("client", "/run/gms/snapshot-v1.sock"),
        ("manager", client, "vmm", 3),
        ("workspace_hook", workspace, pools),
        ("runtime", manager, pools),
    ]


def test_backend_orders_v1_lifecycle(vllm_modules, monkeypatch) -> None:
    backend, _worker_module = vllm_modules
    events = []
    runtime = SimpleNamespace(
        pools=SimpleNamespace(prepare_snapshot=lambda: events.append("v1_sleep")),
        manager=SimpleNamespace(wake=lambda: events.append("v1_wake")),
    )
    monkeypatch.setattr(backend, "current_runtime", lambda: runtime)

    instance = backend.GMSV1SleepModeBackend()
    instance.suspend()
    instance.resume()

    assert events == ["v1_sleep", "v1_wake"]
    assert instance.state() == "RUNNING"


def test_worker_rejects_partial_lifecycle_and_exits_on_transition_failure(
    vllm_modules,
    monkeypatch,
) -> None:
    _backend, worker_module = vllm_modules
    events = []

    def fail_sleep(_instance, level=1):
        events.append(("sleep", level))
        raise RuntimeError("partial suspend")

    def fail_wake(_instance, tags=None):
        events.append(("wake_up", tags))
        raise RuntimeError("partial resume")

    monkeypatch.setattr(worker_module.Worker, "sleep", fail_sleep)
    monkeypatch.setattr(worker_module.Worker, "wake_up", fail_wake)
    worker = object.__new__(worker_module.GMSV1Worker)

    with pytest.raises(ValueError, match="whole-engine"):
        worker.sleep(2)
    with pytest.raises(ValueError, match="partial-tag"):
        worker.wake_up(["weights"])
    assert events == []

    with pytest.raises(SystemExit, match="1"):
        worker.sleep()
    with pytest.raises(SystemExit, match="1"):
        worker.wake_up()
    assert events == [("sleep", 1), ("wake_up", None)]


def test_vllm_memory_accounting_adds_only_parameter_cache(vllm_modules) -> None:
    _backend, worker_module = vllm_modules
    pools = SimpleNamespace(
        parameter=SimpleNamespace(
            snapshot=lambda **kwargs: [
                {"total_size": 100, "allocated_size": 80},
                {"total_size": 200, "allocated_size": 150},
            ]
        ),
        private=object(),
    )

    assert worker_module._vllm_model_memory_bytes(pools, 350) == 420

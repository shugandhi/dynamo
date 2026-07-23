# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
from typing import cast

import pytest
from _v1_fakes import V1FakeVMM
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.v1.client.memory_manager import (
    AccessClass,
    SnapshotMemoryManager,
)
from gpu_memory_service.v1.client.rpc import AllocationClient
from gpu_memory_service.v1.errors import GMSError
from gpu_memory_service.v1.errors import FatalGMSError
from gpu_memory_service.v1.server.allocations import AllocationStore

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _manager(monkeypatch):
    vmm = V1FakeVMM()
    store = AllocationStore("GPU-0", vmm, 3)
    monkeypatch.setattr(SnapshotMemoryManager, "_gpu_identity", lambda self: "GPU-0")
    service = cast(AllocationClient, store)
    return vmm, store, SnapshotMemoryManager(service, vmm, 3)


def test_sleep_wake_and_retire_ownership(monkeypatch) -> None:
    vmm, store, manager = _manager(monkeypatch)
    parameter_va = manager.allocate_parameter(65)
    private_va = manager.allocate_private(33)
    before = {mapping.base: mapping for mapping in manager.mappings}
    parameter_handle, private_handle = [
        event[3] for event in vmm.events if event[0] == "create"
    ]

    manager.sleep()

    assert set(vmm.reservations) == {parameter_va, private_va}
    assert not vmm.mapped
    assert parameter_handle in vmm.server_handles
    assert private_handle not in vmm.server_handles

    manager.wake()
    after = {mapping.base: mapping for mapping in manager.mappings}

    assert after[parameter_va].allocation_id == before[parameter_va].allocation_id
    assert after[private_va].allocation_id != before[private_va].allocation_id
    assert after[parameter_va].access is AccessClass.PARAMETER_RO
    assert after[private_va].access is AccessClass.PRIVATE_RW
    assert vmm.access[parameter_va] is GrantedLockType.RO
    assert vmm.access[private_va] is GrantedLockType.RW

    manager.retire()

    assert not vmm.server_handles
    assert not vmm.imports
    assert not vmm.reservations
    for mapping in after.values():
        with pytest.raises(Exception, match="unknown allocation"):
            store.export(mapping.allocation_id)


@pytest.mark.parametrize("failure", ["ro_transition", "sleep_unmap", "wake_map"])
def test_lifecycle_failures_are_atomic_and_fatal(monkeypatch, failure) -> None:
    vmm, store, manager = _manager(monkeypatch)
    first = manager.allocate_parameter(64)
    second = manager.allocate_parameter(64)
    private = manager.allocate_private(64)

    if failure == "ro_transition":
        vmm.fail_access_call = vmm.access_calls + 2
    elif failure == "sleep_unmap":
        vmm.fail_unmap.add(second)
    else:
        manager.sleep()
        vmm.fail_map_call = vmm.map_calls + 3

    with pytest.raises(FatalGMSError) as first_failure:
        manager.wake() if failure == "wake_map" else manager.sleep()
    with pytest.raises(FatalGMSError) as replay:
        manager.retire()

    assert replay.value is first_failure.value
    attempted_unmaps = {event[1] for event in vmm.events if event[0] == "unmap"}
    assert {first, second, private} <= attempted_unmaps
    assert not vmm.server_handles
    assert not vmm.imports
    assert not vmm.reservations


def test_uncertain_wake_rollback_retains_exact_ownership(monkeypatch) -> None:
    vmm, store, manager = _manager(monkeypatch)
    manager.allocate_parameter(64)
    private_va = manager.allocate_private(64)
    manager.sleep()
    vmm.fail_map_call = vmm.map_calls + 2
    failed_import = vmm.next_import + 1
    vmm.fail_release[failed_import] = 10

    with pytest.raises(FatalGMSError, match="wake failed"):
        manager.wake()

    private = next(
        mapping
        for mapping in manager.mappings
        if mapping.access is AccessClass.PRIVATE_RW
    )
    assert private.base == private_va
    assert failed_import in vmm.imports
    fd = store.export(private.allocation_id)
    os.close(fd)
    # Independent parameter cleanup still ran.
    parameter = next(
        mapping
        for mapping in manager.mappings
        if mapping.access is AccessClass.PARAMETER_RO
    )
    with pytest.raises(Exception, match="unknown allocation"):
        store.export(parameter.allocation_id)


def test_allocation_cannot_commit_during_snapshot_transition(monkeypatch) -> None:
    vmm, _store, manager = _manager(monkeypatch)
    manager.allocate_parameter(64)
    manager.begin_snapshot_preparation()

    synchronize_entered = threading.Event()
    finish_synchronize = threading.Event()
    original_synchronize = vmm.synchronize

    def blocked_synchronize():
        synchronize_entered.set()
        assert finish_synchronize.wait(10)
        original_synchronize()

    vmm.synchronize = blocked_synchronize
    failures: list[Exception] = []

    def sleep():
        try:
            manager.sleep()
        except Exception as exc:
            failures.append(exc)

    def allocate():
        try:
            manager.allocate_private(64)
        except Exception as exc:
            failures.append(exc)

    sleeping = threading.Thread(target=sleep)
    sleeping.start()
    assert synchronize_entered.wait(10)
    allocating = threading.Thread(target=allocate)
    allocating.start()
    finish_synchronize.set()
    sleeping.join(timeout=10)
    allocating.join(timeout=10)

    assert not sleeping.is_alive()
    assert not allocating.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], GMSError)
    assert "not awake" in str(failures[0])
    assert len(manager.mappings) == 1
    assert not vmm.mapped
    assert not vmm.imports

    manager.wake()
    assert set(vmm.mapped) == {manager.mappings[0].base}

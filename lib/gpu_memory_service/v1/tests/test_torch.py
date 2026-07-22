# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from gpu_memory_service.v1.client import AllocationPools, Manager
from gpu_memory_service.v1.errors import GMSError
from gpu_memory_service.v1.protocol import AccessClass, Allocation, Generation, Mapping
from gpu_memory_service.v1.registry import Registry
from gpu_memory_service.v1.tests.fakes import VMM
from gpu_memory_service.v1.torch import (
    TorchPools,
    discover_parameter_mappings,
    validate_buffer_mappings,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]

GENERATION = Generation("generation", "GPU-0")


class Storage:
    def __init__(self, identity, pointer, size):
        self._cdata = identity
        self.pointer = pointer
        self.size = size

    def data_ptr(self):
        return self.pointer

    def nbytes(self):
        return self.size


class Parameter:
    device = SimpleNamespace(type="cuda")

    def __init__(self, storage):
        self.storage = storage

    def untyped_storage(self):
        return self.storage


class Model:
    def __init__(self, *parameters, buffers=()):
        self._parameters = parameters
        self._buffers = buffers

    def parameters(self):
        return iter(self._parameters)

    def named_buffers(self, *, remove_duplicate):
        assert remove_duplicate is False
        return iter(self._buffers)


class CallbackOnlyManager:
    device = 0
    _mappings: dict[int, Mapping] = {}

    def allocate_parameter(self, size: int) -> int:
        raise AssertionError

    def allocate_private(self, size: int) -> int:
        raise AssertionError

    def free(self, base: int) -> None:
        raise AssertionError


def mapping(
    base=0x1000,
    size=256,
    allocation_id="allocation",
    access=AccessClass.PARAMETER_RO,
):
    return Mapping(
        Allocation(GENERATION, allocation_id, size, size, access),
        base,
        size,
    )


def test_tied_parameters_dedupe_storage_without_mutation() -> None:
    storage = Storage(7, 0x1040, 64)
    first, second = Parameter(storage), Parameter(storage)
    record = mapping()

    found = discover_parameter_mappings(Model(first, second), (record,))

    assert found == (record,)
    assert first.storage is storage and second.storage is storage


def test_distinct_parameter_storages_in_one_mapping_dedupe_mapping() -> None:
    record = mapping()
    model = Model(
        Parameter(Storage(1, 0x1010, 32)),
        Parameter(Storage(2, 0x1080, 64)),
    )

    assert discover_parameter_mappings(model, (record,)) == (record,)


def test_duplicate_mapping_records_are_canonicalized() -> None:
    record = mapping()
    model = Model(Parameter(Storage(1, 0x1040, 32)))

    assert discover_parameter_mappings(model, (record, record)) == (record,)


def test_complete_containment_and_parameter_domain_are_required() -> None:
    crossing = Model(Parameter(Storage(1, 0x10F0, 32)))
    with pytest.raises(GMSError, match="crosses"):
        discover_parameter_mappings(
            crossing,
            (mapping(size=256), mapping(base=0x1100, allocation_id="next")),
        )

    private = mapping(access=AccessClass.PRIVATE_RW)
    with pytest.raises(GMSError, match="parameter domain"):
        discover_parameter_mappings(
            Model(Parameter(Storage(1, 0x1040, 32))), (private,)
        )


def test_unexplained_live_parameter_allocation_fails_capture() -> None:
    with pytest.raises(GMSError, match="no registered Parameter"):
        discover_parameter_mappings(
            Model(Parameter(Storage(1, 0x1040, 32))),
            (mapping(), mapping(base=0x2000, allocation_id="extra")),
        )


def test_qwen_inv_freq_buffer_must_not_overlap_parameter_backing() -> None:
    parameter = mapping(base=0x1000, size=0x100)
    inv_freq = Parameter(Storage(2, 0x1080, 0x20))
    model = Model(buffers=(("rotary_emb.inv_freq", inv_freq),))

    with pytest.raises(GMSError, match="inv_freq.*overlaps parameter backing"):
        validate_buffer_mappings(model, (parameter,))


def test_aliased_buffer_storage_is_allowed_outside_parameter_backing() -> None:
    storage = Storage(2, 0x2080, 0x20)
    inv_freq = Parameter(storage)
    alias = Parameter(storage)
    model = Model(
        buffers=(
            ("rotary_emb.inv_freq", inv_freq),
            ("rotary_emb.inv_freq_alias", alias),
        )
    )

    validate_buffer_mappings(model, (mapping(base=0x1000, size=0x100),))


def test_partially_overlapping_buffer_storage_is_rejected() -> None:
    parameter = mapping(base=0x1000, size=0x100)
    model = Model(
        buffers=(("rotary_emb.inv_freq", Parameter(Storage(2, 0x0FF0, 0x20))),)
    )

    with pytest.raises(GMSError, match="overlaps parameter backing"):
        validate_buffer_mappings(model, (parameter,))


def test_allocator_free_callback_failure_is_latched() -> None:
    pools = AllocationPools(CallbackOnlyManager())
    pools.free(0x1000, 64, 0, 0)

    assert pools.failure is not None


def test_callback_only_manager_cannot_construct_torch_capture_pools() -> None:
    with pytest.raises(TypeError, match="capture-capable V1 Manager"):
        getattr(sys.modules[TorchPools.__module__], "TorchPools")(CallbackOnlyManager())


def test_capture_manager_constructs_torch_pools_and_prepares_capture(
    monkeypatch,
) -> None:
    from gpu_memory_service.client.torch import extensions

    class AllocatorExtension:
        __file__ = "allocator.so"

        @staticmethod
        def init_module(malloc, free):
            assert callable(malloc)
            assert callable(free)

    class PluggableAllocator:
        def __init__(self, path, malloc, free):
            assert (path, malloc, free) == (
                "allocator.so",
                "my_malloc",
                "my_free",
            )

        def allocator(self):
            return object()

    class Cuda:
        CUDAPluggableAllocator = PluggableAllocator

        @staticmethod
        def device(device):
            assert device == 3
            return nullcontext()

        @staticmethod
        def MemPool(*, allocator):
            assert allocator is not None
            return object()

        @staticmethod
        def empty_cache():
            return None

    vmm = VMM()
    registry = Registry("GPU-0", vmm, 3)
    monkeypatch.setattr(Manager, "_gpu_identity", lambda self: "GPU-0")
    manager = Manager(
        registry,
        vmm,
        3,
        artifact_id="artifact",
        generation_id="generation",
    )
    base = manager.allocate_parameter(64)
    record = manager.mappings[0]
    monkeypatch.setattr(extensions, "_allocator_ext", AllocatorExtension)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=Cuda))

    pools = TorchPools(manager)
    parameters = pools.prepare_capture(
        Model(Parameter(Storage(1, base, record.allocation.requested_size)))
    )

    assert pools._manager is manager
    assert pools._allocator.manager is manager
    assert parameters == (record,)
    assert manager._sealed is True
    assert manager._imports == {}

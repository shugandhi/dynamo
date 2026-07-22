# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
from gpu_memory_service.v1.protocol import AccessClass, Allocation, Generation, Mapping
from gpu_memory_service.v1.qwen_e2e import (
    containing_mapping,
    identity_digest,
    mapping_evidence,
    move_parameters,
    move_registered_buffers,
    restore_evidence,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class Storage:
    _cdata = 31


class Parameter:
    _cdata = 23

    def untyped_storage(self):
        return Storage()

    def data_ptr(self):
        return 0x1234


class Buffer:
    _cdata = 41

    def __init__(self, storage=None):
        self.storage = storage or Storage()
        self.moves = []

    def untyped_storage(self):
        return self.storage

    def data_ptr(self):
        return 0x5678

    def to(self, device):
        self.moves.append(device)
        return Buffer(self.storage)


class Module:
    def __init__(self, **buffers):
        self._buffers = buffers


class Model:
    def __init__(self):
        self.parameter = Parameter()
        self.rotary = Module(inv_freq=Buffer())
        self.alias = Module(inv_freq_alias=self.rotary._buffers["inv_freq"])
        self._buffers = {}
        self.parameter_move_buffer_values = None

    def named_parameters(self, *, remove_duplicate):
        assert remove_duplicate is False
        return iter((("weight", self.parameter),))

    def named_buffers(self, *, remove_duplicate):
        assert remove_duplicate is False
        return iter(
            (
                ("rotary.inv_freq", self.rotary._buffers["inv_freq"]),
                ("alias.inv_freq_alias", self.alias._buffers["inv_freq_alias"]),
            )
        )

    def modules(self):
        return iter((self, self.rotary, self.alias))

    def to(self, device):
        self.parameter_move_buffer_values = [
            value for module in self.modules() for value in module._buffers.values()
        ]
        assert device == "cuda:0"
        return self


def _mapping(
    allocation_id: str,
    base: int,
    access: AccessClass,
) -> Mapping:
    generation = Generation("generation", "gpu")
    return Mapping(
        Allocation(generation, allocation_id, 64, 64, access),
        base,
        64,
    )


def test_identity_digest_covers_every_identity_layer() -> None:
    model = Model()
    before = identity_digest(model)

    model.parameter._cdata = 24
    after = identity_digest(model)

    assert before["algorithm"] == "sha256"
    assert before["model"] == id(model)
    assert before["parameter_bindings"] == 1
    assert before["buffer_bindings"] == 2
    assert before["digest"] != after["digest"]


def test_identity_digest_includes_registered_buffers() -> None:
    model = Model()
    before = identity_digest(model)

    model.rotary._buffers["inv_freq"]._cdata = 42
    after = identity_digest(model)

    assert before["digest"] != after["digest"]


def test_qwen_style_buffer_move_deduplicates_aliases_outside_parameter_move() -> None:
    model = Model()
    original = model.rotary._buffers["inv_freq"]

    move_registered_buffers(model, "cuda:0")
    moved = model.rotary._buffers["inv_freq"]
    move_parameters(model, "cuda:0")

    assert original.moves == ["cuda:0"]
    assert moved is model.alias._buffers["inv_freq_alias"]
    assert moved is not original
    assert model.parameter_move_buffer_values == [None, None]
    assert model.rotary._buffers["inv_freq"] is moved
    assert model.alias._buffers["inv_freq_alias"] is moved


def test_mapping_helpers_are_sorted_and_require_one_owner() -> None:
    private = _mapping("private", 0x2000, AccessClass.PRIVATE_RW)
    parameter = _mapping("parameter", 0x1000, AccessClass.PARAMETER_RO)

    assert mapping_evidence((private, parameter)) == [
        {
            "allocation_id": "parameter",
            "base": 0x1000,
            "size": 64,
            "reservation_size": 64,
            "access": AccessClass.PARAMETER_RO.value,
        },
        {
            "allocation_id": "private",
            "base": 0x2000,
            "size": 64,
            "reservation_size": 64,
            "access": AccessClass.PRIVATE_RW.value,
        },
    ]
    assert containing_mapping((parameter, private), 0x2010, AccessClass.PRIVATE_RW) == (
        private
    )
    with pytest.raises(RuntimeError, match="has 0 private_rw mappings"):
        containing_mapping((parameter,), 0x2010, AccessClass.PRIVATE_RW)


def test_restore_evidence_checks_fresh_backing_at_stable_va() -> None:
    identity = {"digest": "identity"}
    parameters = [{"allocation_id": "parameter", "base": 0x1000}]
    private_identity = {"tensor": 1, "tensor_impl": 2, "storage_impl": 3, "data_ptr": 4}
    old_private = {
        "allocation_id": "old",
        "base": 0x2000,
        "size": 64,
        "reservation_size": 64,
        "access": AccessClass.PRIVATE_RW.value,
    }
    capture = {
        "tokens": [1, 2],
        "output": "result",
        "identity": identity,
        "parameter_mappings": parameters,
        "private_mapping": old_private,
        "private_identity": private_identity,
        "gpu": "gpu",
        "server_process": (10, 20),
    }
    new_private = dict(old_private, allocation_id="new")

    evidence = restore_evidence(
        capture,
        tokens=[1, 2],
        output="result",
        identity=identity,
        parameters=parameters,
        private=new_private,
        private_identity=private_identity,
        private_write=True,
        gpu="gpu",
        server_process=(10, 20),
    )

    checks = {
        key: value
        for key, value in evidence.items()
        if key.endswith("_equal")
        or key
        in {
            "private_backing_fresh",
            "private_va_equal",
            "private_write",
            "same_gpu",
        }
    }
    assert all(checks.values()), SimpleNamespace(**evidence)

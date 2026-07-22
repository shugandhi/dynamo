# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from gpu_memory_service.v1.tests.test_deployment import (
    _checkpoint_cleanup_pod,
    _evidence,
    _render,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_render_uses_image_pull_secret() -> None:
    manifests = _render(
        name="gms-v1-test",
        namespace="test",
        node="gpu-node",
        image="registry.example/gms:test",
        image_pull_secret="registry-secret",
        checkpoint_pvc="snapshot-pvc",
        checkpoint_path="/checkpoints",
        device_class="gpu.nvidia.com",
        runtime_class="nvidia",
    )

    assert manifests[1]["spec"]["imagePullSecrets"] == [{"name": "registry-secret"}]


def test_qwen_render_uses_model_cache_and_offline_standard_loader() -> None:
    manifests = _render(
        name="gms-v1-qwen",
        namespace="test",
        node="gpu-node",
        image="registry.example/gms:test",
        image_pull_secret="registry-secret",
        checkpoint_pvc="snapshot-pvc",
        checkpoint_path="/checkpoints",
        device_class="gpu.example.com",
        runtime_class="nvidia",
        harness="qwen",
        model_cache_pvc="model-cache-pvc",
        model_cache_path="/models",
        model="Qwen/Qwen3-0.6B",
    )

    claim, pod, _snapshot = manifests
    engine = next(
        container
        for container in pod["spec"]["containers"]
        if container["name"] == "engine"
    )
    env = {item["name"]: item["value"] for item in engine["env"]}
    volumes = {item["name"]: item for item in pod["spec"]["volumes"]}

    assert (
        claim["spec"]["spec"]["devices"]["requests"][0]["exactly"]["deviceClassName"]
        == "gpu.example.com"
    )
    assert pod["metadata"]["annotations"]["nvidia.com/snapshot-target-containers"] == (
        "engine"
    )
    assert engine["command"] == [
        "python3",
        "-m",
        "gpu_memory_service.v1.qwen_e2e",
    ]
    assert env == {
        "DYN_SNAPSHOT_CONTROL_DIR": "/snapshot-control",
        "GMS_V1_MODEL": "Qwen/Qwen3-0.6B",
        "HF_HOME": "/models",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    assert volumes["model-cache"]["persistentVolumeClaim"]["claimName"] == (
        "model-cache-pvc"
    )


def test_render_rejects_unknown_harness() -> None:
    with pytest.raises(ValueError, match="unknown GMS V1 harness"):
        _render(
            name="gms-v1-test",
            namespace="test",
            node="gpu-node",
            image="registry.example/gms:test",
            image_pull_secret="registry-secret",
            checkpoint_pvc="snapshot-pvc",
            checkpoint_path="/checkpoints",
            device_class="gpu.nvidia.com",
            runtime_class="nvidia",
            harness="unknown",
        )


def test_evidence_accepts_identical_duplicates_and_rejects_conflicts() -> None:
    first = 'GMS_V1_EVIDENCE {"phase":"restore","output":[1]}'
    assert _evidence(first + "\n" + first, "restore") == {
        "phase": "restore",
        "output": [1],
    }

    with pytest.raises(AssertionError, match="conflicting restore evidence"):
        _evidence(
            first + '\nGMS_V1_EVIDENCE {"phase":"restore","output":[2]}',
            "restore",
        )


def test_evidence_rejects_malformed_payload() -> None:
    with pytest.raises(AssertionError, match="malformed GMS V1 evidence"):
        _evidence("GMS_V1_EVIDENCE {", "restore")


def test_checkpoint_cleanup_pod_uses_image_pull_secret() -> None:
    pod = _checkpoint_cleanup_pod(
        name="gms-v1-test",
        namespace="test",
        node="gpu-node",
        image="registry.example/gms:test",
        image_pull_secret="registry-secret",
        checkpoint_pvc="snapshot-pvc",
        checkpoint_path="/checkpoints",
    )

    assert pod["spec"]["imagePullSecrets"] == [{"name": "registry-secret"}]

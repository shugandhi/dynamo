# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("gms-v1-snapshot")
    group.addoption("--gms-v1-harness", choices=("toy", "qwen"), default="toy")
    group.addoption("--gms-v1-kube-context")
    group.addoption("--gms-v1-namespace")
    group.addoption("--gms-v1-node")
    group.addoption("--gms-v1-image")
    group.addoption("--gms-v1-image-pull-secret")
    group.addoption("--gms-v1-checkpoint-pvc")
    group.addoption("--gms-v1-checkpoint-path", default="/checkpoints")
    group.addoption("--gms-v1-device-class", default="gpu.nvidia.com")
    group.addoption("--gms-v1-runtime-class", default="nvidia")
    group.addoption("--gms-v1-model-cache-pvc")
    group.addoption("--gms-v1-model-cache-path", default="/model-cache")
    group.addoption("--gms-v1-model", default="Qwen/Qwen3-0.6B")

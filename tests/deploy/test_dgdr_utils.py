# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoGraphDeploymentRequest lifecycle helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.deploy.dgdr_utils import DGDRCleanupError, DGDRTestConfig, ManagedDGDR

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def initialized_manager() -> ManagedDGDR:
    manager = ManagedDGDR(DGDRTestConfig(namespace="test-namespace", image="test"))
    manager.custom = MagicMock()
    manager.core = MagicMock()
    manager.batch = MagicMock()
    manager.apiextensions = MagicMock()
    return manager


async def test_cleanup_reports_all_failures_and_retains_failed_names() -> None:
    manager = initialized_manager()
    manager._created_names = ["first", "second", "third"]
    calls = []

    async def cleanup_name(name: str, failed: bool) -> None:
        calls.append((name, failed))
        if name != "second":
            raise RuntimeError(f"could not delete {name}")

    manager._cleanup_name = AsyncMock(side_effect=cleanup_name)

    with pytest.raises(DGDRCleanupError) as error:
        await manager.cleanup(failed=True)

    assert calls == [("third", True), ("second", True), ("first", True)]
    assert [name for name, _ in error.value.failures] == ["third", "first"]
    assert manager._created_names == ["first", "third"]


async def test_profiling_failure_diagnostics_include_job_and_pod_logs() -> None:
    manager = initialized_manager()
    assert manager.batch is not None
    assert manager.core is not None

    job = MagicMock()
    job.to_str.return_value = "profiling job"
    manager.batch.read_namespaced_job = AsyncMock(return_value=job)

    pod = MagicMock()
    pod.metadata.name = "profiling-pod"
    pod.spec.init_containers = []
    pod.spec.containers = [SimpleNamespace(name="profiler")]
    pod.to_str.return_value = "profiling pod"
    manager.core.list_namespaced_pod = AsyncMock(
        return_value=SimpleNamespace(items=[pod])
    )
    manager.core.read_namespaced_pod_log = AsyncMock(return_value="profiler logs")

    await manager._log_diagnostics({"status": {"profilingJobName": "profiling-job"}})

    manager.batch.read_namespaced_job.assert_awaited_once_with(
        "profiling-job", "test-namespace"
    )
    manager.core.list_namespaced_pod.assert_awaited_once_with(
        "test-namespace", label_selector="job-name=profiling-job"
    )
    manager.core.read_namespaced_pod_log.assert_awaited_once_with(
        "profiling-pod",
        "test-namespace",
        container="profiler",
        tail_lines=300,
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live DRA + Dynamo Snapshot deployment test for GMS V1."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import pytest
import yaml

pytestmark = [
    pytest.mark.dynamocheckpoint,
    pytest.mark.k8s,
    pytest.mark.deploy,
    pytest.mark.post_merge,
    pytest.mark.e2e,
    pytest.mark.gpu_1,
    pytest.mark.timeout(1800),
]

_SNAPSHOT = ("nvidia.com", "v1alpha1", "podsnapshots")
_RESOURCE_CLAIM_TEMPLATE = (
    "resource.k8s.io",
    "v1",
    "resourceclaimtemplates",
)
_CLEANUP_TIMEOUT = 180


async def _wait(description: str, fetch, predicate, timeout: int = 600):
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = await fetch()
        if predicate(last):
            return last
        await asyncio.sleep(2)
    raise AssertionError(f"timed out waiting for {description}; last={last!r}")


def _status(pod: Any, container: str):
    for status in pod.status.container_statuses or []:
        if status.name == container:
            return status
    raise AssertionError(f"container status {container!r} not found")


def _evidence(log: str, phase: str) -> dict[str, object]:
    prefix = "GMS_V1_EVIDENCE "
    found: list[dict[str, object]] = []
    for line in log.splitlines():
        if line.startswith(prefix):
            try:
                value = json.loads(line.removeprefix(prefix))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"malformed GMS V1 evidence line: {line}") from exc
            if not isinstance(value, dict):
                raise AssertionError(f"GMS V1 evidence is not an object: {value!r}")
            if value.get("phase") == phase:
                found.append(value)
    if not found:
        raise AssertionError(f"{phase} evidence not found in engine log:\n{log}")
    if any(value != found[0] for value in found[1:]):
        raise AssertionError(f"conflicting {phase} evidence in engine log:\n{log}")
    return found[0]


def _snapshot_ready(value: dict[str, Any]) -> bool:
    conditions = value.get("status", {}).get("conditions", [])
    for condition in conditions:
        if condition["type"] == "Failed" and condition["status"] == "True":
            raise AssertionError(f"PodSnapshot failed: {condition}")
        if condition["type"] == "Ready" and condition["status"] == "True":
            return True
    return False


async def _create_resource_claim_template(
    custom: Any, namespace: str, body: dict[str, Any]
) -> object:
    return await custom.create_namespaced_custom_object(
        group=_RESOURCE_CLAIM_TEMPLATE[0],
        version=_RESOURCE_CLAIM_TEMPLATE[1],
        namespace=namespace,
        plural=_RESOURCE_CLAIM_TEMPLATE[2],
        body=body,
    )


async def _get_resource_claim_template(
    custom: Any, namespace: str, name: str
) -> object:
    return await custom.get_namespaced_custom_object(
        group=_RESOURCE_CLAIM_TEMPLATE[0],
        version=_RESOURCE_CLAIM_TEMPLATE[1],
        namespace=namespace,
        plural=_RESOURCE_CLAIM_TEMPLATE[2],
        name=name,
    )


async def _delete_resource_claim_template(
    custom: Any, namespace: str, name: str
) -> object:
    return await custom.delete_namespaced_custom_object(
        group=_RESOURCE_CLAIM_TEMPLATE[0],
        version=_RESOURCE_CLAIM_TEMPLATE[1],
        namespace=namespace,
        plural=_RESOURCE_CLAIM_TEMPLATE[2],
        name=name,
    )


def _checkpoint_cleanup_pod(
    name: str,
    namespace: str,
    node: str,
    image: str,
    image_pull_secret: str,
    checkpoint_pvc: str,
    checkpoint_path: str,
) -> dict[str, Any]:
    base = PurePosixPath(checkpoint_path)
    if not base.is_absolute() or ".." in base.parts:
        raise ValueError("checkpoint path must be absolute without parent traversal")
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name + "-checkpoint-cleanup",
            "namespace": namespace,
            "labels": {"app.kubernetes.io/name": "gms-v1-snapshot-e2e-cleanup"},
        },
        "spec": {
            "activeDeadlineSeconds": 120,
            "restartPolicy": "Never",
            "imagePullSecrets": [{"name": image_pull_secret}],
            "nodeSelector": {"kubernetes.io/hostname": node},
            "tolerations": [
                {
                    "key": "nvidia.com/gpu",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                },
                {
                    "key": "dedicated",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                },
            ],
            "containers": [
                {
                    "name": "cleanup",
                    "image": image,
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["/bin/rm"],
                    "args": ["-rf", "--", str(base / name)],
                    "volumeMounts": [
                        {
                            "name": "checkpoint-storage",
                            "mountPath": str(base),
                        }
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "checkpoint-storage",
                    "persistentVolumeClaim": {"claimName": checkpoint_pvc},
                }
            ],
        },
    }


async def _checkpoint_cleanup(
    core: Any,
    exceptions: Any,
    body: dict[str, Any],
) -> None:
    name = body["metadata"]["name"]
    namespace = body["metadata"]["namespace"]
    failure: Exception | None = None
    try:
        await core.create_namespaced_pod(namespace, body)

        async def read_cleanup_pod():
            return await core.read_namespaced_pod(name, namespace)

        pod = await _wait(
            "checkpoint cleanup Pod completion",
            read_cleanup_pod,
            lambda value: value.status.phase in {"Succeeded", "Failed"},
            timeout=_CLEANUP_TIMEOUT,
        )
        if pod.status.phase != "Succeeded":
            try:
                log = await core.read_namespaced_pod_log(
                    name, namespace, container="cleanup"
                )
            except Exception as exc:
                log = f"<cleanup log unavailable: {exc}>"
            raise AssertionError(
                f"checkpoint cleanup Pod ended in {pod.status.phase}: {log}"
            )
    except Exception as exc:
        failure = exc
    finally:
        try:
            try:
                await core.delete_namespaced_pod(
                    name,
                    namespace,
                    grace_period_seconds=0,
                )
            except exceptions.ApiException as exc:
                if exc.status != 404:
                    raise

            async def cleanup_pod_deleted():
                try:
                    return await core.read_namespaced_pod(name, namespace)
                except exceptions.ApiException as exc:
                    if exc.status == 404:
                        return None
                    raise

            await _wait(
                "checkpoint cleanup Pod deletion",
                cleanup_pod_deleted,
                lambda value: value is None,
                timeout=_CLEANUP_TIMEOUT,
            )
        except Exception as exc:
            if failure is None:
                failure = exc
            else:
                failure = RuntimeError(
                    f"{failure}; cleanup Pod deletion also failed: {exc}"
                )
    if failure is not None:
        raise failure


def _render(
    name: str,
    namespace: str,
    node: str,
    image: str,
    image_pull_secret: str,
    checkpoint_pvc: str,
    checkpoint_path: str,
    device_class: str,
    runtime_class: str,
    harness: str = "toy",
    model_cache_pvc: str | None = None,
    model_cache_path: str = "/model-cache",
    model: str = "Qwen/Qwen3-0.6B",
) -> list[dict[str, Any]]:
    templates = {"toy": "snapshot.yaml", "qwen": "qwen-snapshot.yaml"}
    try:
        template = templates[harness]
    except KeyError as exc:
        raise ValueError(f"unknown GMS V1 harness {harness!r}") from exc
    text = (
        Path(__file__)
        .parents[1]
        .joinpath("deploy", template)
        .read_text(encoding="utf-8")
    )
    values = {
        "__NAME__": name,
        "__NAMESPACE__": namespace,
        "__NODE__": node,
        "__IMAGE__": image,
        "__IMAGE_PULL_SECRET__": image_pull_secret,
        "__ARTIFACT_ID__": name,
        "__CHECKPOINT_PVC__": checkpoint_pvc,
        "__CHECKPOINT_PATH__": checkpoint_path,
        "__DEVICE_CLASS__": device_class,
        "__RUNTIME_CLASS__": runtime_class,
        "__MODEL_CACHE_PVC__": model_cache_pvc or "",
        "__MODEL_CACHE_PATH__": model_cache_path,
        "__MODEL__": model,
    }
    for key, value in values.items():
        text = text.replace(key, value)
    return list(yaml.safe_load_all(text))


@pytest.mark.asyncio
async def test_gms_v1_dra_snapshot_deployment(
    request: pytest.FixtureRequest,
) -> None:
    """Capture one container, preserve its sidecar, and restore in place."""
    from kubernetes_asyncio import client, config
    from kubernetes_asyncio.client import exceptions

    options = {
        "harness": request.config.getoption("--gms-v1-harness"),
        "context": request.config.getoption("--gms-v1-kube-context"),
        "namespace": request.config.getoption("--gms-v1-namespace"),
        "node": request.config.getoption("--gms-v1-node"),
        "image": request.config.getoption("--gms-v1-image"),
        "image_pull_secret": request.config.getoption("--gms-v1-image-pull-secret"),
        "checkpoint_pvc": request.config.getoption("--gms-v1-checkpoint-pvc"),
        "checkpoint_path": request.config.getoption("--gms-v1-checkpoint-path"),
        "device_class": request.config.getoption("--gms-v1-device-class"),
        "runtime_class": request.config.getoption("--gms-v1-runtime-class"),
        "model_cache_pvc": request.config.getoption("--gms-v1-model-cache-pvc"),
        "model_cache_path": request.config.getoption("--gms-v1-model-cache-path"),
        "model": request.config.getoption("--gms-v1-model"),
    }
    required = set(options)
    if options["harness"] == "toy":
        required.remove("model_cache_pvc")
    missing = [key for key in required if not options[key]]
    if missing:
        pytest.fail(
            "explicit GMS V1 deployment options are required: " + ", ".join(missing),
            pytrace=False,
        )

    await config.load_kube_config(context=options["context"])
    api_client = client.ApiClient()
    core = client.CoreV1Api(api_client)
    custom = client.CustomObjectsApi(api_client)
    name = f"gms-v1-{uuid4().hex[:12]}"
    render_options = {key: value for key, value in options.items() if key != "context"}
    manifests = _render(name=name, **render_options)
    claim, pod_body, snapshot = manifests
    namespace = str(options["namespace"])
    cleanup_pod = _checkpoint_cleanup_pod(
        name=name,
        namespace=namespace,
        node=str(options["node"]),
        image=str(options["image"]),
        image_pull_secret=str(options["image_pull_secret"]),
        checkpoint_pvc=str(options["checkpoint_pvc"]),
        checkpoint_path=str(options["checkpoint_path"]),
    )
    claim_name = str(claim["metadata"]["name"])
    pod_created = snapshot_created = claim_created = False
    try:
        await _create_resource_claim_template(custom, namespace, claim)
        claim_created = True

        async def read_resource_claim_template():
            return await _get_resource_claim_template(custom, namespace, claim_name)

        await _wait(
            "ResourceClaimTemplate availability",
            read_resource_claim_template,
            lambda value: value is not None,
        )
        pod = await core.create_namespaced_pod(namespace, pod_body)
        pod_created = True

        async def read_pod():
            return await core.read_namespaced_pod(name, namespace)

        pod = await _wait(
            "engine capture readiness",
            read_pod,
            lambda value: any(
                condition.type == "Ready" and condition.status == "True"
                for condition in (value.status.conditions or [])
            ),
        )
        server_before = _status(pod, "gms-server")
        engine_before = _status(pod, "engine")
        snapshot_spec = snapshot["spec"]
        snapshot_spec["source"]["podRef"]["uid"] = pod.metadata.uid
        await custom.create_namespaced_custom_object(
            group=_SNAPSHOT[0],
            version=_SNAPSHOT[1],
            namespace=namespace,
            plural=_SNAPSHOT[2],
            body=snapshot,
        )
        snapshot_created = True

        async def read_snapshot():
            return await custom.get_namespaced_custom_object(
                group=_SNAPSHOT[0],
                version=_SNAPSHOT[1],
                namespace=namespace,
                plural=_SNAPSHOT[2],
                name=name,
            )

        result = await _wait(
            "PodSnapshot Ready",
            read_snapshot,
            _snapshot_ready,
        )
        failed = [
            condition
            for condition in result.get("status", {}).get("conditions", [])
            if condition["type"] == "Failed" and condition["status"] == "True"
        ]
        assert not failed

        pod = await _wait(
            "engine restart into restore standby",
            read_pod,
            lambda value: (
                _status(value, "engine").container_id != engine_before.container_id
                and _status(value, "engine").state.running is not None
            ),
        )
        server_standby = _status(pod, "gms-server")
        assert server_standby.container_id == server_before.container_id
        assert server_standby.restart_count == server_before.restart_count

        body = {
            "metadata": {
                "labels": {
                    "nvidia.com/snapshot-is-checkpoint-source": None,
                    "nvidia.com/snapshot-capture-eligible": None,
                    "nvidia.com/snapshot-is-restore-target": "true",
                }
            }
        }
        await core.patch_namespaced_pod(name, namespace, body)

        pod = await _wait(
            "in-place restore completion",
            read_pod,
            lambda value: (
                (value.metadata.annotations or {}).get(
                    "nvidia.com/snapshot-restore-status.engine"
                )
                == "completed"
            ),
        )
        pod = await _wait(
            "restored engine readiness",
            read_pod,
            lambda value: _status(value, "engine").ready is True,
        )
        server_after = _status(pod, "gms-server")
        assert server_after.container_id == server_before.container_id
        assert server_after.restart_count == server_before.restart_count

        current = await core.read_namespaced_pod_log(
            name, namespace, container="engine"
        )
        try:
            previous = await core.read_namespaced_pod_log(
                name, namespace, container="engine", previous=True
            )
        except exceptions.ApiException:
            previous = ""
        capture = _evidence(previous + "\n" + current, "capture")
        restore = _evidence(current, "restore")
        assert restore["output"] == capture["output"]
        assert restore["server_process"] == capture["server_process"]
        assert restore["output_equal"] is True
        assert restore["private_backing_fresh"] is True
        assert restore["server_process_equal"] is True
        if options["harness"] == "toy":
            assert restore["identity"] == capture["identity"]
            assert restore["identity_equal"] is True
            assert restore["parameter_allocations_equal"] is True
            assert restore["parameter_allocations"] == capture["parameter_allocations"]
            assert restore["private_allocation_before"] == capture["private_allocation"]
            assert restore["private_allocation_after"] != capture["private_allocation"]
        else:
            capture_private = capture["private_mapping"]
            restore_private = restore["private_mapping"]
            assert isinstance(capture_private, dict)
            assert isinstance(restore_private, dict)
            assert restore["tokens"] == capture["tokens"]
            assert restore["identity"] == capture["identity"]
            assert restore["private_identity"] == capture["private_identity"]
            assert restore["gpu"] == capture["gpu"]
            assert restore_private["allocation_id"] != capture_private["allocation_id"]
            assert {
                key: restore_private[key]
                for key in ("base", "size", "reservation_size")
            } == {
                key: capture_private[key]
                for key in ("base", "size", "reservation_size")
            }
            assert restore["token_equal"] is True
            assert restore["identity_digest_equal"] is True
            assert restore["parameter_mappings_equal"] is True
            assert restore["private_identity_equal"] is True
            assert restore["private_va_equal"] is True
            assert restore["private_access"] == "private_rw"
            assert restore["private_write"] is True
            assert restore["same_gpu"] is True
            assert restore["parameter_mappings"] == capture["parameter_mappings"]
    finally:
        test_failure = sys.exc_info()[1]
        cleanup_errors: list[Exception] = []
        if snapshot_created:
            try:
                await custom.delete_namespaced_custom_object(
                    group=_SNAPSHOT[0],
                    version=_SNAPSHOT[1],
                    namespace=namespace,
                    plural=_SNAPSHOT[2],
                    name=name,
                    body=client.V1DeleteOptions(),
                )

                async def snapshot_deleted():
                    try:
                        return await custom.get_namespaced_custom_object(
                            group=_SNAPSHOT[0],
                            version=_SNAPSHOT[1],
                            namespace=namespace,
                            plural=_SNAPSHOT[2],
                            name=name,
                        )
                    except exceptions.ApiException as exc:
                        if exc.status == 404:
                            return None
                        raise

                await _wait(
                    "PodSnapshot deletion",
                    snapshot_deleted,
                    lambda value: value is None,
                    timeout=_CLEANUP_TIMEOUT,
                )
            except Exception as exc:
                cleanup_errors.append(exc)
        if pod_created:
            try:
                await core.delete_namespaced_pod(
                    name,
                    namespace,
                    grace_period_seconds=0,
                    body=client.V1DeleteOptions(),
                )

                async def pod_deleted():
                    try:
                        return await core.read_namespaced_pod(name, namespace)
                    except exceptions.ApiException as exc:
                        if exc.status == 404:
                            return None
                        raise

                await _wait(
                    "deployment Pod deletion",
                    pod_deleted,
                    lambda value: value is None,
                    timeout=_CLEANUP_TIMEOUT,
                )
            except Exception as exc:
                cleanup_errors.append(exc)
        try:
            await _checkpoint_cleanup(core, exceptions, cleanup_pod)
        except Exception as exc:
            cleanup_errors.append(exc)
        if claim_created:
            try:
                await _delete_resource_claim_template(custom, namespace, claim_name)

                async def resource_claim_template_deleted():
                    try:
                        return await _get_resource_claim_template(
                            custom, namespace, claim_name
                        )
                    except exceptions.ApiException as exc:
                        if exc.status == 404:
                            return None
                        raise

                await _wait(
                    "ResourceClaimTemplate deletion",
                    resource_claim_template_deleted,
                    lambda value: value is None,
                    timeout=_CLEANUP_TIMEOUT,
                )
            except Exception as exc:
                cleanup_errors.append(exc)
        try:
            await api_client.close()
        except Exception as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            detail = "\n".join(f"{type(exc).__name__}: {exc}" for exc in cleanup_errors)
            if test_failure is None:
                raise RuntimeError(f"GMS V1 cleanup failed:\n{detail}") from (
                    cleanup_errors[0]
                )
            request.node.add_report_section("call", "GMS V1 cleanup errors", detail)

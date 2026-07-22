# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic CUDA engine for the GMS V1 Dynamo Snapshot deployment test."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.common.vmm import get_vmm

from .client import Manager
from .errors import FatalGMSError
from .protocol import AccessClass
from .rpc import RPCClient
from .torch import TorchPools


def _identity(model: object, private: object) -> dict[str, int]:
    return {
        "model": id(model),
        "weight": id(model.weight),  # type: ignore[attr-defined]
        "tied": id(model.tied),  # type: ignore[attr-defined]
        "weight_tensor_impl": int(model.weight._cdata),  # type: ignore[attr-defined]
        "tied_tensor_impl": int(model.tied._cdata),  # type: ignore[attr-defined]
        "weight_storage": int(model.weight.untyped_storage()._cdata),  # type: ignore[attr-defined]
        "tied_storage": int(model.tied.untyped_storage()._cdata),  # type: ignore[attr-defined]
        "weight_ptr": int(model.weight.data_ptr()),  # type: ignore[attr-defined]
        "tied_ptr": int(model.tied.data_ptr()),  # type: ignore[attr-defined]
        "private_ptr": int(private.data_ptr()),  # type: ignore[attr-defined]
    }


async def _standby() -> None:
    while True:
        await asyncio.sleep(3600)


async def run(
    device: int,
    socket_path: str,
    artifact_id: str,
    standby_marker: str,
) -> None:
    marker = Path(standby_marker)
    if marker.exists():
        await _standby()

    import torch

    from dynamo.common.snapshot.lifecycle import SnapshotConfig

    config = SnapshotConfig.from_env()
    if config is None:
        raise RuntimeError("DYN_SNAPSHOT_CONTROL_DIR is required")
    torch.cuda.set_device(device)
    rpc = RPCClient(socket_path)
    server_process = rpc.process_evidence()
    manager = Manager(
        rpc,
        get_vmm(),
        device,
        artifact_id=artifact_id,
    )
    pools = TorchPools(manager)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.arange(16, dtype=torch.float32, device="cuda").view(4, 4)
            )
            self.tied = torch.nn.Parameter(self.weight.view(-1), requires_grad=False)

        def forward(self, value):
            return value @ self.weight

    with pools.parameter_pool():
        model = Model()
    with pools.private_pool():
        private = torch.ones(4, device="cuda")
    before = model(private.view(1, 4)).detach().cpu().tolist()
    identity = _identity(model, private)
    parameters = pools.prepare_capture(model)
    parameter_ids = [mapping.allocation.allocation_id for mapping in parameters]
    old_private = next(
        mapping
        for mapping in manager.mappings
        if mapping.allocation.access is AccessClass.PRIVATE_RW
        and mapping.base <= identity["private_ptr"] < mapping.end
    )
    rpc_open = True
    try:
        print(
            "GMS_V1_EVIDENCE "
            + json.dumps(
                {
                    "phase": "capture",
                    "output": before,
                    "identity": identity,
                    "parameter_allocations": parameter_ids,
                    "private_allocation": old_private.allocation.allocation_id,
                    "server_process": server_process,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        # The connected stream belongs to the checkpoint source and is
        # excluded from CRIU. Reconnect only after capture/restore.
        rpc.close()
        rpc_open = False

        class _Pause:
            async def pause(self):
                return None

            async def resume(self):
                return None

            def mark_resumed(self):
                return None

        restored = await config.run_lifecycle(_Pause())
    except Exception as cause:
        failures: list[Exception] = []
        cleanup = rpc if rpc_open else None
        if cleanup is None:
            try:
                cleanup = RPCClient(socket_path)
            except Exception as exc:
                failures.append(exc)
        if cleanup is not None:
            manager.service = cleanup
            manager._abandon_capture(failures)
            cleanup.close()
        if failures:
            raise FatalGMSError(
                f"capture failed ({cause}) and cleanup failed: {failures[0]}"
            ) from cause
        raise
    if not restored:
        marker.write_text("captured", encoding="utf-8")
        os._exit(0)

    restored_rpc = RPCClient(socket_path)
    restored_server_process = restored_rpc.process_evidence()
    manager.service = restored_rpc
    manager.wake(f"reader-{artifact_id}")
    private.fill_(1)
    after = model(private.view(1, 4)).detach().cpu().tolist()
    restored_identity = _identity(model, private)
    restored_parameters = [
        mapping.allocation.allocation_id
        for mapping in manager.mappings
        if mapping.allocation.access is AccessClass.PARAMETER_RO
    ]
    new_private = next(
        mapping
        for mapping in manager.mappings
        if mapping.allocation.access is AccessClass.PRIVATE_RW
        and mapping.base <= restored_identity["private_ptr"] < mapping.end
    )
    evidence = {
        "phase": "restore",
        "output_equal": after == before,
        "identity_equal": restored_identity == identity,
        "parameter_allocations_equal": restored_parameters == parameter_ids,
        "private_backing_fresh": (
            new_private.allocation.allocation_id != old_private.allocation.allocation_id
        ),
        "private_access": new_private.allocation.access.value,
        "server_process_equal": restored_server_process == server_process,
        "server_process": restored_server_process,
        "output": after,
        "identity": restored_identity,
        "parameter_allocations": restored_parameters,
        "private_allocation_before": old_private.allocation.allocation_id,
        "private_allocation_after": new_private.allocation.allocation_id,
    }
    print("GMS_V1_EVIDENCE " + json.dumps(evidence, sort_keys=True), flush=True)
    if not all(
        (
            evidence["output_equal"],
            evidence["identity_equal"],
            evidence["parameter_allocations_equal"],
            evidence["private_backing_fresh"],
            evidence["private_access"] == AccessClass.PRIVATE_RW.value,
            evidence["server_process_equal"],
        )
    ):
        raise RuntimeError("snapshot evidence mismatch")
    Path(f"{standby_marker}.restored").write_text("restored", encoding="utf-8")
    await _standby()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--socket-path")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--standby-marker", required=True)
    args = parser.parse_args()
    asyncio.run(
        run(
            args.device,
            args.socket_path or get_socket_path(args.device, "snapshot-v1"),
            args.artifact_id,
            args.standby_marker,
        )
    )


if __name__ == "__main__":
    main()

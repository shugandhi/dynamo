# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
from contextlib import nullcontext
from importlib.util import find_spec
from types import SimpleNamespace

import pytest
from _deps import HAS_CUDA
from _v1_fakes import V1FakeVMM
from gpu_memory_service.v1.client.memory_manager import SnapshotMemoryManager
from gpu_memory_service.v1.client.torch.allocator import (
    SnapshotTorchPools,
    _AllocatorCallbacks,
)
from gpu_memory_service.v1.errors import FatalGMSError
from gpu_memory_service.v1.server.allocations import AllocationStore

HAS_VLLM = find_spec("vllm") is not None


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.none
@pytest.mark.gpu_0
def test_void_free_callback_failure_surfaces_at_snapshot_preparation(
    monkeypatch,
) -> None:
    vmm = V1FakeVMM()
    store = AllocationStore("GPU-0", vmm, 0)
    monkeypatch.setattr(SnapshotMemoryManager, "_gpu_identity", lambda self: "GPU-0")
    manager = SnapshotMemoryManager(store, vmm, 0)
    base = manager.allocate_parameter(64)

    class Cuda:
        @staticmethod
        def device(device):
            return nullcontext()

        @staticmethod
        def empty_cache():
            pass

    pools = SnapshotTorchPools.__new__(SnapshotTorchPools)
    pools._torch = SimpleNamespace(cuda=Cuda)
    pools._manager = manager
    pools._allocator = _AllocatorCallbacks(manager)
    pools._condition = threading.Condition()
    pools._active_scopes = 0
    pools._preparing = False
    pools.device = 0
    pools.parameter = object()
    pools.private = object()

    pools._allocator.free(base, 63, 0, 0)

    with pytest.raises(FatalGMSError, match="allocator free"):
        pools.prepare_snapshot()
    assert not vmm.server_handles
    assert not vmm.imports
    assert not vmm.reservations


@pytest.mark.post_merge
@pytest.mark.integration
@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.skipif(not HAS_CUDA or not HAS_VLLM, reason="CUDA and vLLM are required")
def test_real_cuda_torch_vllm_snapshot_lifecycle() -> None:
    """Use a subprocess because Torch allocator callbacks are process-global."""
    code = textwrap.dedent(
        """
        import os
        import tempfile
        import threading

        import torch

        from gpu_memory_service.common.vmm import get_vmm
        from gpu_memory_service.v1.client.memory_manager import (
            AccessClass,
            SnapshotMemoryManager,
        )
        from gpu_memory_service.v1.client.rpc import AllocationClient
        from gpu_memory_service.v1.client.torch import SnapshotTorchPools
        from gpu_memory_service.v1.errors import GMSError
        from gpu_memory_service.v1.integrations.vllm import install_vllm_integration
        from gpu_memory_service.v1.server.allocations import AllocationStore
        from gpu_memory_service.v1.server.rpc import AllocationRPCServer
        from vllm.v1.worker.workspace import WorkspaceManager

        torch.cuda.set_device(0)
        vmm = get_vmm()
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "gms-v1.sock")
            store = AllocationStore(gpu_uuid, vmm, 0)
            with AllocationRPCServer(path, store) as server:
                thread = threading.Thread(
                    target=server.serve_forever, daemon=True
                )
                thread.start()
                client = AllocationClient(path)
                try:
                    manager = SnapshotMemoryManager(client, vmm, 0)
                    pools = SnapshotTorchPools(manager)
                    workspaces = WorkspaceManager(torch.device("cuda:0"))
                    install_vllm_integration(workspaces, pools)

                    second_manager = SnapshotMemoryManager(client, vmm, 0)
                    try:
                        SnapshotTorchPools(second_manager)
                    except RuntimeError as error:
                        assert "another owner" in str(error)
                    else:
                        raise AssertionError(
                            "a second callback owner constructed Torch pools"
                        )

                    with pools.parameter_pool():
                        storage_owner = torch.arange(
                            16, device="cuda", dtype=torch.float32
                        )
                        first = torch.nn.Parameter(storage_owner.view(4, 4))
                        second = torch.nn.Parameter(
                            storage_owner[4:12], requires_grad=False
                        )
                        workspace = workspaces._ensure_workspace_size(4096)

                    parameter_mapping = next(
                        mapping
                        for mapping in manager.mappings
                        if mapping.base <= first.data_ptr() < mapping.end
                    )
                    workspace_mapping = next(
                        mapping
                        for mapping in manager.mappings
                        if mapping.base <= workspace.data_ptr() < mapping.end
                    )
                    assert parameter_mapping.access is AccessClass.PARAMETER_RO
                    assert workspace_mapping.access is AccessClass.PRIVATE_RW
                    old_workspace_id = workspace_mapping.allocation_id
                    before = (
                        id(first),
                        id(second),
                        int(first._cdata),
                        int(second._cdata),
                        int(first.untyped_storage()._cdata),
                        int(second.untyped_storage()._cdata),
                        first.data_ptr(),
                        second.data_ptr(),
                        int(workspace._cdata),
                        int(workspace.untyped_storage()._cdata),
                        workspace.data_ptr(),
                    )

                    pools.prepare_snapshot()
                    manager.wake()

                    after = (
                        id(first),
                        id(second),
                        int(first._cdata),
                        int(second._cdata),
                        int(first.untyped_storage()._cdata),
                        int(second.untyped_storage()._cdata),
                        first.data_ptr(),
                        second.data_ptr(),
                        int(workspace._cdata),
                        int(workspace.untyped_storage()._cdata),
                        workspace.data_ptr(),
                    )
                    assert after == before
                    assert before[4] == before[5]
                    assert first.detach().cpu().tolist() == [
                        [0.0, 1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 7.0],
                        [8.0, 9.0, 10.0, 11.0],
                        [12.0, 13.0, 14.0, 15.0],
                    ]
                    assert second.detach().cpu().tolist() == [
                        4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0
                    ]

                    restored_workspace = workspaces._ensure_workspace_size(1024)
                    assert restored_workspace is workspace
                    restored_mapping = next(
                        mapping
                        for mapping in manager.mappings
                        if mapping.base <= workspace.data_ptr() < mapping.end
                    )
                    assert restored_mapping.base == workspace_mapping.base
                    assert restored_mapping.allocation_id != old_workspace_id
                    assert restored_mapping.access is AccessClass.PRIVATE_RW
                    workspace.fill_(7)
                    assert torch.all(workspace == 7)
                    try:
                        workspaces._ensure_workspace_size(8192)
                    except GMSError:
                        pass
                    else:
                        raise AssertionError(
                            "workspace growth used a destroyed private pool"
                        )
                    allocation_ids = [
                        mapping.allocation_id for mapping in manager.mappings
                    ]
                    manager.retire()
                    for allocation_id in allocation_ids:
                        try:
                            client.export(allocation_id)
                        except GMSError:
                            pass
                        else:
                            raise AssertionError("retire left physical backing")
                finally:
                    client.close()
                    server.shutdown()
                    thread.join(timeout=10)
                    assert not thread.is_alive()
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)

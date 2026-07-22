# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.integration,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def test_real_cuda_pool_teardown_and_workspace_restore() -> None:
    """Use a subprocess because the allocator callback singleton is process-wide."""
    code = textwrap.dedent(
        """
        import os
        import tempfile
        import threading

        import torch

        from gpu_memory_service.common.vmm import get_vmm
        from gpu_memory_service.v1.client import Manager
        from gpu_memory_service.v1.registry import Registry
        from gpu_memory_service.v1.rpc import RPCClient, RPCServer
        from gpu_memory_service.v1.torch import TorchPools, discover_parameter_mappings
        from gpu_memory_service.v1.vllm import install_workspace_routing
        from gpu_memory_service.v1.protocol import AccessClass
        from vllm.v1.worker.workspace import WorkspaceManager

        torch.cuda.set_device(0)
        vmm = get_vmm()
        gpu = str(torch.cuda.get_device_properties(0).uuid)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "gms-v1.sock")
            registry = Registry(gpu, vmm, 0)
            with RPCServer(path, registry) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                rpc = None
                try:
                    rpc = RPCClient(path)
                    manager = Manager(
                        rpc,
                        vmm,
                        0,
                        artifact_id="cuda-test",
                        generation_id="cuda-test",
                    )
                    pools = TorchPools(manager)
                    workspace_manager = WorkspaceManager(torch.device("cuda:0"))
                    install_workspace_routing(
                        workspace_manager, pools.private_pool
                    )
                    with pools.parameter_pool():
                        base = torch.arange(16, device="cuda", dtype=torch.float32)
                        first = torch.nn.Parameter(base.view(4, 4))
                        second = torch.nn.Parameter(base[4:12], requires_grad=False)
                        workspace = workspace_manager._ensure_workspace_size(4096)
                    with pools.private_pool():
                        private = torch.ones(4, device="cuda")

                    class Model(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.first = first
                            self.second = second

                    model = Model()
                    before = (
                        id(first),
                        id(second),
                        int(first.untyped_storage()._cdata),
                        int(second.untyped_storage()._cdata),
                        first.data_ptr(),
                        second.data_ptr(),
                        private.data_ptr(),
                        int(workspace._cdata),
                        int(workspace.untyped_storage()._cdata),
                        workspace.data_ptr(),
                    )
                    parameter_records = discover_parameter_mappings(
                        model, manager.mappings
                    )
                    parameter_ids = {
                        record.allocation.allocation_id
                        for record in parameter_records
                    }
                    workspace_record = next(
                        record
                        for record in manager.mappings
                        if record.base <= workspace.data_ptr() < record.end
                    )
                    assert workspace_record.allocation.access is AccessClass.PRIVATE_RW
                    assert (
                        workspace_record.allocation.allocation_id
                        not in parameter_ids
                    )
                    assert workspace_record.base not in {
                        record.base for record in parameter_records
                    }
                    old_workspace_allocation = (
                        workspace_record.allocation.allocation_id
                    )
                    pools.prepare_capture(model)
                    manager.wake("cuda-test-reader")
                    after = (
                        id(first),
                        id(second),
                        int(first.untyped_storage()._cdata),
                        int(second.untyped_storage()._cdata),
                        first.data_ptr(),
                        second.data_ptr(),
                        private.data_ptr(),
                        int(workspace._cdata),
                        int(workspace.untyped_storage()._cdata),
                        workspace.data_ptr(),
                    )
                    assert before == after
                    assert first.detach().cpu().tolist() == [
                        [0.0, 1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 7.0],
                        [8.0, 9.0, 10.0, 11.0],
                        [12.0, 13.0, 14.0, 15.0],
                    ]
                    assert second.detach().cpu().tolist() == [
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        8.0,
                        9.0,
                        10.0,
                        11.0,
                    ]
                    assert len(parameter_records) == 1
                    assert before[2] == before[3]
                    restored_workspace = workspace_manager._ensure_workspace_size(1024)
                    assert restored_workspace is workspace
                    new_workspace_record = next(
                        record
                        for record in manager.mappings
                        if record.base <= workspace.data_ptr() < record.end
                    )
                    assert new_workspace_record.base == workspace_record.base
                    assert (
                        new_workspace_record.allocation.allocation_id
                        != old_workspace_allocation
                    )
                    assert (
                        new_workspace_record.allocation.access
                        is AccessClass.PRIVATE_RW
                    )
                    workspace.fill_(7)
                    assert torch.all(workspace == 7)
                    manager.retire()
                    assert not registry._generations
                finally:
                    if rpc is not None:
                        rpc.close()
                    server.shutdown()
                    thread.join(timeout=10)
                    assert not thread.is_alive()
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)

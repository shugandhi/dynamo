# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen Snapshot E2E using the standard Transformers model loader."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.common.vmm import get_vmm

from .client import Manager
from .errors import FatalGMSError
from .protocol import AccessClass, Mapping
from .rpc import RPCClient
from .torch import TorchPools

if TYPE_CHECKING:
    from collections.abc import Iterable

_DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
_DEFAULT_PROMPT = "The capital of France is"


def _buffer_registries(model: object) -> list[tuple[object, dict[str, object]]]:
    registries = []
    for module in model.modules():  # type: ignore[attr-defined]
        registries.append((module, dict(module._buffers)))  # type: ignore[attr-defined]
    return registries


def move_registered_buffers(model: object, device: str) -> None:
    """Move each registered buffer object once, preserving all registration aliases."""
    moved: dict[int, object] = {}
    for module, buffers in _buffer_registries(model):
        for name, buffer in buffers.items():
            if buffer is None:
                continue
            replacement = moved.get(id(buffer))
            if replacement is None:
                replacement = buffer.to(device)  # type: ignore[attr-defined]
                moved[id(buffer)] = replacement
            module._buffers[name] = replacement  # type: ignore[attr-defined]


def move_parameters(model: object, device: str) -> None:
    """Use the standard model migration with registered buffers excluded."""
    registries = _buffer_registries(model)
    for module, buffers in registries:
        module._buffers.update(dict.fromkeys(buffers))  # type: ignore[attr-defined]
    try:
        migrated = model.to(device)  # type: ignore[attr-defined]
        if migrated is not model:
            raise RuntimeError("standard model migration returned another model")
        for module, buffers in registries:
            current = module._buffers  # type: ignore[attr-defined]
            if current.keys() != buffers.keys() or any(
                value is not None for value in current.values()
            ):
                raise RuntimeError("model migration changed registered buffers")
    finally:
        for module, buffers in registries:
            current = module._buffers  # type: ignore[attr-defined]
            current.clear()
            current.update(buffers)


def identity_digest(model: object) -> dict[str, object]:
    """Digest CRIU-owned model-state object, TensorImpl, StorageImpl, and VA IDs."""
    parameters = []
    for name, parameter in model.named_parameters(remove_duplicate=False):  # type: ignore[attr-defined]
        storage = parameter.untyped_storage()
        parameters.append(
            (
                name,
                id(parameter),
                int(parameter._cdata),
                int(storage._cdata),
                int(parameter.data_ptr()),
            )
        )
    buffers = []
    for name, buffer in model.named_buffers(remove_duplicate=False):  # type: ignore[attr-defined]
        storage = buffer.untyped_storage()
        buffers.append(
            (
                name,
                id(buffer),
                int(buffer._cdata),
                int(storage._cdata),
                int(buffer.data_ptr()),
            )
        )
    payload = (id(model), parameters, buffers)
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    return {
        "algorithm": "sha256",
        "digest": hashlib.sha256(encoded).hexdigest(),
        "model": id(model),
        "parameter_bindings": len(parameters),
        "buffer_bindings": len(buffers),
    }


def mapping_evidence(mappings: "Iterable[Mapping]") -> list[dict[str, object]]:
    """Return stable allocation IDs and CRIU-owned virtual address ranges."""
    return [
        {
            "allocation_id": mapping.allocation.allocation_id,
            "base": mapping.base,
            "size": mapping.allocation.aligned_size,
            "reservation_size": mapping.reservation_size,
            "access": mapping.allocation.access.value,
        }
        for mapping in sorted(mappings, key=lambda value: value.base)
    ]


def containing_mapping(
    mappings: "Iterable[Mapping]", pointer: int, access: AccessClass
) -> Mapping:
    found = [
        mapping
        for mapping in mappings
        if mapping.allocation.access is access and mapping.base <= pointer < mapping.end
    ]
    if len(found) != 1:
        raise RuntimeError(
            f"pointer 0x{pointer:x} has {len(found)} {access.value} mappings"
        )
    return found[0]


def restore_evidence(
    capture: dict[str, object],
    *,
    tokens: list[int],
    output: str,
    identity: dict[str, object],
    parameters: list[dict[str, object]],
    private: dict[str, object],
    private_identity: dict[str, int],
    private_write: bool,
    gpu: str,
    server_process: tuple[int, int],
) -> dict[str, object]:
    """Build all machine-checkable Snapshot invariants in one pure helper."""
    old_private = capture["private_mapping"]
    if not isinstance(old_private, dict):
        raise TypeError("capture private mapping evidence must be a dictionary")
    return {
        "phase": "restore",
        "tokens": tokens,
        "output": output,
        "identity": identity,
        "parameter_mappings": parameters,
        "private_mapping": private,
        "private_identity": private_identity,
        "gpu": gpu,
        "server_process": server_process,
        "token_equal": tokens == capture["tokens"],
        "output_equal": output == capture["output"],
        "identity_digest_equal": identity == capture["identity"],
        "parameter_mappings_equal": parameters == capture["parameter_mappings"],
        "private_identity_equal": private_identity == capture["private_identity"],
        "private_backing_fresh": (
            private["allocation_id"] != old_private["allocation_id"]
        ),
        "private_va_equal": (
            private["base"] == old_private["base"]
            and private["size"] == old_private["size"]
            and private["reservation_size"] == old_private["reservation_size"]
        ),
        "private_access": private["access"],
        "private_write": private_write,
        "same_gpu": gpu == capture["gpu"],
        "server_process_equal": server_process == capture["server_process"],
    }


def _private_identity(value: object) -> dict[str, int]:
    storage = value.untyped_storage()  # type: ignore[attr-defined]
    return {
        "tensor": id(value),
        "tensor_impl": int(value._cdata),  # type: ignore[attr-defined]
        "storage_impl": int(storage._cdata),
        "data_ptr": int(value.data_ptr()),  # type: ignore[attr-defined]
    }


def _infer(
    model: object, tokenizer: object, prompt: str, device: str
) -> tuple[list[int], str]:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")  # type: ignore[operator]
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.inference_mode():
        generated = model.generate(  # type: ignore[attr-defined]
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=8,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,  # type: ignore[attr-defined]
        )
    tokens = [int(token) for token in generated[0].cpu().tolist()]
    output = tokenizer.decode(tokens, skip_special_tokens=True)  # type: ignore[attr-defined]
    del generated, attention_mask, input_ids, inputs
    return tokens, output


async def _standby() -> None:
    while True:
        await asyncio.sleep(3600)


async def run(
    device: int,
    socket_path: str,
    artifact_id: str,
    standby_marker: str,
    model_name: str,
    prompt: str,
) -> None:
    marker = Path(standby_marker)
    if marker.exists():
        await _standby()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dynamo.common.snapshot.lifecycle import SnapshotConfig

    config = SnapshotConfig.from_env()
    if config is None:
        raise RuntimeError("DYN_SNAPSHOT_CONTROL_DIR is required")
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    target = f"cuda:{device}"
    gpu = str(torch.cuda.get_device_properties(device).uuid)
    rpc = RPCClient(socket_path)
    server_process = rpc.process_evidence()
    manager = Manager(rpc, get_vmm(), device, artifact_id=artifact_id)
    pools = TorchPools(manager)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    move_registered_buffers(model, target)
    with pools.parameter_pool():
        move_parameters(model, target)
    model.eval()
    with pools.private_pool():
        private = torch.zeros(4096, dtype=torch.uint8, device=target)

    tokens, output = _infer(model, tokenizer, prompt, target)
    identity = identity_digest(model)
    private_identity = _private_identity(private)
    parameters = pools.prepare_capture(model)
    parameter_records = mapping_evidence(parameters)
    old_private = containing_mapping(
        manager.mappings, private_identity["data_ptr"], AccessClass.PRIVATE_RW
    )
    capture: dict[str, object] = {
        "phase": "capture",
        "model_name": model_name,
        "prompt": prompt,
        "tokens": tokens,
        "output": output,
        "identity": identity,
        "parameter_mappings": parameter_records,
        "private_mapping": mapping_evidence((old_private,))[0],
        "private_identity": private_identity,
        "gpu": gpu,
        "server_process": server_process,
    }
    rpc_open = True
    try:
        print("GMS_V1_EVIDENCE " + json.dumps(capture, sort_keys=True), flush=True)
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
    private.fill_(37)
    private_write = bool(torch.all(private == 37).item())
    restored_tokens, restored_output = _infer(model, tokenizer, prompt, target)
    restored_identity = identity_digest(model)
    restored_private_identity = _private_identity(private)
    new_private = containing_mapping(
        manager.mappings,
        restored_private_identity["data_ptr"],
        AccessClass.PRIVATE_RW,
    )
    evidence = restore_evidence(
        capture,
        tokens=restored_tokens,
        output=restored_output,
        identity=restored_identity,
        parameters=mapping_evidence(
            mapping
            for mapping in manager.mappings
            if mapping.allocation.access is AccessClass.PARAMETER_RO
        ),
        private=mapping_evidence((new_private,))[0],
        private_identity=restored_private_identity,
        private_write=private_write,
        gpu=str(torch.cuda.get_device_properties(device).uuid),
        server_process=restored_server_process,
    )
    print("GMS_V1_EVIDENCE " + json.dumps(evidence, sort_keys=True), flush=True)
    required = (
        "token_equal",
        "output_equal",
        "identity_digest_equal",
        "parameter_mappings_equal",
        "private_identity_equal",
        "private_backing_fresh",
        "private_va_equal",
        "private_write",
        "same_gpu",
        "server_process_equal",
    )
    if not all(evidence[key] for key in required) or (
        evidence["private_access"] != AccessClass.PRIVATE_RW.value
    ):
        raise RuntimeError("Qwen snapshot evidence mismatch")
    Path(f"{standby_marker}.restored").write_text("restored", encoding="utf-8")
    await _standby()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--socket-path")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--standby-marker", required=True)
    parser.add_argument(
        "--model", default=os.environ.get("GMS_V1_MODEL", _DEFAULT_MODEL)
    )
    parser.add_argument(
        "--prompt", default=os.environ.get("GMS_V1_PROMPT", _DEFAULT_PROMPT)
    )
    args = parser.parse_args()
    asyncio.run(
        run(
            args.device,
            args.socket_path or get_socket_path(args.device, "snapshot-v1"),
            args.artifact_id,
            args.standby_marker,
            args.model,
            args.prompt,
        )
    )


if __name__ == "__main__":
    main()

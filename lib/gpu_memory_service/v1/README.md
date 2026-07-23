<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPU Memory Service V1

GMS V1 is an experimental, Dynamo Snapshot-only GPU memory owner. It assumes
one persistent GMS sidecar process on the same physical GPU as the
checkpointed process. There is no compatibility mode or model-loading API.

## Ownership

The checkpointed client owns:

- Tensor, TensorImpl, StorageImpl, aliases, views, and model topology;
- virtual addresses and one canonical record per allocator segment;
- the `parameter_ro` versus `private_rw` access class;
- imports, mappings, pool lifetime, and fail-stop sleep/wake/retire state.

The sidecar owns only:

- a random process-incarnation nonce and the physical GPU UUID;
- allocation ID to aligned size, physical handle, and retained export FD.

`hello`, `allocate`, `export`, and `free` are the complete RPC protocol.
Allocation IDs are caller-generated. Repeating `allocate` with the same ID and
size or repeating `free` is safe after response loss. Reusing an ID with a
different size fails.

The client records the sidecar nonce and GPU UUID on its first connection.
Every transport reconnect and every wake verifies both values before an
allocation mutation or mapping. A restarted sidecar or different physical GPU
is terminal.

## Snapshot lifecycle

Torch uses two MemPools backed by the same pluggable allocator callbacks:

- parameter allocations enter the `parameter_ro` domain;
- private/runtime allocations enter the `private_rw` domain.

`SnapshotTorchPools.prepare_snapshot()` accepts no model. It destroys and
evicts both pools, observes the allocator's void-return free callback latch,
synchronizes the GPU, changes the complete locally recorded parameter domain
to read-only, unmaps and releases each local import once, and frees private
physical backing. Pool destruction and allocator callbacks define allocation
identity; production code does not inspect Parameters, buffers, or storages.

After CRIU restore, `SnapshotMemoryManager.wake()` reimports the surviving
parameter backing read-only at its preserved VA. It creates a fresh allocation
ID and physical backing for each private mapping, then maps that backing
read-write at the preserved VA.

Cleanup failures are fail-stop. The manager retains imports or physical
allocation IDs whenever ownership cannot be proved released, while continuing
independent cleanups in reverse order.

## vLLM

V1 requires every private/runtime (mutable) allocation to be created during
initialization or warmup, **before** snapshot preparation, so it is captured in
the private pool and restored with fresh backing at its preserved VA on wake.

`install_vllm_integration()` runs before model construction and routes the one
validated mutable path: active-DBO `WorkspaceManager._ensure_workspace_size`
growth. Growth enters the private pool even beneath an outer parameter pool. A
no-growth call bypasses a destroyed private pool; growth after pool destruction
fails. Reinstalling the same pool owner is idempotent; a second owner or a
replaced hook fails before use.

Kernels that lazily allocate scratch during serving `forward()` — e.g. certain
quantized MoE paths such as Marlin MoE / Humming — are **not** supported by V1.
The private MemPool is destroyed at snapshot preparation and never recreated,
so a post-wake `forward()` would allocate into a dead pool. Supporting them
would require a private-pool post-wake lifecycle that is out of scope. The
validated dense target (Qwen3-0.6B) never exercises these paths.

The outer parameter pool is confined to vLLM model construction, weight load,
and post-load creation of immutable weights. It must exit before memory
profiling, KV-cache allocation, warmup, or CUDA graph capture. The model-shaped
EP preparation at the end of `GPUModelRunner.load_model()` initializes modular
communication backends. Its model walk selects MoE layers but does not create
another model-owned mutable tensor: routing tables were already constructed
with the model, and DeepEP/NIXL/MORI communication backing is returned by raw
backend `get_handle()` objects. Those raw NCCL/communication allocations bypass
Torch MemPools and remain the Snapshot/backend hook's responsibility, not GMS
allocation records.

## Commands and tests

The only V1 console command is:

```text
gms-v1-server --device 0 [--socket-path PATH]
```

Behavioral tests live with the existing GMS suite in
`lib/gpu_memory_service/tests/test_v1_*.py`. No test engine, manifest, shell
asset, or model-specific workload is installed as product data.

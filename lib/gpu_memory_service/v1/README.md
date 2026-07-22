# Snapshot-only GPU Memory Service V1

V1 has one fail-stop contract: Dynamo Snapshot/CRIU restores the engine
process and all Python, `TensorImpl`, `StorageImpl`, alias, and module topology;
an unchanged sidecar preserves exact parameter allocation IDs and physical
backing on the same GPU.

Torch 2.11 uses two dedicated CUDA `MemPool`s. Parameter allocations are
writable while loading, become read-only as one complete set before the
artifact is published, and preserve their backing across capture. Private
allocations are discarded during sleep and receive fresh read-write backing at
the CRIU-preserved virtual addresses after restore.

These are generic allocation domains, not model- or backend-specific memory
types. The server owns only allocation IDs and physical handles/backing. It
never receives virtual addresses, tensor identities, shapes, or model
metadata. The checkpointed client remains the sole owner of tensors,
`TensorImpl`s, `StorageImpl`s, aliases, and virtual addresses.

Capture destroys the dedicated pools so Torch evicts inactive segments.
Registered CUDA `Parameter` storages are deduplicated by `UntypedStorage`
identity, resolved by complete range containment to one parameter mapping, and
then deduplicated by exact containing allocation. Sleep/wake process each
canonical allocation mapping once; V1 never serializes or reconstructs tensor
topology. Registered buffers are never parameter-backed: capture rejects every
live CUDA named-buffer storage range that overlaps a parameter mapping.

## vLLM workspace allocation scope

The pinned vLLM 0.25.1 adapter is an explicit, instance-local binding from
`WorkspaceManager` growth to a caller-supplied allocation scope:

```python
from gpu_memory_service.v1.vllm import install_workspace_routing
from vllm.v1.worker.workspace import current_workspace_manager

install_workspace_routing(current_workspace_manager(), pools.private_pool)
```

Only growth enters the supplied scope, so an outer parameter scope is
overridden for the allocation itself. A no-growth call does not enter the scope
and therefore remains valid after capture destroys the pools. Growth after
destruction fails instead of allocating through an ambient allocator.
Installation with the same manager and scope is idempotent; conflicting
ownership or a replaced allocation hook fails. V1 itself has no workspace
allocation class or workspace lifecycle state.

The binding is intentionally installed once for one V1 generation in one
engine process. V1 does not support uninstalling it or reusing the process for
a later generation; process teardown ends the binding's lifetime.

## Qwen3-0.6B E2E scope

`qwen_e2e.py` loads `Qwen/Qwen3-0.6B` on CPU with the standard Transformers
`AutoModelForCausalLM.from_pretrained()` loader. It moves registered buffers,
including `rotary_emb.inv_freq`, through the native CUDA allocator while
preserving repeated registrations of the same buffer object, then activates
the V1 parameter domain only for standard Parameter migration. It does not use
legacy `--load-format gms`, a meta-model, tensor metadata, a model manifest, or
custom tensor reconstruction. It performs deterministic greedy inference with
`use_cache=False`, captures through `SnapshotConfig.run_lifecycle`, and emits
`GMS_V1_EVIDENCE` JSON proving:

- generated token and decoded output equality;
- one digest over model, Parameter and registered-buffer bindings,
  `TensorImpl`, `StorageImpl`, and data-pointer identities;
- stable parameter allocation IDs and virtual address ranges;
- a deliberate private probe with a fresh allocation ID at the same
  read-write virtual address and a successful post-wake write;
- the same physical GPU UUID; and
- GMS sidecar PID and startup-nonce continuity.

This is intentionally an in-process Transformers Qwen model test. It validates
real Qwen parameter/storage Snapshot behavior without introducing worker
subprocess ownership ambiguity; it does not claim coverage of vLLM scheduling,
KV cache, CUDA graphs, or request serving. The separate vLLM integration and
real-CUDA subprocess tests cover workspace allocation routing.

## Commands

Start the non-checkpointed sidecar:

```bash
gms-v1-server --device 0 --socket-path /gms/gms-v1.sock
```

Run the engine inside a Dynamo Snapshot target container:

```bash
gms-v1-e2e \
  --device 0 \
  --socket-path /gms/gms-v1.sock \
  --artifact-id "${CHECKPOINT_ID}" \
  --standby-marker /state/captured
```

This toy harness remains the fast lifecycle smoke test.

## DRA + Snapshot deployment test

The collected test creates one DRA `ResourceClaimTemplate`, one two-container
Pod, and one `PodSnapshot`. It snapshots only `engine`, restores that container
in place, proves the `gms-server` container never restarted, and machine-checks
inference, object/storage/data-pointer identity, stable parameter allocation
IDs, and fresh private backing.

It requires explicit cluster placement and never evicts workloads:

```bash
KUBE_CONTEXT=<context> \
NAMESPACE=<namespace> \
NODE=<gpu-node> \
IMAGE=<torch-2.11-dynamo-image> \
IMAGE_PULL_SECRET=<registry-pull-secret> \
CHECKPOINT_PVC=<snapshot-pvc> \
lib/gpu_memory_service/v1/deploy/run.sh
```

Optional variables are `CHECKPOINT_PATH` (default `/checkpoints`),
`DEVICE_CLASS` (default `gpu.nvidia.com`), and `RUNTIME_CLASS` (default
`nvidia`). The test deletes only its uniquely named resources.

The Qwen deployment uses the same raw Pod, PodSnapshot, engine-only capture,
shared DRA claim, persistent GMS sidecar, and checkpoint PVC topology. It also
requires an offline model-cache PVC and an explicit exact DeviceClass name:

```bash
KUBE_CONTEXT=<context> \
NAMESPACE=<namespace> \
NODE=<gpu-node> \
IMAGE=<torch-2.11-transformers-dynamo-image> \
IMAGE_PULL_SECRET=<registry-pull-secret> \
CHECKPOINT_PVC=<snapshot-pvc> \
MODEL_CACHE_PVC=<qwen-cache-pvc> \
DEVICE_CLASS=<exact-device-class-name> \
lib/gpu_memory_service/v1/deploy/run-qwen.sh
```

Optional Qwen variables are `MODEL` (default `Qwen/Qwen3-0.6B`),
`MODEL_CACHE_PATH` (default `/model-cache`), `CHECKPOINT_PATH` (default
`/checkpoints`), and `RUNTIME_CLASS` (default `nvidia`). The cache must already
contain the model because both Hugging Face Hub and Transformers offline modes
are enabled.

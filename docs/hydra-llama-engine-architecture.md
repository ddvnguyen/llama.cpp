## Context

The target Hydra runtime is **per-GPU llama-engine processes**, each
of which is a complete llama-server that can also act as a ggml-RPC
backend (rpc-server). There is **no weight transfer** between
engines: every engine mmap's the model GGUF from disk, and the
COMBINE-head uses upstream `ggml-RPC` to call per-token graph
compute + per-tensor `RPC_CMD_RESOLVE_TENSOR(name)` against the
peer's local mmap.

This design supersedes the design that PR #31 was trying to
implement. PR #31 is being left open as a draft; specific
upstream-touched hunks from it will be cherry-picked into a separate
Phase 1 PR (see "Cherry-pick from #31" below). The design rationale
and gap analysis are in the **Comparison** section at the end of
this body.

## Hard constraints

- **C1**: minimal diff in upstream `ggml/src/ggml-rpc/*`,
  `ggml/src/ggml-cuda/*`, `ggml/src/ggml-backend.cpp`. The four
  small upstream additions (≈71 LOC total) can each be a focused
  upstream PR. Bulk of new code (~250 LOC) lives in
  `tools/llama-engine/hydra_rpc/`.
- **C2**: no weight transfer over the wire. Peer loads tensors from
  its own mmap; `RPC_CMD_RESOLVE_TENSOR` (PR #20) sends the tensor
  name only.
- **C3**: SOLO mode must not regress. No new global locks, no
  mandatory setup, no per-op overhead vs upstream.
- **C4**: cross-host (P100) unchanged. All Hydra-specific behavior
  is same-host only.
- **C5**: each engine can load any model from a known path on
  demand. No per-process model lock-in.
- **C6**: fail-soft wrapping is allowed **only in teardown paths**
  (buffer_free during model unload, scheduler teardown). Normal
  ops keep `RPC_STATUS_ASSERT` / `GGML_ABORT` on failure.
- **C7**: protocol dispatch uses **one byte** (ggml-RPC uses
  `0x00–0x11`, Hydra uses `0x30–0x46`; ranges don't overlap).
  No magic, no per-conn `detach()`, bounded thread pool.

## Goals (measurable)

| # | Goal | Target | Current |
|---|------|--------|---------|
| G1 | DENSE 27B layer-split COMBINED decode | ≥ 50 tok/s on 5060+3060 | 0 (broken) |
| G2 | MoE 35B-A3B layer-split COMBINED decode | ≥ 45 tok/s on 5060+3060 | 0 (broken) |
| G3 | SOLO 5060 Ti decode (no regression) | ≥ 190 tok/s (≤ 5% from 200) | 200 tok/s |
| G4 | Per-engine model load (cold, NVMe) | ≤ 25 s for 27B Q4 | not measured |
| G5 | Per-engine model load (warm, page cache) | ≤ 5 s for 27B Q4 | not measured |
| G6 | Runtime SOLO ↔ COMBINED switch (same model) | < 100 ms (no reload) | not implemented |
| G7 | Multi-model profile switch (DENSE ↔ MoE) | < 60 s end-to-end | not measured |
| G8 | P100 cross-host P/D split | no regression (28 tok/s) | 28 tok/s |
| G9 | `git diff upstream/master -- 'ggml/src/ggml-rpc/*' 'ggml/src/ggml-cuda/*' 'ggml/src/ggml-backend.cpp'` | ≤ 4 functions / 75 LOC | PR #31 = 200+ LOC across 15 files |

## Architecture

```
                  Hydra.Core (C#)
                       │
        HTTP: {"mode":"combine","peer":"localhost:9504","model":"DENSE-27B-Q4","split":"21/44"}
                       │
        ┌──────────────▼─────────────────────────────┐
        │   llama-engine A (host: GPU 0 = 5060 Ti)    │
        │   role: COMBINE-head                         │
        │   ports: HTTP=8080, RPC=9504                │
        │                                              │
        │   ┌─────────────────────────────────────┐   │
        │   │  HTTP server (upstream)              │   │
        │   │  --port                              │   │
        │   │  /health, /v1/chat/completions,      │   │
        │   │  /control/load_model (A→B),         │   │
        │   │  /control/set_expert_mode, etc.      │   │
        │   └─────────────────────────────────────┘   │
        │   ┌─────────────────────────────────────┐   │
        │   │  Hydra RPC (tools/llama-engine/     │   │
        │   │  hydra_rpc/) --rpc-port              │   │
        │   │  dispatch: 1-byte peek               │   │
        │   │   0x0E → ggml_backend_rpc_handle_   │   │
        │   │          client (ggml-RPC compute)  │   │
        │   │   else  → hydra_handle_connection   │   │
        │   │          (Hydra control)            │   │
        │   │  bounded thread pool (size 2)       │   │
        │   └─────────────────────────────────────┘   │
        │   mmap: DENSE-27B-Q4.gguf                  │
        │   tensor_split: layers 0–20 on local GPU    │
        │                  layers 21–43 on RPC peer   │
        └──────────────┬──────────────────────────────┘
                       │  ggml-RPC over loopback
                       │  RPC_CMD_GRAPH_COMPUTE (per token)
                       │  RPC_CMD_SET_TENSOR (peer loads from mmap)
                       │  RPC_CMD_RESOLVE_TENSOR (zero-copy binding)
                       │  NO weight data on the wire
        ┌──────────────▼──────────────────────────────┐
        │   llama-engine B (host: GPU 1 = 3060)       │
        │   role: COMBINE-peer                        │
        │   ports: HTTP=8081, RPC=9505                │
        │                                              │
        │   ┌─────────────────────────────────────┐   │
        │   │  HTTP server (upstream)              │   │
        │   │  --port                              │   │
        │   └─────────────────────────────────────┘   │
        │   ┌─────────────────────────────────────┐   │
        │   │  Hydra RPC (tools/llama-engine/     │   │
        │   │  hydra_rpc/) --rpc-port              │   │
        │   │  dispatches ggml-RPC ops to the      │   │
        │   │  local GPU; rejects Hydra control    │   │
        │   │  (peer is passive)                   │   │
        │   └─────────────────────────────────────┘   │
        │   mmap: DENSE-27B-Q4.gguf                  │
        │   exposes GPU 1 as ggml-RPC backend        │
        │   layers 21–43 resident on GPU 1            │
        └─────────────────────────────────────────────┘
```

## Inside the merged Hydra RPC server

```
┌─────────────────────────────────────────────────────────────┐
│  hydra_rpc::start({port, backends, hydra_ctx, pool_size})  │
│                                                              │
│  bind+listen(server_fd, port)                                │
│      │                                                       │
│      ▼                                                       │
│  accept_thread (single, blocked in accept())                │
│      │                                                       │
│      ├── accept() returns conn_fd                            │
│      │                                                       │
│      ├── pool.try_enqueue([conn_fd]{                         │
│      │       recv(conn_fd, &b, 1, MSG_PEEK)                 │
│      │       ├── b == 0x0E && !backends.empty()             │
│      │       │     → ggml_backend_rpc_handle_client(...)    │
│      │       ├── b != 0x0E && hydra_ctx != nullptr          │
│      │       │     → hydra_handle_connection(...)          │
│      │       └── else → ::close(conn_fd)                    │
│      │   })                                                  │
│      │   (drops connection if pool queue is full)           │
│      │                                                       │
│      └── loop until stop_requested                          │
│                                                              │
│  pool: bounded_thread_pool<size=2, max_queue=64>            │
│  shutdown: ::shutdown(server_fd, SHUT_RDWR) + atomic flag   │
└─────────────────────────────────────────────────────────────┘
```

## Mode-to-port mapping

| Mode | `--port` | `--rpc-port` | Backends exposed | Hydra control |
|---|---|---|---|---|
| SOLO (no control) | 8080 | 0 (off) | — | — |
| SOLO with control | 8080 | 9504 | none | yes |
| COMBINE head | 8080 | 9504 | none (head's GPU; not exposed) | yes |
| COMBINE peer (model) | 8080 | 9505 | local GPU | no (passive) |
| COMBINE peer (no model) | 8080 | 9505 | local GPU | no (passive) |

## Per-process model loading (A tells B)

```
A                                                    B
│                                                    │
│ POST /control/load_model                           │
│ {                                                  │
│   "model": "/models/DENSE-27B-Q4.gguf",            │
│   "keep_layer_range": {"lo": 21, "hi": 43}         │
│ }                                                  │
├───────────────────────────────────────────────────►│
│                                                    │ parse, set keep_layer_range,
│                                                    │ unload current model (if any),
│                                                    │ llama_model_load_from_file()
│                                                    │   - mmap (full or partial)
│                                                    │   - loader creates tensors
│                                                    │     only for layers 21..43
│                                                    │   - register tensors for RPC
│                                                    │
│ HTTP 200 OK                                        │
│ {                                                  │
│   "loaded": true,                                  │
│   "n_tensors": 712,                                │
│   "size_mb": 8192                                  │
│ }                                                  │
│◄───────────────────────────────────────────────────┤
│                                                    │
│ A then loads its own model (layers 0–20 on local   │
│ CUDA), uses RPC_CMD_RESOLVE_TENSOR to bind         │
│ layers 21–43 to B's mmap addresses.                │
```

The `keep_layer_range` filter is implemented via an extension of
the existing `tensor_buft_overrides` mechanism (no upstream struct
change). The pattern `layer:N-M` matches tensors whose `tn.bid`
(blok index) is in `[N, M]`.

## Phased plan

### Phase 0: Design (this PR). No code.

Get sign-off on goals G1–G9, constraints C1–C7, non-goals, phase
ordering, and the cherry-pick list below.

### Phase 1: Cherry-pick the safe pieces from PR #31

Open a fork PR "fork: Hydra llama-engine Phase 1 — cherry-pick from
#31" with ONLY the items in the KEEP list. Every change must be
in a fork-isolated file or one of the four upstreamable small APIs
in C1.

**KEEP from #31 (upstreamable small APIs):**
- `ggml_backend_rpc_handle_client` decl + impl in
  `ggml-rpc.cpp` + `ggml-rpc.h` (~20 LOC)
- `ggml_backend_rpc_remove_server` decl + impl in
  `ggml-rpc.cpp` + `ggml-rpc.h` (~30 LOC)
- `set_keepalive` in `transport.cpp` (~20 LOC)
- `MSG_NOSIGNAL` on `send` in `transport.cpp` (1 LOC)

**KEEP from #31 (fork-isolated):**
- `tools/llama-engine/llama-engine.cpp`:
  - `extract_hydra_capability_flags`
  - `wait_for_peer_ready` + `--peer-health-url` flag
  - Staged `llama_engine()` init order (HTTP early → model →
    RPC → ready)
  - `startup_stage` state machine + staged `/health` body
  - Per-engine HTTP server with full routes
- `include/llama-hydra.h` + `src/llama-hydra.cpp`:
  - `llama_hydra_clear_combined_bindings`
- `src/llama-context.cpp`:
  - `hydra_remove_combined_rpc_backend`
- `tools/server/server-http.cpp`:
  - `/health` exempt from `is_ready` middleware

**NEW in Phase 1 (the v4 module):**
- `tools/llama-engine/hydra_rpc/hydra_rpc.{h,cpp}` — merged
  accept loop with 1-byte dispatch and bounded thread pool
- `tools/llama-engine/hydra_rpc/bounded_thread_pool.{h,cpp}` —
  small template, size-N pool
- `tools/llama-engine/llama-engine.cpp`:
  - Collapse two start calls (`start_shared_backend_rpc_server` +
    `start_rpc_server`) into one `hydra_rpc::start` call
  - Delete `start_shared_backend_rpc_server` (replaced)
  - Delete `start_backend_rpc_peer_server` (replaced)
- `tools/server/server-context.cpp`:
  - Delete `server_context::start_rpc_server` (replaced)
- `tools/llama-engine/llama-engine.cpp`:
  - Remove `--ggml-rpc-port` flag (use `--rpc-port` only)
  - Add `--peer-health-url` auto-derivation from `--rpc-engine`
    (replace RPC port with HTTP port)
  - Make `wait_for_peer_ready` timeout configurable
  - Update `infra/hydra-core/config/workers.json` and
    `infra/hydra-head/config/node-rtx.yaml` to drop
    `--ggml-rpc-port`

**REJECT from #31:**
- `RPC_STATUS_ASSERT` → fail-soft replacement (16+ places):
  REJECT. C6 says fail-soft only in teardown. Inference path
  keeps `GGML_ABORT`.
- `response = {}` zero-inits (3 places): REJECT. Pointless.
- `ggml-cuda/fattn.cu` `GGML_ABORT` removal: REJECT. Masks
  a real bug.
- Per-connection `std::thread::detach()`: REJECT. Replaced by
  bounded thread pool in the new module.
- `socket_t::from_fd`: REJECT. v4 doesn't need it; the new
  module calls `ggml_backend_rpc_handle_client` directly.
- `llama_hydra_load_combined_experts`,
  `llama_hydra_rebind_combined_experts`,
  `s_hydra_combined_bindings`, `ffn_*_exps_rpc` fields:
  REJECT. COMBINED-OT expert-split is a dead path; remove.
- `llama_hydra_validate_quant_parity` (Phase C) +
  `SWAP_QUANT` validation block: REJECT. Split to a separate
  PR per `ddvnguyen/hydra_vortex#394`.

**Verifications (run before opening the Phase 1 PR):**
- `git diff upstream/master -- 'ggml/src/ggml-rpc/*'
  'ggml/src/ggml-cuda/*' 'ggml/src/ggml-backend.cpp'` is
  ≤ 75 LOC (G9)
- `cmake --build build_sm86_sm120 --target llama-engine`
  succeeds (fat binary)
- SOLO 5060 Ti tok/s is unchanged from 200 (G3)
- The four small upstream APIs each have a doc comment
  explaining what they do and why they're public

**Estimated Phase 1 effort**: 1 week.

### Phase 2: COMBINED correctness

The unified accept loop + per-process model loading is in place
from Phase 1. Now make layer-split COMBINE actually work on the
3060 + 5060 Ti pair.

- **Add the new buft for resolve-mode**:
  `ggml_backend_rpc_resolve_buffer_type` in
  `tools/llama-engine/hydra_rpc/` (fork-isolated). For tensors
  placed on the peer's RPC device, the loader uses
  `RPC_CMD_RESOLVE_TENSOR` (already in PR #20) to get a pointer
  to the peer's mmap-resident tensor. No weight transfer.
- **Fix the two diagnostic-flagged paths from PR #31**:
  - `ggml-rpc.cpp:1601` "device index out of range" — ensure
    the peer's `ggml_backend_rpc_start_server_with_backends`
    uses the same backends list the head sees
  - `ggml-rpc.cpp:1649` "truncated graph" — call
    `RPC_CMD_RESOLVE_TENSOR` (already in
    `ggml_backend_rpc_bind_remote_tensor`) for every tensor
    in the peer's layer range before the first graph dispatch
- **E2E test**: 200-token decode through COMBINED on the
  3060 + 5060 Ti pair with DENSE 27B at Q4 returns coherent
  text; no device-index or truncated-graph errors. G1
  partially met.

**Estimated Phase 2 effort**: 1–2 weeks.

### Phase 3: Per-engine model loading (A tells B)

- Add `keep_layer_range` extension to `tensor_buft_overrides`
  in `src/llama-model-loader.cpp::create_tensor` (fork-isolated;
  no upstream struct change). Pattern `layer:N-M` matches
  tensors with `tn.bid ∈ [N, M]`.
- Add `POST /control/load_model` route to B's HTTP server.
  Body: `{model, keep_layer_range}`. B's handler unloads
  current model, sets the range, calls
  `llama_model_load_from_file`.
- A's `start_combined_with_peer` orchestrator parses the
  split, computes B's layer range, sends the load command.
- A and B can do their loads in parallel; A's
  `RPC_CMD_RESOLVE_TENSOR` only succeeds after B acks.

G4, G5 met. **Estimated Phase 3 effort**: 1 week.

### Phase 4: Hydra.Core orchestration (parent repo)

The C# side sends the right config to the right engine at the
right time. Fully in `ddvnguyen/hydra_vortex`:
- `WorkerSchedulerService` picks the head engine based on
  request model + mode
- `MultiEngineRouter.Select` extended to emit the per-engine
  config (model path, mode, peer endpoint, split)
- `WorkerInfo` carries the peer's RPC endpoint + current
  loaded model
- Profile switch (DENSE ↔ MoE) is `WorkerConfig.Reload()`
  with the new env vars; bounded by G7 (≤ 60 s)

G7 met. **Estimated Phase 4 effort**: 1 week.

### Phase 5: Profile + decide

After Phase 4, run `nsys` + `ncu` on a single forward pass on
the 3060+5060 Ti pair in COMBINED layer-split mode:
- If `RPC_CMD_GRAPH_COMPUTE` dispatch is > 20% of per-token
  time, proceed to Phase 6 (AF_UNIX)
- If graph dispatch is fine, stop here. SOLO G3 + COMBINED
  G1/G2 met.

**Estimated Phase 5 effort**: 2 days.

### Phase 6 (optional): AF_UNIX fork wrapper

Wrap `ggml_backend_rpc_*` in a Hydra-side file that opens an
`AF_UNIX` socket to `/var/run/hydra-<node>.sock` instead of TCP
loopback. Wire format unchanged. ~2× lower dispatch latency on
same-host. Cross-host stays on TCP. Fork-isolated in
`tools/llama-engine/hydra_rpc/`. **Estimated**: 1 week.

### Phase 7+ (deferred): shared memory, CUDA IPC

Only if Phase 5 shows the per-token path is still on the
critical path after Phase 6.

## Multi-model profile switch

Per C5, two models don't co-reside. The switch is:
1. Hydra.Core picks a worker (head engine A) for the new model.
2. A: unload current model + RPC peer, load new model, set up
   new RPC peer (~25 s for 27B Q4).
3. B: same, in parallel.
4. Both engines advertise new capabilities to Hydra.Core.
5. Hydra.Core resumes routing.

End-to-end: ≤ 60 s (G7). Reload is parallelized; wall-clock is
`max(A_load, B_load) + RPC rebind`.

We do NOT keep both models resident. The reload is the dominant
cost.

## Non-goals

- COMBINED-OT (expert-split with weight dual-loading). Layer-
  split only. `llama_hydra_load_combined_experts` and the
  related fields are dead paths; remove.
- A new ggml-RPC opcode set. We use upstream's opcodes + the
  Hydra protocol unchanged. The new thing is the unified server,
  not a new wire.
- COMBINED for MoE 35B as a throughput win over SOLO 5060 Ti
  (~48 vs 200 tok/s, by math). The MoE COMBINED win is
  resource-sharing (free the 5060 Ti), not throughput.
- Simultaneous DENSE 27B + MoE 35B residency (28 GB VRAM, not
  enough for both at usable quants).
- A custom CUDA shared-context multi-GPU.
- "Seamless" < 100 ms model swap between DENSE and MoE. The
  model reload dominates (~25 s); we don't try to keep both
  resident.

## Cross-repo links

- `ddvnguyen/hydra_vortex#376` — startup crash. Closed by Phase 2.
- `ddvnguyen/hydra_vortex#392` — Unified RPC Server epic.
  Updated to point at this design.
- `ddvnguyen/hydra_vortex#394` — split Phase C out of #31.
- `ddvnguyen/hydra_vortex#393` — per-request peer trust gate.
- `ddvnguyen/hydra_vortex#353` — COMBINED first-PREFILL crash.
  Closed by Phase 2.

## PR #31 outcome

- PR #31 is **superseded** by this design.
- The fix-hunks cherry-pick in Phase 1 is the new starting point.
- PR #31 is left open as a draft for reference; not merged.
- The data-integrity risk from PR #31's broad `RPC_STATUS_ASSERT`
  fail-soft replacement is real — that change is not adopted in
  the Phase 1 PR.

## Comparison: v4 vs PR #31

| Metric | v4 design | PR #31 |
|---|---|---|
| Files changed | 4 (3 new in fork, 1 deleted) + 4 small upstream additions | 15 files, 1255 ins / 707 del |
| Upstream-touched LOC | ~71 (4 small functions) | ~200+ (broad fail-soft, zero-inits, fattn.cu, from_fd) |
| Fork-isolated LOC | ~250 (new module) + ~150 net in llama-engine.cpp / server-context.cpp | ~800+ |
| Correctness (fail-soft scope) | Fail-stop in inference; fail-soft only in teardown | Broad fail-soft in 16+ places (data corruption risk) |
| DoS surface | Bounded thread pool (size 2) | Per-conn `std::thread::detach()` (unbounded) |
| No-weight-transfer property | Delivered (new buft + RPC_CMD_RESOLVE_TENSOR) | Not delivered (still uses RPC_CMD_ALLOC_BUFFER + RPC_CMD_SET_TENSOR) |
| Per-process model loading | Delivered (POST /control/load_model + tensor_buft_overrides layer extension) | Not delivered |
| Build verification | Phase 1 requires `cmake --build build_sm86_sm120 --target llama-engine` | Author admits "reviewed by inspection, not compiled" |
| Test coverage | Phase 1 + Phase 2 add unit + integration tests | None added |
| Config migration | Phase 1 updates workers.json + node-rtx.yaml | Not done |

The v4 design is ~3× smaller in upstream-touched LOC, removes
the data-corruption risk, and delivers the architectural
properties (no-weight-transfer, per-process model loading) that
PR #31 was supposed to deliver but didn't.

## Open questions

1. **Model storage location.** Each engine mmap's the same
   GGUF. Same path on every engine? Shared mount? Different
   paths? (Affects G4 cold-load time.)
2. **Engine startup model config.** Does each engine start
   with `--model X --port 8080` and that's it, with mode/peer
   sent per-request by Hydra.Core? Or is `--model` fixed at
   startup and only mode/peer is per-request? (Affects G6
   design.)
3. **Per-request payload to engine A.** The example has
   `mode`, `peer`, `model`, `split`. Is that complete? Does
   the peer (engine B) need any per-request config, or is
   it static-at-startup?
4. **Where does the split ratio come from?** Operator-
   configured in `workers.json` per (model, GPU pair), or
   auto-derived from each engine's reported VRAM?
5. **Per-request peer switch.** If A is currently in COMBINED
   with B and a new request asks for COMBINED with C, do we
   tear down B and bring up C inline (blocks the request), or
   queue and tear down async? (Affects request latency tail.)
6. **`tensor_buft_overrides` extension syntax.** Is the
   `layer:N-M` pattern sufficient, or do we need a more
   expressive syntax (e.g., `layer:N-M.*.ffn_.*=CPU` for
   subset of tensors within a layer range)?
7. **B's `/control/load_model` auth.** Per #393, the
   per-request `peer` field needs a trust gate. The
   `/control/load_model` endpoint has the same surface.
   Default: only accept from loopback (`--control-allow-from
   127.0.0.1`). Confirm.

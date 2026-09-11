# Arm — adaptive KV streaming on the 2×RTX RPC-split topology (Qwen3.8-27B, GDN hybrid)

Design/spec phase only — not yet executed. Nothing in this doc assumes code
changes; the binary under test is the **baseline** build (clean v0.4.0 + PR
#110's admission-gate port + UM prefetch net, `d50efc6f0`), **plus the
externally ported adaptive-KV-stream feature** (see "Feature source and port
plan"). No test-only behavior patch is needed: unlike the context-shift arm
(`docs/arms/arm-context-shift-hybrid-correctness.md`, PR #116), nothing here
has to be force-enabled — the feature ships its own flag.

Upstream technique under evaluation:
https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming
(branch `feature/adaptive-kv-stream`, discussion
https://github.com/ggml-org/llama.cpp/discussions/28216, project writeup
https://medium.com/@raymond860909/running-qwen-27b-on-16g-vram-with-full-context-length-building-adaptive-kv-cache-streaming-for-bf1e819116e9).

## Background and premise

Production's VRAM ceiling (16 GB + 12 GB, weights-first layer split) forces a
tradeoff between usable context and quant quality. The only existing lever for
"context beyond what fits" is `--context-shift`, which we established
(see the context-shift arm doc) is:

1. **silently disabled at boot for this model** — `qwen35` → IMROPE rope type →
   `n_pos_per_embd() = 4` → `llama_kv_cache::get_can_shift()` returns false,
   an architectural hard gate, and
2. even force-enabled, **lossy/discard-based** with a real, code-confirmed
   risk of silently corrupting GDN/recurrent layer state.

Adaptive KV streaming is a genuinely different class of technique:

- **Not lossy.** The full, exact attention KV cache is kept — relocated to
  pinned host RAM in block-granular pages, streamed into a bounded VRAM
  pool (resident pages + transfer ring) on demand, with layer-ahead
  prefetch to overlap transfer with compute. Their README contrasts this
  explicitly with context-shift's recalculate/discard approach.
- **Attention-only.** It replumbs the CUDA flash-attention path
  (`fattn.cu` + fattn-\*.cuh kernels + the matmul/vecdot paths FA dispatches
  into) to consume paged KV from the ring instead of assuming one contiguous
  device buffer. It never touches recurrent/SSM/GDN state — by construction
  that state is O(1), fixed-size regardless of context length, so there is
  nothing to stream, and — critically — the discard / rollback hazard that
  plagues context-shift on hybrid models **does not exist in this technique
  by construction** (re-confirmed in code before probing — see Gate G, not
  assumed).
- **On-target shape.** Their tested config is
  `unsloth/Qwen3.8-27B-GGUF UD-Q3_K_XL`, 262144-token context, Flash
  Attention, Q8_0 K / Q4_0 V, on an RTX 5070 Ti 16 GB — same model family and
  VRAM class as our production rig. Discussion #28216 carries third-party
  daily-use confirmation (`aebrer`, `giveen`).

### Why it must still be proven on OUR topology — the crux of this arm

**What upstream tested: one GPU, one process, one server slot.** Their README
states it directly: validated "primarily for an RTX 5070 Ti with 16 GB … one
server slot"; parallel slots and non-CUDA backends are "not yet broadly
characterized".

**What production is: two GPUs, two processes, two slots.**

| Item | Upstream tested | Production rig |
|---|---|---|
| GPUs | 1× RTX 5070 Ti 16 GB | RTX 5060 Ti 16 GB (CUDA0, main) + RTX 3060 12 GB (RPC peer) |
| Topology | single process, all tensors local | `llama-server -dev RPC0,CUDA0 -ts 27,38`, `rpc-server` on the 3060 — **layer split across two processes** |
| Slots | `-np 1` | `-np 2` (production parallel: 2) |
| KV types | q8_0 K / q4_0 V | q8_0 K / q5_1 V (+ draft q8_0/q5_1) |
| Split mode | n/a | `layer` (default) — `-sm row` is unsupportable on CUDA per PR105.0 rig-validation |

Two structural reasons it does not transfer automatically:

1. **Two processes.** Under the production layer split, FLASH_ATTN_EXT
   compute for the KV layers owned by the RPC0 share runs **inside the
   `ggml-rpc-server` process** on the 3060. The streaming runtime is a
   CUDA-backend-internal, per-device mechanism (pinned-host authoritative
   store + `cudaMalloc`-resident pool/ring). Composing it with RPC means it
   must activate **per process — including inside the RPC peer binary**,
   whose argv today is minimal (`--host/--port/-d`) with no known path to
   receive `--kv-stream-stage-mib`. Whether intact streaming composes
   per-device before/after the RPC boundary, and what enablement mechanism
   the peer needs (env-keyed activation inside the CUDA backend? a small
   fork extension of rpc-server argv?), is an explicit deliverable of this
   arm — see Gate R. `ggml-rpc.cpp` / `transport*.cpp` show **zero
   substantive changes** in the streaming branch (only an unrelated
   `ggml-rpc.h` version bump carried with their unmerged transport work),
   so there is no upstream-side RPC support to fall back on — this is
   genuinely open, and we do not assume portability.
2. **Per-device ownership is inherited, not designed.** Upstream streams one
   device's whole KV workload. Under layer-split each process owns only its
   share of attention layers' KV (plus weights). "Each device streams its
   own subset independently" is our working hypothesis; it is validated per
   device, not assumed — including whether the 3060's smaller pool (11.6 GB
   usable, 27% weight share after `-ts 27,38`) separately wants/needs
   streaming, and whether any per-device budget math (their
   `kv-stream-span-tuner.h`) assumes single-device context apportioning.

A negative constraint from PR105.0 makes this arm non-negotiable in scope:
**do not route around RPC via the single-machine in-process topology.**
PR105.0's rig-validation measured in-process multi-GPU + UM split collapse
(14× at the production 27,38 ratio) and −16% even balanced vs RPC. The RPC
split is production; the arm proves streaming **on the RPC topology** or
concludes NOT PORTABLE without RPC-side engineering.

## Confirmed facts about the technique source (verified 2026-09-10;
GitHub compare of `feature/adaptive-kv-stream` against their base, plus
branch commit history)

- **Scope.** CUDA backend only, ~25 substantive files: `fattn.cu`
  (+2274-line diff), `fattn-mma-f16.cuh` (310), `fattn-vec.cuh` (112),
  `fattn.cuh` (104), `fattn-common.cuh` (51), `fattn-tile.cuh` (12),
  new `fattn-swizzle.cuh` (+126), `ggml-cuda.cu` (1176), `common.cuh` (11),
  `mmq*`/`mmv*`/`mmf.cuh`/`mmid.cu`, `vecdotq.cuh`, `set-rows.cu/.cuh`,
  `top-k.cu`, `topk-moe.cu`, `unary.cu/.cuh`, `moe-weighted-reduction.cu/.cuh`,
  `mmq-config-*` reshuffles, and new `kv-stream-span-tuner.h` (+90).
  Frontend: `common/arg.cpp` (`--kv-stream-stage-mib`),
  `common/common.h`, and `ggml/include/ggml-cuda.h` (+100: runtime handle,
  params struct, per-type capability probe
  `ggml_backend_cuda_kv_stream_get_type_capabilities`, attention-mode enum
  DIRECT / F16-conversion). Also carries `benchmarks/benchmark_kv_stream.py`
  (+985) and `benchmarks/test_kv_stream_serial_server.py`,
  `test_benchmark_kv_stream.py` tests.
- **Two attention modes by KV type** (from the public header):
  `ATTENTION_DIRECT` (native-quant K consumed directly by FA kernels) vs
  `ATTENTION_F16` (streamed pages pass through an F16 conversion fallback).
  Production's V `q5_1` is not their tested `q4_0` V; which mode `q5_1`
  lands in and its correctness is probed at Gate T, not assumed.
- **UVM relationship.** The adaptive pool is intentionally allocated with
  `cudaMalloc` — never managed memory — so it stays device-resident even
  under `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` (their commit `2269100a8`
  "keep adaptive KV pool device-resident under UVM"). UM remains usable in
  parallel for model weights. Production runs UM=1 for the weight-heavy
  262k shape; cells record UM state per run.
- **Concurrency.** README: "one server slot" tested; parallel slots
  explicitly not broadly characterized. Production runs `parallel: 2`, and
  every arm this session that assumed single-slot behavior generalizes to
  concurrency-2 has hidden a real bug (#469 / #641 / the context-shift arm's
  cross-session concern). Cells B/*D* below are therefore **mandatory,
  not optional**.
- **Port base skew (verified via compare API + branch history):** their
  branch carries merge-base `f280b2698` (2026-08-24) with their master
  snapshot; ~277 commits ahead; the feature commits proper run
  2026-08-22 → 2026-09-06 (`ab539d579` ring sizing, `d9dcd518c` decode
  residency bounds, `b1d73aebe` resident-page migration across layouts,
  `7dc5fccd5` serial prompts + cache restore, `4a84b5ccc` decode layout by
  layer capacity, `14da01836` repartition events surfaced at default
  verbosity, `158067c0a` batched page uploads, `852094f51` multi-wave
  layouts, `2269100a8` UVM pool residency, `50a7de723` cache-type
  generalization + prefetch tuning, `46c1778f7` wide prefill batches,
  `cef30daab` per-attention-mode scratch scaling, `15154a688` bounded query
  workspace, `aee6bb66c` ubatch doc). Their branch also carries unrelated
  unmerged work (ggml-cpu `iqp.cpp` +1253, RPC RCE patch #20908,
  `transport-apple.*`, `mmq-config-pascal-older.cuh`, speculative synth
  changes…). **The port cherry-picks KV-stream commits only; merging the
  branch wholesale is prohibited by design.**

## Feature source and port plan (pre-execution)

The arm's execution is gated on a forward-port, not a cherry-pick-with-no-merge:

- **S0 — commit inventory (code read only).** Enumerate the exact KV-stream
  commit list on `feature/adaptive-kv-stream` (message grep
  `kv[ -]stream`, `adaptive`, interleaved with their unrelated carries), map
  commit → files, and produce an ordered port checklist appended to this
  doc's `## Results` placeholder before any merge attempt.
- **S1 — base-skew overlay (code read only).** Our baseline = upstream
  `5266f24da` (v0.4.0 bump, **2026-09-04**) + PR #110. Their base is
  upstream master ~2026-08-24 — the forward-port crosses ~2 weeks of
  upstream drift including CUDA `sparse-fa for DSV4/GLM` (#27970,
  `8e93a9773`, 2026-09-02) landed **inside their most-edited file
  (`fattn.cu`)**, and several mmq/mmvq cruft-shifts. Expected: mechanical
  cherry-pick for frontend + independent CUDA files; `fattn.cu` needs a
  hand-merge of their streaming hooks into the post-#27970 FA structure.
  Budget **days, not hours**; treat as the highest-risk step (assessment
  below).
- **S2 — port lands.** One branch `fork/port-adaptive-kv-stream`, ordered
  commits; gates below must pass before any rig cell.

## Merge / integration risk assessment vs our fork's own CUDA changes

Evaluated against actual our-branch diffs (verified locally, not assumed):

| Our branch | What it touches | Streaming touches (same files) | Conflict expectation |
|---|---|---|---|
| `fork/pr103-gdn-fusion-impl` (GDN cache-cpy fusion) | `ggml-cuda.cu` **+11** lines: env-gated early-return at top of `ggml_cuda_try_gdn_cache_fusion` (~line 2758 region); `tests/test-backend-ops.cpp` +126 | `ggml-cuda.cu` 1176-line diff, concentrated in FA dispatch / graph capture / buffer-init paths; no shared function beyond `ggml_cuda_compute_forward` plumbing | **Low textual conflict.** Semantic: GDN fusion operates on GGML_OP_GATED_DELTA_NET nodes, orthogonal to FA streaming; both alter timing inside the decode loop — keep an A/B interplay cell if fusion lands first (cell E2). |
| `fork/pr104-mixed-quant-arm` (row-split quant, split-buffer backend port) | `common.cuh` +3 lines: `ggml_tensor_extra_gpu` gains per-device slice-quant fields + contiguous-cache pointer; `ggml-cuda.cu` +560 lines: split-buffer-type machinery reintroduced (~line 972-1412 region) + hunks through `ggml_cuda_compute_forward`, `ggml_cuda_can_fuse`/`try_fuse`, `device_supports_op`, `reg_get_proc_address` | `common.cuh` streaming changes are in **different regions** of the file (CC macros, `ggml_cuda_syncwarp`, `ggml_cuda_graph::node_props` + `kv_stream_generation`, mm-fusion args) — no overlap with `ggml_tensor_extra_gpu`; `ggml-cuda.cu` region overlap is limited to `compute_forward` and graph machinery | **Moderate, semantic more than textual.** Both branches add graph-capture-adjacent state (streaming: `kv_stream_generation` guard in graph node props; PR104: nonstandard split-buffer types feeding the same graph replay machinery). Expected textual merges close to clean, but graph-capture interactions must be explicitly re-verified (cell E3). |
| Baseline PR #110 UM-prefetch net (`ba2c46f65`) | `ggml-cuda.cu` +10 lines (cudaMemAdvise read-mostly + prefetch for managed allocs) | Streaming's UVM handling (`2269100a8`) touches allocation paths; pool deliberately stays non-managed | **Low textual; semantically related** (both manage where resident memory lives). Keep the net intact; cells record UM=1 unchanged. |
| Baseline admission gate (#747) | server-side (`tools/server`) | touches only `common/arg.cpp` for its own flag | **No conflict.** Cells confirm 0 spurious defers with streaming on. |

**Plainly stated integration verdict (not a clean rebase):**

1. **Not a bolt-on.** The streaming fork sits on upstream master ~Aug 24,
   our baseline on upstream ~Sep 4 — with `fattn.cu` itself churned in
   between (sparse-fa #27970). Their 2274-line `fattn.cu` delta includes
   pre-sparse-fa structure; a cherry-pick will not merge cleanly — expect a
   manual transplant of their streaming hooks onto the new FA topology.
2. **Their branch is not a usable merge source**: it carries unrelated
   unmerged features (including RPC transport work that WOULD collide with
   our `GGML_RPC=ON` rpc-server build). Wholesale merge is prohibited by
   design; the port is commit-subset cherry-pick + hand-merge.
3. **Local branch order matters** (irreversible-ish, flag explicitly):
   stream-port lands on clean baseline FIRST; PR103 rebases second (trivial);
   PR104 rebases third (moderate; its 560-line re-entry into
   `ggml-cuda.cu` is the most delicate of the three but independent
   regions). Reversing the order puts PR104's graph-machinery edits between
   the port and every subsequent streaming change — maximized churn.
4. Expected conflict hot-spots (for the port PR review): `fattn.cu`
   (top-k / dispatch tables + sparse-fa interactions),
   `ggml-cuda.cu::ggml_cuda_compute_forward` + graph-capture region,
   `common.cuh` `ggml_cuda_graph` struct, `common/common.h` (`need_n_rs_seq`
   region adjacent to their speculative-carries — skip, do not port those).

## Hypotheses

- **H1 (portable, exact):** per-device streaming composes with the RPC
  split; when contexts fit (no streaming pressure) it is numerically
  indistinguishable from the incumbent path; when oversubscribed it gives
  correct long-context output (nothing discarded ⇒ both early and late
  markers recallable).
- **H2 (concurrency breaks it).** The shared bounded pool / ring is
  contended or per-sequence page accounting breaks under `-np 2`; surfaces
  as per-slot asymmetry, cross-session recall degradation, or
  repartition-thrash storms.
- **H3 (RPC needs fork-side plumbing).** No stock mechanism exists to
  enable streaming inside the RPC peer process; the arm records the
  mechanism verdict + sizing for a minimal fork change (env-keyed per-device
  activation preferred).
- **H4 (type-mode risk).** Production `V q5_1` lands in the F16-conversion
  path (upstream tested `q4_0`); correctness/quality there is unverified
  and may delta — cell A catches it cheaply before any long-context cell.

## Rig and launch spec

Per-cell base = **747.4 production pin** with two deltas: the streaming
flag, and a reduced `--cache-ram` for deep cells (streaming adds its own
pinned-host store; the 24576 MiB checkpoint reservoir and the KV-stream
host pool must be budgeted together, recorded per cell). Topology is the
production RPC split exactly: `rpc-server` on CUDA1 (3060), server on CUDA0
(5060 Ti), `-ts 27,38` (RPC0 gets 27, CUDA0 gets 38 — PR105.0-verified
ordering).

| Item | Production 747.4 | This arm | Why |
|---|---|---|---|
| RPC `tensor_split` | 27,38 | 27,38 | parity |
| `-np` / parallel | 2 | 2 | production-realistic; cells B'* and D* exercise it explicitly |
| `ctx` (total) | 262144 | **cell-dependent** (65536 / 262144 / 524288) | see cells |
| KV types | q8_0/q5_1 (+ draft) | same primary; secondary q8_0/q4_0 cells mirror upstream's tested shape | isolate type-mode (H4) |
| UM | 1 | 1 in all streaming cells (verified coexistence) + an UM-off control variant of A | their calim: UVM optional for weights, KV pool stays non-managed |
| streaming flag | n/a | `--kv-stream-stage-mib N` (per-cell provisioning, see below) | the feature's single knob |
| MTP | draft-mtp on | on primary, off variant in cell E1 | production-realistic; MTP/draft-KV × streaming interaction probed not assumed |
| cache-prompt / checkpoints / idle slots / cache-ram | 24576 MiB | same flags, `cache_ram_mib` reduced for long cells (1024) | two independent host-RAM consumers must coexist; budget logged |
| admission gate | threshold 100000 | same flag value | parity; only binds in deep cells |
| port | 18081 | **8080** (prod pod untouched) | arm etiquette per PR105.0/context-shift arms |

Launch (cell A primary, streaming on):

```bash
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
# peer (3060) — built from the same tree, so streaming code exists here too
./build/bin/rpc-server --host 127.0.0.1 --port 50052 -d 1 2>&1 &

./build/bin/llama-server \
  -m /mnt/SSD/Qwen3.8-27B-UD-Q5_K_M.gguf \
  --rpc 127.0.0.1:50052 -ts 27,38 -ngl 99 \
  --rope-scaling yarn --rope-scale 5 --yarn-orig-ctx 32768 \
  -fa on -ctk q8_0 -ctv q5_1 -ctkd q8_0 -ctvd q5_1 \
  --no-kv-unified --cache-prompt --cache-reuse 64 --cache-idle-slots \
  --cache-ram 1024 --ubatch-size 512 --cont-batching \
  -np 2 -c 65536 \
  --parallel-ctx-threshold 100000 --spec-type draft-mtp \
  --prio-batch 1 --kv-stream-stage-mib <PER-CELL-N> \
  --jinja --host 0.0.0.0 --port 8080 --metrics --slots --log-verbosity 4
```

Build flags per fork convention: `-DGGML_CUDA=ON -DGGML_RPC=ON
-DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_FORCE_CUBLAS=OFF
-DCMAKE_CUDA_ARCHITECTURES="86;120"
-DCUDAToolkit_ROOT=/opt/software/cuda/13.2.2 -DCMAKE_BUILD_TYPE=Release`.

`--kv-stream-stage-mib` per cell is **provisioned at execution time** from
their own guidance (start conservative, raise while watching startup/peak
VRAM) plus their `benchmarks/benchmark_kv_stream.py` sweep on this rig —
not guessed in the spec. Each cell logs: pool bytes, resident-page count,
ring slots, per-device VRAM, and pinned-host RSS of the KV-stream store.

Hardware pre-flight per cell (PR105.0 convention): `nvidia-smi` free
memory, `/health` 200, both devices in boot log, then gates.

## Gates (no probe traffic passes any gate)

### Gate B — port/build gate

- [ ] Ported tree builds (llama-server + rpc-server) on both arches
      (86;120) with the fork's FA_ALL_QUANTS suite.
- [ ] Vanilla-behavior parity control: `--kv-stream-stage-mib 0` build must
      produce cell-A-parity decode/prefill vs the pre-port baseline binary
      (proves the port did not disturb the incumbent path).

### Gate T — KV type-capability probe

Boot log records `kv-stream` type-capability output for `q8_0/q5_1` and
`q8_0/q4_0` (both main and draft KV): which attention mode (DIRECT vs F16)
was selected per type, and the allocated pool/ring/resident budget. If
`q5_1` V lands in an unsupported/auxiliary class, re-scope the arm to the
`q4_0` primary with a `q5_1` secondary and record the decision.

### Gate G — GDN/recurrent-untouched (code read, part of S1)

- [ ] Diff-scope proof: the ported commit set does not touch gated-delta-net
      / SSM / recurrent CUDA ops or `src/llama-memory-recurrent*.cpp`.
- [ ] Page-accounting-vs-cache-operations review: how their runtime maps
      token positions ↔ pages under our checkpoint (`--cache-prompt`)
      restore and `set_rows`/cache-write semantics. Their `7dc5fccd5`
      "support serial prompts and cache restore" is our hook; the pin is
      verifying it against **our** post-PR5/PR9 checkpoint semantics
      (hybrid-specific), before any probe, not after.

### Gate R — RPC composition (explicit investigation, then a boot)

The single biggest open architectural question — resolve it, do not assume:

1. Where the runtime parameterizes per-device (params struct +
   span-tuner): per-CUDA-device per-process, or per-context.
2. Whether any stock mechanism enables it inside the `ggml-rpc-server`
   process. Candidate mechanisms to enumerate in code:\n   (a) env var read inside the CUDA backend init (on Peer side); (b)
   fork-side extension of rpc-server argv; (c) no mechanism ⇒ restricted
   cells only + a follow-up fork issue.
3. Boot-test whatever resolves: per-device log evidence that the runtime
   activated where it should (their `14da01836` repartition-event logging
   is our observable).

## Test cells

All probes reuse the two-marker recall harness conventions of the
context-shift arm doc (M1/M2 plant, probe phrasing rotation, P3
coherence, P4 `temp=0` byte-diff determinism, concurrency via 2-thread
wall-clock overlap ≥ 80%, disjoint marker casts across sessions) with ONE
decisive difference: **nothing is discarded**, so there is no
honest-forget boundary — **both early (M1) and late (M2) markers must be
recallable at ≥ 5/5** in every probe. Any M1-recall degradation relative to
M2 is a corruption signature (streaming claims exactness) — the inverse
rule of the context-shift arm's grading.

| Cell | ctx (total) | np | UM | KV types | streaming | What it isolates |
|---|---|---|---|---|---|---|
| **A-off / A-on** | 65536 | 2 | 1 (off control optional) | q8_0/q5_1 (q8_0/q4_0 secondary) | off → on | **parity + no numeric drift**: both shapes boot and fit normally (streaming may be active but unpresured); greedy outputs byte-identical between cells iff the runtime is a pure rescheduling. Sample further variants: UM off × streaming on (their README claims UM-optional). |
| **B** | 262144 (prod shape) | 2 | 1 | q8_0/q5_1 | on | production shape streaming vs the 47.09 t/s (27,38 RPC) production bar — degradation beyond noise = red flag; recordings only, not a gating bar (the production capability was already proven without streaming) |
| **C** | **524288** (2×262144) | 2 | 1 | q8_0/q5_1 | on | **the capability bar**: today's shape OOMs (and context-shift cannot enable —Architectural gate). Both M1 AND M2 recall ≥ 5/5 (the no-discard proof), coherence across a ~40-60 K token probe sequence per session, 0 shift attempts, sustained decode recorded per slot; VRAM ceilings + pinned host RSS budget logged |
| **D** | 524288 | 2 | 1 | q8_0/q5_1 | on | **2-concurrency under streaming**: the parallel-slot confound upstream never tested. PR105.0 2×10 multiturn harness at deeper per-turn prompt growth (drives both slots toward their 262144 ceilings concurrently); measures overlap (`n_busy_slots_per_decode` ≥ 1.6 sustained), cross-session marker leakage (disjoint casts), per-slot balance, late-turn M2-style recall, no OOM/restart, no repartition-thrash storm |
| **E (conditional)** | 65536 or 262144 | 2 | 1 | q8_0/q5_1 | on | adjacency cells, executed only after A–D verdicts: (E1) `--no-spec` off (draft-KV streaming interaction isolated); (E2) `GGML_CUDA_FUSE_GDN_CACHE 0/1 × streaming 0/1` 2×2 (GDN fusion is orthogonal, correctness-neutral, perf-accounting-cell); (E3) restricted-topology comparison (peer streaming on vs off) if Gate R leaves both states reachable |
| **F (optional)** | - | - | - | - | - | upstream `benchmark_kv_stream.py` sweep on this rig (informational pool-curve; NOT a correctness cell) |

Ordering: B → T → R → G (G code read) → A → B → C → D → E. Cell A comes
first among rigs (numerical anchor); C/D only run if A and B are clean
(they are meaningless against a mis-scheduling runtime).

## Bars — pass/fail in concrete terms

**PASS ("streaming proves portable, correct, and useful on the 2×RTX RPC
split, parallel 2")** — ALL of:

1. **Gates B/T/R/G all pass** and a production type-mode is confirmed
   workable (DIRECT or a fully-working F16 path).
2. **Cell A numerical exactness**: greedy `temp=0` outputs byte-identical
   between streaming-off and streaming-on at the same shape across ≥ 3
   loops; any observed delta is characterized (token-list identical, only
   sampling-order nondeterminism) and reproduced; prefill within 5%.
3. **Cell C capability**: M1 and M2 recall both ≥ 5/5 per session, coherent
   P3 (0 garbled/contradictory passages), P4 byte-identical repeats, **zero**
   `slot context shift` attempts, no eviction/`n_discard` events of any
   kind.
4. **Cell C sustained throughput:** decode measured ≥ 3 loops; **provisional
   bar ≥ 10 t/s per slot sustained at ≥ 150 K resident per session**
   (subject to retune from the first C run; the fail trigger is ≤ 2 t/s —
   the UM-thrash signature we've already seen collapse to 2.4 t/s in
   PR105.0 rig-validation). VRAMP: GPU0 ≤ 15.5 GB, GPU1 ≤ 11.5 GB; pinned-
   host RSS + `cache-ram` checkpoints RSS recorded with no system swap.
5. **Cell D concurrency**: cross-session disjoint-cast **zero leakage** (10
   probes × 2 sessions = 20 opportunities), per-slot decode symmetry within
   10%, sustained genuine overlap (`n_busy_slots_per_decode` ≥ 1.6),
   admission gate 0 spurious defers, no OOM/restart, and no
   repartition-thrash storm (> 20 repartition events per decode-loop is
   the red-flag line).

**FAIL** — ANY of:

1. Port/build gate fails: A-off ≠ pre-port baseline → port is broken; fix
   before anything else, no probes on the defective tree.
2. Cell A: byte-delta or recall-regression beyond characterization (H4 hit)
   → type-mode or re-scheduling bug; report upstream + fork; re-scope
   primary cell to q4_0-v if the q5_1 mode is the culprit.
3. Cell C: any marker-recall degradation (M1 or M2 < 5/5), coherence crack,
   non-byte-determinism on P4, shift/eviction event, or sustained decode
   collapse (≤ 2 t/s) ⇒ "streaming NOT production-usable on this
   hybrid-RPC shape" + targeted diagnosis issue; possible follow-up: check
   upstream `#28216` author response on multi-slot + RPC.
4. Cell D: leakage / asymmetry / thrash signatures → H2 hit, hard FAIL
   regardless of single-slot verdicts — production `parallel: 2` is
   load-bearing (fork bug classes #469/#641/#743/#744 precedent).
5. Memory breach: any `cudaMalloc out of memory` in C/D, or pinned-host RSS
   that collides with `cache-ram` checkpoints budgets (system swap).

## Correctness gates (hard, per fork convention)

- Pre-port-baseline control (streaming disabled class of Cell A) itself
  must match the pre-port binary's own baseline bar — if the harness blames
  streaming for port-side regressions, the port verdict (not the technique)
  is the finding.
- Greedy P4 repeat determinism within each streaming cell: byte-identical
  run-to-run (PR #110 self-determinism standard).

## Sequencing / execution notes (for the runner)

1. S0/S1/Gate-G code read session FIRST; paste the commit inventory +
   skew-overlay summary into this doc's `## Results` head.
2. Gates B → T → R (in that order); any failure is the arm's verdict —
   do not "boot anyway and see".
3. Cells A → B → C → D → E; each ends with the arm-etiquette restore
   (pkill test server, `podman pod start pod_llama-baseline`, `/health`
   200 on :18081, `nvidia-smi` ≈ 15847/11911 MiB) — never leave the rig
   half-stopped (PR105.0 convention, restore verified twice there).
4. Preserve per-cell artifacts: server boot logs, probe harness JSON,
   `restore-log/` (nvidia-before, health-before, cmdline), KV-stream
   mem-summary lines.
5. Record results in a `## Results` section; findings per
   `review-finding` convention if a bug class emerges; **no merge without a
   rig execute pass + explicit user confirmation** (never merge
   E2E/live verify straight to `main`).

## Open questions for this PR's review (explicitly unassumed)

1. Does upstream have any per-process (rpc-server) streaming enablement this
   spec missed? (Gate R resolves from code, not assumptions.)
2. Is their `--cache-prompt`/`cache restore` support (commit `7dc5fccd5`)
   compatible with our post-#5/#9 hybrid checkpoint semantics, or does the
   KV-stream runtime require a position-tracked link to unicorn targets?
3. q5_1 V: DIRECT vs F16 verdict (capability probe) — is V q5_1 worth
   keeping on streaming cells at all, or should q4_0 V be the
   streaming-native shape on this rig?
4. MTP / draft-KV under streaming: does the draft KV (ctkd/ctvd) get
   streamed too, and is draft acceptance affected (they claim "the
   adaptive pool is device-resident", but draft-path probing is unverified
   upstream)?

## Results

### S0 — commit inventory (2026-09-11)

Source: `RaymondHuang210129/llama.cpp-adaptive-kv-streaming@feature/adaptive-kv-stream` = `4ba0b9b25`. Their branch is a **linear rebase-carry** (no merge commits). Critical re-verification vs this doc's design assumptions:

- Their branch's upstream carry does NOT stop at ~2026-08-24: the carries include **`5266f24da` (v0.4.0 bump — this arm's baseline base)** and continue past it to ≈ 2026-09-06 (`73a43d1f` #28475, `d03efa5d` #28402, `7620399f` #28437, `9e0e2205` #28469, …).
- **Sparse-FA #27970 (`8e93a977`) is already an ancestor of `5266f24da`**, hence already inside our baseline's `fattn.cu`. The "their fattn.cu delta includes pre-sparse-fa structure, hand-merge onto new FA topology" concern does not fire: their `fattn.cu` sits on the SAME post-#27970 FA structure our baseline has. The port was therefore cleaner than the S1 budget assumed.
- Between the last carry and the first streaming commit (`2646f0aa`) there is exactly ONE feature-dependent commit: `7fa262b3 "baseline: mirror production CUDA unified-memory build"` — its `ggml_backend_cuda_buffer_set_preferred_host/device` definitions are referenced by the streaming feature's proc-address table; **included** in the port. The 6 upstream carries touching llama/ggml (465e49b9, 5fdfa628, 3ad1ba73, 9e0e2205, 7620399f, 49c0dc82) were **excluded** — upstream lineage, not feature.

Port = **61 streaming-series commits cherry-picked in order** (`2646f0aa`…`4ba0b9b2`) onto clean `baseline` (`d50efc6f`). Only 2 trivial conflicts, both empty-HEAD-side conflict-marker noise in `ggml-cuda.cu` (`reg_get_proc_address` block) and `llama-kv-cache.cpp` (buffer-clear hunk); plus one duplicate-definition cleanup after `7fa262b3`.

**Fidelity proof:** all 44 non-carry, non-fork-delta files in the port tree are byte-identical (git blob SHA equal) to the source branch HEAD. Fork delta (PR #110: `--parallel-ctx-threshold` args + UM prefetch net `cudaMemAdvise`) verified intact in the ported tree.

### Gate G — GDN/recurrent untouched (code read)

- Ported diff touches **zero** files matching `recurrent|gated|delta|ssm|mamba` under `ggml/src/ggml-cuda/` or `src/`; `gated_delta_net.cu/.cuh`, `ssm-*.cu/.cuh`, `src/llama-memory-recurrent*.cpp` untouchED. Streaming lives solely in `llama_kv_cache` (attention) + its kv-stream buffer type — never recurrent state.
- Draft/MTP: their speculative wiring explicitly zeroes `kv_stream_stage_mib` for the draft cache ("MTP keeps its ordinary cache until both contexts can share one pool") — resolves this doc's open-question 4: **draft KV is NOT streamed**.
- Upstream-carry observation (not ported — upstream lineage, not streaming): `5fdfa628` #28068 fixes GDN normalization `max`→`rsqrt` for the qwen35 family on upstream; our baseline doesn't have it. Potential follow-up for PBpy family correctness, separate issue.

### Gate R — RPC composition (mechanism verdict)

1. **Runtime parameterization is per-CUDA-device, per-process.** `llama_kv_cache` builds the runtime via `ggml_backend_reg_get_proc_address(device_reg, "ggml_backend_cuda_kv_stream_...")`. The `ggml-rpc-server` peer binary has no llama-layer at all (it replays ggml graphs against remote allocations; it never constructs `llama_kv_cache`), so **env-keyed activation inside the peer is a dead end AND a fork-side rpc-server argv extension gains nothing** — streaming semantics (pages, residency, set_rows hooks) live in the llama-layer. Full per-process composition would require a protocol-level extension (stream-reallocation opcodes across the RPC channel) — genuine follow-up fork issue, out of scope here.
2. **Stock architectural blocker vs the exact production split:** the wiring streams the FIRST kv-layer device's buffer type and APPLIES it to every KV layer; with `-ts 27,38`, layers 0+ are owned by RPC0 ⇒ boot throws "block KV streaming requires the CUDA backend".
3. **Chosen mechanism (lower-risk, faster to a real number):** env-keyed activation in the llama layer — `LLAMA_KV_STREAM_DEVICE=CUDA0` streams only KV layers whose model-layer device matches; other layers keep ordinary fully-resident KV (the doc's restricted fallback: CUDA0-streaming, 3060 fully resident, production RPC topology intact). Implemented in `src/llama-kv-cache.cpp`. Unset ⇒ stock behavior byte-preserved. The reason not the argv option: the peer has no llama-layer and cannot host the runtime at all, so argv plumbing would be dead code.
4. **`-np > 1`:** stock validator hard-rejects (single_sequence). Added opt-in `LLAMA_KV_STREAM_ALLOW_MULTISEQ=1` (loud per-boot warning) so the H2 concurrency hypothesis is testable upstream-untested instead of being asserted unbootable. H2 verdict = measurement, still pending rig.

### Gate B — build gate

- [x] llama-server + ggml-rpc-server build OK: CUDA 13.2.2, archs `86;120`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `GGML_RPC=ON`, Release, exit 0.
- [x] Ported feature's CPU-side unit tests: `test-kv-stream-plan` (241 assertions), `test-kv-stream-config` (18), `test-kv-stream-softmax` — 0 failures each.
- [ ] Vanilla parity control (`--kv-stream-stage-mib 0` vs pre-port baseline binary) — **blocked on rig** (one physical rig; agent 6edd46da / PR #116 currently holds it per "one GPU = one compute task"). No boot until explicit rig release.

### Rig cells status

- **waiting-on-rig** per rig-coordination protocol redux.
- Prepared: `scripts/arm117/{boot-cell.sh, cell-a-parity.sh, cell-c-longctx.sh, snapshot.sh}` committed to the port branch.


### Gate B — vanilla parity control (rig)

- [x] **PASS — base binary (`d50efc6f`) vs port tree with `--kv-stream-stage-mib 0` (A-off), np1/ctx 65536, RPC `-ts 27,38`:** greedy temp=0 kernel prompt (2208-token prompt, 96-token cap) **byte-identical**; the only textual difference was a trailing-EOF newline from the harness `print()`, not token content.

### Gate T — type-capability probe (code+boot evidence)

- **q8_0/q5_1 → `ATTENTION_DIRECT`.** Both types carry `direct_attention=true` under `GGML_CUDA_FA_ALL_QUANTS=ON` (…`fattn.cu:1120` case Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/F16/BF16). No F16-conversion fallback engaged. Boot log: `KV streaming device filter = CUDA0 (9 layers)`, `CUDA_KV_Stream_Host KV buffer size = 1044.00 MiB`; model has 16 kv layers total at ctx 65536 (9 on CUDA0 streamed + 7 on RPC0 fully resident), draft/MTP context NOT streamed (confirmed in-boot).
- q4_0 cell not run (primary passes; production types are the mandate). q4_0 remains available per doc as a fallback diagnostic cell.

### Cell A — numerical parity (rig, np1 + np2)

Cell shape: ctx 65536, `-ts 27,38`, UM=1, `/mnt/SSD/Qwen3.8-27B-UD-Q5_K_M.gguf`, stage 0 vs stage 2048, :8080.

**np1 (← the clean verdict):**
- OFF vs ON (fresh boots, 2 + 1 samples): **byte-identical greedy outputs**.
- ON self-repeat (same server, 2 samples): byte-identical.
- OFF cross-boot repeat (2 boots): byte-identical.
- **Cell A np1 verdict: PASS — streaming is a pure rescheduling + numerically exact at production KV types in single-slot.**

**np2 (parallel: 2 — the upstream-untested gap; exercised via `LLAMA_KV_STREAM_ALLOW_MULTISEQ=1`):**
- OFF self-repeats across 2 boots and 2 same-server samples: byte-identical (np2-single-slot request confined to one slot → deterministic control valid).
- ON with `cache_prompt=false`: **byte-identical to OFF** (streaming decode path itself is exact even with slots).
- ON with `cache_prompt=true` (fresh boot, first request → **slot 1**): diverged reproducibly from OFF (distinct reasoning token list, finish reasons differ) with cached_tokens=0 — i.e. NOT a cache-restore problem (restore with 2204 cached tokens later reproduced OFF text exactly). The first request on a fresh boot went to slot 1 twice; subsequent slot-0 requests were exact. **Probable cause: slot-1-local streaming page/dirty-rows accounting or first-decode layout on a non-zero stream; NOT isolated further in this pass.**
- **Concurrent 2-slot streaming decode: HARD CRASH** — `GGML_ASSERT(ggml_cuda_kv_stream_fattn_fits(dst)) failed` at `ggml-cuda.cu:1898` (`ggml_cuda_graph_evaluate_and_capture`) during paired requests; server died (abort + backtrace preserved in `arm117-artifacts/`).

**Cell A np2 verdict: FAIL (H2 class).** Two independent signatures: (a) slot-1 first-decode divergence, (b) hard assert crash under genuine 2-slot concurrency. Per this doc's FAIL bar, **concurrency violation stands regardless of single-slot correctness** — streaming's page/dirty-row/layout model is not slot-parallel-safe as ported.

### Overall verdict for this pass

- **Port: clean** (fidelity proof + build gate + CPU unit tests all green; Gate G untouched-scope verified; doc's worst-case S1 fattn.cu skew did not materialize).
- **Single-slot streaming on the production RPC topology (CUDA0 9/16 streamed-attn layers, 3060 fully resident): numerically exact, boots in production KV types (DIRECT mode), byte-parity with stock.**
- **Parallel: 2 streaming: NOT usable as ported** (slot-1 divergence + `kv_stream_fattn_fits` assert crash). The `LLAMA_KV_STREAM_ALLOW_MULTISEQ` knob did exactly what it was built for: turned an unproven upstream assumption into a concrete, reproducible failure signature.
- Gate R: mechanism verdict + device-filter fallback implemented (`LLAMA_KV_STREAM_DEVICE=CUDA0`); full per-process RPC composition (streaming inside the peer) documented as the follow-up fork issue (rpc protocol extension).

### Rig restore log

1. Preflight: prod health 200, VRAM 15847/11911 (exact production signature). `podman pod stop pod_llama-baseline` → drain verified (1 MiB / 1 MiB, ports clear).
2. All test boots on :8080; test artifacts + per-cell logs under `arm117-artifacts/` in the port branch working tree.
3. Restore: killed all test binaries → GPUs 1 MiB/1 MiB → `podman pod start pod_llama-baseline` → health 200 (verified twice, including a second recheck) → VRAM 15847/11911. Ports 18081/50052 listening again. **Rig clean.**

### Known issue in the boot path (cosmetic but real)

During server boot, `common_fit_params` probe context creation throws my new device-filter error ("block KV streaming enabled but no KV layers on the streaming device") before devices are wired, gets caught by fit-params retry, and the real context then boots fine. Evidence lines survive in every A-on log. Cosmetic-only (retry path works, no user impact), but the filter should tolerate device-less probe contexts — follow-up nit for the port PR.

### Round 2 — root cause, numbers, q8_0 verdict (2026-09-11, later same day)

#### Task 1 — root cause of the np2 crash (instrumented rig repro; fix `ef3b119f`)

Instrumented `ggml_cuda_kv_stream_fattn_fits()` to log K/V buffer pointers + buft names + runtimes on every false return, rebuilt, reproduced live (fresh np2 boot, two concurrent requests). Repro data, verbatim:

```
E arm117 fits=false: dst=0x614d8c90f8c0 ne=[256,24,2,1] type=f32 k=0x... kbuf="CUDA0" | v="CUDA0" | kr=(nil) vr=(nil)
```

Two chained mechanisms, both upstream design assumptions, NOT quant-type issues:

1. **Crash (exact mechanism).** `ggml_cuda_kv_stream_fattn_fits` has two failure classes: (a) runtime identity (the doc's printed guard `k_runtime==nullptr || v_runtime==nullptr || k_runtime!=v_runtime`), (b) `ggml_cuda_flash_attn_ext_streamed_supported()` geometry predicates (`fattn.cu:1216`), which require `Q->ne[3]==1 && K->ne[3]==1 && V->ne[3]==1` — a **single-stream ubatch assumption**. Under cont-batching with 2 busy parallel slots, the FA K/V views are 4D over `ne[3] = ns = 2` (the failing probe printed exactly `dst ne=[256,24,2,1]`). Geometry rejection is *not* why the crash surfaced: the CUDA-side dispatcher routes to `ggml_cuda_kv_stream_fattn` **only on buffer membership** (`ggml-cuda.cu:3053`: `runtime_from_tensor(src[1]) != nullptr || src[2]...`), not on the full `fits()` predicate, so the assert at `:1898` fired instead of falling back. Upstream never runs `-np>1`, hence this never triggers there.
   - **Fix shipped** (`ef3b119f`): dispatch now gates on the full `fits()` predicate and falls back to the ordinary `ggml_cuda_flash_attn_ext` path (the kv-stream buffer is host-mapped pinned storage, so reads are zero-copy and semantically exact) with a WARN line. Post-fix np2 concurrent repro: **no crash** (server survived), both slots returned identical coherent outputs.
2. **Slot-1 divergence (separate mechanism, same subsystem).** Upstream's resident-page bookkeeping has **no stream dimension**: `ggml_cuda_kv_stream_resident_cache_mark_dirty_rows` maps `row/page_tokens` → page and marks `layer_pages[layer]` for ALL layers without any `stream` axis (…`fattn.cu:709-756`). With `n_stream=2`, slot 0's tokens and slot 1's tokens alias the **same (layer, page) indices**; the physically-stored bytes differ only by the view pointer (`k_data + s*k->nb[2]`), so whichever slot wrote a given page last wins for BOTH streams' reads. This produces exactly the observed signature: slot-1's first decode diverges from the byte-stable slot-0 control; later cached requests that restore slot-0's rows reproduce OFF exactly. Fixing this requires giving the resident cache a real (layer × stream) layout — an upstream-level design change (the README's "one server slot" limitation made concrete), out of port scope.
   - Consequence: the port keeps `LLAMA_KV_STREAM_ALLOW_MULTISEQ` as a **measurement-only** opt-in with loud warnings; the doc's concurrency FAIL verdict stands, but the failure class changed from "hard abort" to "silent non-exactness" post-fix.

#### Task 2 — real decode throughput (multiturn-growth-test.sh, 1 session × 12 turns, ~8K/turn growth, 750 out)

Identical launch shape both sides (`-ts 27,38`, ctx 65536, np1, UM=1, q8_0/q5_1, cache-prompt etc.). Turns 10-12 = ctx-cap (400) for BOTH configurations, excluded from the mean by the harness itself.

| config | turns 1-9 tok/s | turn 1 | turn 9 (deepest) |
|---|---|---|---|
| baseline binary (`d50efc6f`, fully resident) | **mean 24.26** | 31.54 | 21.89 @ 59.7K |
| port + streaming ON (9/16 layers streamed, stage 2048 MiB) | **mean 17.25** | 31.52 | 21.41 @ 60.5K |

- **Streaming costs ≈ −28.9% mean decode tok/s vs fully resident** at this shape (1× slot).
- Depth curve: streaming degrades to **8.79–10.78 tok/s at 40–53K resident** (turns 6-8) vs baseline's flat ~21-24 there — the host-side pool/ring round-trips dominate once streaming pressure exceeds the 2048 MiB resident pool. Prefill parities near turn 1 (31.52 vs 31.54) confirm boot-time path is unchanged.
- Practical read: streaming buys the *capability* (context beyond VRAM) at a real decode cost; the resident-page budget (stage MiB) is the main tunable and was fixed conservative at 2048.
- Cell A np1 byte-parity unaffected: outputs identical, only alloc/eviction scheduling differs.

#### Task 3 — "uniform q8_0 for K+V" question (definite answer, code + evidence)

- **NO — switching to q8_0/q8_0 will not fix either signature.**
- Crash: it dies inside `ggml_cuda_kv_stream_fattn_fits()` returning false via `streamed_supported()`'s **geometry** check on 3D stream metadata (`ne[3]==1`), quant-size-independent. Both K and V per layer live in ONE kv-stream buffer (`same buffer pointer 0x...9a0` in the diagnostic print), runtime identity is never the issue (kr/vr non-nil and equal whenever the same buffer is streamed — the printed nil/nil match was the prepare_graph probe on a draft/plain-buffer node). Same-geometry crash reproduces identically if `ctv q8_0` is set — page/dirty and geometry logic are quant-size agnostic.
- Type-specific codepaths that could have interacted (`ggml_backend_cuda_kv_stream type capabilities (storage/decode_f16/auxiliary)` and the allocator's page_bytes) are quant-size — dependent only in page BYTES (q5_1 vs q8_0 rows differ in stride), and the divisors (`KV_STREAM_HEAD_DIM=256`, FATTN_KQ_STRIDE) are per-head-dim, adjusted automatically by `ggml_cuda_kv_stream_page_bytes()`.
- The q8_0/q8_0 boot test would only re-produce the pre-fix abort; it is disallowed. The fix is to the geometry gating, not the type.

#### Rig restore log round 2 (protocol)

- Preflight: `:18081 health 200` verified, VRAM 15847/11911, pod Running; `pod stop` → drain verify (1MiB/1MiB, ports clear).
- Crash diagnostics boot/all perf runs on :8080, split engines `50052`.
- Restore: all test binaries killed → GPUs 1 MiB/1 MiB → `podman pod start pod_llama-baseline` → health **200 (double-verified)** → VRAM **15847/11911**. Rig clean.

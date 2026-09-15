# Decode look-ahead expert prefetch for llama.cpp - design note

Goal: hide the host/NVMe -> GPU transfer of MoE experts behind compute on the
2-GPU rig, porting the mechanisms used by colibri and FreeToken.

Status: design only, no code. Requesting review of the approach and the scope
boundary before implementing.

## Current-state evidence (why the obvious path is closed)

The fork already has a grouped early-router, but on this rig it cannot be used:

- Decode runs the legacy cached MMID path (`DECODE_LEGACY`), not `DECODE_GROUPED`.
  The certified decode group reason is `route(8)` (`GROUP_REASON_ROUTE`).
- Cause: on layer-split, the routing `argsort` (the source of the top-K ids) lives
  in the other GPU's split, so `moe_candidate_discover_route` cannot find the ids
  nodes before the MMID node in the same split graph. The fork then deliberately
  fails closed to the cached MMID (layer-split keeps argsort on one device).
- Conclusion: `DECODE_GROUPED` is intentionally disabled for layer-split, not
  broken. Enabling it is not a toggle; it needs the ids on the MoE split.

Also, the expert cache is currently almost empty at decode:

- `--fit on` with `--moe-expert-cache-size` does NOT work: the cache installs a
  tensor buft override, and `common_fit_params` aborts with
  `failed to fit params ... tensor_buft_overrides already set by user, abort`.
  So fit does not run and placement falls back to the default offload.
- With `--fit on` alone, only 1 of 48 layers (`blk.47`) lands in the cached
  buffer, so there is nothing for a prefetcher to feed.
- The working combination must be `--fit off` plus manual placement, keeping the
  28.8 GB PLE embedding host-side (`--n-cpu-moe`, `--override-tensor`) and setting
  `--moe-expert-cache-size` by hand. Establishing this residency is a prerequisite
  for both halves below.

## What the references actually do

### colibri (C, JustVugg/colibri) - router-driven look-ahead (PILOT)

Source: `c/colibri.c`.

- `la_predict(m, target, h, kind)` (c:6343): predicts a layer's top-K by running the
  SAME routing pipeline on a hidden state: `rmsnorm(h, layer.post_ln)` -> `router`
  matmul -> `sigmoid + router_bias` -> greedy top-K.
  - kind 1 = PILOT: `target = L+1`, using L's post-attention state (stale, before
    MoE(L)). Measured recall 71.6% (GLM-5.2), 75.8% on its trace; previous-token
    routing only 41.3%.
  - kind 2 = PILOT_TWO: approximate MoE(L) by computing only L's SHARED expert on the
    normalized state (resident, no disk), add it to h, then run L+1's router. Corrects
    the dominant part the stale state is missing; +2.3-3.1%.
- Invocation (c:6979): inside the per-layer loop, immediately AFTER attention, when
  `li+1` is sparse: `pilot_prefetch(m, li+1, x, S)`.
- `pilot_prefetch` (c:6690): for each position, run the predicted router, then for each
  predicted expert NOT already resident in the FUTURE layer's cache, enqueue (layer, eid)
  onto a lock-free SPMC ring; dedicated I/O worker threads consume it and issue WILLNEED
  (`PILOT=1`, hint-only default) or a real cross-layer `pread` into `ecache[L+1]`
  (`PILOT_REAL=1`).
- Knobs: `PILOT`, `PILOT_K` (prefetch only top-k, head of ranking more reliable; default 6
  real / 8 hint), `PILOT_TWO`, `PILOT_REAL`, `PILOT_WORKERS`, `PILOT_EVICT_GUARD=1`.
- Safety invariants (c:1462-1478):
  1. pilot writes ONLY future layers (`layer > g_cur_moe_layer`); it never touches the
     layer the main thread is computing.
  2. main waits (cond var) for in-flight pilot loads on the layer it is about to compute.
  3. eviction guard: a speculative load may evict a resident expert only if the resident
     is not genuinely warm (>=2 demand accesses and hotter than the speculation by
     hysteresis).

### FreeToken (Python/CUDA, local) - cache + overlap, look-ahead only in prefill

Source: `freetoken/python/freetoken`.

- `OffloadMoeCache` (`moe/offload_cache.py`): slot cache keyed by flat `id = layer*E +
  expert`, LRU, bank-layout aware (nvfp4 etc.).
- Prefill: explicit one-layer look-ahead double buffering - `_prefill_routed`
  (`layers/moe.py:388`) calls `cache.prefetch_prefill_layer(L)`;
  `prefetch_prefill_layer(L+1)`; `wait_prefill_layer` before use. `prefill_hit_d2d`
  gathers already-resident experts device-side instead of re-streaming them.
- Decode: NO cross-layer look-ahead. `_decode_routed` (`layers/moe.py:279`) calls
  `cache.ensure_experts(L, topk_ids)` on demand: hits served from the slot cache, misses
  streamed over PCIe (fused multi-bank `cudaMemcpyBatchAsync`).
- Hybrid: `decode_target="hybrid"` fans each layer's misses out - a capped/fractional
  subset is fetched over PCIe, the rest computed on CPU; `hybrid_fetch_fraction =
  pcie_bw/cpu_bw` balances fetch time and CPU GEMV time for perfect overlap. Deep
  principle: every decode miss is ultimately a PCIe transfer, so latency is bounded by
  the link - look-ahead cannot remove it unless the prediction is cheap and early enough.

Take-aways for us:

- Cross-layer look-ahead helps only if the prediction is computed EARLY (during L's
  attention/MoE) and the transfer overlaps existing compute.
- colibri's predictive router is the transferable decode mechanism; FreeToken's
  double-buffer and PCIe/CPU split bound the misses that prediction does not cover.
- Both keep the idea device-agnostic: prediction produces (layer, expert) ids; a loader
  moves bytes. That maps cleanly onto llama.cpp.

### Upstream context (ggml-org discussion #24528, leloch's RFC)

The fork's expert cache descends from this RFC; upstream never merged it, which is why
the fork carries it. Its comment thread produced measurements that directly shape this
design:

- Decode routing skew is real and independently corroborated: top 10% of experts take
  ~80% of hits (Gini ~0.76, issue #20757). But noonghunna's capacity sweep shows the
  demand LRU keeps improving nearly linearly to ~31% pool coverage before flattening
  (~0.56 -> 0.29 -> 0.01 marginal t/s per GB) - the hot head is NOT a small fixed set
  (SharkWipf's "top ~1000 experts" reading was an under-warming artifact). So a warmed
  demand cache already absorbs most of the win; look-ahead prefetch's job is the miss
  TAIL, and its expected gain is bounded by that residual. Validate against a warmed
  pool, never a cold one.
- Sync points kill: the RFC's Metal slot-pool experiment was 2x slower than vanilla at
  97-99% hit rate, purely from per-layer syncs; batot1's GTX 1080 Ti sweep regressed at
  every budget. Direct constraint on PR-A: no full-stream synchronize in any demand or
  prefetch path; dependency on in-flight prefetch copies must be event-based only.
- leloch's v2 ablation: packing two H2D copies per dispatch into one removed ~40,000
  H2D ops from matched traces. PR-A must use the cache's batched copy path
  (`cudaMemcpyBatchAsync`, moe-cache.cu:5033) instead of one copy per eid where the
  banks allow it.
- Hybrid hit/miss execution (hits on GPU, misses stay on CPU) is measured, not
  hypothetical: +10-57% across 13 models with prefill bit-untouched. This is measured
  evidence for the FreeToken-bound follow-up below, and it composes with speculation:
  MTP worth +28% without the cache, +51% with it.
- Benchmark footguns to bake into our validation: an under-warmed pool reads as
  saturated; requested budgets are silently VRAM-capped (read the granted-capacity log,
  not the flag); the cache must never silently bypass (upstream added a one-shot warning
  for batch-bound bypass; our consumer must log, not no-op silently, when a prefetch
  target cannot be cached).

## Mapping onto the llama.cpp fork

Existing machinery already present:

- CUDA slot cache with a real prefetch consumer:
  `ggml_cuda_moe_cache_acquire_locked(..., is_prefetch, ...)` (moe-cache.cu:13186)
  schedules the H2D on a dedicated `copy_stream`, tracks `slot_prefetched`, and exposes
  `phase_prefetch_*` counters.
- Producer entry point stub (zero callers):
  `ggml_backend_cuda_moe_prefetch_experts(device, tensor_name, eids, n_eids, use_l2,
  is_decode)` declared in ggml/include/ggml-cuda.h:68 and implemented as a no-op at
  moe-cache.cu:14683.
- Legacy cached decode path `ggml_cuda_mul_mat_id_cached` (ggml-cuda.cu:3261) is what
  actually runs on this rig; grouped/early-router is disabled for layer-split.
- CLI: `--moe-expert-cache-size` / `--moe-expert-cache-l2-pinned-mb`;
  `mparams.moe_expert_cache_slots`.
- Gating: the entire feature (consumer + producer) ships behind a new `--moe-lookahead`
  CLI param, default off. No env master switch. See "Param gates" below.

Proposed implementation, two halves:

A. Consumer (self-contained, low risk)

   Implement `ggml_backend_cuda_moe_prefetch_experts`: resolve the per-device cache for
   `tensor_name`, and for each eid call `ggml_cuda_moe_cache_acquire_locked(...,
   is_prefetch=true, wait_for_compute=false)` with the expert row pointer obtained from
   the registered tensor base + eid * expert_stride. Reuse the eviction guard semantics
   (never evict a slot that a running GEMM is reading; do not clobber warmer residents).

B. Producer (the real work, two options)

   Option P1 - graph-level look-ahead (colibri PILOT, layer device-agnostic):

   In the decode graph, right after attention of layer L, add: `rms_norm(x, L+1.post_ln)`
   -> `mul_mat(L+1.router)` -> (sigmoid + bias) -> `top_k` -> a new `GGML_OP_MOE_PREFETCH`
   node carrying the predicted ids for layer L+1. In the CUDA backend that node calls
   `ggml_backend_cuda_moe_prefetch_experts` before layer L's MoE runs. Cost: about 4-5
   small matmul nodes per layer (shared-expert two-step optional). Pros: works on
   layer-split, matches colibri, no grouped-path dependency. Cons: touches
   llama-graph.cpp + a new op + backend dispatch.

   Option P2 - reuse the fork's early-router, but for the LEGACY path: teach the legacy
   cached mmid to consume a next-layer prediction instead of requiring DECODE_GROUPED.
   Smaller graph change but entangled with the grouped plan; does not help layer-split
   route ids.

Recommendation: P1 + A. P1 is the faithful port of colibri's PILOT and sidesteps the
layer-split grouped blocker entirely. Add FreeToken's bound as a follow-up: when predicted
misses exceed what PCIe can move in the layer's compute window, route the overflow to CPU
(the existing legacy host path already computes experts on the host, so this is a policy
knob, not new math). Upstream RFC #24528 measured exactly this hybrid: +10-57% decode on
13 models forced to spill, prefill bit-untouched - and confirmed the levers compose with
speculative decoding (MTP +28% standalone, +51% with the cache active).

## Invariants to preserve (from colibri)

1. Prefetch MUST target only future layers; never mutate the cache slots of the layer
   being computed.
2. Never let a speculative fill evict a slot the current GEMM is reading; never evict a
   genuinely warm resident for a speculation.
3. A missed prediction must be harmless (fall back to the demand path); a failed
   speculative load must not abort the request.
4. Prediction must be cheap relative to the layer's compute (about 4 small matmuls), and
   must run EARLY (right after attention) so the copy overlaps the MoE.
5. Bound speculative bandwidth (`PILOT_K` / fraction) so wrong predictions do not starve
   demand.

## Validation plan

- Unit: predict-then-check recall on a captured decode graph vs. the demand routing ids
  (expect about 70-76% top-K on this model; compare to the previous-token baseline).
- Instrumented run: `phase_prefetch_hits/used/h2d_bytes` from the existing telemetry;
  assert prefetch_used > 0 and decode t/s improves on the warm apex config (baseline
  21 t/s plain / 26 t/s MTP, 0 disk I/O).
- A/B with `--moe-lookahead 0` (param default) to isolate the gain; guard against
  regressions on non-MoE models.
- All decode A/B arms MUST run on a warmed pool (per upstream #24528: an under-warmed
  cache reads as saturated; bigger pools need proportionally more varied traffic).
  Measure the granted pool size from the cache log, not the flag value.
- Prefill/prompt-processing regression check on every A/B (upstream comparison showed
  PP can regress in cache builds even when prefill is bit-untouched: -6% to -14% PP).
  Our contract: PP unchanged when the switch is off, PP unchanged when on
  (look-ahead emits decode-only graph nodes).

## Risks / open questions

- Expert tensors must actually be in the CUDA-MoE-cached buffer for any of this to matter.
  Today with `--fit on` only 1 of 48 layers is; host-resident experts must be routed
  through the cache (the loader warns to use `--fit off` + manual cache size). This is a
  prerequisite for both A and P1 and should be verified first.
- Layer-split: the router/`post_ln` tensors of L+1 and the hidden state x must be on
  (or reachable from) the device issuing the prefetch. If x is on the other GPU's split,
  the prediction needs a small D2D/peer transfer of the hidden state (1 x D floats) -
  cheap.
- The prediction graph nodes add per-layer overhead; must confirm it stays under the win.
- Fork policy: this is a new op + multi-file change; needs maintainer-facing design
  scrutiny and an explicit go-ahead before coding.

## Implementation plan (PR-A consumer + PR-P1 producer)

Both PRs stack on `feat/763-reconcile-qwen4exp-mtp` (PR #120) and target the epic
`baseline-flash-next` after #120 lands. Both default OFF. The consumer is inert
until the producer exists. Line numbers below are on the feat/763 tip.

Param gates (CLI, not env; follows the `--moe-expert-cache-*` knob conventions):

- `--moe-lookahead N` master gate and predicted width in one knob, default 0 = off.
  N > 0 enables look-ahead for main target decode with predicted width N
  (colibri: 6 real / 8 hint; recommended first A/B value 8).
- `--moe-lookahead-two` PILOT_TWO shared-expert correction, default off.
- Arg validation: error if `--moe-lookahead > 0` and `--moe-expert-cache-size` is 0
  (residency is the prerequisite; fail loudly, no silent no-op).
- Plumbing: `common_params::n_moe_lookahead` beside `n_moe_expert_cache_slots`
  (common.h:543), `mparams.moe_lookahead` at common.cpp:1713, then a backend setter
  `ggml_backend_cuda_moe_set_lookahead()` mirroring
  `ggml_backend_cuda_moe_set_l2_pinned_cache_size()`. No draft inherit
  (no `--spec-draft-moe-lookahead`): look-ahead is main-target-only, matching the
  grouped-decode restriction.
- Debug-only env (never a gate): `GGML_CUDA_MOE_LOOKAHEAD_DEBUG=1` recall + prefetch
  telemetry. Env cannot flip behavior.

### PR-A - consumer (self-contained, low risk)

Template: `ggml_cuda_moe_grouped_context::prefetch_legacy_siblings` (moe-cache.cu:8529).
It already does the acquire loop we need. P1 only changes the target from same-layer
siblings to the next layer's expert tensors.

1. `moe-cache.cuh` - add to `ggml_cuda_moe_grouped_context`:
   `void prefetch_legacy_layer(const ggml_tensor * experts, const int32_t * eids, int n_eids, bool use_l2, bool is_decode);`
2. `moe-cache.cu` - implement it:
   - `auto lease = acquire_legacy_cache(experts);` installs the next layer's per-layer
     pool on demand (needs `op == GGML_OP_NONE`, `ne[2] > 0`, `nb[2] > 0`).
   - `cudaStream_t cs = ggml_cuda_moe_cache_copy_stream(cache);`
   - for each eid: `ggml_cuda_moe_cache_acquire(cache, data + eid*nb[2], nb[2], cs, use_l2, is_decode, /*is_prefetch=*/true, /*pin=*/false);`
   - prefetch the gate/up/down siblings too via `prefetch_legacy_siblings` on the lease.
   - batch the H2D: prefer one batched copy (the demand path's `cudaMemcpyBatchAsync`
     pattern, moe-cache.cu:5033) over one async copy per eid - upstream v2 removed
     ~40k H2D ops by packing copies per dispatch.
3. `moe-cache.cu:14629` - replace the no-op `ggml_backend_cuda_moe_prefetch_experts`:
   resolve the device's `ggml_cuda_moe_grouped_context` and call `prefetch_legacy_layer`.
   Keep the exported signature (`ggml/include/ggml-cuda.h:68`).

Invariants: `is_prefetch=true` (drives the `phase_prefetch_*` counters),
`wait_for_compute=false`, `pin=false`; an acquire returning -1 is ignored; never touch
the cache of the layer being computed.

Acceptance A: built and run with `--moe-lookahead 8`; `phase_prefetch_hits/used` and
`phase_prefetch_h2d_bytes` move; decode output is byte-identical (prefetch cannot change
results); with the param at its default there is no hit-rate regression.

### PR-P1 - producer (graph-level PILOT)

1. Predicted ids reuse existing ops, no new math:
   `rms_norm(cur, L+1.ffn_norm)` -> `build_lora_mm(L+1.ffn_gate_inp, .)` -> `sigmoid`
   (+`ffn_gate_inp_b`) -> `top_k`. This mirrors `build_moe_ffn` (llama-graph.cpp:2067,
   gate at :2099, sigmoid at :2122).
2. New side-effect op `GGML_OP_MOE_PREFETCH`:
   - `src[0]` = next layer's expert tensor (`ffn_gate_up_exps` or `ffn_gate_exps`),
     `src[1]` = predicted ids (i32).
   - constructor in `ggml.h`; CPU backend no-op; CUDA dispatch in
     `ggml_cuda_compute_forward`: read `src[1]` D2H, call `prefetch_legacy_layer`.
   - output aliases `src[1]` so the scheduler keeps the node in order (no dead-node removal).
3. Graph insertion in the per-model layer loop (`llm_build_*`), decode only, only when
   `il+1` is MoE: emit the prediction + op right after layer L's attention residual,
   before L's MoE, so the H2D overlaps L's MoE.
4. PILOT_TWO (optional): add L's shared-expert output to `cur` before the norm.

Open questions to settle before P1 coding:

- Q1: can `acquire_legacy_cache` install L+1's pool while L executes (generation/epoch
  guards)? Probe first.
- Q2: layer-split - L and L+1 can be on different devices; the op must run where L+1's
  cache lives (device 0 today). `cur` may need a 1 x n_embd peer copy.
- Q3: CUDA graph capture - keep the side-effect op out of captured graphs first
  (steady-state capture is already off in this config).
- Q4: draft/MTP contexts - gate to target decode only.

### Validation

- recall (predicted vs demand ids), `phase_prefetch_used`, decode t/s on fit-off
  N=144 (baseline 32.66 t/s), sdb read 0. Warm the pool first; log granted slots.
- A/B with `--moe-lookahead 0`; no regression on dense models; PP unchanged on/off.

## PR-P1 outcome (measured) - 2026-09-14

PR-P1 is implemented, measured and **stopped**. Branch `feat/moe-lookahead-p1`
(tip `4f681738b`, 21 files, +464/-5) is pushed and PR #127 is open against
`feat/763-reconcile-qwen4exp-mtp` as an unmerged experimental record. Everything
below was measured on the rig (Qwen3.8-Flash-Next-APEX-I-Mini, RTX 5060 Ti,
`--moe-expert-cache-size 84`, `-b 1 -ub 1` so every ubatch is a decode row,
`--moe-lookahead 8`).

Decision: park. The three blockers stack and each alone disqualifies the current
shape. Blocker 3 is structural rather than tuning, blocker 2 leaves nothing to feed,
and blocker 1 is a numerics-contract violation whose fix lives outside the prediction.
Deep surgery in three independent subsystems is not justified by a gain bounded to the
residual miss tail.

### Blocker 1 - the producer changes model output

Same prompt and flags, only `--moe-lookahead` differs. Isolation matrix, one build per
row:

| configuration | PPL | all-logits md5 |
| --- | --- | --- |
| look-ahead 0 (baseline) | 3.0613 | `fc1da5b9e41d249e7da51791a934d561` |
| look-ahead 8, full producer | 2.9434 | `8e0a9d10e9095ce7806f645e7edbba0e` |
| look-ahead 8, prediction only, prefetch op removed | 2.9434 | `8e0a9d10e9095ce7806f645e7edbba0e` |
| look-ahead 8, prediction reads a trunk tensor, no extra chain | 3.1849 | `4ba9f8e7a60ca9830b0dcbed073ffc3e` |
| look-ahead 8, gate MUL_MAT only, no argsort, no op | 3.1377 | `9bbf44eb3b98b5503ed2d1f9b028dda3` |
| look-ahead 8, neutral COPY node only | 3.0613 | `fc1da5b9e41d249e7da51791a934d561` |
| look-ahead 0, `GGML_CUDA_DISABLE_GRAPHS=1` | 3.0613 | `fc1da5b9e41d249e7da51791a934d561` |

All deterministic across reruns. Conclusions:

- The trigger is the extra `MUL_MAT` that reads `ffn_gate_inp`. Removing the argsort and
  the prefetch op does not remove the divergence; the op itself contributes nothing.
- The neutral-COPY row is byte-identical to the baseline, so the backend is not merely
  shape-sensitive. The divergence tracks readers of a router weight specifically, not
  node count - consistent with a router census perturbation rather than a scheduling one.
- Disabling CUDA graphs on the baseline reproduces the baseline exactly, so lost graph
  capture is not the cause.

Fix location, if this track is ever revisited: an exclusion mechanism for
prediction-subgraph nodes in the route/candidate census (the
`moe_candidate_discover_route` / `use_counts` machinery that also had to be taught to
ignore `GGML_OP_MOE_PREFETCH` sources, otherwise the grouped-decode certificate rejects
the graph with `graph=unproven(14)`). It is not a prediction-math problem.

The earlier byte-identical-logits reading recorded in this document is refuted. It came
from a build in which the producer never ran.

### Blocker 2 - the producer is inert; Q1 answered negatively

`acquire_legacy_cache()` installs a new pool only while the target group's authority is
`GGML_CUDA_MOE_GROUP_AUTHORITY_LEGACY` with admission open. On this rig it is not, so the
reserve-time preinstall fails for all 144 targets and every prediction is dropped:

```
E moe-cache: look-ahead could not install 144 of 144 MoE expert cache pools;
             the target layers are not under legacy cache authority, so look-ahead prefetch is inert
W moe-cache: look-ahead prefetch found no installed pool for the target layer, prediction dropped
```

Q1 (can `acquire_legacy_cache` install the next layer's pool while the current layer
executes?) is therefore answered **no**, not because of the generation/epoch guards, but
because the authority regime refuses the install outright. The plan's "preinstall every
pool at model load" is also wrong as written: the candidate snapshot does not exist until
after `sched_reserve()`, so the install has to be attempted there or later.

### Blocker 3 - throughput, and it is structural

| arm | decode t/s |
| --- | --- |
| look-ahead 0 | 43.23 |
| look-ahead 8 | 37.71 (37.57 / 37.61 / 37.68 across runs) |

The ids readback synchronizes the stream, which forces `use_cuda_graph = false` for every
graph containing the op - the precedent is `ggml_cuda_mul_mat_id_needs_sync` at the same
decision point. Without the exclusion the run dies:
`E CUDA error: operation not permitted when stream is capturing`.

### Future direction (not now)

If the track is re-approached, the only architecture-level fix for blocker 3 is the
fork's own early-router copy-worker plus device-to-host mailbox pattern (poll worker with
`cuStreamWriteValue32`, no in-graph readback, `moe-cache.cu` around the copy-stream
setup). That makes the producer a side-channel that the legacy path consumes, which this
document originally rejected - the measured cost of the in-graph readback flips that
judgment, but the work stays gated on grouped-plan / layer-split work.

The pool-install question is settled as a design invariant, not a bug. `acquire_legacy_cache`
admits a new record only under `group_authority.authority == LEGACY && !admission_closed`, and
that latch exists to bind pool creation to a certified execution: an unauthenticated install
would let a caller claim VRAM pools that certification never proved, the overcommit class
README.md:25 already warns about ("fit accounting does not include these pools"). Within a
decode step, layer L+1 is by definition not the authoritative group - its authority publishes
when its own demand path runs, which is the moment a prefetch would already be too late - so
cross-layer install is unreachable BY DESIGN. `preinstall_legacy_pools` returning `-1` for all
144 targets is the design working; do not patch it to open authority.

If the track reopens, the seam to change is authority publication at graph reserve /
certification time, where the full layer inventory and the slot budget are known up front
(consistent with the recorded correction that the candidate snapshot only exists after
`sched_reserve()`). That is a deliberate change to the certification contract and needs its
own review, which is one more reason this producer stays parked rather than patched.

### Reopen-transport ruling (mailbox vs device-side) - 2026-09-14

Reviewer/decider ruling on how blocker 3 would be fixed if the track reopens.

**(a) Mailbox approved as the transport; device-side gather rejected for this codebase.** The
mailbox reuses two components the fork already ships and exercises: the early-router copy worker
(`moe-cache.cu:4913-5140` - `cudaHostAllocMapped` buffers, `moe_early_router_publish_copy`,
`cuda::atomic_ref<..., thread_scope_system>` poll, `cuStreamWriteValue32` / `cuStreamWaitValue32`)
and the async paging half (`ggml_cuda_moe_cache_prefetch_locked`). The transport work is then
glue, not architecture. The FreeToken-style alternative - a device kernel that reads the
predicted ids from device memory and gathers slabs from host-mapped banks - is architecturally
superior but requires device-resident slot management: LRU selection, the LFRU eviction guard,
and slot booking/rollback are host-side logic under `cache->mu` today (`select_victim_locked`,
`install_fill_locked`). A gather kernel therefore means a second cache implementation and
reintroduces the demand/speculative drift that sharing those two functions eliminated. FreeToken
built its device-side cache from day one; this fork did not. Revisit only if slot management ever
moves to the device wholesale. Supporting fact: recall is not the weak part (86.21% at width 8,
about 22x chance), so the next unit of work belongs to transport and authority, not prediction.

**(b) Deleting the `use_cuda_graph = false` rule does not suffice.** The op is structurally
capture-incompatible as written, and capture-compatible after the redesign. The graph must not
contain the pageable D2H memcpy, `cudaStreamSynchronize`, or any host-side slot booking. It may
contain the prediction subgraph, a publish kernel writing ids into `cudaHostAllocMapped` memory,
and capture-legal stream memory ops. The `ggml_cuda_mul_mat_id_needs_sync` precedent forces
no-capture because the demand path performs the readback; a mailbox producer has no in-graph
readback and does not need the exclusion, but the exclusion must remain for any non-mailbox
fallback. Capture-replay hazard: a captured `cuStreamWaitValue32` bakes an expected value, so on
replay it either spins forever (value not yet reached) or passes instantly (already passed). It
must be refreshed per step through graph exec update - the fork already runs a per-step update
for MoE graphs - or replaced by a polling kernel reading a monotonic device counter. Never
capture a raw static-value wait.

**(c) Minimal change set.** `ggml-cuda.cu`: rewrite `ggml_cuda_moe_prefetch` (3883-3926) to
publish instead of reading back, and gate the capture exclusion (4561) on mailbox mode rather
than on op presence. `moe-cache.cu` / `.cuh`: add a prefetch job type to the early-router copy
worker - the worker resolves the target tensor, calls `acquire_legacy_cache` (authority still
gates, by design), runs the existing paging path, and records a per-job event on the copy stream
that the demand path consumes through its existing `cudaStreamWaitEvent`. Untouched: the paging
half, the cache authority machinery, the route/candidate census exclusion, `build_moe_lookahead`,
and the consumer surface.

**(d) Required guards and instrumentation.** Bounded job ring (depth about 2) with drop-newest and
a one-shot log; counters for published / dropped_overflow / dropped_authority / consumed.
Sequence-keyed ring slots, replay-idempotent publish, recycle only after consumption is
confirmed. Ordering: `__threadfence_system()` before the sequence write on the device side,
acquire through `cuda::atomic_ref<..., thread_scope_system>` on the worker. No unconditional
waits: a dropped prediction must never leave the demand path waiting on a semaphore that will
never be published, hence per-job events recorded only when copies were enqueued, plus a
`waited_empty` counter. Verify ordering when a prefetch batch and a demand staged batch are in
flight on the same copy stream. Graph churn: the per-step `graph_update_required` rate must
settle to zero in steady state.

**Sequencing (unchanged).** The mailbox fixes blocker 3 only; blocker 2 (authority) and blocker 1
(census) still gate. Reopening is three ordered workstreams: (1) census exclusion for the
prediction subgraph (issue #128), (2) authority publication at graph reserve/certification time
(the seam above), (3) the mailbox transport. One review, not three PRs. The track stays parked.

### Corrections this round made to the plan above

- `ffn_gate_up_exps` is null for every layer of this model, so the producer targets
  `ffn_up_exps`. The consumer resolves the layer by tensor name, so any of the layer's
  cached expert tensors works.
- Q3: capture is not merely "already off in this config"; the op has to force it off.
- Q4: the draft/MTP gate is `cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT`, because the
  execution certificate that encodes the same distinction is assembled at compute time.
- `qwen35moe.cpp` is wired mechanically and is unvalidated - no model on this rig to test
  it against. It stays in the branch as part of the experimental record.

### Prediction accuracy (measured 2026-09-14)

Horizon: the producer looks ahead exactly **one MoE layer, never a token**. Layer L's
post-attention state predicts layer L+1's selection for the same token, at each of the
model's 47 layer transitions per decode step, so one prediction per layer per step.
`--moe-lookahead N` is the number of experts predicted per layer, not a token count.

Method: the demand path cannot score this. During decode only one layer (`blk.47`, the
layer holding the legacy cache lease at `--moe-expert-cache-size 84`) reaches the
host-visible ids read; every other layer executes in the certified grouped path and its
selection never surfaces to the host. The ground truth was therefore built in-graph: each
layer's own router matmul on its real FFN input, top-k, routed through a measurement node
and compared against the prediction recorded for that same layer. Decode rows only, ground
truth pinned at the model's real selection (`n_expert_used = 10`), 23,936 scored
(step, layer) pairs per run, 2 x 256-token generations.

| predicted width | recall (of predicted) | coverage (of the 10 actually used) |
| --- | --- | --- |
| 2 | 97.37% | 19.47% |
| 4 | 95.13% | 38.05% |
| 6 | 91.50% | 54.90% |
| 8 | 86.21% | 68.97% |
| 10 | 79.26% | 79.26% |

Random overlap for 10 of 256 experts is 3.9% of the predicted width, so width 8 is about
22x chance - the prediction is genuinely informative, not noise. RTX 3060 (same model, same
prompts, width 8): 85.85% recall / 68.68% coverage, the same within numerical noise.

Reference point from this design's own source: colibri reports 71.6% PILOT recall on
GLM-5.2 (75.8% on its trace), using the same stale state - layer L's post-attention hidden
state, before L's MoE. The model and its expert count differ, so this is not a like-for-like
comparison, but at comparable prediction widths the producer here is at least in that range
(91.50% at width 6, 86.21% at width 8). Prediction quality is therefore not a reason to keep
the track parked; the blockers are.

Two traps found while making this measurable, both of which had kept it unmeasured:

- The committed debug instrument logs with `GGML_LOG_INFO`, which never reached the server
  log at default verbosity; only `fprintf(stderr, ...)` appeared. The env gate itself was
  fine. Any future recall work should use a log path that is actually emitted.
- Scoring inside `ggml_cuda_mul_mat_id_cached` only ever saw `blk.47`, because every other
  layer takes the `cache == nullptr` early return during decode. Recall must be scored
  outside that path.

Refinement implied by the numbers: the model routes to **10** experts, so `--moe-lookahead
8` predicts fewer experts than are used - 69% coverage - while width 10 reaches 79%. Note
also that the measurement runs add graph nodes and are therefore valid only for the recall
numbers, not as output-equivalence runs (see blocker 1).

### Phase attribution and off-lease recall (instruments A/B, temporary, measured 2026-09-14)

Two temporary, env-gated instruments were run on the RTX 3060 (ctx 81920,
`--moe-expert-cache-size 42`, one 13,946-token prompt, 256 decode tokens, no
look-ahead) and then reverted; none of this is committed engine code.

Method. The backend receives scheduler slices rather than the whole graph: those
slices carry no leafs and their matmuls report `ne[1] == 1` in both regimes, so a
step cannot be classified as prefill or decode from graph shape. They are separated
by inter-step wall time instead - prefill ubatches take ~4.6 s and decode steps
~97 ms, a ~47x gap, boundary 500 ms. CUDA events are opened per maximal run of
same-(phase, layer) nodes and read once per step after one stream sync (that sync
costs ~2.6% of throughput: 10.58/9.23 t/s instrumented against 10.98/9.47
uninstrumented). The attribution arm must run with `GGML_CUDA_DISABLE_GRAPHS=1`,
because replay steps never visit the host dispatch loop where events are recorded;
the graphs-on control on this rig is 11.42 t/s against 10.98 t/s graphs-off (-3.9%),
so the shares below are quoted for the graphs-off regime.

Decode attribution (256 steps, 96.90 ms/step, 10.43 t/s; the phase rows sum to
95.7 ms, so the split covers the step):

| bucket | ops | ms/step | share |
| --- | --- | --- | --- |
| ATTN | flash-attn, indexer, SSM/GDN, rope, softmax | 1.41 | 1.5% |
| MOE | `MUL_MAT_ID`, `MOE_PREFETCH`, `ARGSORT`, `TOP_K` | 67.18 | 69.3% |
| DENSE | every other matmul and elementwise op | 23.80 | 24.6% |
| OTHER | copy/cont/reshape/view/permute/transpose | 3.30 | 3.4% |
| PLE | `GET_ROWS` on `per_layer_token_embd` | 0.00 | 0.0% |

Dispatch state for this arm: `mode_legacy=3 mode_direct=253 mode_capture=0
mode_replay=0` - these are DIRECT-dispatch numbers by construction. The buckets are
op sets, not architectural phases: the MOE row is exactly the MoE op set (which is
what the park turns on), while DENSE absorbs everything else, so attention work that
is not one of the named ATTN ops (GDN elementwise in particular) sits in DENSE.

Prefill attribution (28 ubatches, 4555 ms/ubatch): ATTN 42.07 ms (0.9%), MOE
4005.38 ms (87.9%), DENSE 367.91 ms (8.1%), OTHER 11.49 ms, PLE 0.00.

Per-layer decode MoE (ms/layer/step): blk.00 2.91, blk.01 2.48, blk.12 1.75,
blk.24 1.12, blk.36 1.11, blk.47 1.55. MoE cost is spread across the layers - the
grouped path executes for all of them - and the ATTN buckets stay below 0.02 ms per
layer at this batch size.

Device state in the same window (`dmon -s ump`, decode): SM 95-99%, DRAM controller
19-26%, 90-94 W, 55-56 C, 10,851 MiB resident of 12,288 MiB; SM clock 2122-2145 MHz.
The PCIe link is **gen4 x4** (the card is x16-capable, the slot is wired x4), about
7.9 GB/s theoretical. Expert H2D traffic over the arm is 337.6 MiB/token
(90.62 GB in 256 tokens) = 3.71 GB/s = ~47% of that ceiling, and the grouped
telemetry reports `calls=12240 ready=12240 ready_min=255 completed=12240`.

Do NOT read those grouped counters as staging residency. `ready` and `completed`
count **plan admission** (`state = GROUPED_ACTIVE`, moe-cache.cu:12177, 11215, 11939),
not data in the banks: they say every step's plan was admitted, and nothing about
whether the slabs were resident. Reading `ready_min=255` as "staging already complete,
so the traffic never gates the kernel" is the error that produced the earlier "the
limiter is MoE kernel execution, not PCIe" verdict, and that verdict is withdrawn.

The existing counters cannot settle it, because the grouped path has no copy-wait
accounting at all. `copy_wait_event_*` is recorded only by the legacy `moe-cache-phase`
line, whose decode row covers **9 of the 12,240** grouped calls
(`phase=decode ops=9 ... copy_wait_events=9 copy_wait_event_ms=0.007`), and the
`copy_wait_event_ms=0.000` row in the same log is the shutdown flush (`ops=0`).
Neither is evidence about the grouped path.

What the log does fix arithmetically: 91,316,684,800 H2D bytes over 255 steps =
358 MiB/step, ~45 ms/step at the 7.88 GB/s link, against a 92 ms step and a 67 ms MoE
bucket whose `MUL_MAT_ID` kernels account for only 460 us x 48 = 22 ms/step. The MoE
bucket residual and the serialised transfer time are the same order, so both "transfer
is already hidden" and "transfer is the residual" fit the numbers in hand. The grouped
phase timer plus the consumption-side residency counter (plan/prepare/remap/wait/impl/
finish split, `resident_at_entry` against `copied`) are what separate them; the
decision rule is the regression of per-op microseconds on `copied`.

Recall measured off the legacy lease (instrument B). The committed instrumentation
scores predictions only inside the legacy cached demand path, so it only ever sees
`blk.47`; scoring in the dispatch loop for every `MUL_MAT_ID` node instead (the ids
live in `src[2]`), with the ids read back asynchronously and scored once per step,
gives over the last 32 decode steps: 141 node-predictions scored, 1,410 used expert
ids, 909 used-and-predicted, 501 used-not-predicted - 64.5% of the used experts were
predicted and 80.6% of the predictions were used. This confirms, independently of the
in-graph measurement above and on the host-visible layer, the same order of magnitude
(that measurement averages 86.2% of predicted / 69.0% coverage across all layers).
The other two populations are structural zeros in this configuration:
predicted-not-copied (install refusals) = 0 and copied-not-used (prefetched slabs
evicted without a hit) = 0, because nothing is ever offered to the installer - the
decode phase line reads `ops=9` (of 12,240 grouped calls) with `legacy cache authority`
printed once, i.e. the consumer is inert and every prediction is simply unused. The drop counters are now
printed (`prefetch_dropped`, `evicted_prefetched_unused` on the phase line) instead
of being counted silently, so a future reopen does not have to rediscover them.

Conclusion relevant to the park. A decode step is 69% MoE execution on a saturated
SM (95-99%) with prefetch traffic fully hidden; the producer covers one layer, and
the host-visible prediction quality is 64.5% of the used experts. Prediction width,
pinning or a device-side transport cannot move a limiter that is arithmetic
throughput on the 3060, so the park stands.

Independent finding recorded this round: the rig's `-ot per_layer_token_embd=CPU` is
redundant and inert. Layer-input tensors already land in the CPU buflist, and the lazy
path returns before user override matching runs, so the flag changes nothing; PLE time
is 0.00 ms/step in the attribution above.

### Review corrections (advisory review of issue #129, 2026-09-14)

Four claims from this section were put to review; three were upheld, with one caveat that
changes how the numbers must be quoted, and two pieces of arithmetic were falsified.

1. The MOE bucket is **not kernel-pure**. The phase instrument opens events per maximal run
   of same-(phase,layer) nodes, so host-side gaps fall inside the window. In decode the
   legacy-dispatch ops alone contribute `op_cpu_ms=16.761` over 9 ops (1.86 ms/op) and
   `ids_d2h_ms=10.920` over 2 syncs, i.e. roughly 5-9 ms/step of the 67.18 ms bucket. The
   conclusion holds because the 45 direct-dispatch layers (no id readback, no lease acquire)
   independently show ~1.2-1.4 ms per ~21.5 MiB, the same 15-18 GB/s as the aggregate, but a
   kernel-only rate does not yet exist and must be produced before any kernel campaign is
   priced.
2. Decode at M=1 already selects MMVQ, not MMQ: `ggml_cuda_moe_use_mmq` requires
   `n_tokens > 1`, so there is no mis-selection to harvest there.
3. The measured -8..-13% of `--moe-lookahead` is **not prefetch overhead**. Under direct
   authority the producer installs 0 of 144 pools ("look-ahead could not install 144 of 144
   MoE expert cache pools ... prefetch is inert"), so the whole penalty is the cost of forcing
   `use_cuda_graph=false` plus the ids readback sync. Fixing the transport removes that
   penalty; it does not by itself add throughput here.
4. The 94% wasted speculative installs are a **prefill lease-path** phenomenon (M=512 rows,
   flat router, `l1_evictions=123,151` against 129,073 fills); decode prediction quality is
   separately measured as decent (64.5% used-and-predicted, 80.6% predicted-and-used). The
   prefill waste is not evidence about decode prediction.
5. Falsified in earlier drafts of this work: the link-bound acceptance table. With the miss
   set at 1.0817e9 B/token (`1031 MiB`) against a 7.88 GB/s gen4 x4 link, the binding
   fraction is `miss_fraction * 1.0817e9 * tps <= 7.88e9`: today's 32.7% miss binds at about
   22 t/s, so 30 t/s requires <=24% and 48 t/s requires <=15%. An earlier statement of 74%
   and 46% for those rates was wrong by roughly 3x.
6. Reviewer's ordering: raised M via the existing MTP/draft path ranks ahead of split-K,
   because the same work at M=512 costs 8.9 ms/token-equivalent against 96.9 ms at M=1 and
   that machinery already exists and is certified in this fork, while split-K targets an
   efficiency ceiling not yet separated from booking overhead. Split-K itself is feasible
   without disturbing the pool-install authority invariant: it changes capability plumbing
   and plan admission, not `acquire_locked`/install.

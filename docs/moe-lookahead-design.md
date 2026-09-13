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
knob, not new math).

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
- A/B with prediction disabled (env kill-switch) to isolate the gain; guard against
  regressions on non-MoE models.

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

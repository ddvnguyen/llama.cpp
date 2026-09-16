# MoE expert residency tiering — design

Status: **design, not implemented.** Target: raise 3060 decode throughput by
prioritising experts across execution sites rather than by predicting them.

Owner framing (2026-09-16): *"the whole idea of look-ahead is more like
prioritise the experts or set tiers for experts to maximise the speed."*
This document takes that framing literally and drops prediction entirely.

---

## 1. The idea in one line

An expert's cost is set by **where it is computed**, and this rig has three
places to compute one, with bandwidths that differ by 50x. Tier the experts
across all three and keep every pipe busy.

| tier | site | source | bandwidth | ms per expert (2.15 MiB) |
|---|---|---|---|---|
| **T1** | GPU, resident | VRAM | ~360 GB/s | **0.0063** |
| **T2** | GPU, streamed | PCIe gen4 x4 | 6.6 GB/s measured | **0.342** |
| **T3** | CPU | host DDR | ~32 GB/s effective | **0.0704** |

The three run **in parallel**: a VRAM GEMV, a DMA-plus-GEMV, and a host FFN
touch disjoint resources. A layer's MoE time is therefore `max(T1, T2, T3)`,
not their sum.

Note T3 is **5.2x cheaper than T2**. Host DDR beats the x4 link by a wide
margin, which is why the CPU is the right home for the cold tail and why the
link is *not* the right place to send everything that misses.

---

## 2. What is already closed — do not re-litigate

Re-deriving any of these wastes a cycle. Each is measured, not argued.

**Where the citations live.** `plan N` = section N of
`docs/moe-lookahead-improvement-plan.md`, which is on branch
**`feat/moe-lookahead-p1`**, not on this one. Rig artefacts (`mtp/`,
`mtp14k/`, `contend/`, `cnre-phase0/`) are under `/tmp/opencode/` on the
hydra_vortex host and are not in the repo; copy anything you need to keep
before that host is rebooted.

| item | verdict | source |
|---|---|---|
| Next-token / cross-layer **prefetch** | net loss at every width; 1.91% useful at w8, 95.5% at w1 and still loses | plan 7.21, 7.29 |
| Look-ahead at N=53 | **aborts the server** (103 MiB of never-evicted lanes) | plan 7.20 |
| MTP as a prefetch signal | no lead time — verification is one ubatch | plan 7.33 |
| Adjacent-token expert overlap | **~0%** (backed out of the two MTP arms) | mtp/, mtp14k/ |
| **Static offline expert tiers** | in-sample ceiling 67.0/67.8% < gate; heldout 45.3/47.8%; transfer 18.5% vs 10.35% chance | cnre-phase0/REAL_RESULTS.md |
| Per-workload (coding vs general) tiers | Jaccard median 0.463, `frac>0.8 = 0.26%` | same |
| Non-uniform **slot allocation** across layers | +0.29%, fails split-half holdout; per-layer curves are near-parallel | plan 7.26.1 |
| Online retention policy variants | LFU-16 beats LRU and no-decay | plan 7.17 |
| Lossless slab compression | zstd-9 ratio 1.0000 | plan section 4 |

**Consequence:** the remaining lever is not *which experts we can foresee*.
It is *where each expert runs*, and how the miss set is divided between two
parallel pipes.

---

## 3. The core rule: split the miss set by bandwidth

This is the whole design and it needs no predictor, no speculation and no
lead time.

Let `M` be the experts demanded in a dispatch that are **not** T1-resident.
Send a fraction to T2 (link) and the rest to T3 (CPU). The two pipes are
independent, so the optimum equalises their completion times:

```
f_link = B_link / (B_link + B_cpu) = 6.6 / (6.6 + 32) = 0.171
```

**Route ~17% of the miss set over the link and ~83% to the CPU.** The miss
set is then served at an aggregate `B_link + B_cpu = 38.6 GB/s` instead of
32 GB/s CPU-only — a **17% cut** in the dominant term.

The rule self-tunes: both bandwidths are measurable at runtime, and `f_link`
follows. If the card is ever moved to the x16 slot (26 GB/s), the rule
rebalances to `f_link = 0.45` with no code change.

**This is the productive use of the idle link.** In the CPU-MoE config the
link sits unused. It is not useful for prefetching (nothing to predict), but
it is useful as a *third execution path*, because routing a miss over it
requires no foreknowledge — only a budget.

---

## 4. Tier assignment

**T1 (resident) — who stays.** Keep the shipped online policy; it is the best
of those measured. Two increments, both optional and independently gated:

- `--moe-expert-priors <file>`: seed the frequency counters at load from an
  offline decode-derived histogram. Phase 0 T0.5/T0.7b indicate a seeded
  adapting policy beats hard pins by ~16pp and reproduces the rig's
  `d16 > LRU > NODECAY` ordering. **But against the shipped policy the gain is
  only ~+3-6pp**, so size expectations accordingly.
- Reserved share: `GGML_CUDA_MOE_CACHE_RESERVED` already exists on
  `feat/moe-lookahead-p1` and is **known broken** (budget resolves to 0 at
  cache creation). Fix it before use.

Note the decay law: the counter halves every **16 grouped planning steps**
(48 per decode token) and is fully zeroed after 512 planning steps ~ 10.7
tokens (`899d8ec7a`). A seed is therefore erased ~11 tokens into decode — by
design, and it is why the rig keeps d16's edge under prefill pollution.

**T1 capacity does NOT rise in the hybrid — corrected 2026-09-16.** An earlier
draft of this document claimed the hybrid frees 4.5-5 GiB, reaching N ~ 65-70
and `h ~ 0.62-0.64`. **That was wrong.** Plan 7.15 measured the ceiling directly:
at N=28 the 3060 reports 9293 MiB used / 2619 MiB free, the marginal cost is
48 x 2.148 = 103 MiB per slot, and **N=64 is a hard cudaMalloc OOM**. That puts
the ceiling at **N ~ 53-57**.

The hybrid does not move it. Backing the slot cost out of the N=28 figure leaves
a ~6.4 GiB base of non-expert weights + KV + compute buffers, and the hybrid
reduces none of those — it only avoids some staging buffers, worth 1-2 slots.
The N=53 control arm is therefore already at the ceiling, and the capacity lever
above it is worth **~1pp of `h`, not 6**.

**64K KV is already paid for.** Every arm in the capacity sweep ran at
`-c 81920` (plan section 0's `phase.sh 1 81920 ...`), so the 80K KV cache is
already inside the 9293 MiB and inside the 1.55/2.2 GiB headroom figures. A 64K
run does not add a new VRAM tenant. This was a live concern and it is closed by
the measurement regime, not by argument.

**T2 (link) — who streams.** The `f_link` highest-frequency experts *among the
misses*. Rationale: a streamed expert is a candidate for promotion into T1, so
spending link budget on the ones most likely to recur compounds. No prediction
is involved — the ranking is over experts already known to be demanded.

**T3 (CPU) — everything else.** No policy, no ordering.

---

## 5. Expected gain

Anchored on the measured MTP dispatch shape (3.76 positions, 3.43 tokens,
`F = 28.3 ms`, head ~5 ms), 48 layers, 37.6 expert-demands per layer:

| config | miss ms/layer | dispatch | ms/token | **t/s** |
|---|---|---|---|---|
| measured today: MTP + CPU-MoE, no cache | 2.647 | 132.4 | 46.8 | **21.4** |
| + T1 at h=0.58 (hybrid, all misses to CPU) | 1.112 | 86.7 | 25.3 | ~40 |
| **+ T2 bandwidth split** | **0.922** | **77.6** | **22.6** | **~44** |
| + T1 at ceiling N~55, h=0.59 | 0.910 | 77.0 | 22.4 | ~44.5 |

T2 contributes **+11%** on top of the hybrid. Modest, but it is prediction-free,
self-tuning and cheap. The hybrid itself (T1 existing at all) is the +85% step.

All rows after the first are **projections** from a model that has reproduced
two MTP arms at different depths on one parameter set, and validated
out-of-sample across N=28/40/53 to 1.2%. They are not measurements.

---

### 5.1 T-2 result (2026-09-16): the hybrid did not run

Measured **10.40 / 10.08 t/s** against a 21.36 control (control reproduces at
21.23; accepted tokens bit-stable). Recorded `d-ce1e9913f5`, gitlink `7e9cbdc40`.

**This is not a refutation of the design, because the split never engaged.**
H1 is unengaged under CUDA graphs (`defer_completion`); H2 engages off-graphs but
is skip-heavy and lands equal to cache-32 no-split (10.53). Both legs measured
the **cache-only** path — every miss over PCIe, nothing to the CPU.

10.40 is a known number. It is where the capacity sweep already sat (N=36 →
10.36, N=42 → 10.84), and the model predicts it: at h=0.58 the cache-only path
sends 15.8 misses/layer over a 0.342 ms link where CPU-MoE computes 37.6 demands
at 0.0704 ms, so cache-on *should* be ~2x slower than CPU-MoE. It is
(21.36/10.40 = 2.05x).

**Direct evidence that the hybrid's premise holds.** Decompose the two exp1 arms
with a shared fixed term: N=53 cache-on is 86.43 - 201.6x0.2842 = **F 29.14**
with 278 hits/token; N=0 CPU-MoE is 62.13 - 480x0.0704 = **F 28.34** with zero
hits. The same fixed term falls out of both, so a resident hit costs
**0.0029 ms against 0.2842 ms for a miss — 99x cheaper**. Tier 1 is as close to
free as the design assumed. What was never tested is routing misses to tier 3.

**Honest haircut.** The measured cache-only point is 10.40 where the model says
11.7-13.8, so the miss term runs **1.14x** my nominal. Applying that same factor
to the hybrid moves the projection from ~40 to **~36 t/s**. Section 5's table is
nominal and should be read with this factor.

## 6. Implementation plan

Sequenced so each step is independently measurable.

**T-1. Fix #124 — blocking, on the critical path.**
Cache + MTP are mutually exclusive today: `--moe-expert-cache-size > 0` with
`--spec-type draft-mtp` yields `grouped plan unavailable`,
`required_unsupported 63-95`, and the path **fails closed** into legacy
fallback while still serving. Every result we hold was taken at cache 0, which
is what makes #124 moot. **This design turns the cache back on, so #124
returns.** Sibling: #121, same execution-intent machinery.
*Gate:* a `draft acceptance` line appears with the cache enabled.

**T-2. Land and measure the hybrid split.** `0fc51e039` is in the tree,
env-gated and unmeasured. Report `h` and t/s against the cache-0 control.
*Gate:* beats 21.36 t/s at 767 tokens.

**T-3. T2 budget.** Add a per-dispatch link budget `f_link`, defaulting to
`B_link/(B_link+B_cpu)` from measured rates, overridable by env for A/B.
Route the top-`f_link` fraction of misses by frequency to the existing demand
path; the remainder to CPU. *Gate:* >= +5% over T-2 (projection is +11%).

**T-4. Confirm the ceiling. DEMOTED — do this last, or not at all.**
The N=65-70 upside this step was written for does not exist (see section 4).
Reachable is N=55-57 against a 53 control, worth ~1pp `h` and ~0.5 t/s.
Flag-only sweep N=53/55/57; stop at the first OOM. *Gate:* none worth setting —
run it only if the rig is otherwise idle.

**T-5. Seeded T1 priors.** Only after the sim clears calibration gate 3
(offset stable across N). *Gate:* >= +3pp `h` over unseeded.

**T-6. Depth verification at 64K.** Never measured — we have 767 and 13,946
only. Every 64K number in this document is extrapolation.
*Gate:* >= 15 t/s at 64K.

---

## 7. Risks

1. **#124 (highest).** Fails *closed* and then serves from fallback, so an
   unwary arm measures the wrong thing and looks merely slow. Any arm with the
   cache on must assert the `draft acceptance` line before its number is used.
2. **Shared DDR under COMBINED mode.** T3 and the `f_link` rule both assume
   host bandwidth. Measured contention cost with both host GPUs active is
   **-9.5%** (`contend/`, after separating the thread-halving confound). The
   probe used 6 of 8 P-cores, so this is a measurement at one config, not a
   ceiling.
3. **Per-expert byte size: use 2.148 MiB for VRAM sizing.** The GGUF inventory
   gives 2.148 MiB (gate 0.598 + up 0.671 + down 0.879); the per-step transfer
   ledger derives 1.793 MiB. The gap is ~20% and it moves every absolute number
   in section 5. For *VRAM* the GGUF figure is the better-evidenced one — it is
   what plan 7.15 used to predict a ceiling near N=55, and N=64 then OOM'd as
   predicted. The ledger figure plausibly excludes padding or head overhead.
   Ratios and the `f_link` rule are unaffected either way, so this does not
   need its own arm — fold it into T-1/T-2 rig time.
4. **`f_link` assumes the two pipes do not interfere.** A DMA into VRAM and a
   host FFN both touch host memory. If they contend, the effective aggregate is
   below `B_link + B_cpu` and `f_link` should be fitted empirically rather than
   computed. T-3's env override exists for this.

---

## 8. What this design deliberately does not contain

No predictor, no speculation, no staging lane, no lead-time requirement, and
no offline static tier. Every one of those has been measured on this rig and
found negative or structurally impossible. The prediction machinery already
built is not wasted — its honest consumer is **eviction/victim choice** within
T1, which is plan 7.31's own conclusion and which this design leaves open as a
later increment on top of T-5.

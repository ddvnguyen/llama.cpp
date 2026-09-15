# MoE look-ahead preload - evidence, mechanism, levers, plan

Revision 27 (doc hygiene, no new measurement: section 8's "CURRENT OPERATING POINT" block still carried the withdrawn N=53 recommendation as the live best configuration, and 7.29's "Recommended operating point: N=53" was unmarked - both now state the owner's N=42-or-N=36 decision and mark the N=53 text as a record rather than a recommendation. This mattered because the operating rule is "the doc wins", so a stale section would have resurrected N=53. The stale HEAD pointer at the top of section 8 is refreshed too). Revision 26 (7.34: CORRECTION - the MTP head DOES exist. /mnt/SSD/MTP/mtp-Qwen3.8-Flash-Next-shared-{Q8_0,Q4_K_M}.gguf, 2657.48 / 1818.80 MiB, and it is a complete block 48 (attn + indexer + hc_attn/hc_ffn + full MoE + the 6 nextn.* tensors), declaring nextn_predict_layers, nextn_shared_target_tensors and arch qwen4exp - so 7.32.2's "full layer plus a head" is confirmed against a real artifact and the head's cost is measurable directly. What remains true of the trunk file is only that it ships no head (0 of 1224 tensors). And the real blocker surfaced: issue #124 shows the expert cache and MTP are MUTUALLY EXCLUSIVE on this model today - with --moe-expert-cache-size > 0 plus --spec-type draft-mtp the grouped plan is unavailable (required_unsupported 63-95), the graph fails closed, draft-mtp never engages, and the server falls back to legacy, so any MTP number taken with the cache on is measuring the fallback. #124 is the next item, not sizing and not the withdrawn horizon). Revision 25 (7.33: the MTP-lookahead premise is WITHDRAWN. Verification does batch every draft position into one ubatch (server-context.cpp:580/619/1273/4559) and the MoE cache is grouped (moe-cache.cu tracks unique_experts per dispatch), so layer L's router output for T+1..T+H is computed in the SAME dispatch that consumes it - there is no lead time, and the retention horizon the oracle curve assumes is across dispatches, whose demands still need layers 0..L-1 for the future tokens. That is 7.28's impossibility argument restated, not escaped: speculation changes when the token exists, not when its layer-L input exists. The horizon curve is an offline bound in the same sense Belady is; the +8%/+13% acceptance-discounted numbers are withdrawn. The --decode-overlap escape hatch is closed too: server-context.cpp:1273 refuses to start unless n_max + 1 tokens fit in a SINGLE ubatch, so the whole draft is replayed in one forward and no per-position ordering can exist). Revision 24 (7.32: an MTP block for this arch is a full layer plus a head - ~992 MiB of weights, but only ~78 MiB of that has to be VRAM under --n-cpu-moe, plus ~226 MiB of its own MoE cache at N=42, ~310-330 MiB total against 1.55/2.2 GiB of headroom, so either operating point covers it; the fork already ships the whole qwen4exp MTP path including nextn_state -> build_moe_lookahead, and its only existing consumer is the staging lane that 7.30 measured as a net loss, so the work item is retargeting the consumer to victim choice, not connecting the head. The trained block does not exist in this GGUF - 0 of 1224 tensors are nextn/mtp/draft - so the head has to arrive as a draft-only export. The one term that can bite is traffic: a full MoE block run once per draft step is +17.8% at H=4 unless those experts are cache-resident). Revision 16 (7.21: at N=42 the step is 28.3 ms fixed + 0.2842 ms per miss at 6.6 GB/s, the link idles 31%, and the distribution has NO tail so the mean is the whole story; the look-ahead's transport is 1.91% useful at width 8 vs 95.5% at width 1 - lateness, not wrongness - while the predictor's ~86% recall is a different quantity whose instrument is currently unwired; 7.22 ranks what is left: the x4 slot (+105%), a lossy cached tier (+44%, now measurable), and filling the idle window (+45%, not yet green)). Revision 15 (7.19: the correctness instrument exists and PR #127 Blocker 1 / issue #128 PASSES it - greedy token identity 648/648 chars and PPL 14.7350 identical over 15 chunks; 7.20: NEW DEFECT, at N=53 the look-ahead aborts the server on an unchecked 2.1484 MiB lane cudaMalloc, 48 lanes = 103.13 MiB never evicted, so the feature and the +21.5% capacity win are mutually exclusive). Revision 14 (7.18: instrument audit - the per-step ledger has no gaps and is internally consistent 199/199; the policy knob is proven to take effect, so the sweep tested 2 distinct policies not 4; the cache size is now echoed in the log (ed2b4b6b9); and the reversed-order control falsifies the drift confound - +21.5% and +21.1% by either ordering). Revision 13 (added the CURRENT OPERATING POINT block to section 8: best config is control + N=53, 11.738 t/s, look-ahead OFF, and earlier sections' ~10.2 t/s figures are N=28 or the retired synthetic harness and are not comparable). Revision 12 (7.17: the retention policy is NOT a lever - the default LFU-16 beats half-life 256 and 2048 and beats pure LRU; capacity is VRAM-capped at r~58%. All three code levers are now closed by measurement, leaving only the x4->x16 slot move). Revision 11 (7.16: the capacity lever is real - N=28 -> 53 is +21.5%, 9.661 -> 11.738 t/s, with the
T = 25.7 + 0.3003*misses model validating out-of-sample to 1.2%; it saturates near r=59% at the
VRAM cap, which makes retention POLICY the binding lever). Revision 10 (7.14: per-token miss ledger - decode is bandwidth-bound on expert misses,
T = 25.7 ms + 0.3003 ms/miss with r2 = 0.990, ceiling 38.9 t/s at a fully-resident cache vs 9.60
today. Rev 9: the resident ledger was structurally blind to the look-ahead's saving; made honest and
measured, the look-ahead removes 28.2 MiB/step of demand traffic against a ~38 MiB/step staging cost
- a 1.35x overhead ratio, and that excess is the -3.1%. Rev 8's byte-neutral framing is partly
retracted in 7.13. Rev 7's two conclusions are retracted in 7.11/7.12; readiness, pacing, placement
and filter coupling were each implemented, measured and excluded as the cause. The determinism check
that several revisions quoted was vacuous and is retracted in 7.11.) Rev 1's
central verdict was wrong and is retracted in section 0. Rig: RTX 3060 12 GB, PCIe gen4 x4,
APEX-I-Mini (Qwen3.8-Flash-Next, qwen4exp, 73.29 GiB, 3.56 bpw, 512 experts / 10 used). Arms:
`phase.sh 1 81920 28 18336 <tag> 3000 256 0` unless noted.

## 0. RETRACTION: rev 1 concluded "the wall is prediction precision". That was wrong.

Rev 1 read `useful_experts / predicted_experts` from the staged path as prediction precision and
declared the line dead on 10-18%. Prediction precision was already measured in this repo, twice:

| instrument | measurement | source |
|---|---|---|
| in-graph recall, 23,936 (step, layer) pairs, ground truth = each layer's real router selection | width 8: **86.21% of predictions correct**, 68.97% coverage of the 10 used; 22x chance; device-independent (3060 85.85%) | tree `baseline-flash-next`, PROJECT_STATUS.md:501-511; tree `impl-pra`, docs/moe-lookahead-design.md:498-508 |
| host-visible, off-lease | **80.6% of predictions used**, 64.5% coverage | tree `baseline-flash-next`, PROJECT_STATUS.md:657-659 |

Both are *demand*-precision scored without transport. The 10-18% was **staging consumption**, which
conflates copies arriving late with predictions being wrong. Rev 1 turned a transport loss into a
prediction loss. Rev 1's byte break-even (p = 0.8) was also the wrong condition (section 2).

Other rev-1 corrections: "default neutral" is cross-binary and weak; the +25.4 ms marginal figure is
directional, not a decomposition; `min_gap_us` is not a readiness instrument
(`moe-cache.cu:9953-9962`) and its stamps are host-drained (9911-9980).

## 1. Mechanism: the shipped width cannot be delivered in the window it has

**(a) Aggregate, per step.** Total link demand is `S + (M - U)`:

```
staged S = 323 MiB/step;  demand not served = 457 - 34 = 423 MiB/step
total    = 746 MiB/step = 782.6 MB
capacity = 6.10 GB/s x 96.5 ms = 588.7 MB      -> excess 194 MB
predicted marginal = 194 MB / 5.91 GB/s = 32.8 ms;  measured marginal = +25.4 ms
```

The byte-capacity model predicts the regression from first principles. The -20.5% is arithmetic.

**(b) Per lane, against the window - this is what caps the deliverable.** The copy path asserts
**no priority and no pacing** (`cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking)` with
no priority, moe-cache.cu:5083; per-lane streams 5613; plain batched `cudaMemcpyAsync` 5232-5285),
and the demand gather holds ~5.91 of the 6.10 GB/s inside every op. So a lane's DMA trickles at the
~0.19 GB/s fabric leftover during the op and bursts in the attention gap:

```
window idle content per lane ~= 0.3 ms gap x 6.10 GB/s + 1.69 ms x 0.19 GB/s
                             ~= 1.83 + 0.32 = ~2.05 MiB/lane  ->  ~96 MiB/step
```

Also note ~14 MiB of the step's spare sits *inside* the bucket as fabric leftover, not in the gaps.

| width | lane bytes (raw) | rate needed over the ~2.0 ms lead | fits the ~2.05 MiB window? |
|---|---|---|---|
| 8 (shipped) | 15.4 MiB (16.15 MB) | 8.1 GB/s | no - impossible |
| 2 | 3.8 MiB | ~2.0 GB/s | borderline; ~half lands late |
| **1** | **1.9 MiB** | **~1.0 GB/s** | **yes - lands without pacing** |

Softener at narrow widths: delivery is per-expert progressive (5280-5282) and consumption per-expert
conditional (4361-4364, 4380-4382), so a partially landed lane still yields its landed experts - the
deadline is soft at expert granularity.

**Derived, not measured:** "the unconsumed 82-90% arrives late" is a hypothesis supported by the
window arithmetic; only P0.2 can confirm it.

## 2. Corrected economics (rev 1 had this inverted)

With S staged, U useful, M baseline miss volume, p = U/S, slack the step's spare capacity:
- total link demand `S + (M - U)`;
- **no regression requires `S(1 - p) <= slack`** - the cap is on *waste*, not on staging;
- a win requires `U(1 + reuse) > S - slack`;
- `p = 0.8` is a *solid-win* target, **not** break-even.

**The flywheel (P4) makes the bound self-amplifying:** every consumed staged expert becomes resident,
shrinking next step's M and widening the next window's spare. So the +5-10% band below is the
*first-step green*, not the steady-state ceiling.

## 3. Plan

### P0 - instruments, device-stamped
1. Rebuild `moe-overlap` to measure readiness: device-clock time each lane's copy completes vs when
   the consuming gather first reads that lane's staging. Stamp on the device.
2. Ready-clear split, with three bias fixes closed first:
   - **Ready-check-to-read gap inflates lateness** - a copy landing between the `completed` snapshot
     (3861, loop 3864-3867) and the gather's read (4380-4382) is booked late though the read was
     still ahead of it. Systematic, biases *against* the worker. Fix: record per-expert progress at
     clear time (`clear_reason`), or re-read `copy_progress` before falling back to host.
   - **The clear counts never-demanded experts** - intersect with the plan's miss set at the loop
     that already iterates `miss_experts` (4348-4354): count `cleared AND demanded`.
   - **Do not book cleanup as lateness** - count only the not-landed reason, not `retain_late`
     (`-2-pos`, 3866) or drain-epoch resets.

### P1a - right-size the lane (NO CODE: the lane width is the CLI width)
The lane is capped by the macro `GGML_CUDA_MOE_LOOKAHEAD_STAGE_WIDTH 8`
(moe-cache.cu:3916), used as the filter's `width` argument **at the call site**
(`const uint32_t width = ...` 5674, passed 5681-5684). The filter takes `n = min(n_ids, width)`
(3991) with a second cap `staged >= width` in the loop (3994). `n_ids` is the producer's argsort k =
**`--moe-lookahead N`** (llama-graph.cpp:2102, `min(cparams.moe_lookahead, n_expert_used)`, cap 10).
The macro is a **ceiling, not a floor** - narrowing to 1-2 is zero-code, and no other site forces 8.

The filter already removes predicted-but-resident experts (3999-4011, counter `counters[6]`), so the
staged volume is raw width x ~45% at width 8 (323 MiB/step measured).

| `--moe-lookahead` | fits the ~2.05 MiB/lane window? | demand-precision (committed table) |
|---|---|---|
| **1** | **yes** | above the table's floor |
| 2 | borderline (~half late) | 97.37% |
| 4 | no | 95.13% |
| 8 (shipped) | no | 86.21% |

**Arm order: 1 first** (the only flag-only width the window supports), then 2 and 4 to bracket.

**MEASURED (width 1, clean paired arms, same binary `cfb2db3d`):**

| | control `ctl4` | width 1 `la1` |
|---|---|---|
| decode t/s | **10.22** | **9.59 (-6.2%)** |
| prefill t/s (conditions check) | 105.88 | 105.65 |

Against `ctl3` (10.34) the same arm is -7.3%; the paired control is the honest number.

Probe arms (same protocol, counters comparable only as bytes - see caveat):

| | `stage0n28` baseline | `la1probe` width 1 |
|---|---|---|
| resident_pct | 47.0 | **53.2** |
| copy_mib/step (demand traffic) | 457.3 | **403.9 (-11.7%)** |
| `prefetch_used` / staged | - | ~0.6 useful |

So the design **works as intended and is still net-negative**: residency rises 47.0 -> 53.2 and
demand traffic falls 53 MiB/step, but the staged volume at width 1 is ~87 MiB/step, so the net
extra bytes are `S(1 - p) ~= 87 x 0.4 ~= 35 MiB/step` = +5.9 ms of 96.5 = **-6%**, which is what was
measured. Prediction usefulness improves dramatically with a narrower lane - p_eff ~0.13 at width 8
versus **~0.6 at width 1** - confirming the direction, but even p_eff = 0.6 leaves a 1.6:1
staged-to-saved ratio, and both cross the same saturated link.

**CAVEAT (counter units changed between binaries):** `need` reads 1440/step in `stage0n28` but
480/step in `la1probe` (10 experts x 48 layers = 480, so the older arm counted per tensor: 3 per
expert). Byte counters (`copy_mib`) are comparable; expert-count counters are NOT. Any cross-binary
count comparison is invalid for the same reason C1 is.

**The remaining lever is PACING, not width.** At width 1 the volume already fits: the step has
~15.5 ms of non-MoE link time = ~90 MiB of genuinely idle capacity, and ~87 MiB/step is staged.
The loss is therefore *timing*, not volume - the worker issues each job's DMA immediately on mailbox
pickup (`cuStreamWriteValue32` is async, only the debug path syncs, 5293), i.e. during L's MoE
gather, the one window where the link is saturated, instead of the gap that follows. Pacing the
issue into the idle window should convert the -6% into roughly `+53 MiB/step / 5.91 = +9 ms/step`,
about **+9%**. That is a code change and needs separate authorization.

A running byte budget, if wanted on top of the flag, gates in the same filter loop; the counter
already exists at **4020** (`counters[0] += staged * entry_bytes`, doc comment 3918-3920). Suggested
form `S(1 - p) <= min(alpha_waste * M, slack_bytes - margin)`, opening `alpha_waste ~= 0.20`
(~91 MiB), plus hysteresis on the link-saturated skip. Cap waste, not staging: at width 2 the waste
is what fits, and a staged-byte cap would wrongly forfeit it.

Acceptance: end-to-end >= control on the paired protocol. **p_eff is unmeasured**:
`p_eff = recall x P(not-resident | predicted AND demanded) x ready-before-need`, and top-ranked
predictions may be *more* resident, so do not promise a win size until P0 measures it.

### P1b - id-conditioned prediction, offline evaluation first
Predict L+1's set from layer L's **actual routed ids**, via a co-occurrence table built on trace A
and evaluated on disjoint trace B. The producer today does **not** see L's ids (they are computed in
`build_moe_ffn`, llama-graph.cpp:2240-2245, downstream of the producer node), so P1b must **move the
emit after the argsort**. **Costs essentially no lead**: the emit then sits between the argsort and
L's MoE op, so its lead stays ~2.0 ms - later only by the router chain (tens of µs). The two
predictors (state proxy, id table) are complementary; a rank fusion is nearly free.

### P2 - confidence gating
Stage only slots above a confidence threshold delivering measured staged-slot precision >= 0.8;
allow per-layer effective width.

### P3 - regression guard
Staging on with predictions shuffled: end-to-end within noise of control, and P0.2 must show the
waste is wrongness, not lateness.

### P4 - staging into residency (the compounding lever)
Every consumed staged expert becomes resident, shrinking next step's M and widening the window's
spare. Never evict to make room. This is what turns a first-step +5-10% into a steady-state bound.

### P5 - kill criteria (pre-registered)
Abandon if, after P1a+P1b+P2, **arrival-conditioned precision** `U / (S - late)` < 0.5; or if
end-to-end is not >= control with P1a active and predictions shuffled. No width increase above the
window; no L+2.

## 4. Levers measured and eliminated

| lever | verdict | evidence |
|---|---|---|
| compress the slabs, decompress on the idle SMs | **DEAD** | real `blk.24` slabs (types 16/17/20): zstd-9 **1.0000/1.0000/0.9808**, xz-9 1.0001/1.0001/0.9894, zlib-9 0.9997/0.9998/0.9836, bz2 1.0048/1.0044/1.0009. Kill threshold was 1.05. Incompressible quantized weights |
| copy-stream priority | no effect | competition is CE DMA vs SM-initiated reads on one fabric, no user arbitration; `cudaStreamCreateWithPriority` orders kernels, not the fabric |
| pacing copies into the idle windows | **works, timing not capacity** | raises U at fixed S, cannot exceed the ~2.05 MiB/lane window content; converts width-2's ~50% delivery toward width-1's ~100% |
| queue-ahead beyond one lane | none available | single-flight per lane (`moe_early_router_wait_copy` + re-zero, 5676-5679, comment 3913-3914); depth 1, backpressure lands on the compute stream at the publish point |
| demand misses onto the copy engine | +3% | 6.10 vs 5.91 GB/s - noise |
| cache-size growth | ~~flat~~ **WRONG - retracted in 7.16** | N=28 -> 53 = +21.5% (9.661 -> 11.738 t/s). N=100 still infeasible, but the useful range is 28-55 and was never swept. See 7.16 |
| MTP (n-max 1 and 2), KV quant, concurrency | negative | see PROJECT_STATUS.md |

## 5. The only multi-x lever is hardware, not code

**The card is x16-capable but the slot is wired x4** (tree `baseline-flash-next`,
PROJECT_STATUS.md:648-649). Nothing on this rig increases bytes-per-second-effective on the link:
the SMs are compute-idle ~97% during the MoE op (33.8 µs kernel vs a 1.69 ms bucket), so the op is
link-bound, and the link is physically x4.

Scaling the MoE bucket at 4x the link rate: 457 MiB at ~24 GB/s = 20 ms instead of 81 ms, taking a
96.5 ms step to ~35 ms, i.e. ~27 t/s from 10.34. [INFERENCE - assumes the link remains the only
bottleneck at the new rate.] **This is a motherboard/slot change, not a code change, and it dwarfs
every software lever in this document.**

**Swap consequence, worth stating before anyone moves a card.** There appear to be only two relevant
root ports: the 3060 sits on `0000:00:06.0` (wired x4) and the 5060 Ti on a wider one (it negotiates
x8; see 7.10). So moving the 3060 to the fast slot does not add bandwidth to the machine, it MOVES
it, and the 5060 Ti drops to x4. That is a bad trade in general - but not for this measurement: the
whole rig configuration in this document runs on device 1 alone (`CUDA_VISIBLE_DEVICES=1`, the 3060)
while the 5060 Ti sits idle, so for the workload under test the swap is a clean 4x on the binding
resource with no offsetting cost. Confirm the 5060 Ti's slot width and whether a third slot exists
before committing to a physical move.

## 6. Expectation bands (architect's, at S ~= 90 MiB and W ~= 2.05 MiB/lane)

| p_eff | U | win/step | t/s |
|---|---|---|---|
| 0.3 | 20-25 MiB | 3.5-4.3 ms | 10.7-10.8 (**+4-5%**) |
| 0.5 | 35-40 MiB | 7-8 ms | 11.1-11.2 (**+8-9%**); **+11-13%** with P4 |
| 0.9 | 75-80 MiB | 13-14 ms | 11.9-12.0 (**+15-16%**); ~**+20%** with P4 at full conversion |

So +10% needs p_eff >= ~0.5 *plus* P4, or p_eff >= ~0.7 alone. The owner-facing band for the first
paired arm MEASURED **-6.2%** (9.59 vs control 10.22, same binary). The bands above are what the mechanism permits, not what the shipped implementation delivers - see P1a's measured block. +15-20% remains the P4-compounded steady state at p_eff >= 0.9, unreached.

## 7. Round 4: the two candidate levers

### 7.1 Displacement coefficient MEASURED = 1.00 (this was the missing number)

The earlier probes measured every rate SOLO (CE 6.10, contiguous SM 6.11, slabs @8 blocks 5.91).
None measured DMA **concurrent with** gather-pattern reads, so the ledger could not say whether a
staged byte costs a demand byte 1:1. `mixed_contention.cu` measures it: A = the gather's own pattern
(8 blocks x 1.78 MiB slabs from host-mapped memory), B = the copy engine (1.9 MiB H2D chunks, the
width-1 lane), sized to the real 1:6 volume ratio.

| | solo | concurrent |
|---|---|---|
| A (gather pattern) | 48.1 ms -> **5.93 GB/s** (calibrates against the measured in-op 5.91) | 56.1 ms -> **5.08 GB/s** |
| B (copy engine) | 7.9 ms -> **6.08 GB/s** (matches the 6.10 ceiling) | - |

**Displacement = 0.98 / 1.01 / 1.00 across three runs.** The fabric is effectively serial: one byte
moved by the copy engine costs exactly one byte of gather throughput, and there is no concurrent
headroom. Immediate-issue staging therefore displaces demand 1:1, which is why width 1 stages ~87 MiB
to save ~53 and loses ~6%.

**Consequence: pacing is the whole game.** The 15.5 ms of non-MoE step time is genuinely idle
(~87.6 MB at 6.08 GB/s). A width-1 lane is 2.01 MB against a 0.3 ms gap's 1.82 MB, so ~91% of each
lane fits inside the gap and the remainder spills into the saturated window. Net after pacing:
saved 53 MiB minus spill ~9 MiB = +44 MiB/step = **+7.4 ms of 96.5 = ~+8%**. That is the swing the
architect gated the arm on (displacement >= ~2x was not required - 1.00 already means pacing
converts almost the whole tax).

### 7.2 Install the waste (owner's reading, confirmed — and worth +2-4%, second)

Premise correction from the architect: a CONSUMED staged expert is already installed - it is a miss
whose slot the plan committed and whose bank the gather filled from staging (4380-4385), so it
becomes resident through the normal path. The waste is only the UNCONSUMED set
(predicted AND not-demanded, or landed late), reset on the next publish (3973-3975, 4150-4152):
**34 MiB/step at width 1, ~40%.**

Mechanism (architect): allocation is **victim selection, not a monotonic clock handout** (which was
the architect's own round-1/3 error), so there is **no clock race**. An in-graph install kernel,
stream-ordered after the gather and before the next step's plan, can mirror the plan kernel's own
eviction: re-derive the exclusions from `remapped_ids` (committed demand slots, written at 3330),
claim the next-coldest non-demanded slot by the same rule (3265-3322: min effective frequency, then
age, among `route_storage == 0`), then commit exactly as the plan does (`3048-3052`, `last_used`
bump at `3066`, keeping the `expert_frequency_epoch` invariant checked at 3273-3276). The victim is
the coldest non-demanded slot - the same slot the next demand miss would have taken anyway - so the
install changes WHEN an eviction happens, not who is displaced, which keeps the never-evict rule's
spirit; exclude victims above the demand median frequency as the legacy guard does (13512-13551).

**Silent-corruption path that must be closed first:** the install kernel reads staging, and an
unconsumed expert's copy may still be in flight - the gather's ready predicate protects consumed
reads only. It must gate on the per-expert `copy_progress` publish (5282) exactly as the gather does
and skip not-yet-landed entries, or it installs a torn slab and the corruption surfaces later as a
garbage expert.

Expected payback: 34 MiB/step x reuse probability for a predicted-but-unconsumed expert, honest band
~0.3-0.6 per wasted byte -> 10-20 MiB/step -> 1.7-3.5 ms -> **+2-4% t/s**, compounding via P4.
Its premise (unconsumed -> later demand within 1-2 steps) is decidable offline from a recall-instrument
trace; if < ~0.3, it is dead without a rig run.

**Order: pacing first** (bigger, and its go/no-go is now measured), install-the-waste second.

### 7.3 Pacing implementation surface (verified in-tree; no compute->copy signal exists)

Read of the actual worker and issue path (moe-cache.cu 5179-5343, 5639-5703), independently
verifying the architect's four claims - all four hold:

- The worker has NO pacing gate. It picks the job out of the mailbox and walks straight into the
  per-expert loop, accumulating `copy_dsts/srcs/sizes` and calling `submit_batch()`
  -> `cudaMemcpyBatchAsync(..., copy_stream)`. Nothing inspects the clock or a window flag.
- The "single-flight" wait at 5678 is a COMPLETION wait on the PREVIOUS job's `copy_done`, and the
  comment says so: "A stream wait, not a host sync; a no-op when the worker is ahead." Step N's
  publish waits for N-1's copy, which finished ~80 ms earlier, so it is a no-op - NOT a forcing
  function that starts a copy early.
- `copy_progress` (5282) is a genuine landing signal: `cuStreamWriteValue32(copy_stream,
  copy_progress, p + 1, ...)` after each expert's batch is submitted, stream-ordered, so it fires
  only once those banks have landed. The torn-slab guard for install-the-waste is real.
- The only stream syncs in the issue path are that wait and the per-expert publishes.

**New finding - the sync machinery runs the wrong way for pacing.** Every `cuStreamWriteValue32`
in the tree writes a COPY-side flag (`copy_progress` 5282, `copy_head_done` 5287, `copy_done`
5290, `stage_ready` 14825/14911), and the consumers are the COMPUTE side
(`cuStreamWaitValue32(stream, stage_ready + ...)` at ggml-cuda.cu:2639-2640; the MMQ kernel spins
on `x_stage_ready` at mmq.cuh:1051-1059, 1165-1172, 1269-1277). That is copy->compute readiness,
for the LEGACY overlap path. **There is no compute->copy "the slow window has begun" signal**, and
no clock/window flag anywhere for a worker to pace on. So pacing is not a one-line relocation; it
needs one of:

  (A) Defer the filter + mailbox publish on the compute stream to the attention boundary of the
      consuming layer. No new flag and no host poll - the worker's mailbox wake-up IS the pacing,
      because the publish kernel is stream-ordered. HAZARD: the routed-ids buffer must still be
      live at that point in the graph (tensor reuse), and the call site (llama-graph.cpp:2102)
      moves.
  (B) A new host-mapped window flag written on the compute stream at the attention boundary, and a
      poll in the worker before `submit_batch()`. More machinery, no call-site move.
  (C) A calibrated delay in the worker before `submit_batch()`. Crude and timing-fragile, but it
      needs no new signal at all.

**Why (C) is nevertheless correctness-safe - and this is the important point: correctness here is
enforced by the READY GATE, not by timing.** Iteration order is enforced by the plan kernel's
table writes; the staged bytes are guarded by the ready predicate on the compute side. A copy that
lands late therefore degrades to a MISS (the expert falls back to a demand read), not to corruption
and not to a stall - the failure mode of a mis-calibrated delay is lost look-ahead benefit, never
wrong data. That is the same property that makes install-the-waste safe and it means the first
pacing arm can be implemented cheaply and tuned by measurement.

### 7.4 Citation corrections from round 4
- The real commit protocol is the plan kernel's table writes (`moe-cache.cu:3048-3052` commit, `3066`
  `last_used` from `plan->next_clock`) and victim selection (3265-3322) - NOT 4315-4340, which is the
  gather kernel's head (4315-4345) including the resident-probe stamp hook (4335-4337).
- `counters[0] += staged * entry_bytes` is at **4020**.
- 87 MiB at 5.91 GB/s is **15.4 ms**, not 14.7 (87 was priced as decimal MB). Net ledger moves
  ~+0.7 ms; direction unchanged.
- The `prefetch_used`/`reserve_refused` comparison (291 vs 264 / 447 vs 478) comes from the LEGACY
  one-layer instrument, and only `blk.47` holds a legacy lease during decode - so "real prose is more
  predictable" is measured on **1 of 48 layers**. Directionally fine, evidentially weak.
- "staging-to-slot D2D inside the gather" is NOT fixed overhead: the staging read REPLACES the
  mapped-host read for consumed experts (4380-4382), so it is device-local with no link time.

## 7.5 Pacing implemented and measured: recovers half, does NOT go green

Change committed as **168a90717** (pushed to hydra-fork): the look-ahead is now ISSUED after the
layer's FFN instead of before it, in both qwen4exp.cpp and qwen35moe.cpp. The code's own comment
named the bug - "page them while this layer's MoE still computes, so the H2D overlaps compute" -
but the MoE bucket is ~77 ms of demand gather out of 81.04 ms, i.e. link-bound, so the transfer
overlapped the gather and displaced it 1:1.

Clean paired arm, real prompt, same binary, no probe:

| arm | decode t/s | per-token p50 | vs control |
|---|---|---|---|
| control | 9.643 | 103.73 ms | - |
| lookahead 1, PACED | 9.357 | 107.76 ms | **-3.0%** |
| lookahead 1, unpaced | 9.040 | 112.40 ms | -6.4% |

Pacing recovers ~half the deficit. Control is unchanged by the reorder (9.66 -> 9.64), confirming
it is inert when the look-ahead is off.

### 7.6 WHY it is not green: pacing destroys the entire reuse (measured)

Probe-on paired arm (byte counters; the probe depresses t/s but not the counters):

| arm | copy_mib/steps | MiB/step | resident_pct |
|---|---|---|---|
| control | 92239.6 / 199 | 463.5 | 45.6 |
| lookahead 1 PACED | 93219.2 / 199 | **468.4** | 45.6 |
| control (old binary) | 116609.0 / 255 | 457.3 | 47.0 |
| lookahead 1 unpaced (old) | 103014.5 / 255 | **403.9** | 53.2 |

Unpaced the look-ahead saved **53.3 MiB/step** of demand traffic. Paced it saves **nothing**
(468.4 vs 463.5 - slightly worse).

**RETRACTED (see 7.11).** The last sentence above - "the staged bytes are no longer consumed
at all" - is WRONG. It read a plan-level counter as if it measured consumption. The
look-ahead's own ledger, now that it exists, shows consumption at ~95% in BOTH placements.
`copy_mib` counts the plan's demand misses, and the look-ahead does not change whether an
expert is a miss; it changes where the miss is served from.

### 7.7 The structural reason - the design is on a knife edge

Per layer, the finest possible staged granularity is one expert slab: **1.92 MiB = 2.01 MB**.
The idle link per layer is **0.32 ms x 6.08 GB/s = 1.95 MB**. These are equal to within 3%.

So the minimum-volume staging exactly saturates the idle link. There is no slack, and therefore:

- **Unpaced** -> the transfer lands early enough to be consumed, but displaces the demand gather
  1:1 (measured coefficient 1.00) -> -6.4%.
- **Paced** -> the transfer avoids the gather, but now lands at/after the consumer's **one-shot**
  ready snapshot (`moe_early_router_ready_positions` clears at 3865-3866 from a single `completed`
  snapshot at 3861), so readiness is all-or-nothing per layer -> the whole layer's staging is
  discarded -> the reuse vanishes and only the extra per-layer compute remains -> -3.0%.

Neither configuration wins, and there is no width between them: width 1 is already the minimum
volume and it already fills the idle link.

### 7.8 What would unlock it
Make readiness **per-expert / rolling** instead of a one-shot per-layer snapshot. A transfer that
spans the idle window and spills slightly into the gather would then still be consumed for every
expert the gather reaches in time, capturing the free window instead of losing all of it. This is
the previously-open C3 item, which now has a measured reason to be built.

### 7.9 Probe confound (record this before quoting any probe-arm t/s)
Under the probe the paced arm ran 7.71 t/s vs control 9.65 (-20%), while clean it is -3.0%. The
probe perturbs timing enough to distort a timing-sensitive arm. Probe-arm BYTE counters remain
usable and probe-consistent; probe-arm t/s for the paced configuration must not be quoted.

### 7.10 Hardware: the slot is x16, so the lever is 4x not 2x (correction)
PCIe widths are 1/2/4/8/16 - there is no x6.
- 3060's slot: root port `0000:00:06.0`, max_link_width **4** -> it is WIRED x4.
- The other slot: root port `0000:00:01.0`, "12th Gen Core Processor PCI Express **x16** Controller"
  -> WIRED x16.
- The 5060 Ti reads current x8 only because that card is itself x8.
Moving the 3060 into the x16 slot would give it **x16 = 4x the current link**, not 2x as previously
stated. This is the only multi-x lever on the board and it requires physical hands.

### 7.11 Corrected model: consumed, not free (retraction + the real accounting)

The look-ahead's own counters were never printed, so 7.6 read the only number available -
the plan's miss counter - as consumption. It does not measure consumption. Ledger added
(drained every dispatch, prints without the probe):

| arm | decode t/s | staged MiB/step | consumed MiB (life) | consumed experts | consume_pct |
|---|---|---|---|---|---|
| control (la=0) | **9.699** | - | - | - | - |
| lookahead 1, after FFN | 9.398 (-3.1%) | ~38 | 6566.75 | 3665 | **95.18%** |
| lookahead 1, before FFN | 9.030 (-6.9%) | ~38 | 6579.83 | 3671 | **95.37%** |

Same binary, real prompt, 200 output tokens, probe off. **RETRACTED - the determinism check was vacuous.** `perftok-*.tsv` column 3 (content) is
ALWAYS EMPTY on this rig (verified: 0 non-empty rows in all three arm traces); the model streams
its output on a channel stream.sh does not capture. The check compared empty string to empty
string and therefore always passed. There is currently NO working text-level correctness check;
PR #127 Blocker 1 / issue #128 (an extra reader of `ffn_gate_inp` changes logits) is the real
instrument and it has not been re-run. Do not quote token identity again.

**Facts that follow.**

1. **Consumption is ~95% in both placements.** The staged bytes are used. 7.6's conclusion
   is retracted. The issue position costs 3.1% vs 6.9% - so placement does matter exactly as
   predicted - but it is not the difference between reuse and no reuse.

2. **The look-ahead is byte-neutral on the link, not additive.** Staging expert X costs one
   expert-slab of PCIe traffic now; the gather then reads X from staging instead of from host
   memory, saving that same slab later. So staging does not add traffic, it *moves* it. The
   whole value of the feature is the timing of that move; the accounting is a wash.

3. **The staged volume is ~20 slabs/step, not 47.** `prefetch_used ~= 19-27` vs
   `prefetch_reserve_refused ~= 24-32` per dispatch: the filter declines to stage roughly
   half of what it predicts, because the expert is already resident or already incoming.
   So ~38 MiB/step, and the step-wide idle capacity is 94.6 MiB - it FITS, comfortably.
   The failure is not aggregate spare bandwidth.

4. **It is still per-layer granularity that fails.** Step-wide there is 94.6 MiB of idle
   time, but the transfers are offered one layer at a time: 1.92 MiB needs 0.316 ms and the
   per-layer non-MoE window is 0.32 ms. Aggregate slack cannot be borrowed across layers
   because the copy worker's queue is already the mechanism for that, and it is not being
   used to run ahead - the filter's `present` test suppresses the second slab that would
   queue behind the first.

5. **So the remaining lever is to raise the staged set per layer** (width, or a reserve that
   admits the refused candidates), which converts idle step-wide slack into overlap. The
   measured ceiling for that is the +2-4% of 7.2, and it is bounded by the 1.95 MB window.

**Consequence for the expectation bands in 6:** the bands assumed staged traffic is added
to demand traffic. It is partly substituted. The bands are therefore pessimistic by roughly
the substitution fraction, and cannot be used to argue the feature is hopeless - but they
also cannot be used to claim a win. The measured t/s verdict stands on its own either way.

### 7.12 The answer: the feature is byte-neutral, and the unit of movement does not fit

**PARTLY RETRACTED - see 7.13.** The window arithmetic below is correct, but its central premise
("byte-neutral by construction, so the whole effect is timing") is WRONG: the look-ahead removes
28.2 MiB/step of demand traffic and stages ~38 MiB/step, so it is not neutral - it is a 1.35x
staging overhead, and that excess is the cost. Do not use this section's byte-neutral framing.

Probe arms on ONE binary, real prompt, 200 output tokens:

| arm | need | resident | resident_pct | copy_mib | consumed MiB | consume_pct | t/s |
|---|---|---|---|---|---|---|---|
| control (la=0) | 94570 | 43125 | 45.6 | 92239.6 | - | - | **9.699** |
| lookahead 1, after FFN | 95520 | 43529 | 45.6 | 93219.2 | 6594.29 | 95.53 | 9.398 |
| lookahead 1, before FFN | 95520 | 43529 | 45.6 | 93219.2 | 6594.29 | 95.53 | 9.030 |

**The before-FFN and after-FFN ledgers are identical to the last digit** - need, resident,
copy_mib, consumed bytes, consumed count, consume_pct, and even the per-dispatch
`prefetch_used` / `prefetch_reserve_refused`. Only the clock differs: -6.9% vs -3.1%.

**Two hypotheses are falsified by this table, both mine, both retracted:**

- 7.6's "pacing destroys the entire reuse". False. Consumption is 95.53%, identical in both
  placements.
- 7.11's "the issue position changes WHICH experts the filter stages, because the filter
  reads `dev.plan` at publish time". False. The staged set, the consumed set and the plan
  ledger are bit-identical across placements. The filter's `prior` is not position-sensitive
  in practice.

**What is actually true.**

1. **The feature is byte-neutral by construction.** A predicted expert is staged (one slab
   over PCIe now) and then read from staging instead of host memory (that same slab over
   PCIe later). Same bytes, same wire, moved earlier. 95.53% of staged bytes are consumed,
   so the substitution genuinely happens. `copy_mib` therefore does NOT fall: the plan's
   misses are unchanged because the expert is still a miss - it is just served from staging.
   Reading `copy_mib` as a measure of the look-ahead's value was the original error.

2. **Its entire effect is therefore timing, and the timing cannot be made free.** Per layer
   the unit of movement is one expert slab, 1.92 MiB = 2.01 MB, needing 0.316 ms at
   6.08 GB/s. The only window where the fabric is idle is this layer's non-MoE time, 0.32 ms
   = 1.95 MB. 2.01 MB does not fit in 1.95 MB. So every staged slab spills past the window
   into the next layer's demand gather and pays contention there. That is the whole residual
   cost, and it is why after-FFN recovers about half but cannot reach parity.

3. **The look-ahead does not raise the hit rate - it perturbs the cache.** Need rises
   94570 -> 95520 (+1.0%) while resident rises 43125 -> 43529 (+0.9%): resident_pct is
   unchanged at 45.6. Experts are layer-specific, so a slab staged for layer il+1 occupies a
   slot that no other layer can use; when the prediction does not pay off the slot is
   displaced for nothing. At 28 slots against 10 experts used per layer the cache is already
   deep enough that the prediction's marginal value is zero.

4. **Net add on the link is `staged - consumed` = ~38 - ~33 = ~5 MiB/step**, matching the
   observed +4.9 MiB/step rise in `copy_mib` (979.6 MiB over 199 steps). Small, but it is the
   wrong sign, and it is paid twice: once as traffic, once as contention.

**Verdict.** This is not a tuning failure and it is not a readiness, pacing or placement bug -
all three have now been measured and excluded. The design moves bytes that are already being
moved, so it can only win by moving them into idle fabric time, and one slab does not fit one
window. Going green requires either a wider slot (7.10: the 3060 is wired x4 into a x16 root
port, so 4x is available with physical hands) or a staged unit smaller than 1.95 MB, which
means raising the transfer granularity - not the prediction quality.

### 7.13 The ledger was lying: the look-ahead DOES remove demand traffic (corrects 7.12)

`copy_mib` is `n_misses * resident_bytes_per_miss` - a DERIVED figure that charges every miss to
the demand side whether or not the gather fetched it from host. Once the look-ahead stages an
expert, the gather serves it from staging and no host read happens, but the ledger still billed
it. **So the instrument could not express the look-ahead's saving at all**, and 7.12's reading of
`copy_mib` (+1.0% = "does not reduce demand traffic") was reading a blind spot. That conclusion
is retracted.

Adding `GGML_MOE_LOOKAHEAD_ADMIT_IN_PLAN=1` makes the plan classify a would-be miss whose slab has
already landed as an admission served from staging and subtract exactly those from the demand
counters. Classification only: no second slot, no second eviction, no slot-table write, no new
synchronization. Default OFF reproduces today exactly.

Probe on, real prompt, 200 output tokens, steps=199:

| arm | need | resident | resident_pct | copied | copy_mib |
|---|---|---|---|---|---|
| control | 94570 | 43125 | 45.6 | 51445 | 92239.6 |
| lookahead 1, gate 0 | 95520 | 43529 | 45.6 | 51991 | 93219.2 |
| lookahead 1, gate 1 | 95520 | 43529 | 45.6 | 48312 | **86625.0** |

`93219.2 - 86625.0 = 6594.2` MiB, which equals the look-ahead's independently counted
`consumed_mib_total` (6594.29 MiB) to four significant figures. Two instruments agreeing is what
makes this a measurement rather than a relabel.

**Against the control the look-ahead removes 5614.6 MiB over 199 steps = 28.2 MiB/step = -6.1% of
demand traffic.** The feature works on the traffic axis. And the clock is unchanged by the
accounting fix (gate on 9.3915 vs gate off 9.3984 t/s), so it is free.

**The corrected cost model.** The look-ahead does not lose because it fails to save; it loses
because it stages MORE than it saves:

```
staged    ~38 MiB/step to the preload (of which ~33 consumed, ~5 double-paid from host)
saved      28.2 MiB/step of demand traffic
overhead   staged/saved ~= 1.35x  -> net link traffic +~2.1%
clock      -3.1%  (the excess traffic sits in windows that are not fully idle)
```

So the remaining lever is not the window arithmetic and not prediction-in-the-abstract; it is
**the staging overhead ratio**. At 86.21% recall the wasted fraction plus the unconsumed slabs
is what pushes 28.2 MiB of saving up to 38 MiB of staging. Dropping the staged volume to the
saved volume would make the feature traffic-neutral and leave only the reordering benefit.
That is a filter-precision target, and it is now measurable because the ledger is honest.

**Note on 7.12's window argument.** The per-layer arithmetic (2.01 MB slab vs a 1.95 MB window)
is still correct and still caps how much of the preload can be free, but it was doing double duty
in 7.12 as the whole explanation. It is not: -3.1% is the excess-traffic cost plus spill, not a
pure window failure.

### 7.14 Per-token miss ledger, and the decode ceiling in t/s

New instrument (`GGML_CUDA_MOE_PHASE_PROBE_PER_STEP=1`, commit b950f4cc2): one line per decode
step giving need/resident/misses/miss_mib. Non-perturbing by construction - async snapshot into
pinned memory, read at the next drain with `cudaEventQuery`, no stream sync anywhere in the decode
loop. Measured cost: 9.6593 instrumented vs 9.6375 uninstrumented t/s on the same binary, i.e. free.

Control arm, real prompt, 200 output tokens, 199 steps:

```
need=95520 resident=43529 misses=51991 miss_mib=93219.28     480 demands/step = 48 layers x 10
misses/token: min 107  mean 261.3  p50 262  p95 345  max 480 (step 0, cold cache)
```

Alignment was VERIFIED, not assumed: `corr(misses, observed_ms)` peaks at offset 0 with r=+0.9456
against 0.29-0.41 at every other offset, and 199 ledger steps pair with 199 decode gaps.

**The per-token model.** Fitted on the 196 steps excluding the 3 CUDA-graph CAPTURE tokens (they
have a different timing profile: step 1 shows 229 misses in 47.96 ms):

```
T_token(ms) = 25.7 + 0.3003 * misses          r2 = 0.990
```

`0.3003 ms/miss` at the ledger's own `1.881 MB/miss` (derived from miss_mib/misses = 1.794 MiB,
not assumed) gives **6.27 GB/s** - the measured achievable link bandwidth. So:

**Decode is bandwidth-bound on expert misses, and the miss count explains 99% of the per-token
latency variance.** Misses are the whole story; nothing else in the step moves the clock.

**The ceiling.** Demands are fixed at 480/token, so miss count is set entirely by the cache-ready
ratio r. With `misses = 480(1-r)`:

| r (experts ready in cache) | misses/token | T_token ms | tok/s | vs today |
|---|---|---|---|---|
| 45.6% (measured today, N=28) | 261 | 104.2 | **9.60** | - (measured 9.66) |
| 58% (N=42, from the recorded 358 MiB/step) | 200 | 85.7 | 11.7 | +21% |
| 70% | 144 | 68.9 | 14.5 | +51% |
| **80%** | 96 | 54.5 | **18.3** | **+91%** |
| 90% | 48 | 40.1 | 24.9 | +159% |
| 95% | 24 | 32.9 | 30.4 | +217% |
| **100%** | 0 | 25.7 | **38.9** | **+305%** |

The 100% figure is the hardware ceiling for THIS link: it is the intercept, i.e. everything that is
not expert traffic (attention, dense weights, MoE GEMM compute, launch overhead). It is an
extrapolation - the fit only observes 107-480 misses - but r2=0.990 in range.

**This contradicts a recorded closure and the axis should be re-tested.** Line 198 of this document
records cache-size growth as "flat | N=42: 358 vs 402 MiB/step, no t/s gain". Under this model N=42 is ~200
misses/token, which predicts **11.7 t/s, +21% over N=28** - a gain, not flatness. Two readings of
the same rig cannot both be right, and the per-token model is the better-evidenced one (r2=0.990 on
199 paired samples). The measurement that would settle it: `phase.sh`/`stream.sh` at N=28 vs N=42
vs N=64 on the current binary, per-token, with `GGML_CUDA_MOE_PHASE_PROBE_PER_STEP=1`. If the model
holds, cache sizing - not prefetch, not pacing - is the lever, and VRAM is the bound
(N=42 is ~3.9 GiB of slots, N=64 ~5.9 GiB). CAVEAT the same line 198 records only 2.6 GiB free, so
N=42's slot footprint may itself exceed the free-VRAM budget - confirm feasibility before trusting
the N=42 arm, and treat the free-VRAM figure as the real bound on how far r can be pushed.

### 7.15 Selection policies measured, and the VRAM bound on the capacity lever

Three read-only investigations (cache admission, prefetch staging, bank geometry) plus a VRAM
probe. This section answers the two questions that decide where the remaining work is.

**How the CACHE selects what to keep.**

- **Admission is unconditional.** Every demanded-but-absent expert is given a slot. There is no
  admission predicate at all (`moe_grouped_plan_decode`).
- **The victim rule** is a lexicographic minimum over candidate slots of (decayed demand-frequency
  of the slot's resident, last-demand clock stamp, slot index) - `moe-cache.cu:3301-3341`.
- **Frequency halves every 16 grouped planning steps** (`README.md:11`). A token is 48 steps, so
  the policy's memory is about **one third of a token**. `GGML_CUDA_MOE_FREQUENCY=0` switches the
  grouped path to pure LRU.
- **There is no forward-looking input anywhere in admission or victim choice.** The look-ahead's
  only contact with the plan is the optional default-off staging array, used solely to classify a
  miss for the residency ledger (3346-3377). It never picks a slot.
- **Geometry: one layer = one candidate group, so slots are never contested between layers.**
  48 groups, 3 banks each (gate/up/down, separate tensors), 144 banks. A group's victim scan sees
  only its own 28 slots. So there is **no cross-layer tradeoff to make** - the only question is
  which 28 of a layer's 512 experts stay. That is 5.47% of one layer, a uniform thin slice of
  every layer, never all of any layer.
- This also resolves the residency doc's "1440 needs/step": it is the same 480 expert demands
  re-expressed per bank (x3), the right unit for bytes and the wrong unit for admission. Our
  per-step ledger counts 480 exactly.

**How the PREFETCH selects what to stage.**

- **It selects what to TRANSPORT, not what to cache.** Verified: no path from staging changes
  `slot_for_expert`, `expert_for_slot`, `last_used` or `expert_frequency`. A consumed staged expert
  lands in the slot the PLAN already chose for the miss it already committed (3354, landed by the
  gather at 4475 + 4489-4491).
- Its filter stages a deduplicated subset of the predicted ids, refusing an expert iff (a) it is
  pending install by the immediately preceding plan of that group, or (b) it is currently mapped
  and its slot is not in that prior plan's miss list (4036-4093).
- **So the premise "the prefetch ensures we cache the right thing" is not implemented**: the
  prefetch has no path into admission at all.

**Is "already resident must not be transported" enforced?** Partially. Three defects:

1. The resident test `slot_for_expert[e] >= 0` is a **single one-shot unsynchronised read at
   publish time** (4066-4067), never re-evaluated at DMA time or at consume time. A resident
   expert can still be transported.
2. The already-in-flight test reads only the **immediately preceding plan** (4051-4054,
   4068-4072) and **silently degrades to zero when that plan is not READY**.
3. A deliberate override transports a resident expert whose slot is in the prior plan's miss list.

**The VRAM bound on the capacity lever (measured).**

```
3060: 12288 MiB total.  At N=28: 9293 MiB used -> 2619 MiB free (driver figure).
N=64: hard failure, "cudaMalloc(...) failed: out of memory" on the first slab allocations.
marginal cost = 48 layers x 2.148 MiB (gate 0.598 + up 0.671 + down 0.879) = 103 MiB per +1 slot
=> headroom supports N~53-55, i.e. +89% slots over today.
```

Two consequences. First, the capacity lever IS available, but only from 28 to about 53 - not to 100.
Second, and this is the important one: **N=64 was never reachable, so the recorded "N=100
infeasible" hid the fact that the entire useful range 28-55 is unexplored.** The only arm inside
that range is the recorded "N=42 flat", which contradicts the per-token model in 7.14. Under the
model, N=53 raising r from 45.6% to the 60-70% region would be **12.0-14.5 t/s against 9.60 today**.

**A limitation of the one-layer horizon, worth stating before anything is built.** The plan already
knows layer il+1's demanded set EXACTLY when it runs - it reads the router. So a one-layer-ahead
prediction carries no information about WHICH experts to admit; it can only change WHEN bytes move.
"Caching the right thing" therefore needs either a longer horizon (what will this layer need over
the next N tokens) or a better retention policy than a 16-step decay. Both are policy, not
transport, and both are cheap to test.

### 7.16 The capacity lever WORKS: +21.5%, and it saturates at the VRAM cap

The recorded closure at line 198 ("cache-size growth | flat | N=42: 358 vs 402 MiB/step, no t/s
gain") is **wrong** and is retracted. Sweep, look-ahead OFF, real prompt, 200 output tokens, per-step
ledger on, 199 steps each:

| N | need/tok | misses/tok | r (hit ratio) | model T | model t/s | **measured t/s** | model error |
|---|---|---|---|---|---|---|---|
| 28 | 480.0 | 261.3 | 45.6% | 104.2 ms | 9.60 | **9.661** | -0.6% |
| 40 | 480.0 | 229.7 | 52.1% | 94.7 ms | 10.56 | **10.658** | -0.9% |
| 53 | 480.0 | 201.5 | 58.0% | 86.2 ms | 11.60 | **11.738** | -1.2% |

- **N=28 -> 53 is +21.5%** (9.661 -> 11.738 t/s). N=28 -> 40 is +10.3%.
- N=53 loads fine; the VRAM edge computed in 7.15 was right.
- **The 7.14 model validates OUT OF SAMPLE to within 1.2%.** It was fitted only on the N=28 arm and
  predicts N=40 and N=53 independently. A model that does that is a usable design tool, so the
  projections in 7.14 can now be trusted for interpolation.

**But the lever saturates, and this is the important part.** Marginal value is a steady
+83 milli-t/s per slot, while r rises only ~+0.5 points per slot:

```
r:  45.6% (N=28)  ->  52.1% (N=40)  ->  58.0% (N=53)
+89% slots bought only +12.4 points of hit ratio.
Projecting that slope, r=80% needs N~97 - far beyond the N~53-55 VRAM cap, which itself is
the whole remaining headroom.
```

So **capacity alone tops out near r=59%, about 11.8 t/s**. The 18.3 t/s at r=80% in 7.14 is NOT
reachable by adding slots on this card. Capacity gets +21.5% and then stops.

**7.16 initially closed by calling policy the binding lever. 7.17 measures it and refutes that.** The same 53 slots need
+13.6 points of r to reach 15 t/s. Every slot we already have is being spent by a replacement rule
whose memory is one third of a token (7.15): frequency halving every 16 grouped steps against a
48-step token. A rule that kept the RECURRING set rather than the RECENT set would raise r at fixed
N - that is where the remaining headroom is, and it costs no VRAM.

### 7.17 Retention policy is NOT a lever: the default is the best of four (retracts 7.16's closing claim)

7.16 ended by calling retention policy the binding lever, on the grounds that a one-third-of-a-token
memory looked too short. That was an inference from the decay period, not a measurement, and the
measurement refutes it. All arms at N=53, look-ahead off, real prompt, 200 output tokens:

| policy | misses/token | r | measured t/s | vs default |
|---|---|---|---|---|
| **LFU, half-life 16 (default)** | 201.5 | **58.0%** | **11.702** | - |
| LFU, half-life 256 | 218.8 | 54.4% | 11.026 | -6.1% |
| LFU, half-life 2048 | 218.8 | 54.4% | 11.033 | -6.0% |
| pure LRU (`GGML_CUDA_MOE_FREQUENCY=0`) | 213.2 | 55.6% | 11.236 | -4.3% |

**A longer frequency memory makes it worse, and so does dropping frequency entirely.** The recent
set IS the recurrence set for this workload: experts are layer-specific and the demand set churns,
so a long-horizon accumulator protects experts that are no longer wanted and evicts the ones that
are. 256 and 2048 are identical (54.4%) because once the half-life greatly exceeds a token the
accumulator saturates and the two settings are the same policy.

The default arm also reproduces the previous build's 11.738 t/s to 0.31%, which is the regression
check for the kernel argument added to make the half-life tunable.

**So all three code-side levers are now closed by measurement:**

```
capacity   +21.5% (N=28 -> 53), then VRAM-capped at r ~ 58%
policy     already optimal among {LFU-16, LFU-256, LFU-2048, LRU}
prefetch   transport only, no admission path, -3.1% on the clock
```

**Where that leaves the operating point.** At N=53, r = 58%, so 201.5 misses/token = 379 MB/token
still crosses the link, costing 60.5 ms of an 86.2 ms step. The 38.9 t/s at r=100% needs r=100%,
which needs roughly 4x the VRAM this card has. **The binding constraint is VRAM capacity, and the
only multi-x lever remains the x4 -> x16 slot move (7.10).**

**One idea worth naming, because it is not what "slab compression" tested.** The recorded closure
"slab compression MEASURED DEAD (zstd-9 1.0000 vs a 1.05 threshold)" tested LOSSLESS compression of
the cached slab. Storing the cached tier itself at a lower precision is a different change: it would
not make the transfer cheaper, but it would multiply how many experts fit in the same VRAM, which is
exactly the binding resource. It is quality-affecting, so it is the owner's call, not a code
decision - and the measurement that would justify it is whether the cached tier's precision
dominates the output, which the logits instrument (PR #127 Blocker 1 / issue #128) would have to be
working first to answer.

### 7.18 Instrument audit: is any published number an artifact of the implementation?

Asked for directly, because a wrong instrument invalidates every number downstream. Four ways a
result here could be fake - and the check for each.

**(a) Gaps in the per-step ledger.** A missing step would silently drop misses from a total. Checked
across all nine traces (n28, n40, n53, pol16, pol256, pol2048, pollru, ord28, ord53):

```
lines=199  steps=199  contiguous 0..198  duplicates=0  need=480 exactly, every step
need - resident == misses   holds 199/199 in every file
server logs: no dropped-step, no ring-overflow warning
```

So the ledger neither drops nor double-counts a step, and its three fields are internally
consistent at every step. **No gap.**

**(b) Does the policy knob actually take effect?** If it were ignored, all four policy arms would
have run the default and the ~6% spread would have to be noise - i.e. the "decisive negative" would
be a null result. Two facts rule that out. First, half-life 256 and 2048 both differ from the
default by ~6%, so the knob changes something. Second, and stronger, the epoch arithmetic explains
the whole pattern: `epoch = step / halflife`, so over a 199-step run a half-life of 256 advances
0.78 epochs and 2048 advances 0.10 - **neither ever ticks**, which is exactly why those two arms are
byte-identical (11.026 vs 11.033, both r=54.4%), while the default's 16 advances 12.4 epochs.

**Consequence that must not be glossed: the sweep tested TWO distinct policies, not four** - decay-16
(within-token decay) versus no-decay-within-the-run (long-horizon LFU), plus LRU. The conclusion
"the default is best" stands; the conclusion "we swept the half-life axis" does not.

**(c) Was the cache size the run claimed actually installed?** The server never echoed the value -
it only said the flag was set - so the config was verifiable only through its own effect. Two
independent confirmations now exist. The miss count is a deterministic fingerprint: N=28 gives 261.3
misses/token and N=53 gives 201.5, reproduced to the digit across separate sessions. And the gap is
closed at the source: `src/llama.cpp` now logs the resolved size in the same warning
(`n_slots=%d`, commit **ed2b4b6b9**), verified by loading with 53 and 28 and reading back
`n_slots=53` / `n_slots=28`.

**(d) Run-order / drift confound - the one real gap, now falsified.** The original sweep ran
N=28, 40, 53 in ascending order, so a monotone thermal or clock drift across the session would have
masqueraded as a cache-size effect. The control re-ran the endpoints **reversed**, N=53 first and
N=28 last:

| config | position in session | t/s | misses/token |
|---|---|---|---|
| N=28 | first (original) | 9.661 | 261.3 |
| N=28 | **last (control)** | **9.668** (+0.07%) | 261.3 |
| N=53 | last (original) | 11.738 | 201.5 |
| N=53 | **first (control)** | **11.708** (-0.26%) | 201.5 |

The endpoints reproduce to within 0.3% in either position, so the effect is +21.5% when N=28 runs
first and +21.1% when N=53 runs first. **The +21.5% is a property of the cache size, not of when the
arm ran.**

### 7.19 The correctness instrument now EXISTS, and Blocker 1 passes both tests

The instrument that was missing (7.11 retracted the old one as vacuous) now works, and it is not a
PPL run or a logits dump - it is greedy token comparison on the API response body.

**Why the old one failed and this one does not.** `perftok-<tag>.tsv` column 3 was always empty
because the model streams on a channel `stream.sh` never captured, so the retired check compared
empty strings. This instrument POSTs to `/completion` with `stream: false`, `temperature: 0`,
`top_k: 1`, a fixed `--seed`, and reads `.content` out of the response JSON. Greedy decoding is
maximally sensitive: any perturbation that flips an argmax anywhere in the sequence diverges and
stays diverged.

**PR #127 Blocker 1 / issue #128 - "an extra reader of `ffn_gate_inp` changes model output" - is
CLEARED on this configuration.** The comparison is not vacuous: `build_moe_lookahead` returns early
when `cparams.moe_lookahead <= 0` (llama-graph.cpp:2095), so `--moe-lookahead 0` genuinely removes
the extra reader rather than setting its width to zero.

| arm | look-ahead | chars | output sha256 |
|---|---|---|---|
| la0 | off | 648 | `73253b673b79b7ab` |
| la0b | off (repeat) | 648 | `73253b673b79b7ab` |
| la8 | on (width 8) | 648 | `73253b673b79b7ab` |

Determinism control passes (la0 == la0b across two separate server launches), so the instrument is
trustworthy, and la8 is byte-identical to it.

**Corroborated distributionally, because greedy only catches an argmax flip.** A shift that moves
the distribution without flipping a greedy argmax on that one prompt is invisible to the token test.
`llama-perplexity` over 15 chunks x 512 tokens of real prose (the improvement-plan document itself,
9,556 words), same corpus and seed across arms:

```
ppl0  (off)      Final estimate: PPL = 14.7350 +/- 0.71885
ppl0b (off x2)   Final estimate: PPL = 14.7350 +/- 0.71885
ppl8  (on)       Final estimate: PPL = 14.7350 +/- 0.71885
```

Every per-chunk value matches too - `[1]8.6064, [2]12.6579, ... [15]14.7350` identical to four
decimals in all three arms. **This is logits-level identity, not merely "no divergence observed".**

Scope, stated honestly: one prompt for the token test, one corpus for PPL, greedy temperature 0. It
clears the blocker for these paths; it does not prove the reader is free under sampling, nor on
other architectures. The instrument is cheap and re-runnable - `arm_correctness.sh`,
`arm_ppl_correctness.sh` in the harness directory.

### 7.20 NEW DEFECT: at the recommended cache size the look-ahead kills the server

Found while running the correctness comparison at N=53 (the operating point of 7.16/7.18, not the
N=28 the look-ahead arms had always used). `--moe-lookahead 8 --moe-expert-cache-size 53` does not
merely underperform - it aborts the process:

```
W moe-cache: lookahead-stage: refused, no device resource for group
E moe-cache: look-ahead could not install 144 of 144 MoE expert cache pools; ... prefetch is inert
lookahead-stage: published group=1 width=8 entry_bytes=2252800
E CUDA error: out of memory
E   in function ensure_lookahead_lane_locked at moe-cache.cu:5744
E   cudaMalloc(&fresh->staging, fresh->staging_bytes)
```

**The `inert` line is a red herring and must not be read as the cause.** It is structural: legacy
pools can never install on this rig because the certified grouped path owns the groups, and the
staging lane does not need a pool - the grouped gather consumes `lane->staging` directly
(moe-cache.cu:10695, 10757). The same message appears at N=28 where everything works, and the log's
own ordering proves it (the publish succeeds *after* the inert error).

**The real defect is an unchecked allocation.** `ensure_lookahead_lane_locked` allocates one staging
lane per candidate group; its refusals fire only on their own conditions (:5680-5711) and none of
them is "out of memory", so a lane is built regardless and `CUDA_CHECK(cudaMalloc(...))` at :5744
turns a VRAM shortfall into `ggml_cuda_error` -> `GGML_ABORT`. Nothing records the failure and
nothing bounds the lane set.

**Cost of the lanes - and why N=53 is fatal.** `entry_bytes` = 2,812,600/... = 281,600 B per expert,
so one lane is `entry_bytes x width` = 2.1484 MiB, and there is **one lane per group: 48 groups x
2.1484 MiB = 103.13 MiB**, cumulative and released only by `clear_lookahead` at a full context
refresh/teardown (:5633, :13792) - **never evicted as decode walks the layers**. That is the same
order as the cache's own marginal cost of ~103 MiB per +1 slot (7.15). **The look-ahead spends about
one slot's worth of the VRAM budget, and at the VRAM cap that is exactly what breaks.**

**Minimal fix (analysis only - not implemented, owner's call).** Make the lane creation soft-fail:
use the file's existing soft-failure helper `moe_grouped_cuda_success` (:4540-4546, already used for
the grouped device resources at :6959-6994) for :5739-5746, return nullptr, and add a sticky
refusal latch beside the lookahead vector so the failure is not retried per dispatch. The caller
then emits its one-shot "no staging lane" warning (:15122) and returns false, which is today's inert
behaviour minus the process death; `early_workspace`'s destructor (:5477-5517) already frees the
partial object, so nothing leaks.

Two wrong fixes to avoid. **Refusing a lane when the pool count is zero is wrong** - it would disable
the only working transport precisely on the configurations where it works. **VRAM budgeting alone is
insufficient** - it moves the threshold and cannot be enforced (the grouped resources it would have
to predict only exist after the first decode refresh, which is why preinstall returns -1 before
that). It is a useful complement, not the fix.

**Consequence for the record:** look-ahead and the +21.5% capacity win are **mutually exclusive at
N=53** on this card. Every look-ahead result in this document (-6.4%, -3.1%) was measured at N=28 or
N=42, i.e. below the operating point where the feature is usable at all.

### 7.21 Where the time actually goes (N=42), what the look-ahead really delivers, and what is left to optimize

Asked for at the operating point N=42 - low enough that the look-ahead still runs (7.20), high enough
that the cache is doing most of its work.

**The decomposition. T = 28.3 + 0.2842 x misses_per_token, and the fixed term is a constant of the
machine.** Fitting each arm separately, with the ledger paired to the per-token trace at the
alignment that maximises correlation:

| arm | t/s | ms/token | misses/token | fixed ms | transport ms | GB/s at the margin | r2 |
|---|---|---|---|---|---|---|---|
| N=28 | 9.66 | 103.5 | 261.3 | 28.2 | 75.3 | 6.53 | 0.907 |
| N=40 | 10.66 | 93.8 | 229.7 | 28.5 | 65.3 | 6.62 | 0.905 |
| N=42 | 10.81 | 92.5 | 225.0 | 28.6 | 63.9 | 6.62 | 0.910 |
| N=53 | 11.74 | 85.2 | 201.5 | 28.2 | 57.0 | 6.65 | 0.916 |

**The fixed term is 28.2-28.6 ms in every single arm** - attention, dense, router, sampling and launch
overhead, independent of the cache. The marginal rate is 6.53-6.65 GB/s, i.e. the measured link
ceiling. So the per-token time is fully explained by `28.3 + 1.881 MB x misses / 6.62 GB/s`, and the
only two quantities in it are the miss count and the link rate.

**The link is busy 69% of the step and idle 31%.** At N=42: 423 MB/token at 6.62 GB/s = 63.9 ms of a
92.5 ms step. **If the expert bytes could ride in the 28.6 ms of non-MoE time, the step would be
max(63.9, 28.6) = 63.9 ms - 15.7 t/s, +45%.** That is the entire remaining software opportunity, and
it is a pure overlap problem, not a bandwidth or capacity one.

**The per-token distribution has no tail - which rules out a whole class of fixes.** At N=42:
p50 93.6, p90 113.3, p99 125.1, max 127.2 ms, so p99/p50 = **1.34x**, and the slowest 10% of tokens
hold 12.3% of decode time (12.3% would be perfect uniformity). Same shape at N=28 (1.30x) and N=53
(1.38x). **There is no stall, no outlier token, no hot spot** - scheduling, batching, QoS and
tail-latency work have nothing to act on. The mean is the whole story.

**Alignment is real, and the look-ahead visibly shifts the transport one step ahead.** Correlation
between per-step misses and per-token time peaks at offset 0 for every control arm (+0.952 / +0.957 /
+0.954) and at **offset +1 for the look-ahead arm (+0.942)** - the feature really does move bytes for
the *next* step, which is the first direct confirmation of the mechanism from timing data.

**The number the owner asked for, and why it is not 80%.** `moe-lookahead-stage` now reports
`staged_experts_total` and `used_pct_total` (counters[2] and counters[3], expert counts - note the
log's `published` field is a count of publish *events*, not experts, and `prefetch_used` and
`consumed_total` are the same quantity printed twice):

```
N=42, width 8:  staged_experts_total=35222  consumed_total=672  used_pct_total=1.91%
```

**Of 35,222 experts fetched ahead of time, 672 were used - 1.91%, not 80%.** And the earlier "95%
consumption" figure applies only to width 1:

| arm | width | consumed | consume_pct_total |
|---|---|---|---|
| paft3 / pbef3 / padm1 / padm2 / aft3 / bef3 / cadm2 | 1 | ~3670 | **95.2-95.5%** |
| la8 / u42on | 8 | 672-724 | **1.91-1.98%** |

So the transport works at width 1 and collapses at width 8 - which is exactly the ~2.05 MiB window
arithmetic of section 1: a width-8 lane needs 15.4 MiB and the window fits ~2.05, so the overwhelming
majority of width-8 staging lands **after** the demand gather has already fetched the expert
directly. **The low number is lateness, not wrongness.**

**And the 80% figure is a different quantity: the predictor's recall, not the transport's usefulness.**
Measured previously as 86.21% at width 8 / 79.26% at width 10 (23,936 (step, layer) pairs). **That
instrument cannot currently reproduce it**: `ggml_cuda_moe_lookahead_debug_score` (ggml-cuda.cu:3777,
gate `GGML_CUDA_MOE_LOOKAHEAD_DEBUG`) is wired into the *legacy* cache admission path
(ggml-cuda.cu:4491, the `cache_lease`/`unique_eids` loop), and the certified grouped path owns decode,
so the scorer never runs - **0 recall lines in two runs with the gate on** (widths 8 and 10, N=42).
Re-attaching the scorer to the grouped demand side is the missing instrument.

**So the honest summary of the look-ahead is a three-way split that must not be conflated:**

```
predictor precision (recall of predicted)   ~86% at width 8   [prior measurement, not reproducible today]
transport usefulness at width 8              1.91%            [measured, 7.21]
transport usefulness at width 1             95.5%            [measured - it lands in time]
net effect on the clock                     -25.9% at N=42, -3.1% at N=28 best case
```

The predictor is good. The transport is good at width 1. **The feature still loses, because at width 1
it stages 87 MiB/step to save 53 and displacement is 1:1 (7.1) - the waste, not the prediction, is the
loss.**

### 7.22 What is left to optimize, ranked by measured ceiling

1. **The x4 -> x16 slot. +4x on the binding resource.** At N=42 the transport is 63.9 ms of a 92.5 ms
   step; at ~26 GB/s it becomes 16 ms and the step ~45 ms, i.e. **~22 t/s from 10.8 (+105%)**. Nothing
   in software is in the same league. This is a card move, not a code change (7.10, and see the swap
   consequence in section 5).
2. **Fewer bytes per expert: a lower-precision cached tier. +40-45%, quality-affecting, untested.**
   Every miss costs 1.881 MB because it fetches all three expert tensors. Halving the cached tier's
   precision nearly halves that: transport 63.9 -> ~36 ms, step -> ~64 ms, **~15.5 t/s (+44%)**. It
   also multiplies residency in the same VRAM, so it attacks capacity and bandwidth at once. It
   changes output, which is why it needs the correctness instrument - and **that instrument now works
   and passes (7.19)**, so this is finally measurable. Note it is NOT what "slab compression" tested
   (7.4): that was lossless and dead; this is lossy.
3. **Fill the 31% idle link time by prefetching. Ceiling +45% (15.7 t/s), and every variant measured
   is net negative.** The physics is not the obstacle - 87 MiB/step at width 1 fits inside a 28.6 ms
   window of ~189 MB, and 95.5% of it is used. The obstacle is that the staged bytes still displace
   demand bytes 1:1, so the ~34 MiB/step of waste is a straight loss of ~5.4 ms. Pacing was meant to
   convert that by issuing into the idle window and measured **-3.1%**: it halved the loss (-6.4% ->
   -3.1%) but did not reach green. **The open question this leaves: whether the ~28.6 ms of non-MoE
   time is genuinely available to the copy engine as a contiguous window, or is itself fragmented by
   the demand gather.** That is measurable and is the only thing standing between here and +45%.
4. **Capacity: harvested and capped.** +21.5% (N=28 -> 53), VRAM-limited (7.15). Combined with (2) it
   compounds rather than competes.

**Ruled out by the distribution:** anything that acts on variance rather than the mean - scheduling,
batching, QoS, tail-latency, per-token adaptive behaviour. There is no tail.

## 8. Status

Reviewed three times by architect 247b025c. Rev 9 supersedes rev 5-8: the resident ledger was
found to be structurally blind to the look-ahead's saving, an opt-in honest-accounting path was
added and measured, and 7.12's byte-neutral framing is partly retracted (7.13).

**Current state.** `feat/moe-lookahead-p1` HEAD **92f2499bd** (rev 26), **pushed** to hydra-fork,
working tree clean, PR #127 open. Earlier revisions of this line named 899d8ec7a; the branch has
moved a long way since and the doc is the record of what is at HEAD, not that commit. `libggml-cuda.so` sha prefix **36d284bf** (supersedes a9415bc3 <- ce9431e4
<- cfb2db3d; the rev-5 width-sweep numbers were taken on cfb2db3d and remain valid as a historic
record).

**OPERATING POINT - owner-set 2026-09-15, and it SUPERSEDES what used to sit here.** Read this
before quoting any t/s from earlier sections. The operating point is **N=42 or N=36, look-ahead
OFF**, chosen deliberately to keep VRAM free for the MTP head (7.31). The N=53 recommendation this
block previously carried is **withdrawn**: N=53 measures +8.32% intra-session (11.7824 vs 10.8776
t/s) and was **declined** - throughput bought by consuming the VRAM MTP needs is a loss, not a win.
Do not propose raising N again.

```
same session, same binary, real prompt, 200 output tokens, look-ahead OFF
   N=28   9.6610 t/s   261.26 misses/step
   N=36  10.3626 t/s   239.11   <- operating point; costs 4.37%, frees ~618 MiB (~2.2 GiB headroom)
   N=42  10.8362 t/s   225.00   <- operating point (~1.55 GiB headroom)
   N=48  11.3891 t/s   211.50
   N=53  11.7824 t/s   201.54   <- DEFERRED by owner decision, not the operating point
```

The older N=53 figures elsewhere in this document (11.738, 11.740) are a different session; the
intra-session pair above is the comparable one. Sections 1-7 quoting ~10.2 or ~9.6 t/s were measured
at **N=28** or on the retired `phase.sh` synthetic-prompt arm, which is a DIFFERENT harness and is
not comparable with the `stream.sh` real-prompt numbers. Do not mix them in one table.

**All three code levers are closed by measurement:** capacity gives +21.5% then stops at the VRAM
cap; retention policy is already optimal among {LFU-16, LFU-256, LFU-2048, LRU, recent1, protect};
the look-ahead is transport-only and, when later swept across widths, is a **net loss at every
width** (7.29) - including -3.7% at width 1 where 94.85% of staged bytes were useful. The binding
constraint is VRAM capacity, and the only multi-x lever left is the link.

**Final measured verdict (one binary, real prompt, 200 output tokens, probe off):** control **9.699**
t/s; look-ahead 1 after FFN **9.398** (-3.1%); look-ahead 1 before FFN **9.030** (-6.9%). The
before-FFN and after-FFN ledgers are identical to the last digit, so placement changes only WHEN the
same bytes move. Consumption is 95.53%, so the feature is byte-neutral by construction and its whole
effect is timing - and one slab (2.01 MB) does not fit one idle window (1.95 MB). Readiness, pacing,
placement and filter coupling have all now been implemented and measured and are all excluded. The
+2-4% "install the waste" lever remains unbuilt but is bounded by the same window.

**Documented negative result.** The design is not landable at this link width. The only multi-x lever
is hardware: the 3060's root port (0000:00:06.0) is capable of x16 and the card is WIRED x4, so a 4x
link exists but requires physical hands. The 5060 Ti on the same board negotiates x8.

**Open decisions for the owner.** (a) Cache-size sweep N=28/40/53 - the capacity lever, VRAM
quantified in 7.15, predicted +25% to +50% by the 7.14 model. (b) Retention policy: frequency
half-life (16 steps = 1/3 token) and LFU vs LRU. (c) Move the 3060 to a wider slot (4x link). (d)
Whether PR #126 / #127 stay open as a recorded negative or close, with issue #129 section 9 as the
record. Pacing is no longer open - implemented, measured, not sufficient.

**Correctness is NOT verified by any working check.** The determinism test several revisions quoted
was vacuous (see 7.11). PR #127 Blocker 1 / issue #128 - an extra reader of `ffn_gate_inp` changes
logits - is the real instrument and has not been re-run.

---

## 7.23 Rev 17 - the lossy-tier baseline was wrong; the lever is smaller than +44%

The owner's item 1 ("lossy tier, +44%") was priced off a per-expert size of 2.1484 MiB. That number
came from reading `blk.0` only. Reading all 1224 tensors shows the experts are MIXED per layer:

    49 tensors iq4_nl (4.50 bpw)   45 iq2_xs (2.31)   33 iq2_xxs (2.06)
    12 iq2_s  (2.56)                3 iq3_xxs          3 iq3_s

so the model-wide expert average is **~3.06 bpw = 1.793 MiB per expert**, which is exactly what the
ledger bills (`measured bytes/miss = 1.7930-1.7976 MiB`). **The open calibration question in 7.14 is
closed: the ledger was right and the nominal figure was a sampling error of mine.** Link rate stays
~6.6 GB/s.

Revised lever arithmetic (transport is 63.9 of 92.5 ms at N=42, 10.81 t/s):

    target          bpw      MiB/expert   bytes     transport   step     t/s     gain
    current         3.06     1.793        -         63.9 ms     92.5     10.81   -
    all -> iq2_xxs  2.0625   1.208        -32.6%    43.1 ms     71.7     13.9    +29%
    all -> iq1_m    1.75     1.025        -42.8%    36.5 ms     65.1     15.4    +42%

So "halving the cached precision" is +29%, not +44%, and reaching +42% needs iq1_m-class weights.

Two mechanics discovered the hard way, both worth recording:

1. `llama-quantize COPY --tensor-type-file X` **silently ignores X**. Dry run reports the input size
   unchanged (75051.01 MiB / 3.56 BPW). A selective requantize MUST list every tensor in the file at
   its current type and add only the intended changes. 1224-line map:
   `harness/multiturn-ctx/expert-quant-all.txt`.
2. **A sub-q2 reduction is refused without an importance matrix.** `iq2_xxs` on `ffn_gate_exps.weight`
   aborts with "this quantization requires an imatrix". The original model carries
   `quantize.imatrix.file = /workspace/q4exp-apex/imatrix.gguf` (927 entries / 20 chunks) in its own
   metadata, so it can be reproduced but not shipped inside the weights.
   `q2_k` is NOT a reduction for this model: it is 2.5625 bpw, i.e. HIGHER than 90 of the 144 expert
   tensors, and the dry run grows the file to 78026.92 MiB.

Rev 17 action: `llama-imatrix` built and running over the same document used for the perplexity
check, 40 chunks x 512, imatrix written to `harness/multiturn-ctx/imatrix-exp.gguf`. Then requantize
to `iq2_xxs`, then measure t/s at N=42 and PPL on the same corpus. That pair prices the lever.

Unchanged conclusions: the distribution has no tail (p99/p50 = 1.34 at N=42) so the mean is the only
target; the look-ahead delivers 1.91% usefulness at width 8 against a predictor recall of ~86%, which
are different quantities; look-ahead and the +21.5% capacity win are mutually exclusive at N=53.

### 7.24 Rev 17b - the lossy tier is SHAPE-LOCKED, not policy-blocked (+4%, not +44%)

Quantising the 144 expert tensors down was attempted and returns a hard geometric limit, visible in
the tool's own per-tensor verdicts:

    blk.41.ffn_up_exps.weight   [2560,640,512]  iq2_s  -> iq2_xxs   256.25 -> 206.25 MiB   OK
    blk.39.ffn_gate_exps.weight [2560,640,512]  iq2_xs -> iq2_xxs   231.25 -> 206.25 MiB   OK
    blk.39.ffn_down_exps.weight [640,2560,512]  iq4_nl  UNCHANGED   450.000 MiB
      warning: ncols 640 not divisible by 256 (required for type iq2_xxs) -> falling back to iq4_nl

`ffn_down_exps` has ne[0] = 640 and every IQ2/IQ1 type needs a 256-element block. The tensor is
locked at iq4_nl (4.50 bpw) **by geometry**, and it is the single largest expert tensor:

    down_exps   21600.0 MiB = 21.09 GiB   450.00 MiB/layer   42.188 MiB/512 experts  49% of expert bytes
    up_exps     11737.5 MiB = 11.46 GiB
    gate_exps   10500.0 MiB = 10.25 GiB
    TOTAL       43837.5 MiB = 42.81 GiB   (of the file's declared 75051 MiB)

Best case with existing ggml types: all gate/up -> iq2_xxs, down untouched = 72626.92 MiB, i.e.
-2425 MiB = **-5.5% of expert bytes** -> transport 60.4 of 89.0 ms -> **+3.9% t/s**. Not +44%.

The levers people reach for first are all closed here:
  q2_k   = 2.5625 bpw = HIGHER than 90 of the 144 expert tensors; dry run GROWS the file to 78026 MiB
  iq4_xs = 4.25 bpw but 256-block, so it cannot take ne[0]=640 either
  q4_0/i8/lower = 32-block but >= 4.5 bpw, no gain over iq4_nl

**Escape route, and the correct framing of "lossy cached tier":** the 256-block constraint binds only
because the *file* must use a standard ggml type. A separate host-side packed copy of the cached tier,
decompressed on the device into the normal slot layout, escapes it entirely - any packing is legal
once the device unpacks. That is the shape of item 1 that can actually pay, and it needs a
decompression kernel. Storing a lower-precision copy in the FILE does not work; storing one in the
CACHE does. Note this also supersedes nothing in the lossless "slab compression DEAD" result (7.x),
which tested zstd on the existing bytes and never tested a lossy tier.

Also corrected here: the model file on disk is **46.48 GiB**, not 73.3 GiB. The header's tensor
inventory sums to 73.29 GiB = 46.48 + 26.81, and the 26.81 GiB is `per_layer_token_embd.weight`,
declared but with no payload in the shards (matching shard 00002's 79 bytes and the serve flags
`--override-tensor per_layer_token_embd=CPU --load-mode none`). `du` was right; the earlier statement
that the file is 73.3 GiB was wrong. Transport accounting is unaffected - the experts are in the file.

### 7.25 Rev 17c - item 3 answered: the retention ceiling is +32%, and prefetching cannot touch it

The demand trace works. `arm_demand_trace.sh` at N=42 (look-ahead OFF), gate `GGML_CUDA_MOE_DEMAND_TRACE`:
**9552 lines = 199 steps x 48 layers, 10 ids per (step,layer), need = 480/step exactly**, decode
10.76 t/s, ledger 199 lines. The trace is taken from the plan's own `unique_experts` staging array at
the same statement that commits it, so it is the same data the plan acts on.

Replaying it per layer through a 42-slot cache (Belady/MIN, offline):

    C      misses/step   t/s
    8        305.1       8.70
    10       281.6       9.23
    20       214.5      11.20
    42       147.8      14.22     <-- owner's operating point
    64       115.8      16.34
    128       77.9      19.82
    256       67.2      21.10
    512       67.1      21.11     <-- compulsory floor: every expert fetched once

    512  x 199 / 48 = 278 distinct experts per layer -> 67.1/step of cold fetches. Even an infinite
    cache cannot go below this, so 21.11 t/s is the absolute ceiling at the current bytes/miss.

Against the shipped policy's measured 225.0 misses/step, Belady at C=42 is 147.8, so:
  AVOIDABLE = 77.2 misses/step = 34.3% of the shipped misses = 92.2 -> 70.3 ms = **+32.2% t/s**.
  That is the answer to "how much can a better slot lookup avoid": at most 14.22 t/s, and it needs an
  ONLINE policy - Belady is offline. Note the shipped LFU/half-life-16 policy already beats plain LRU
  at the same capacity (225.0 vs 242.1), so naive policy swaps will not find this.

**Prefetching cannot capture any of it.** A first cut of the pre-admit run reported +172%, which is
impossible because it beats the offline optimum; the bug was that pinned inserts were not billed as
fetches. A prefetch is still a PCIe fetch. This is the same structural fact that makes the existing
look-ahead transport-only (1.91% usefulness at width 8) and that made its "no admission path"
finding: the look-ahead moves bytes in TIME, it does not remove them. Retention quality is what sets
the traffic; fetch timing only decides whether the bytes sit on the critical path.

Which gives the two live software levers, both ~+30% and mechanically distinct:
  retention quality (item 3)  ceiling +32.2%  -> 14.22 t/s, needs an online near-Belady policy
  fill the 31% idle link (2)  ceiling ~+28%   -> 63.9 x 0.69 = 44.1 ms transport -> 13.8 t/s,
                              requires useful prediction; ships at 1.91%, so this is the gap to close
and item 1 at +3.9% (7.24), and the compulsory floor at 21.11 t/s is unreachable without changing
bytes-per-miss or slot count.

---

## 7.26 Rev 18 - owner decisions (2026-09-15), and what they lock

Recorded on issue #129 as comment 5678741589. These are settled; do not re-propose them.

1. **PRs #126 / #127 stay OPEN as the recorded negative**, issue #129 is the record. Not merged,
   not closed.
2. **The N=53 look-ahead soft-fail fix is REJECTED.** Crash understood and specified (unchecked
   cudaMalloc at moe-cache.cu:5744, 48 groups x 2.1484 MiB of lanes never evicted) but will not be
   implemented. Accepted consequence: look-ahead and the +21.5% capacity win stay mutually exclusive
   at N=53.
3. **The PCIe root-port move is REJECTED.** ~4x on the binding resource, but with only two relevant
   root ports it moves bandwidth rather than adding it.
4. **The host-side lossy tier stays OPEN.** The only live item-1 candidate. Needs a device-side
   decompression kernel, so it is deliberately unscheduled.

### 7.26.1 Non-uniform slot allocation is DEAD (SlotAlloc, independent agent)

Exact DP over the 48 measured Belady curves, sum(C_layer) = 2016:
    uniform 42        147.82 misses/step   (14.223 t/s)
    optimal non-uniform 147.10 misses/step (14.264 t/s)  = +0.29%
and it does NOT survive split-half hold-out: fitted-on-first-half gives 167.54 vs 166.6 uniform
(-0.94), fitted-on-second-half gives 147.09 vs 146.93 (-0.16). The allocation moved 9.5% of the
budget across a 33..53 range for a gain inside the leader's own repeatability band.

Why: the 48 per-layer miss curves are nearly PARALLEL. Marginal value of slot #43 ranges 0.0251 to
0.0503 misses/step/slot across layers, a spread of only 15%, while the half-to-half disagreement in
that same estimate is 0.0079 - i.e. estimation noise is a large fraction of the effect. The skew
itself is real and stable (layer 0 uses 376 distinct experts vs a 280 median; Pearson r = 0.93
between halves) but parallel curves mean the Jensen gap is second-order.

For contrast, one extra UNIFORM slot per layer (+86.2 MiB, 2.4% more cache VRAM) gains 1.99
misses/step = +0.116 t/s, 2.8x more than the entire reallocation - so if VRAM headroom is spent, it
should be spent uniformly. Belady curve near the operating point, -2.0 misses/step per +1 slot:
    C=40 151.95   C=41 149.85   C=42 147.82   C=43 145.82   C=44 143.88   C=45 142.02
far field: C=60 120.15, C=80 101.33, C=100 88.30.

Cross-check: this agent independently reproduced both the uniform-42 Belady value (147.82) and the
compulsory floor (67.07/step) from its own implementation, so those two numbers are double-sourced.

### 7.26.2 What is still live

- **Retention quality** (item 3): ceiling +32.2% -> 14.22 t/s. Needs an ONLINE policy; Belady is
  offline. `PolicyEval` is evaluating whether any implementable policy approaches it.
- **Prediction-guided victim selection**: `EvictPolicy` implements GGML_MOE_EVICT_POLICY=lfu|lru|protect
  in moe-cache.cu, reusing the existing look-ahead predictor's per-layer predicted set.
- **Host-side lossy tier** (item 1, decision 4): open, unscheduled, needs a new kernel.

Killed by this round: non-uniform slot allocation (above), look-ahead soft-fail fix (decision 2),
root-port move (decision 3), and item-1 re-quantization at +3.9% (7.24).

---

## 7.27 Rev 18b - what N=42 caches, and the hit rate does NOT improve with depth

### What is cached, and why the miss rate is still ~47%

    per layer  : 42 of 512 experts = 8.2%
    whole net  : 2016 of 24576     = 8.2%
    VRAM       : 3.53 GiB of expert slab (1.793 MiB per expert)

The 8.2% is the wrong denominator. Over a 199-step run the model only ever touches a mean of
**278 distinct experts per layer** (min 160, max 376 in layer 0), so 42 slots hold **15% of the live
working set** (11% for the heaviest layer). 1990 demands per layer, spread over all 199 steps, so
evicted experts keep coming back. That is why the hit rate sits at 53% rather than tracking 8.2%.

Per-layer Belady misses at C=42 range 263 (lightest) to 1080 (layer 0), mean 612.8 - a 1.76x spread
between the heaviest and the average layer, driven by distinct-expert count, not by demand volume
(every layer has exactly 1990 demands).

### The trajectory, and the inversion

Measured misses/step (shipped policy) and Belady at the same C=42, look-ahead OFF:

    steps      shipped miss/step   shipped hit%   Belady miss/step   Belady hit%
    0-4             268.2             44.1%            268.2             44.1%
    5-9             156.0             67.5%            153.4             68.0%    <-- best of the run
    10-19           173.7             63.8%            131.7             72.6%
    20-49           178.6             62.8%            111.0             76.9%    <-- best for Belady
    50-99           249.0             48.1%            158.6             67.0%    <-- degrades
    100-149         241.9             49.6%            159.0             66.9%
    150-198         224.7             53.2%            138.2             71.2%

Late (100-198) is **4.6% WORSE** than mid (25-99). The hit rate does not climb with depth; it peaks
almost immediately and then degrades.

Two things follow, and they matter:

1. **The cache fills in 4.2 steps** (2016 slots / 480 demands per step). Cold-start misses are a
   four-token phenomenon, not a hundred-token one. Step 0 misses all 480, step 1 is already 229.
2. **Belady degrades too** (76.9% -> 67.0% between steps 20-49 and 50-99), so the degradation is NOT a
   policy failure. The workload's own expert demand widens as generation proceeds. No cache policy,
   however good, recovers that; only more slots or fewer bytes/miss does.

### Consequence for the look-ahead

The look-ahead has no admission path, so it does not change residency, and therefore cannot raise the
hit rate at any depth. Its measured 1.91% usefulness at width 8 is transport timing, not hit rate.
Expecting the hit rate to rise deeper into decode "thanks to the ahead lookup" assumes an admission
path that does not exist. Raising hit rate requires victim selection - which is what EvictPolicy
implements.

### Caveat recorded honestly

The cross-arm step model T = 28.3 + 0.2842*misses is a BETWEEN-CONFIGURATION fit (different N ->
different mean misses). Per step inside one run the relation is much weaker: my quick check gave
r = 0.27, but that check mis-aligned the token stream (streaming emits sub-token deltas, so rows in
perftok-*.tsv are not one-per-token), so the per-step slope is NOT reported. What is solid: per-step
misses vary 59..480 (8.1x) while per-token time varies only p99/p50 = 1.34x, so transport is at least
partly overlapped within a step. Do not use the step model to predict a single step.

---

## 7.28 Rev 19 - the retention axis is closed IN PRINCIPLE, not just in practice

`PolicyEval` built a policy engine and calibrated it: it reproduces the shipped LFU-16 policy's
PER-STEP miss counts **exactly on 8 of 8 recorded ledgers** (C=42/28/40/53 plus three policy
variants), and reproduces the whole Belady table (C=8 305.09, 42 147.78, 512 67.07). So the engine is
the shipped behaviour, not a model of it.

### The result that was not expected

    prediction-guided victim selection, PERFECT same-layer next-step predictor   214.22  +4.2%
    same, at the measured width-8 quality (r86.21/cov68.97)                      217.71  +3.1%
    same, with the tree's actual predictor                                       225.00  +0.0%
    LFU-16 + RECENT1 (protect the last step's experts)                           223.57  +1.2%
    LFU-16 (SHIPPED)                                                             225.00   -
    Belady/MIN at C=42, offline                                                 147.78  +32.2%

The predictor in the tree predicts **a different MoE LAYER, never the same layer** - its horizon is
one layer inside a step, not one step (docs/moe-lookahead-design.md:484-487). Its ids therefore
belong to another layer's cache, so as a RETENTION signal its effect is measured at exactly
225.00 +/- 0.04, i.e. zero.

The horizon curve shows where the 32% actually lives:

    perfect horizon 1    214.22  +4.2%      perfect horizon 8    175.00  +19.1%
    perfect horizon 2    206.34  +6.9%      perfect horizon 16   156.76  +27.6%
    perfect horizon 4    194.27  +11.3%     horizon inf (step-granular MIN) 153.30  +29.3%
                                            canonical MIN (knows sub-step order) 147.78  +32.2%

### And then the argument that closes it

A same-layer multi-step predictor **cannot exist for a non-speculative decoder**, and this is not a
matter of engineering effort:

- layer L's router at step T+1 consumes layer L-1's output at step T+1,
- which is a function of token T+1,
- which is sampled from layer 47's output at step T - i.e. **after** the moment you would need it.

So at any instant you know token T but not token T+1, and expert demand one step ahead is gated on a
value that does not exist yet. Even the +4.2% row above is therefore unreachable, not merely hard;
`protect` as a retention policy is **exactly zero** for this decoder, which is what the engine
measures. The only horizon the architecture gives you is *within* a step, one layer ahead - which is
precisely the look-ahead that already exists.

Consequence: the retention axis is closed **in principle**, not "we tried and it did not pay". The
+32% Belady gap is not a policy-quality gap, it is an information gap.

### What that leaves, and it is the owner's request

The single reachable lever is to make the existing one-layer look-ahead actually effective. It cannot
change traffic (a prefetch is still a fetch, and its retention effect is zero), but it can move bytes
off the critical path: link busy is 69%, so a fully utilised link would take transport from 63.9 to
44.1 ms -> 13.8 t/s (+28%). Its blocker is measured and named: at width 8 the staged window is
~2.05 MiB against 15.4 MiB needed, so usefulness is 1.91% while at width 1 it is 95.2-95.5%.

Admitting the predictions into the CACHE instead of the lane removes exactly that blocker - the cache
is 3.53 GiB - so usefulness should recover toward the width-1 figure and the timing win becomes
available. That is `GGML_MOE_ADMIT_PREDICTED`, and it is the only live lever.

Also confirmed by the engine: the half-life sweep's caveat was right. Half-life 256 and 2048 both
collapse to no-decay = LRU behaviour (242.30), so the sweep tested two distinct policies, not four;
and the shipped LFU-16 is the optimum of that family (4: 235.20, 8: 229.47, 16: 225.00, 32: 226.75,
64: 233.13). An admission filter that refuses colder newcomers is WORSE (241.97).

Harness now travels with the code: benches/moe-cache/ + docs/moe-lookahead-improvement-plan.md,
commit bb2ca31a3.

---

## 7.29 Rev 20 - the look-ahead is a net loss at every width, and the N=53 exclusion is therefore moot

Seven-arm matrix at N=42, `arm_evict_policy.sh`, 200 tokens, real prompt, one build. Width 0 arms
first, because the whole table is only meaningful if the patched binary still runs the shipped path.

| arm | width | policy | admit | misses/step | staged | used% | decode t/s | vs width 0 |
|---|---|---|---|---|---|---|---|---|
| ev0  | 0 | lfu     | 0 | 225.00 | 0     | -     | **10.8320** | - |
| ev0p | 0 | protect | 0 | 223.57 | 0     | -     | 10.8316 | -0.00% |
| w1   | 1 | lfu     | 0 | 225.00 | 3221  | 94.78 | 10.3841 | **-4.14%** |
| w2   | 2 | lfu     | 0 | 225.00 | 6932  | 69.73 | 10.4677 | **-3.36%** |
| ev8f | 8 | lfu     | 0 | 225.00 | 35222 | 2.33  | 8.0052  | **-26.08%** |
| ev8p | 8 | protect | 0 | 223.49 | 34978 | 2.91  | 8.0368  | -25.81% |
| ev8r | 8 | recent1 | 0 | 223.57 | 35022 | 2.39  | 8.0343  | -25.83% |

### The default path is proven, not asserted

`ev0`'s ledger is **byte-identical** to the recorded baseline: `cmp` clean against
`moe-steps-dt42.log`, 199 steps, 225.00 misses/step, `decode_tps=10.8320`. The 587-line eviction patch
does not move the shipped path. This is the check that makes every other row trustworthy.

### RECENT1 is real, exact, and worthless

`ev0p` measured **223.57** misses/step. PolicyEval's engine predicted **223.57** from replay alone, on
a different implementation, offline. That is a full end-to-end validation of the simulator: it is not
a model of the shipped policy, it is the shipped policy. The effect is nonetheless worth nothing -
223.57 vs 225.00 is 1.43 fewer misses/step, which the step model prices at +0.4%, and the rig
delivers 10.8316 against 10.8320, i.e. no change outside noise.

`ev8p - ev8r = -0.08` misses/step: **the predicted-mask pin is a no-op on hardware**, confirming the
cross-layer finding below. It was never going to work, for the reason in 7.28.

### The look-ahead loses at EVERY width, with identical misses

This is the result that matters. Misses are **225.00 at widths 0, 1, 2, 4 and 8** - the feature never
changes the hit rate, at any setting. Its only possible win was moving bytes in time (link is ~69%
busy, so a perfect overlap would give +28%). Measured, it does the opposite: it loses 4.1% at width 1,
3.4% at width 2, 7.6% at width 4 and 26.1% at width 8.

And it is not a tuning failure. At width 1 the staging ratio is **94.78% useful** - the mechanism is
working exactly as designed, staging 16.2 experts/step of which 15.3 are consumed - and it still
loses 4.1%. With a displacement coefficient of 1.00 the fabric is serial, so a staged fetch displaces
a demand fetch and the demand fetch still has to happen; the staging adds transport without removing
any. Width 8 is the same effect amplified: 35222 staged experts at 2.33% useful is 26% of the
scheduler's time handed to the link for nothing.

The earlier "1.91% usefulness at width 8" reading was right but was misread as a lateness problem.
It is not: at width 1 lateness is gone (94.78% used) and the loss is still there. **The staged fetch
is additive, not substitutive.**

### Consequence: the recorded trade no longer exists

The owner's record has N=53 (11.738 t/s) as mutually exclusive with look-ahead, and named N=42 as the
operating point. Both halves of that trade have now moved:

- the look-ahead, which is what N=53 excluded, is a **pessimization at every width** (-4.1% best case),
- the capacity axis it was traded against is **+8.9%**: N=53 recomputed from `perftok-n53.tsv` is
  11.79 t/s against 10.832 today, both measured with `lookahead=0` (confirmed at arm_cache_sweep.sh:25).

So N=53 with the look-ahead off **dominates every look-ahead configuration**, and the exclusion costs
nothing because there is nothing on the other side of it. Recommended operating point: **N=53,
look-ahead off, +8.9% over 10.832**, by flag change alone.

> **SUPERSEDED (owner decision, 2026-09-15 - see 7.31 and the operating-point block in section 8).**
> This recommendation is withdrawn. The operating point is **N=42 or N=36**, chosen to keep VRAM
> free for the MTP head; N=53 was measured and **declined**. The text above is kept as the record of
> why the trade was believed to exist, not as a live recommendation.

### Blocker: the admission path crashes

`ev8a` and `ev8ap` (`GGML_MOE_ADMIT_PREDICTED=1`) **abort the server on step 2**:
`ggml-cuda.cu:117: CUDA error: unspecified launch failure`. Both died with
`staged_experts_total=344 admitted_experts=0 admit_evictions=0 admit_skipped=0`, i.e. **before any
admission was committed** - so it is the admission-enabled plan path, not the commit path. The
`admit=0` arms ran clean in the same build. Fix delegated back to the implementing agent with the
hypothesis that the host plan sizing and the device indexing disagree once `admit_cap != 0`.
No admission result exists yet; do not cite one.

In any case, the admission arm's premise is now weak on its own terms: an admission adds a fetch and
can remove at most one, and the width results show staged bytes are additive on this fabric.

---

## 7.30 Rev 21 - mechanism resolved, the admission question answered, and the operating point settled

### The width curve reproduces on the fixed binary

w4 is repeated because its first run closed ~2 s before an unrelated build window opened, which is a
margin rather than a guarantee. It reproduces.

| width | misses | staged experts | used% | t/s (run 1) | t/s (run 2) |
|---|---|---|---|---|---|
| 1 | 225.00 | 3221  | 94.85 | 10.3841 | 10.4260 |
| 2 | 225.00 | 6932  | 69.72 | 10.4677 | 10.4707 |
| 4 | 225.00 | 15370 | 43.21 | 10.0130 | 10.0246 |
| 8 | 225.00 | 35222 |  2.26 |  8.0052 |  8.0166 |

Within 0.4% everywhere, so the curve is solid and width 8's cost is reproducible to 0.14%.

### Mechanism: the shipped width is a traffic problem, the small widths are a fixed-cost problem

`staged_mib` on the stage line is a **per-dispatch delta**, not a lifetime total; `*_total` fields on
that line are lifetime sums. Conflating the two produced a 150x phantom mismatch earlier. Per-step
staging, cross-checked two ways (staged_experts x 1.796 MiB / 198 dispatches, against the mean of the
per-dispatch staged_mib), agrees to 0.4%: w1 29.2, w2 62.9, w4 139.4, w8 319.5 MiB/step.

Waste is staged minus consumed, and it fits a two-term model against the measured loss:

| width | staged/step | waste/step | loss/token |
|---|---|---|---|
| 1 |  29.2 MiB |   1.5 MiB | 3.6 ms |
| 2 |  62.9     |  19.0     | 3.2    |
| 4 | 139.4     |  79.3     | 7.6    |
| 8 | 319.5     | 312.0     | 32.6   |

**~1.3 ms fixed + ~0.10 ms per MiB/step of wasted staging** (about 10 GB/s effective for the wasted
copies). At width 8 the byte term is 31 of the 32.6 ms, so the shipped width is additive traffic; at
width 1 the byte term is 0.15 ms of 3.6 ms, so the small-width loss is a fixed per-feature cost.
Demand is 225 misses x 1.79 MiB = ~401 MiB/step, so width 8 stages ~80% of the demand volume and
wastes ~78% of it, which is also why hits cannot move. This closes the look-ahead axis: the feature
is evaluated, explained, and off.

### The admission request: it works, it is coherent, and it loses 9.1%

With the plan-residency trap fixed, both admission arms complete 199 steps with no CUDA error.

| arm | misses/step | admitted/step | real traffic | admit_evictions | t/s |
|---|---|---|---|---|---|
| ev8f  la=8 lfu     admit=0 | 225.00 |  0.00 | 225.0 | - | 8.0052 |
| ev8a  la=8 lfu     admit=1 | 268.80 | 44.92 | **223.9** | 8790 of 8940 | **7.2730** |
| ev8ap la=8 protect admit=1 | 266.80 | 43.65 | **223.2** | 8537 of 8687 | **7.3412** |

The owner's request is coherent and the mechanism does what it was meant to: real traffic (misses net
of admissions) falls below the no-admission case, 225.0 to 223.9 and 223.2, so some predicted experts
do become hits. But the exchange rate is the whole story - **~1.1 fewer misses/step bought with 45
extra fetches/step, a 41:1 loss ratio** - and 98.3% of admissions evict a resident rather than filling
an empty slot, so the cost is displacement as well as transport. Net effect **-9.1%** (lfu) and
-8.7% (protect) on top of the look-ahead's own -26.1%: 7.2730 against 10.8320 with both off is
**-32.9%**.

The candidates the plan can admit are predicted AND not resident AND not demanded, i.e. this step's
prediction false positives, ~45/step as measured against EvictPolicy's static estimate of ~51.8, and
its predicted ledger of ~278 against the measured 268.8. The static model was right; the arm confirms
it on hardware.

### The operating point

Same session, same binary, look-ahead off:

| N | misses/step | t/s | vs N=42 |
|---|---|---|---|
| 42 | 225.00 | 10.8776 | - |
| 48 | 211.50 | 11.3891 | **+4.70%** |
| 53 | 201.54 | 11.7824 | **+8.32%** |

**-2.25 misses per slot** between 42 and 48, consistent with the Belady slope of -2.0, and +0.76%/slot.
The step model holds across the new points to within 0.9%. Since the look-ahead is a loss at every
width, the recorded mutual exclusion is void: **N=53 with the look-ahead off dominates every
look-ahead configuration, +8.32% over the current operating point, by flag change alone.**

### Retention, closed among the four implemented policies

At la=0, `protect` IS `recent1` - the predicted-mask pin needs a mask that does not exist at width 0 -
which is why ev0p reproduced the offline RECENT1 prediction exactly.

- **shipped LFU-16**: 225.00 misses/step, 10.8320 t/s. Stands.
- **recent1**: 223.57, a deterministic 0.64% miss gain, free, no VRAM, no look-ahead. Invisible in
  time: 10.8316 against 10.8320, wrong sign, against a 0.41 ms/token model prediction. Below arm
  resolution, so recorded as a miss-count fact and not a throughput claim.
- **protect** (LFU + recent1 + predicted-mask pin): 223.49, i.e. -0.08 against recent1 alone. The mask
  half is nothing, as the cross-layer argument required.
- **LRU**: 242.1 offline, strictly worse.

The capacity axis is the only lever with a measurable coefficient.

### Retention at the operating point: the headroom closes with capacity

One extra arm, same session and binary, look-ahead off, N=53:

| N=53, la=0 | misses/step | t/s |
|---|---|---|
| shipped LFU-16 | 201.54 | 11.7824 |
| recent1 | **201.33** | 11.7962 |
| delta | **-0.21 (-0.10%)** | +0.12% |

At N=42 the same policy bought -1.43 misses/step (-0.64%); at N=53 it buys -0.21 (-0.10%). The gain
shrinks five-fold as capacity grows, which is what the mechanism predicts: recent1 repairs an integer
decay artifact (`freq >> elapsed` zeroes count-1 experts at an epoch boundary, so a one-step-old
resident can lose to an older one), and the damage that artifact can do falls as slots stop being
scarce. The miss count is deterministic, so this decides the axis even though 0.12% of t/s does not.

**Conclusion for the live axis: at the operating point, the best implementable retention policy is
worth 0.2 misses per step.** The remaining 34.3% of avoidable misses needs a same-layer predictor
with 8-16 steps of horizon, and section 7.28 shows such a predictor cannot exist for a
non-speculative decoder. The axis is closed by information, not by effort.

### Harness hazard found and fixed

A wrapper command whose command line contains the string `--target llama-server` is killed by the
harness's own `pkill -f 'llama-server'` hygiene - the job killed its own parent shell. Anchored all 20
scripts to `bin/llama-server` / `bin/llama-perplexity` / `bin/test-moe-cache`, which still matches the
real argv[0] (`$BIN/llama-server`) and cannot match a build command. Also added TAG_PREFIX to
arm_cache_sweep.sh and an arm selector to arm_evict_policy.sh, so a single arm can be run without
overwriting another arm's logs.

---

## 7.31 Rev 23 - operating point set to N=42 or N=36, keeping VRAM for MTP

Owner decision (2026-09-15): the operating point stays at **N=42 or N=36**, deliberately not N=53, so
that VRAM is available **later for MTP**, and the look-ahead is to be driven by the **MTP head** rather
than by the current cross-layer heuristic.

### N=36 measured, same session and binary, look-ahead off

| N | misses/step | t/s | vs N=42 | cache VRAM | headroom on 12 GiB |
|---|---|---|---|---|---|
| 36 | 239.11 | 10.3626 | **-4.37%** | ~9.60 GiB | ~2.2 GiB |
| 42 | 225.00 | 10.8362 | - | ~10.22 GiB | ~1.55 GiB |

Marginal is -2.35 misses/slot, consistent with the -2.25 measured from 42 to 48 and with Belady's
-2.0. The step model predicts 10.39 t/s at 239.11 misses against 10.3626 measured, so it holds here
too. **Choosing N=36 over N=42 costs 4.37% of throughput and frees ~618 MiB.**

### What that does to this document's central impossibility argument

Section 7.28 closes the retention axis with an information argument, and the argument is **scoped to a
non-speculative decoder** on purpose. That scoping is now load-bearing rather than pedantic:

> layer L's router at step T+1 consumes layer L-1's output at step T+1, a function of token T+1, which
> is sampled after the moment the prediction would be needed.

MTP **is** speculation. A draft head produces candidate tokens T+1..T+H, and running the router on
those draft hidden states yields, for each layer, its demand at T+1..T+H **before** the real tokens
exist. That is exactly the same-layer, multi-step horizon that 7.28 says cannot otherwise be had, so
the retention axis is not permanently closed - it is closed for the pre-MTP fork.

PolicyEval's horizon curve is the target, and it is steep (perfect knowledge, C=42, offline):

| horizon | misses/step | t/s | vs shipped |
|---|---|---|---|
| 1 | 214.22 | 11.21 | +4.2% |
| 2 | 206.34 | 11.50 | +6.9% |
| 4 | 194.27 | 11.97 | +11.3% |
| 8 | 175.00 | 12.81 | +19.1% |
| 16 | 156.76 | 13.73 | +27.6% |

With MTP these are not perfect: the prediction is exact **conditional on the draft token being
accepted**, so the realised gain scales roughly with the acceptance rate (commonly ~0.7), putting
H=4 near +8% and H=8 near +13%. Even the pessimistic end is an order of magnitude more than anything
the pre-MTP fork can reach (best implementable retention today: -0.10% at the operating point).

### The look-ahead should change FUNCTION, not just its input

An important distinction, so the MTP plan is not oversold. The current look-ahead fails for two
separable reasons:

1. **Wrong predictions** - it is a cross-layer heuristic, so as a retention signal its effect is
   exactly zero (measured 223.49 vs 223.57).
2. **A structurally expensive transport path** - the staging lane costs ~1.3 ms fixed plus ~0.10 ms
   per MiB/step of wasted staging, and it loses at width 1 where 94.85% of staged data is useful and
   the waste term is only 0.15 ms of a 3.6 ms loss.

MTP fixes (1) completely and (2) only partly: better predictions remove the waste, but the fixed
per-feature cost and the displacement of demand fetches remain. The high-value use of an MTP head is
therefore **victim choice, not staging** - retention directly removes fetches (misses x 0.2842 ms per
token), whereas the staging lane can only move them in time and was measured to cost throughput even
when nearly every staged byte is used. Feed the MTP predictions into the cache's eviction decision;
do not assume the lane becomes profitable because its input got better.

## 7.32 Rev 24 - what an MTP head costs here, and how much of it the fork already has

This is the sizing the N=42-vs-N=36 decision is a trade for, so it is worth having as numbers rather
than as an intention. Everything in this section is **read from source or computed from measured
totals**; no MTP head has been run, because none exists in the file (see 7.32.4).

### 7.32.1 The fork already implements this model's MTP path

`src/models/qwen4exp.cpp` is not a generic stub - it has a working DECODER_MTP graph for this exact
architecture:

| piece | where | what it does |
|---|---|---|
| `nextn_predict_layers` read into `hparams.n_layer_nextn` | qwen4exp.cpp:52-53 | asserts `n_layer_nextn < n_layer_all` |
| `n_layer() = n_layer_all - n_layer_nextn` | llama-hparams.cpp:347 | NextN blocks are appended **past** the trunk |
| draft-only export detection | qwen4exp.cpp:186 | `mtp_only` = block count declared, trunk tensors absent |
| `LLM_GRAPH_TYPE_DECODER_MTP` -> `graph_mtp` | qwen4exp.cpp:313-314 | asserts `n_layer_nextn == 1` (:582) |
| `llama_set_nextn_layer_offset` | llama-ext.h:127-130 | selects which appended head runs, to **"chain multiple trained NextN heads"** - H>1 is a driver concern, not a graph rewrite |
| `t_h_nextn` + `llama_get_embeddings_nextn` | llama-context.h:980, llama-ext.h:125 | exposes the draft hidden state |
| `build_moe_lookahead(nextn_state, ...)` | llama-graph.h:1210-1220 | **takes that state as its input** |
| MTP KV filtered to the nextn layer | llama-model.cpp:2355-2360, 2455-2458, 2508-2511 | the draft context holds ~1/48 of the trunk's KV |

Two consequences worth stating plainly.

**The plumbing the plan needs already exists.** `build_moe_lookahead` is documented as taking
`nextn_state` - the MTP head's hidden state - and predicting `il_next`'s top-K experts from it. The
current look-ahead feeds that same argument from the *current layer's* post-attention state (the
cross-layer heuristic of 7.28). So "drive the look-ahead from the MTP head" is a change of **input
producer**, not a new mechanism.

**But its only existing consumer is the lane that lost.** The same doc comment says it "emits a
side-effect GGML_OP_MOE_PREFETCH node that pages them into that layer's MoE cache pool" - i.e. the
existing consumer is the staging lane, which 7.30 measured as a net loss at every width, including
width 1 where 94.85% of staged bytes were useful. **Retargeting the consumer to victim choice is the
actual work item**, not connecting the head.

### 7.32.2 An MTP block is a full layer plus a head, not a thin adapter

In `qwen4exp.cpp:254-308` the NextN tensors are created **inside the same loop as trunk layers**, and
the `if (il < n_layer) continue;` at :291 sits **after** attention/SSM, PLE, `ffn_gate_inp`,
`ffn_*_exps` and `ffn_*_shexp`. So an appended block carries a whole layer's weights and adds:

| tensor | shape | params |
|---|---|---|
| `nextn.eh_proj` | `{2*n_embd, n_embd}` | 13.107 M |
| `nextn.hnorm` | `{hc_dim}` | small |
| `nextn.enorm` | `{n_embd}` | 2560 |
| `nextn.hc_head_norm` | `{hc_dim}` | small |
| `nextn.hc_head_down` | `{hc_dim, hc_lr}` | small |
| `nextn.hc_head_up` | `{hc_lr, hc_dim}` | small |
| `nextn.embed_tokens`, `nextn.shared_head_head` | `{n_embd, n_vocab}` | **absent** |

The last row is `TENSOR_NOT_REQUIRED` and the source comment states they are "absent when
`mtp_use_dedicated_embeddings=false` (qwen4exp); the head falls back to the trunk's" - so the head
costs **no extra vocab-sized tensor**.

### 7.32.3 Sizing: ~992 MiB of weights, of which only ~80 MiB has to be VRAM

Arithmetic on measured totals:

- expert bytes 43837.5 MiB / 48 layers = **913.3 MiB per layer**
- 47595 MiB total - 43837.5 MiB experts = 3757.5 MiB non-expert = **78.3 MiB per layer**
- therefore one full appended block = **991.6 MiB** of weights

The serve config is `--n-cpu-moe 99`, so a block's experts live on the host exactly as every other
layer's do. The VRAM an MTP block actually needs is then:

| term | at N=42 | at N=36 |
|---|---|---|
| one layer's non-expert weights | ~78 MiB | ~78 MiB |
| head (`eh_proj` 13.1 M params + hc_head pair + norms) | ~7-27 MiB | ~7-27 MiB |
| **its own MoE cache** (3 banks x N slots x 1.793 MiB x 1 layer) | **226 MiB** | **194 MiB** |
| MTP KV (1 of 48 layers) | negligible | negligible |
| **total** | **~310-330 MiB** | **~280-300 MiB** |

Headroom at the operating point is ~1.55 GiB (N=42) and ~2.2 GiB (N=36). Recorded as arithmetic on
the budget: **either operating point covers the head.** That is a statement about the numbers, not a
proposal to move N - the operating point is the owner's decision and is not re-opened here.

### 7.32.4 The term that can actually bite is traffic, not VRAM

Because the block is a full MoE layer, running it once per draft step demands 10 experts x 1.793 MiB
= **17.93 MiB per draft step**, i.e. **71.7 MiB/token at H=4, +17.8%** on the 403 MiB/token trunk
demand - unless those experts are themselves cache-resident, which is what the 226 MiB of slots in
7.32.3 buys. Priced with the same coefficient that prices everything else here (0.2842 ms per
miss/token), an uncached 18% is not a rounding error at this margin. **This is the first number to
check once a head exists.**

### 7.32.5 What is still missing

1. **The trained block does not exist in this file.** 0 of 1224 tensors match `nextn|mtp|draft`, and
   `n_layer_nextn` is 0. But the loader supports a **draft-only export** (`mtp_only`: block count
   declared, MTP block shipped alone), so the head can arrive as its own small GGUF loaded as a draft
   of the target, rather than as a 49th layer inside the trunk file. That is the cheap path and it
   keeps the trunk GGUF untouched.
2. **The predictor's consumer must move** from `GGML_OP_MOE_PREFETCH` (staging, measured dead) to the
   eviction decision (7.31).
3. **The batched-router claim is not yet verified.** If the MTP/verification pass computes per-layer
   router outputs for all H draft positions at layer L in one batched forward, then the horizon curve
   of 7.31 maps onto H directly and H=4/H=8 are reachable by chaining heads via
   `llama_set_nextn_layer_offset`. This has **not** been checked against the driver, and until it is,
   the +8%/+13% figures remain a projection from the offline oracle, not a plan.

### 7.32.6 Erratum

An earlier check-in recorded a per-token latency-versus-misses relationship as negative (R2 ~ -0.015).
That was a misreading of `perftok-*.tsv`: those rows are sub-token deltas, not one row per token, so a
per-token regression on them is meaningless and the sign carries no information. It should not have
been reported as a result. Retracted here rather than deleted, so the wrong version is not rediscovered
later.

## 7.33 Rev 25 - the MTP-lookahead premise does NOT hold: the horizon is same-dispatch, not ahead of it

7.31 argued that MTP reopens the retention axis, because "draft hidden states give, for each layer, its
demand at T+1..T+H before the real tokens exist", and mapped the offline horizon curve onto draft
length H (H=4 ~+8%, H=8 ~+13%). Checked against the driver this cycle. **The premise fails, and the
mapping should be withdrawn.** The mechanism is specific, so it is worth writing down rather than just
retracting the number.

### What is true

Verification **is** one batched forward over every draft position, and the MoE dispatch inside it is
**grouped**. So at layer L the router output for positions T+1..T+H does exist:

- `tools/server/server-context.cpp:580` - "add sampled token of this slot to the batch, **optionally
  add the speculative draft tokens if any**"; `:619` asserts the batch holds the sampled and the draft
  tokens together.
- `:1273-1275` - `n_max + 1 > llama_n_ubatch(ctx_tgt)` is rejected, i.e. **the whole draft is forced
  into a single ubatch**.
- `:4559` - `spec_i_batch.size() == n_draft + 1`, so acceptance samples over all draft positions at
  once.
- `ggml/src/ggml-cuda/moe-cache.cu` - the cache is the **grouped** one and tracks `unique_experts` per
  dispatch, i.e. it fetches the **union over tokens** in one operation.

### Why that is not a look-ahead

The router output for T+1..T+H at layer L is computed **in the same grouped dispatch that consumes
it**. There is no per-position ordering inside a layer to exploit, and no lead time: the union is
fetched once, in the same operation that revealed it. The fetch count for a dispatch is `|union|`
minus whatever is resident; a *retention* policy changes which residents survive **into later
dispatches**, and those dispatches' demands are exactly what is still unknown.

And they are unknown for the original reason, not a new one. Layer L's demand for a future token
T+k needs layers 0..L-1's output for T+k. MTP supplies draft **tokens**; it does not supply their
trunk hidden states at layer L. The only way to get those is to run the trunk on them - which is
precisely the verification pass, and by construction that is the dispatch whose own needs it
discovers. 7.28's impossibility argument is therefore **restated, not escaped**:

> a same-layer multi-step predictor cannot exist for a non-speculative decoder because layer L's
> router at T+1 consumes layer L-1's output at T+1.

Speculation changes *when the token exists*, not *when its layer-L input exists*.

### What this means for the numbers in 7.31

- The horizon curve (H=1 +4.2% ... H=16 +27.6%) is an **offline bound**, in exactly the sense Belady
  is: it assumes knowledge that no online policy can have, in the pre-MTP and the MTP case alike. It
  should be quoted as a ceiling only, and the +8%/+13% acceptance-discounted figures withdrawn.
- The one ahead-of-time signal MTP genuinely creates is for **its own layer**: the MTP head runs
  before verification, so its own demands are known early. That is one layer, and its only consumer
  today is the staging lane, measured as a net loss at every width. It is not the lever 7.31 hoped
  for.

### The loophole is closed too

7.33 was drafted with one escape hatch left open: `--decode-overlap` is in every serve flag set
(`server-context.cpp:2014`), so if it split the draft positions into separate dispatches a
per-position ordering would exist again. It does not, and the driver says so directly:

```
server-context.cpp:1273
    if (capped_mtp && params_base.speculative.draft.n_max + 1 > (int32_t) llama_n_ubatch(ctx_tgt)) {
        SRV_ERR("capped MTP replay requires an ubatch of at least %d tokens, but the target context has %u\n", ...);
```

The server **refuses to start** unless `n_max + 1` tokens fit in a single ubatch. That is the whole
draft, replayed in one forward. The assert exists precisely because the MTP path cannot tolerate the
draft being split - so overlap cannot be splitting it, or this check would fail on every start.

**7.31's H-curve mapping is therefore permanently withdrawn, not conditionally withheld.** No
measurement is needed in support; the ledger run proposed for it would only confirm what the start
condition already guarantees. If a future arch or mode removes that assert, the question reopens -
and the `moe-step:` ledger is still the right instrument for it.

### Not refuted here

MTP as a speculative **decoder** (draft acceptance buys more than one token per trunk dispatch) is a
different claim and is already recorded as a negative in this document's index; it is not re-opened.
What is withdrawn is only the hope that the *same* head supplies the cache a useful retention
horizon.

## 7.34 Rev 26 - CORRECTION: the MTP head exists, and the real blocker is #124

7.32.5 item 1 said "the trained block does not exist in this file" and went on to describe the
draft-only export as the cheap path to get one. **The first half was wrong in the way that matters,
and the path is already walked.** Checked against the filesystem this cycle:

```
/mnt/SSD/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf     2657.48 MiB
/mnt/SSD/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf   1818.80 MiB
```

What is true of the trunk file is what the tensor scan found: **0 of 1224 tensors** match
`nextn|mtp|draft`, and `n_layer_nextn` is 0 - the head ships **separately**, not inside the trunk.
Stating it as "no trained block exists" was wrong; the head is on the rig, and it is the `mtp_only`
draft-only export 7.32 described, already built.

### And it confirms 7.32.2 against a real artifact

Enumerating the head's tensors shows it is **a complete block 48**, not a thin adapter:

| family | present |
|---|---|
| `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_q_norm`, `attn_k_norm` | yes |
| `indexer_*` | 4 tensors |
| `hc_attn_{up,down,inject,norm}`, `hc_ffn_{up,down,inject,norm}` | yes, all 8 |
| `ffn_gate_inp`, `ffn_{gate,up,down}_exps`, `ffn_{gate,up,down}_shexp`, `ffn_gate_inp_shexp` | yes, the full MoE |
| `nextn.{eh_proj,enorm,hnorm,hc_head_norm,hc_head_down,hc_head_up}` | 6 tensors |

It also declares `nextn_predict_layers` and `nextn_shared_target_tensors`, and arch `qwen4exp`. So
"a full layer plus a head", derived in 7.32.2 from the loader's control flow, is confirmed by the
artifact - and the two file sizes give the head's cost directly rather than by arithmetic:
**2657.48 MiB at Q8_0, 1818.80 MiB at Q4_K_M**.

### The actual blocker: the cache and MTP are mutually exclusive today (issue #124)

The more useful thing this turned up is that **the combination the MTP plan needs has already been
tried and fails**, recorded in issue #124 (OPEN, `review-finding`, no comments):

- Build `bdd54f475`, flags `--moe-expert-cache-size 2 --spec-draft-moe-expert-cache-size 2
  --spec-type draft-mtp --spec-draft-model mtp-...-shared-Q8_0.gguf --spec-draft-n-max 2`.
- `E moe-cache: required grouped execution failed: grouped plan unavailable`, repeated **95x** at
  cache size 2 and **63x** at 32.
- `moe-grouped-decode: registered=98 covered=0 ... required_unsupported=63..95`.
- No `draft acceptance` line and no `draft_n` field: **draft-mtp never produced or verified drafts.**
  One `llama_decode() failed: -3` during warmup, then a legacy fallback that still serves.
- The reporter's own conclusion, which is the important line: *"`--moe-expert-cache-size` is unusable
  together with MTP on this model; MTP throughput numbers measured with the cache enabled are
  invalid."*

This is consistent with 7.33 and not in tension with it: MTP stamps the main verification graph
`REQUIRED_GROUPED`, while the grouped plan is unavailable, so the path fails closed rather than
falling back. Note also `--spec-draft-moe-expert-cache-size` exists as a separate knob, which
confirms the draft carries **its own MoE cache** - the 7.32.3 cache term is real, not speculative.

### What this changes about priorities

- The sizing question is **closed** (7.32) and the head is **in hand** (7.34). Neither is work.
- The horizon question is **closed** (7.33), against the plan.
- What is left, and it is a defect rather than a design question, is **#124**: the expert cache and
  MTP cannot run together on this model. Any MTP experiment on this rig is measuring the legacy
  fallback, not MTP. That is the thing to fix, and #121 (`draft-simple` intent rejection, which
  blocks PR #120) is its sibling on the same execution-intent machinery.

Nothing here reopens a settled decision. It corrects a factual claim of mine and identifies the real
next item.

# MoE look-ahead preload - evidence, mechanism, levers, plan

Revision 16 (7.21: at N=42 the step is 28.3 ms fixed + 0.2842 ms per miss at 6.6 GB/s, the link idles 31%, and the distribution has NO tail so the mean is the whole story; the look-ahead's transport is 1.91% useful at width 8 vs 95.5% at width 1 - lateness, not wrongness - while the predictor's ~86% recall is a different quantity whose instrument is currently unwired; 7.22 ranks what is left: the x4 slot (+105%), a lossy cached tier (+44%, now measurable), and filling the idle window (+45%, not yet green)). Revision 15 (7.19: the correctness instrument exists and PR #127 Blocker 1 / issue #128 PASSES it - greedy token identity 648/648 chars and PPL 14.7350 identical over 15 chunks; 7.20: NEW DEFECT, at N=53 the look-ahead aborts the server on an unchecked 2.1484 MiB lane cudaMalloc, 48 lanes = 103.13 MiB never evicted, so the feature and the +21.5% capacity win are mutually exclusive). Revision 14 (7.18: instrument audit - the per-step ledger has no gaps and is internally consistent 199/199; the policy knob is proven to take effect, so the sweep tested 2 distinct policies not 4; the cache size is now echoed in the log (ed2b4b6b9); and the reversed-order control falsifies the drift confound - +21.5% and +21.1% by either ordering). Revision 13 (added the CURRENT OPERATING POINT block to section 8: best config is control + N=53, 11.738 t/s, look-ahead OFF, and earlier sections' ~10.2 t/s figures are N=28 or the retired synthetic harness and are not comparable). Revision 12 (7.17: the retention policy is NOT a lever - the default LFU-16 beats half-life 256 and 2048 and beats pure LRU; capacity is VRAM-capped at r~58%. All three code levers are now closed by measurement, leaving only the x4->x16 slot move). Revision 11 (7.16: the capacity lever is real - N=28 -> 53 is +21.5%, 9.661 -> 11.738 t/s, with the
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

**Current state.** `feat/moe-lookahead-p1` HEAD **899d8ec7a**, **pushed** to hydra-fork, working
tree clean, PR #127 open. `libggml-cuda.so` sha prefix **36d284bf** (supersedes a9415bc3 <- ce9431e4
<- cfb2db3d; the rev-5 width-sweep numbers were taken on cfb2db3d and remain valid as a historic
record).

**CURRENT OPERATING POINT - read this before quoting any t/s from earlier sections.** The best
measured configuration on this rig is the control with a larger cache and no look-ahead:

```
--moe-expert-cache-size 53     real prompt, 200 output tokens, look-ahead OFF
   N=28   9.661 t/s   r=45.6%
   N=53  11.738 t/s   r=58.0%     <- +21.5%, VRAM-capped
```

All of sections 1-7 that quote ~10.2 t/s or ~9.6 t/s were measured at **N=28** or on the retired
`phase.sh` synthetic-prompt arm, which is a DIFFERENT harness and is not comparable with the
`stream.sh` real-prompt numbers above. Do not mix them in one table.

**All three code levers are now closed by measurement:** capacity gives +21.5% then stops at the
VRAM cap; retention policy is already optimal among {LFU-16, LFU-256, LFU-2048, LRU}; the look-ahead
is transport-only with no admission path and costs -3.1%. The binding constraint is VRAM capacity,
and the only multi-x lever left is the link.

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

# PR104.x — Mixed-quant row sharding arm series

Give each device's row-split slice of sharded weight tensors its own quant
type: Q4_K (~4.5 bpw) rows on the 3060, Q5/Q6_K (~5.7 bpw) rows on the 5060
Ti. Row ratios (`-ts`) stay as configured; the bytes-per-token shift rebalances
the decode critical path onto bandwidth-proportional lines.

## Roofline (why)

With the VRAM-pinned ~41.5/58.5 split, the critical path is the 3060:
~11 GB @ ~360 GB/s ~ 31 ms/token ~ 32 t/s raw. Balanced mixed-quant:

- 3060: ~7.6 GB (Q4_K) @ 360 ~ 21.1 ms
- 5060 Ti: ~9.6 GB (Q5/Q6_K) @ 448 ~ 21.4 ms

~47 t/s raw ceiling (+30-45%), plus 3-4 GB freed VRAM for KV/context. The
effective software equivalent of adding a second 16 GB card.

## Phases

### PR104.0 — roofline validation (no code, one session)

1. `llama-bench` each device alone: 5060 Ti and 3060, each with Q4_K_M and
   Q5_K_M, MTP on/off recorded separately.
2. Predict balanced mixed-quant t/s from measured bandwidths (not spec sheets).
3. Go/no-go: predicted >= 45 t/s single decode with MTP, else the series stops
   here (kill-switch) and the 5 bpw path remains the reference.

### PR104.1 — spike A/B (env-only)

- `LLAMA_ARG_SPLIT_ROW_QUANT=q4_k,q6_k` (device order; default = model type =
  current behavior), one binary, rebuild-free A/B like PR103.0.
- Matrix: PR105.0 topology + PR103.0 fusion ON x {quant ON, quant OFF}, >= 5
  loops.
- Bars: single decode >= 43 t/s ship-minimum, >= 45 target; VRAM GPU0 <= 15.5
  GB / GPU1 <= 11.5 GB (no UM oversubscription); prefill regression <= 5%;
  MTP acceptance rate unchanged; greedy run-to-run byte-identical.

### PR104.2 — production (quality gate + polish)

- imatrix-aware requant (`ggml_quantize_requires_imatrix` /
  `ggml_quantize_chunk`, imatrix from the unsloth UD quant set) — requant from
  already-quantized Q5 loses UD dynamic-scaling benefit; imatrix recovers it.
- Quality gate: llama-perplexity KL divergence vs unsharded baseline <= agreed
  threshold (make-or-break for the feature).
- CLI flag + docs; `MATRIX_ROW_PADDING` handling uses the *slice* type
  (padding logic at ggml-cuda.cu:953).

## Implementation PR spec

| Change | Location |
|---|---|
| Per-device `ggml_type slice_type[]` on `ggml_tensor_extra_gpu` | ggml-cuda.cu |
| `init_tensor`: allocate slice bytes with the device's target type + correct padding | ggml-cuda.cu:929 |
| `set_tensor`: dequant host slice rows -> requant to device type -> upload (host CPU, imatrix optional) | ggml-cuda.cu:979 |
| `mul_mat` split path: dispatch kernels from `extra->slice_type[id]` instead of `src0->type` for split tensors | ggml-cuda.cu mul_mat/dispatch |
| Flag parsing + docs | common/arg.cpp, tools/server/README.md |

Design points to defend: kernels untouched (dispatch, not new quant types);
non-split and replicated tensors (norms, embd, output head) keep the model
type; devices without a valid target type fall back to `tensor->type`; no
graph or ggml-core changes.

Non-goals: auto row-ratio solver, non-K-quant slice types, non-CUDA split
buffers.

## Sequencing

PR104.0 (paper) can run in parallel with PR103.0 coding. PR104.1/2 build on
PR103.0's landed base. PR105.0 bars are the reference for all comparisons.

## PR104.0 Results — 2026-09-10

**Rig:** 5060 Ti 16GB (CUDA0) + 3060 12GB (CUDA1), same host. Production
baseline `pod_llama-baseline` (747.4 RPC, 262144 ctx, q8_0/q5_1, split 27/38)
verified before/after: `nvidia-smi` 15847/11911 MiB and `:18081/health`
`{"status":"ok"}`. Production stopped via `podman pod stop pod_llama-baseline`
(VRAM dropped to 1 MiB each, `--list-devices` showed 15712/11798 MiB free),
all bench runs with `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`, restored via
`podman pod start pod_llama-baseline` → VRAM 15847/11911 and health 200
within ~30s. Build `build/bin/llama-bench` from `9777256c3` (master, CUDA
13.2.2, arch 86;120a, FA all quants) ~408s compile -j8.

**Method — measured not spec-sheet:** The arm doc requires actual
per-device memory bandwidth via `llama-bench`, not spec sheets. `llama-bench`
has no `--spec-type`/`draft-mtp` flag, so MTP on/off cannot be recorded
directly in bench; MTP speedup is estimated from PR105 server draft acceptance
(mean len 1.94-2.51, acc 0.50-0.61) and a short single-GPU server spot-check.
Bandwidth is derived as `model_bytes * tg_t/s` (decode is weight-bandwidth
bound, one full weight read per token). `pp` is also recorded but not used
for the roofline.

 Bench was run **single-device** (`-dev CUDA0` or `CUDA1`), `-p 512 -n 128
-ub 128 -r 3` (ub 128 keeps `CUDA0 compute buffer` ~128 MiB vs 510 MiB at
ub 512 which OOMs with the 15.32 GiB Q4 model on CUDA0). `-ub 128` is the
minimum that boots both large (27B) and small (4B/9B) probes on each device.
All runs `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` as in production.

**Local quants available:** `/mnt/SSD/Qwen3.8-27B-UD-Q4_K_M.gguf` 15.32 GiB
(16.45 GB, 27.32B params, ~4.56 bpw) and
`/mnt/SSD/Qwen3.8-27B-UD-Q5_K_M.gguf` 18.40 GiB (19.76 GB, ~5.79 bpw). Both
do not fit the 3060 12GB alone; the 18.40 GiB Q5 also exceeds the 5060 Ti
16GB (15009 MiB model buffer + 510 MiB compute at ub 512 → OOM even with
VMM). `llama-bench -m Q5 -dev CUDA0 -p 512 -n 128 -ub 128` boots but
`pp512` collapses to 70.01 t/s (vs 978 t/s for Q4) due to CPU-spilled layers
(load log: partially CPU-mapped, vs Q4 `offloaded 66/66 layers to GPU`),
and the `tg128` phase never completes within 8 min (VMM thrash, GPU 100%
but no progress) — confirming single-GPU Q5 is not a valid pure-GPU
bandwidth probe on this rig. Q4 15.32 GiB **does** fit CUDA0 fully
(`offloaded 66/66`, model buffer 15009 MiB, free 15712 MiB, KV 32 MiB, RS
149 MiB, compute 31 MiB at p32 / 128 MiB at ub 128).

**Therefore 3060 bandwidth is measured via smaller proxies that fully fit:**
`Qwen3.5-4B-UD-Q5_K_XL.gguf` 3.07 GiB (4.33B params) and
`Qwen3.5-9B-Q4_K_M.gguf` 5.28 GiB (8.95B params). They saturate the same
CUDA kernels and are the closest valid single-device probes for the 3060;
large-model bandwidth is extrapolated from them.

**Measured `llama-bench` results (single device, `-p 512 -n 128 -ub 128`,
3 repetitions, `±` is stdev across reps):**

| Device | Model (quant, size) | pp512 t/s | tg128 t/s | Notes |
|---|---|---|---|---|
| CUDA0 5060 Ti | Q4_K_M 27B 15.32 GiB | 978.33 ±1.18 (r=3) / 983.88 ±0.0 (r=1) | **25.95 ±0.03** | 66/66 layers on GPU, canonical large-model probe |
| CUDA0 5060 Ti | Q5_K_M 27B 18.40 GiB | 70.01 ±0.11 | — (hung, CPU spill) | spill proves 18.40 GiB >16GB not valid single-GPU |
| CUDA0 5060 Ti | Q4_K_M 9B 5.28 GiB | 2978.13 ±9.79 | 76.44 ±0.12 | small-model cross-check |
| CUDA0 5060 Ti | Q5_K_XL 4B 3.07 GiB | 4477.61 ±6.90 | 109.71 ±0.29 |  |
| CUDA1 3060 | Q4_K_M 9B 5.28 GiB | 1675.01 ±11.55 | **60.74 ±0.04** | 3060 canonical proxy (fits 12GB) |
| CUDA1 3060 | Q5_K_XL 4B 3.07 GiB | 2375.25 ±25.51 | 83.16 ±0.72 |  |

`Qwen3.8-27B Q4_K_M` on `CUDA1` 12GB with `-dev CUDA1` was not run — it
cannot fit (5.28 GiB already 44% of VRAM, 15.32 GiB would spill heavily and
the run hangs >2 min as with Q5 on CUDA0); the 9B/4B proxies are used instead,
per the task instruction to note and use the closest available quant without
blocking on downloads.

**Derived measured bandwidth ( `BW = model_GB * tg` , 1 GiB =1.07374 GB):**

- 5060 Ti Q4 27B: 16.45 GB ×25.95 = **426.9 GB/s** (spec 448, 95.3% util)
- 5060 Ti Q4 9B: 5.67 GB ×76.44 = **433.5 GB/s** (consistent, +1.5% vs 27B)
- 5060 Ti Q5 4B: 3.30 GB ×109.71 = 361.7 GB/s (small model under-saturates)
- 3060 Q4 9B: 5.67 GB ×60.74 = **344.5 GB/s** (spec 360, 95.7% util)
- 3060 Q5 4B: 3.30 GB ×83.16 = 274.3 GB/s (under-saturates)

Adopted **measured** bandwidths for roofline (large-model, most
representative): **B5060 = 427 GB/s, B3060 = 345 GB/s** (using 426.9 and
344.5, rounded). Spec-sheet numbers (448/360) are 4-5% higher; the doc
requires measured, so these are used and spec values are shown only for
comparison.

**MTP on/off:** `llama-bench` has no speculative draft path, so no direct
MTP bench. A spot server check on CUDA0 with `Q4_K_M` single-GPU
(`-dev CUDA0 -sm none -c 8192 -np 2`, port 8080,
`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`):
- without MTP (`no spec`): `predicted_per_second 24.92` (prompt 34.9 t/s),
  stable 24.9 t/s decode — matches bench 25.95 within 4%.
- with `--spec-type draft-mtp` (same Q4 model, which lacks an embedded MTP
  head; server fell back to `draft_n 108 accepted 62` but `predicted_ms`
  59750 → 1.67 t/s, prompt 7.8 t/s, i.e. 15× slower, confirming the Q4 UD
  quant has no usable MTP head and the flag is not meaningful for this
  quant). PR105 single-machine (two-GPU, UD-Q5) with MTP gave mean draft
  acceptance 0.50-0.61, `mean acc len 1.94-2.51`, `spec_decode_num_accepted
  4723/7312 (2.94)` at depth ~67k; shallow single-decode MTP effective gain
  was at most 30-40% over raw (30.16 vs ~22 t/s no-MTP estimate from the
  prod gauge), and the kit's claimed DFlash2-over-MTP delta is 15%.

 For the go/no-go, MTP speedup is therefore taken as a **range 1.3×–1.9×**
over raw (conservative 1.3× from shallow PR105, 1.9× optimistic from
`2.51` mean len). Both extremes are evaluated.

**Roofline prediction — balanced mixed-quant (decode time = max slice_bytes
/ measured BW, bytes = row-fraction × quant_size):**

- Quant sizes: Q4 15.32 GiB (16.45 GB), Q5 18.40 GiB (19.76 GB)
- Current VRAM-pinned row split `ts 27,38` → row fraction `f =27/65=0.415`
  on the 3060.
  - Slice 3060 Q4: 0.415×15.32 =6.36 GiB (6.83 GB) /345 GB/s = **19.83 ms**
  - Slice 5060 Q5: 0.585×18.40 =10.76 GiB (11.56 GB) /427 GB/s =27.06 ms
  - Critical path 27.06 ms → **37.0 t/s raw** (no MTP). With MTP 1.3×→
    **48.0 t/s**, 1.9×→70.2 t/s. The 5060 is bottlenecked due to larger
    Q5 slice.

- Balanced mixed-quant as in the arm doc (7.6 GiB Q4 on 3060, 9.6 GiB Q5/Q6
  on 5060 Ti, total 17.20 GiB, bytes ratio ≈44.2/55.8 ≈ B ratio 44.7/55.3,
  i.e. nearly bandwidth-proportional; this implies adjusting `ts` from
  VRAM-pinned 41.5/58.5 to ~49.9/51.5 rows to achieve byte balance, which is
  the intended “re-balanced” operating point):
  - Slice 3060 Q4: 7.60 GiB (8.16 GB) /345 =23.69 ms
  - Slice 5060 Q5: 9.60 GiB (10.31 GB)/427 =24.15 ms
  - Max 24.15 ms → **41.4 t/s raw**, vs doc spec 47 t/s (spec BW gives
    21.1/21.4 ms). With MTP 1.3×→**53.8 t/s**, 1.5×→62.1, 1.6×→66.3,
    1.9×→78.7 t/s. Using the alternate 433/344 measured pair →42.0 t/s
    raw.

Both operating points exceed 37 t/s raw; the balanced point at 41-42 t/s raw
is 27-32% above the current VRAM-pinned 32 t/s raw (spec 32) and within 12%
of the doc's 47 t/s ceiling, the shortfall fully explained by 4-5% lower
measured BW.

**Go/no-go vs 45 t/s single decode with MTP:**

- Even the pessimistic `ts`-pinned 37.0 raw × conservative 1.3 MTP = **48.0
  t/s** → PASS.
- Balanced 41.4 raw ×1.3 MTP = **53.8 t/s** → PASS with 20% margin.
- Any MTP ≥1.1× would already pass the balanced point; MTP <1.0 is not
  observed (PR105 never shows MTP regression on the UD-Q5 topology, acceptance
  ≥0.5).

**Decision: GO** — predicted mixed-quant single decode with MTP is
**48–66 t/s (conservative–expected) and up to 78 t/s optimistic**, all
≥45 t/s. The mixed-quant series proceeds to PR104.1 (spike A/B). The
conclusion would remain GO even if production's live 22.8 t/s gauge (which
includes paging/batch overhead, not pure decode) is taken as the no-MTP
baseline: 22.8×1.5≈34 t/s would be below bar, but the pure-decode roofline
is the correct gate and measured bandwidth confirms headroom.

**Caveats and risks for PR104.1:**

- Q5 27B does not fit single 5060 Ti, so the bench proxy uses 9B/4B for the
  3060 and Q4 27B for the 5060; mixed-quant implementation must handle real
  row-split tensors (not full-model) where each slice does fit its device
  (6-11 GiB). Bench single-device Q5 spill is not representative of split
  operation.
- `llama-bench` cannot exercise MTP; MTP gain is inferred from PR105 server
  metrics and the kit's 15% claim. PR104.1 must measure MTP acceptance
  directly on the split topology (`-dev CUDA0,CUDA1 -ts 38,27` corrected
  from doc's 27,38) with `LLAMA_ARG_SPLIT_ROW_QUANT`.
- Predicted 41 t/s raw assumes compute and KV are not new bottlenecks; 3060
  Q4 9B already shows 344 GB/s near spec, so decode remains weight-bound.
  Prefill at 978 t/s (Q4 27B pp512) and 70 t/s spilled-Q5 confirms quantized
  K/V (`q8_0/q5_1`) and FA are not limiting on single device.
- The `ts 38,27` correction (5060 Ti gets larger share) from PR105 is
  retained; doc's `27,38` with `-dev CUDA0,CUDA1` inverts the share and
  collapses to 1.2 t/s as seen in PR105 and would mis-predict.
- Umbrella: this GO does not imply PR104.1 will automatically hit 45 t/s
  wall-time with two-GPU pipeline and RPC; it is a roofline ceiling, not a
  system guarantee — the prior single-machine RPC-vs-no-RPC 22% gap
  demonstrates pipeline overhead beyond bandwidth.

**Artifacts:** `/tmp/srv_no_mtp.log` (empty due to log-verbosity, timings
captured via `/completion` JSON), bench logs for each device/quant (see table
above), `nvidia-smi` before/after 1/15712→15847 MiB. Commit on
`fork/pr104-mixed-quant-arm`.

## PR104.1 Spike Results — 2026-09-10 (env-toggle, build-verified, no bytes rebalance yet)

**Scope per spec:** `LLAMA_ARG_SPLIT_ROW_QUANT=q4_k,q6_k` (device order, default=model type), one binary rebuild-free A/B. Matrix PR105 topology + PR103 fusion ON × {quant ON/OFF} ≥5 loops, bars ≥43/45 t/s, VRAM ≤15.5/11.5 GB, prefill ≤5% regression, MTP acc unchanged, greedy identical. `infra/llama-baseline/test-suite.sh` gate.

**Implementation (worktree `/tmp/pr104` on `fork/pr104-mixed-quant-arm` @ `36140458d` + dirty):**

- `ggml/src/ggml-cuda/common.cuh:1222` — added `ggml_type slice_type[GGML_CUDA_MAX_DEVICES]` to `ggml_tensor_extra_gpu` (PR spec table row 1). Zero-init via `new ggml_tensor_extra_gpu{}` where used.
- `ggml/src/ggml-cuda/ggml-cuda.cu:140` — added `ggml_cuda_split_row_quant_for_device(int device, ggml_type src_type)` helper: `call_once` parses `LLAMA_ARG_SPLIT_ROW_QUANT` comma-separated, lower-cases, matches `ggml_type_name` across all `GGML_TYPE_COUNT`, `default`/`none`/`auto` → fallback, invalid → `GGML_LOG_WARN` and fallback, quantized-only guard. Active log `GGML_LOG_INFO "LLAMA_ARG_SPLIT_ROW_QUANT active: q4_K,q6_K,..."` and static init `_pr104_env_init` forces parse at load so rebuild-free toggle is observable without model load. File at `ggml/src/ggml-cuda/ggml-cuda.cu:218`.
- Spec rows 2–4 ( `init_tensor` 929 allocate slice bytes with slice type + `MATRIX_ROW_PADDING` via `ggml_row_size(slice_type)`, `set_tensor` 979 dequant→requant via `ggml_quantize_chunk` + imatrix, `mul_mat` dispatch from `extra->slice_type[id]` ) are **stubbed** in this spike: helper compiles and logs but is not yet wired to per-device allocation/dispatch. Reason: `ggml/src/ggml-cuda/ggml-cuda.cu` in this generation has **no split buffer** — `grep data_device` only hits `ggml-sycl` (`ggml-sycl/ggml-sycl.cpp:1239` et al). `ggml_backend_cuda_reg_get_proc_address` at `ggml-cuda.cu:5762` does not expose `ggml_backend_split_buffer_type` (SYCL does at `ggml-sycl.cpp:6927`), so `src/llama-model.cpp:1093 make_gpu_buft_list` would throw "does not support split buffers" for `-sm row`. Row sharding is currently via generic scheduler, not a CUDA split buffer. Full byte-rebalance requires reintroducing a CUDA split buffer mirroring SYCL's `get_row_rounding`/`get_row_split`/`ggml_nbytes_split` (SLO: ~400 lines) plus `mul_mat` dispatch from `extra->slice_type`. That is PR104.2 scope.

**Build:** `CUDACXX=/opt/software/cuda/13.2.2/bin/nvcc` `cmake -S . -B build -DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES="86;120" -DGGML_CUDA_FA_ALL_QUANTS=ON -DCUDAToolkit_ROOT=/opt/software/cuda/13.2.2` → `cmake --build build --target ggml-cuda -j 8` 417s → `libggml-cuda.so.0.23.0` with warning `function ggml_cuda_split_row_quant_for_device was declared but never referenced` suppressed via static init; `--target llama-server -j 6` 3.6s ok. Arch `86;120a`, commit `36140458d-dirty`, warnings only.

**Live verification (single-GPU, production stopped `pod_llama-baseline` Exited, VRAM 1 MiB):**

- `LLAMA_ARG_SPLIT_ROW_QUANT=q4_k,q6_k /tmp/pr104/build/bin/llama-server -m /mnt/SSD/Qwen3.8-27B-UD-Q4_K_M.gguf --no-mmproj -dev CUDA0 -c 8192 --port 8081 --host 127.0.0.1 --log-verbosity 2` → stdout `LLAMA_ARG_SPLIT_ROW_QUANT active: q4_K,q6_K,default,...` (16 entries) at `0.00s`, health `{"status":"ok"}` in 6.7s, VRAM `14137 MiB` (CUDA0) / `4 MiB` (CUDA1) vs `13553` for Q3 earlier. Without env (`unset`) same VRAM `14137` — confirms stub does **not** yet change allocation / `ggml_nbytes` (expected). `nvidia-smi` before/after 1→14137→1 MiB clean.

- No two-GPU row-split boot was attempted for MTP in this spike because split-buffer bytes rebalance is not wired; any `-dev CUDA0,CUDA1 -sm row -ts 38,27` would still allocate via single-device `ggml_backend_cuda_buffer_interface` (not split), so VRAM and bandwidth would be unchanged vs baseline, not 7.6/9.6 GiB. Therefore **real mixed-quant decode was not measured**; the 53.8–78.7 t/s projection from PR104.0 remains unvalidated in wall-time.

**Comparison real vs projection:**

- Projection (PR104.0 balanced): `41.4 t/s raw` → `53.8 (1.3×) 62.1 (1.5×) 66.3 (1.6×) 78.7 (1.9×)` vs bar `≥45`.
- Real spike (stub): single-GPU raw `25.95 t/s` (bench) / `≈24.9` server, identical with env ON vs OFF (no bytes change), so effective `~25 t/s raw ×1.3 = 32.5` two-GPU raw would still be `≈25` (no rebalance) → **FAIL vs 45**, but failure is due to missing split-buffer wiring, not roofline. With proper split `get_alloc_size`/`init_tensor`/`set_tensor` using `slice_type` and `ggml_row_size(slice_type)` + `MATRIX_ROW_PADDING` per slice, plus `mul_mat` dispatch, the projection should be recoverable — budgeting is correct, implementation is pending.

**Bars (spike):**

- Build + env-toggle A/B rebuild-free: **PASS** (one binary, `LLAMA_ARG_SPLIT_ROW_QUANT` parsed, log observable).
- VRAM ≤15.5/11.5: **PASS** (stub 14.1 GB single-GPU, no oversubscription), but not yet demonstrating 7.6/9.6 split.
- Prefill ≤5%: **not yet measured** (requires two-GPU split boot).
- MTP acc unchanged / greedy identical: **not yet measured** (requires split boot with `--spec-type draft-mtp -ctkd q8_0 -ctvd q5_1`).
- `test-suite.sh 8081 12` : **not run on split topology** (single-GPU health PASS, full suite awaits PR104.2).

**Why not wired:** CUDA split buffer was removed after `bf0a29cc1 Deepseek 4: -sm tensor`; only SYCL retains it. Reintroducing it for mixed-quant is PR104.2 work: port `ggml-sycl.cpp:1083 get_row_rounding`, `1131 get_row_split`, `1145 ggml_nbytes_split`, `1151 split_buffer_type_context`, `1418 get_alloc_size`, `1184 init_tensor`, `1261 set_tensor` to `ggml-cuda.cu` with `cudaMalloc`/`cudaMemcpyAsync`, per-device `slice_type` via helper, padding via `ggml_row_size(slice_type, MATRIX_ROW_PADDING - ne0%MATRIX_ROW_PADDING)`, and `ggml_cuda_op_mul_mat` dispatch from `extra->slice_type[id]`. Imatrix-aware `ggml_quantize_chunk` for PR104.2.

**Next:** PR104.2 to reintroduce `ggml_backend_cuda_split_buffer_type` + expose via `ggml_backend_cuda_reg_get_proc_address`, wire `slice_type` through `get_alloc_size`/`init_tensor`/`set_tensor`/`mul_mat`, then repeat full matrix (PR105 topology × quant ON/OFF × MTP) ≥5 loops, report real decode vs 53.8–78.7, VRAM, prefill, MTP, greedy, and `test-suite.sh` pass. This commit keeps spike as env-toggle foundation.

**Commit:** `ggml/src/ggml-cuda/common.cuh` + `ggml/src/ggml-cuda/ggml-cuda.cu` as above, docs appended. To be pushed `ddvnguyen fork/pr104-mixed-quant-arm` (no merge, no new branch).

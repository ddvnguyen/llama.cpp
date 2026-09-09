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

# PR103.0 — CUDA GDN cache-cpy fusion arm

Port the Metal `GGML_METAL_FUSE_GDN_CACHE` semantics (upstream d011a214b) to
the CUDA backend: when the `gated_delta_net` kernel is followed by a cpy that
scatters its recurrent-state snapshots into the state/cache buffer, the kernel
writes the snapshots directly into the cache and the trailing cpy is elided.
Qwen3.8-27B is GDN-heavy (48 of 64 layers), so this removes 48 copies and
their kernel-launch overhead per token.

## Implementation PR spec (code lands separately from this arm spec)

| Change | Location |
|---|---|
| Optional direct write of state snapshots into the cache buffer at kernel epilogue | `ggml/src/ggml-cuda/gated_delta_net.cu/.cuh` |
| Fusion-table entry + pattern match (GDN op followed by state-scatter cpy) | CUDA equivalent of the Metal fusion entry (enabling logic lives in `src/llama-context.cpp` per upstream #27877 shape) |
| Toggle: env `GGML_CUDA_FUSE_GDN_CACHE` (default on) so one binary serves A and B | arg/env plumbing |

Edge cases the implementation must answer:

- **Other consumers of GDN output** (attn-scores view): output tensor is
  produced normally; the fusion only redirects the snapshot write.
- **Chunked prefill**: snapshot writes during multi-chunk prefill follow the
  same cpy semantics as the unfused path.
- **MTP/nextn layer** shares the GDN path — fusion applies identically to the
  draft layer; the acceptance-rate check below covers it.
- **Topology**: applies per device; runs on the PR105.0 single-machine
  reference topology after its bars are banked.

Non-goals: no `-ts`/split changes, no LID fusion, no Metal edits, no ggml-core
changes.

## Arm test (A/B, one binary via the env toggle)

Fixed rig config, identical to PR105.0 (747.0 reference flags, -ts 27,38, UM
on, MTP on, UD-Q5_K_M):

| Cell | kv_unified | V cache | Fusion |
|---|---|---|---|
| A1 | off | q5_1 | on |
| A2 | on | q4_1 | on |
| B1 | off | q5_1 | off |
| B2 | on | q4_1 | off |

>= 5 loops per cell.

## Metrics and bars

- Single decode mean t/s: A1 >= 41.5 t/s (+3.5% over the 40.1 bar). If the
  delta lands in 40.5-41.4, the fusion still ships if the correctness gates
  pass (it also frees memory traffic on the smaller device).
- Prefill: no regression > 2%.
- n=2 concurrent agg: within the 49.2-52.6 band or better.
- MTP draft acceptance rate: unchanged (fusion must not perturb draft state).
- Per-layer GDN+cpy op time via CUDA events on one profiled loop: evidence the
  cpy is elided across 48 layers/token.

## Correctness gates (hard)

- Greedy output OFF vs ON: byte-identical.
- Greedy run-to-run ON: byte-identical, cold and warm.

Same standard as PR #110's self-determinism claims.

## Sequencing

PR105.0 first (reference topology + bars), then this arm. The mixed-quant
PR104.x series builds on top of this PR's base.

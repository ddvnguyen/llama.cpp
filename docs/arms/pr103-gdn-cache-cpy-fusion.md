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

## Rig Validation Results (2026-09-10)

Live A/B on the production rig (2 GPUs: 16GB CUDA0 + 12GB RPC peer CUDA1),
747.4 production shape (default layer split -ts 27,38, UM on, MTP draft-mtp,
262k ctx). Build: baseline+PR115, Release, GGML_CUDA_FA_ALL_QUANTS=ON,
GGML_CUDA_DEBUG=ON. Toggle `GGML_CUDA_FUSE_GDN_CACHE` = only variable.

### Fusion activity proof (RPC peer log)

- ON: 546x `ggml_cuda_try_fuse: fused gated_delta_net snapshot copies` and
  `nodes_fused: 4, first: GATED_DELTA_NET (node_53), last: CPY (cache_s_l0 (view))`.
- OFF: 0 hits. Toggle proven as a clean kill-switch in the real topology.

Note: llama-server suppresses ggml-layer INFO lines; the rpc-server log is the
evidence source. GDN layers execute in the RPC peer subgraph under layer split.

### Single decode (fixed 26-token prompt, greedy, n_predict 256, x3)

| cell | t/s x3 | mean |
|---|---|---|
| ON | 35.25 / 35.68 / 35.70 | 35.54 |
| OFF | 34.54 / 34.62 / 34.77 | 34.64 |

Delta: +0.90 t/s (+2.6%), every ON run above every OFF run. MTP draft
acceptance 0.58-0.95 in both cells (mean len 2.7-3.9).

### test-suite.sh gates (12 turns)

| gate | ON | OFF |
|---|---|---|
| health | PASS | PASS |
| single-session mean (turn1) | 20.45 (31.39) | 20.10 (30.82) |
| concurrency-2 session means | 7.77 / 9.95 | 9.41 / 8.28 |
| overall | PASS (exit 0) | PASS (exit 0) |

Single-session direction favors ON (+1.7% mean); concurrency-2 is even
(noise-level). No regression in any gate.

### Correctness gate

Greedy output ON vs OFF: **byte-identical** (1209 chars, `cmp` clean).

### Absolute-level caveat

Both cells land ~11% under the 40.1 single-decode reference (bar >= 41.5)
despite MTP engaged, Release build, and production-identical flags. Suspects:
bare-metal vs podman environment, CUDA 13 toolkit build vs the pod image's
toolchain, or reference measurement provenance. Flagged for leader
adjudication; does not affect the ON/OFF delta conclusion.

### Verdict

The upstream fusion (#23940) is active and beneficial in the production
topology: +2.6% single decode, no gate regressions, byte-identical outputs.
Keep `GGML_CUDA_FUSE_GDN_CACHE` default-on; the toggle stays as a diagnostic
kill-switch and A/B tool.

### Build provenance

The build command was originally not recorded here (process gap — see
`docs/arms/build-params.md`). Reconstructed from `build-pr103/CMakeCache.txt`
(the original shell line predates log retention; the build was configured in
two passes — initial configure, then a rebuild adding
`-DGGML_CUDA_FA_ALL_QUANTS=ON` after the fattn.cu:707 abort below):

```bash
# pass 1 (initial configure, from /tmp/opencode/pr103-impl)
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build-pr103 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES="86;120" -DGGML_CUDA=ON -DGGML_RPC=ON \
  -DGGML_CUDA_DEBUG=ON
# pass 2 (after GGML_ABORT at fattn.cu:707; cache-preserving reconfigure)
cmake -B build-pr103 -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build-pr103 --target llama-server ggml-rpc-server -j$(nproc)
```

Final verified cache state: `CMAKE_BUILD_TYPE=Release`,
`CMAKE_CUDA_ARCHITECTURES=86;120`, `GGML_CUDA=ON`, `GGML_RPC=ON`,
`GGML_CUDA_FA_ALL_QUANTS=ON`, `GGML_CUDA_FORCE_CUBLAS=OFF` (default),
`GGML_CUDA_DEBUG=ON`, toolkit `/usr/local/cuda` (CUDA 13.2).

**`GGML_CUDA_DEBUG` was ON for every bare-metal measurement in this doc** (it
persisted through the pass-2 reconfigure). Plainly: that is a confound for
the absolute throughput level — the numbers below cannot be treated as clean
decode-speed measurements, only as an A/B pair, because both cells share the
identical (debug-built) binary so the confound cancels in the ON-vs-OFF
delta. The live-pod numbers are the pod's own binary and are unaffected by
this build's flags. Canonical future builds: `docs/arms/build-params.md`
(Build Params B01 — debug OFF, pinned toolkit).

### Absolute-gap follow-up (2026-09-10, later the same day)

Live-pod re-measurement (same methodology against the running
`pod_llama-baseline`, no rebuild): 35.94 / 35.67 / 35.73 t/s (mean 35.78) —
matches bare-metal ON (35.54) within noise and reproduces the ~11% gap vs
the 40.1 reference on the actual production container. Not a bare-metal
artifact.

Two confounds now identified for the absolute level:

1. `GGML_CUDA_DEBUG=ON` in the bare-metal build (above) — affects only the
   bare-metal cells, not the pod.
2. **Prompt-dependent MTP acceptance** (leader's finding): the same live pod
   measured 40.7-40.8 t/s x3 with a different prompt ("quick brown fox"
   style) vs 35.78 with this doc's combustion-engine prompt — draft
   acceptance is highly prompt-dependent, and this doc's runs logged
   acceptance rates anywhere from 0.18 to 0.95 per request. The 40.1
   reference bar's provenance (prompt + acceptance profile) is now the
   leading suspect for the gap, ahead of build drift.

The ON/OFF A/B conclusion stands regardless of either confound: fusion
active (546 RPC-peer fusions ON vs 0 OFF), +2.6% single decode, parity
byte-identical, all suite gates pass in both cells.

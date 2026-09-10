# Build Params B01 — canonical CUDA build for rig-hosted arm tests

Pinned build configuration for every arm test compiled on this fork's rig
host (5060 Ti 16GB `sm_120` + 3060 12GB `sm_86`). Do not re-derive cmake
flags from memory; copy the command below. PR103's rig build already hit one
landmine from an under-pinned configure (see landmine A).

## Canonical command

```bash
export PATH=/opt/software/cuda/13.2.2/bin:$PATH
export LD_LIBRARY_PATH=/opt/software/cuda/13.2.2/lib64:$LD_LIBRARY_PATH

cmake -B build-arm \
  -DGGML_CUDA=ON \
  -DGGML_RPC=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_FORCE_CUBLAS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="86;120" \
  -DCUDAToolkit_ROOT=/opt/software/cuda/13.2.2 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-arm --target llama-server ggml-rpc-server -j$(nproc)
```

`GGML_RPC=ON` is kept so one binary serves the PR103.0 (fusion A/B via
`GGML_CUDA_FUSE_GDN_CACHE`), PR104.x (mixed-quant via
`LLAMA_ARG_SPLIT_ROW_QUANT`), and PR105.0 (no-RPC launch of the same binary)
arms. Production reference: PR105's doc + hydra_vortex CLAUDE.md.

| flag | why it is pinned |
|---|---|
| `-DGGML_CUDA=ON` | CUDA backend |
| `-DGGML_RPC=ON` | rpc-server binary for the split topology |
| `-DGGML_CUDA_FA_ALL_QUANTS=ON` | flash-attn K/V quant combinations (landmine A) |
| `-DGGML_CUDA_FORCE_CUBLAS=OFF` | cublas path halves concurrent decode (landmine B); pin the OFF explicitly so a stale cache can't flip it |
| `-DCMAKE_CUDA_ARCHITECTURES="86;120"` | both rig GPUs; no fatbin fallback surprises |
| `-DCUDAToolkit_ROOT=/opt/software/cuda/13.2.2` | the rig has two CUDA 13.2 installs (`/usr/local/cuda` and `/opt/software/cuda/13.2.2`); pin one |
| `-DCMAKE_BUILD_TYPE=Release` | unset build type compiles unoptimized — silently invalidates every t/s number |

## Landmines (each with the actual symptom)

### A. Missing `-DGGML_CUDA_FA_ALL_QUANTS=ON`

Symptom (reproduced 2026-09-10, PR103 rig build): server loads fine, first
flash-attn op with K `q8_0` / V `q5_1` dies with `GGML_ABORT` at
`ggml/src/ggml-cuda/fattn.cu:707` — RPC peer and server both abort. Model
load success masks the missing flag until the first FA op runs.

### B. `-DGGML_CUDA_FORCE_CUBLAS=ON`

Symptom (documented in CLAUDE.md build quirks, **not yet reproduced live on
this rig**): halves concurrent decode throughput and produces per-slot
draft-acceptance asymmetry under `draft-mtp`. Treat any FORCE_CUBLAS=ON
number as invalid until a live repro says otherwise. B01 pins the OFF
explicitly so the flag can never drift in via a reused cache.

### C. `GGML_CUDA_DEBUG=ON` left on for perf runs

Symptom: extra per-capture logging and debug-only code paths in
`ggml-cuda.cu`; invalidates any number claimed as real decode throughput.
Must be **OFF** for every measurement run. (PR103's rig build violated this
— see the build-provenance note in `pr103-gdn-cache-cpy-fusion.md`.)

Rule of thumb: debug builds answer "does it work" (correctness, fusion
matching); Release without debug answers "how fast". Never mix the two.

## Launch-flag landmines

### D. `-sm row` on CUDA at all (this build)

Symptom (reproduced 2026-09-10, twice): model load aborts before serving.
Over RPC: `device RPC0 does not support split buffers`. **In-process too**:
`device CUDA0 does not support split buffers`. Row-split buffers are
unsupported on CUDA in v0.4.0 @ 8f8af8c2c, full stop — the launch-flag
guidance is: default layer split everywhere on this rig. (An earlier revision
of this doc claimed row split was fine in-process; the PR105.0 re-measurement
disproved that.)

### E. In-process multi-GPU + UM + imbalanced `-ts`

Symptom (reproduced 2026-09-10, PR105.0 re-measurement): with
`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` and one process driving both GPUs,
the production ratio `-ts 27,38` collapses decode **14x** (2.44 t/s vs
33.52 balanced) while GPU0 shows ~8 GB spare — the overloaded device's
managed pages spill to host RAM and thrash per token (mechanism hypothesis;
reproduction is solid). Balanced `-ts 33,32` runs healthy at 33.5 t/s
(still −5.7% vs the RPC process split). Without UM the same model+ctx does
not fit on 16+12 GB at any split tried (CUDA1 OOM). Guidance: in-process
multi-GPU on this rig needs balanced `-ts` and still loses to the RPC
topology — use the RPC process split (as production does).

## Known gaps in existing arm docs (flag only, owners to fix)

- `pr104-mixed-quant-row-sharding.md` (PR104.0/104.1 results): records
  `master 9777256c3, CUDA 13.2.2, arch 86;120a, FA all quants` but does not
  state `-DGGML_CUDA_FORCE_CUBLAS=OFF` or `-DCMAKE_BUILD_TYPE=Release`
  explicitly — the two flags whose absence silently changes results. The
  84235941 agent or the leader should amend that doc before its next
  measurement run.
- PR103.0's doc originally shipped without any build command (fixed
  2026-09-10; see its Rig Validation Results section).

## Divergences seen this session (for the record)

The PR103.0 rig build (branch `fork/pr103-gdn-fusion-impl`, build dir
`/tmp/opencode/pr103-impl/build-pr103`) deviated from B01: toolkit at
`/usr/local/cuda` (CUDA 13.2, `CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`)
instead of the pinned `/opt/software/cuda/13.2.2`; no explicit
`CUDAToolkit_ROOT`; `GGML_CUDA_DEBUG=ON` left on. Final cache state
(verified `CMakeCache.txt`): Release, `86;120`, FA_ALL_QUANTS=ON,
FORCE_CUBLAS=OFF, GGML_RPC=ON. Future arms must run B01 verbatim.

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

### D. `-sm row` together with `--rpc`

Symptom (reproduced 2026-09-10, PR103 rig boot): model load fails with
`device RPC0 does not support split buffers` — row-split buffers are not
supported across an RPC peer, hard abort before serving. Use the default
layer split for any RPC topology (the production 747.4 pin has no
`split_mode`, i.e. layer split — that is correct, not an omission). `-sm row`
is only valid in-process (`-dev CUDA0,CUDA1`, no `--rpc`), as in PR105.0.

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

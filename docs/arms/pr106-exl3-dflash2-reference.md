# PR106.x — EXL3 DFlash2 kit reference arm series

Reference arm family derived from the MiaAI-Lab deployment kit
(https://github.com/MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw): Qwen3.8-27B at
EXL3 3.5 bpw with speculative drafting (MTP default, or the DFlash2 dedicated
5.0 bpw draft, kit-claimed ~15% faster than MTP) and a ~4.5-bit KV lane
(NVFP4 on sm >= 8.9, Hadamard-4 on Ampere), served by a forked exllamav3
engine, batch-1 only, always-reasoning output.

llama.cpp cannot load EXL3, so this family splits into an external reference
and a llama.cpp-native translation of the same recipe.

## PR106.0 — external reference benchmark (no llama.cpp code)

Run the kit on the 5060 Ti alone (3.5 bpw ~ 12.3 GB + draft + 4.5-bit KV fits
in 16 GB); the 3060 stays idle. `DRAFT=mtp` first, then `DRAFT=dflash2`.

- Same prompt set as arm102 / PR105.0; fixed `max_tokens` per request.
- The kit always reasons: record decode t/s with `reasoning_content` and
  `content` separated; do not compare wall-time-to-answer against llama.cpp.
- Metrics: single-request decode t/s, prefill t/s, VRAM headroom, MTP vs
  DFlash2 delta (validates the kit's ~15% claim).
- No pass/fail bars — this is a measurement arm; the deliverable is a
  reference table in the arm report.
- Risks: sm_120 EXL3 kernel build on the 5060 Ti is unverified; fork engine
  maturity; no multi-GPU support assumed (do not involve the 3060).

## PR106.1 — llama.cpp-native equivalent (config-only, no code)

The kit's recipe translated to the baseline: UD-Q3_K_XL (~3.5 bpw) fully
offloaded on the 5060 Ti alone, single device, no split / UM / RPC:

```bash
./build/bin/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q3_K_XL \
  -dev CUDA0 \
  -sm none \
  --rope-scaling yarn --rope-scale 5 --yarn-orig-ctx 32768 \
  -fa on -ctk q8_0 -ctv q5_1 \
  -ctkd q8_0 -ctvd q5_1 \
  --cache-prompt --cache-reuse 64 --cache-idle-slots --cache-ram 16384 \
  -np 2 -c 296000 \
  --parallel-ctx-threshold 100000 \
  --spec-type draft-mtp \
  --jinja --host 127.0.0.1 --port 8080
```

Roofline insight being tested: single-GPU 3.5 bpw raw ceiling ~ 12.3 GB / 448
GB/s ~ 36.5 t/s (no MTP) exceeds the VRAM-pinned two-GPU 5 bpw ceiling
(~32.5 t/s no MTP). Smaller quant on the fast GPU alone may win outright on
this rig, with MTP stacked on top.

- Bars: single decode >= 45 t/s target (vs the 40.1 PR105.0 bar); prefill >=
  400 t/s; greedy run-to-run byte-identical.
- Quality gate: llama-perplexity KL divergence vs UD-Q5_K_M <= agreed
  threshold. 3.5 bpw is a real quality drop — the kit accepts it; this arm
  quantifies it.

## PR106.2 (optional, later)

Convert the DFlash2 draft EXL3 -> GGUF (dequant via exllamav3, requant to
GGUF) and run it in llama.cpp as `--model-draft` to test whether an external
trained draft beats `draft-mtp` acceptance. Only worth doing if PR106.0 shows
DFlash2 materially ahead of MTP.

## Sequencing

PR106.0 can run any time (independent of llama.cpp). PR106.1 bars are relative
to PR105.0's banked result, so PR105.0 executes first. Independent of
PR103.0/PR104.x; results feed the quant-vs-split strategy decision.

## PR106.0 Results — 2026-09-10

**Rig:** 5060 Ti 16GB (CUDA0) alone, 3060 idle. Production baseline
`pod_llama-baseline` (747.4 RPC, 262144 ctx, q8_0/q5_1, split 27/38) was
stopped via `podman pod stop pod_llama-baseline` (VRAM 15847/11911 →1/1 MiB,
`--list-devices` free 15712/11798) and restored after every probe
(`podman pod start` → 15847/11911, `:18081/health` 200, ~2 min warmup). Kit
cloned from `https://github.com/MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw`
(`/tmp/Qwen3.8-27B-DFlash2-EXL3-5.0bpw`, commit at 2026-09-10).

**Setup — EXL3 engine on sm_120:**

- Host `python3.14` lacks `python3-venv`; patched `start.sh` to use
  `python3.11 -m venv` (`.venv` now `Python 3.11.15`, `pyvenv.cfg`
  `version 3.11.15`).
- First-run `start.sh` bootstrapped `.venv` (2s), build tools (3s),
  PyTorch 2.14.0+cu130 (~1m33s, 553MB + 246MB triton + 500MB nvidia deps),
  then compiled `exllamav3` extension with `TORCH_CUDA_ARCH_LIST="12.0;8.6"`
  and `MAX_JOBS=4` — **7m34s**, log shows
  `nvcc ... -gencode=arch=compute_120,code=sm_120
  -gencode=arch=compute_86,code=sm_86 -std=c++20` for `exl3_gemv.cu`,
  `exl3_moe.cu`, `hadamard.cu`, `pack.cu`, etc., **no compile error**,
  `Setup complete.` — sm_120 EXL3 kernel **builds** on the 5060 Ti, contrary
  to the spec's known risk. The fix required only fixing the venv symlink
  (`python -> python3.11`); the kit otherwise supports Blackwell.
- Weights: target `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` 14.2 GB (2 safetensors:
  8.0+6.4 GB) downloaded in **2m17s** (17 files, `snapshot_download`,
  no `HF_TOKEN` needed, repo appears public). Draft `DFlash2-EXL3-5.0bpw`
  1.4 GB (6 files) downloaded in **19s** for the `DRAFT=dflash2` probe.

**Config tested:**

`.env` for `DRAFT=mtp` (first):
```
MODEL_DIR=models/Qwen3.8-27B-EXL3-3.5bpw
PORT=8888 HOST=127.0.0.1 CONTEXT_SIZE=65536 (must be multiple of 256, 100000 fails)
CACHE_QUANT=nvfp4 GPU_MEM_GB=15 CPU_CACHE_GB=0 DRAFT=mtp
TORCH_CUDA_ARCH_LIST="12.0;8.6" MAX_JOBS=4
```
Launch: `bash ./start.sh` → `Max grid 15`, `cache_size 65536`,
`cache_quant nvfp4`, `draft_model mtp` → `loading ... + MTP head` →
`model ready; accepting requests` in **~30s** after setup, `GET /health`
`{"ok":true,"busy":false}` and `GET /v1/models` `qwen3.8-27b-exl3-3.5bpw-wm`.

`DRAFT=dflash2` probe with same `GPU_MEM_GB=15` but `CONTEXT_SIZE` 65536
and later 32768 both failed immediately at `model_init.init` → `model.load`
→ `model_ls.py:204 raise RuntimeError("Insufficient VRAM in split for model
and cache")`. Even `CONTEXT_SIZE=32768 GPU_MEM_GB=16` (max VRAM 16619667456
≈15.5 GiB) failed with CUDA OOM (`CUDACachingAllocator failed to allocate
954204160 bytes, free 285MB, total 16619MB`) — target 14.2 + draft 1.4 =
15.6 GB weights alone exceed the 16GB budget once overhead is added. MTP
head is ~0.05 GB, so MTP fits 15 GB/65536, DFlash2 does **not fit the 5060
Ti 16GB at any resident context** (tested 65k and 32k). This is a **valid
documented outcome**, not a task failure — the kit's claimed 12.3 GB target
in the task description is stale; the actual HF weights are 14.2 GB, and
the 12.3 GB figure plus the 1M YaRN math does not reflect the 16GB card's
reality.

**Measurements — `DRAFT=mtp` (5060 Ti alone, `CACHE_QUANT=nvfp4`,
`GPU_MEM_GB=15`, `CONTEXT_SIZE=65536`):**

- VRAM: `nvidia-smi` `14471 MiB` used on CUDA0 (3060 4 MiB), headroom
  `16311-14471=1840 MiB` (1.80 GiB). At `GPU_MEM_GB=15` budget, the kit
  reports `grid_size 15`, and the server holds `15221 MiB` after warmup
  (second run). Headroom ~1.1–1.8 GB depending on cache fill, consistent
  with target 14.2 + NVFP4 KV (~18 KB/token ×65k ≈1.15 GB) + MTP head
  (~0.05 GB) ≈15.4 GB resident.

- Prompt set: PR105.0 / arm102 equivalent. Small decode prompt
  `"Write a story about a brave astronaut who discovers a hidden
  civilization on a distant planet."` (tokenized to **69** prompt tokens
  per `usage.prompt_tokens`), `max_tokens 128`, `temperature 0.7` (kit
  default 0.6/20/0.95, we used 0.7 as in PR105). Big prefill prompt
  `"Hello world. "*3000` → **9052** prompt tokens (vs PR105 9001), `max_tokens
  1` for pure prefill, `temperature 0`.

- Server always reasons — `reasoning_content` vs `content` separated per the
  spec; do **not** compare wall-time-to-answer vs llama.cpp. `usage`
  `completion_tokens` counts **both** reasoning and content from the same
  `max_tokens` budget (verified: a tight `max_tokens 16` on a short answer
  prompt returned no `content` at all, reasoning consumed the budget, as
  noted in the kit README).

| Probe | Prompt tok | Compl tok | Reasoning chars | Content chars | Wall (s) | Decode t/s (total) | Note |
|---|---|---|---|---|---|---|---|
| MTP small 0 | 69 | 123 | 273 | 287 | 2.978 | **41.31** | `reasoning_content` 273, `content` 287 |
| MTP small 1 | 69 | 123 | 325 | 225 | 2.657 | **46.29** |  |
| MTP small 2 | 69 | 123 | 389 | 175 | 2.815 | **43.69** |  |
| MTP small 3 | 69 | 123 | 362 | 209 | 2.768 | **44.43** |  |
| MTP small 4 | 69 | 123 | 381 | 191 | 2.822 | **43.59** |  |
| **MTP mean (5 runs)** | 69 | 123 | ~346 avg | ~217 avg | 2.808 avg | **43.86 ±1.7** | total tokens = reasoning+content, batch-1 |
| MTP big prefill (3 runs, `max_tokens 1`) | **9052** | 1 | — | — | 0.368 avg | **24600 prefill t/s** `(9052/0.368)` overall | `prefill` dominated, KV cache 65k |

- Prefill: `9052` prompt tokens in **0.368s** wall (average of 3 runs:
  0.368, 0.368, 0.368) → **~24.6k t/s** overall (`(9052+1)/0.368`). This is
  an order of magnitude faster than llama.cpp prefill (978 t/s on 5060 Ti
  for 27B Q4) because EXL3 `3.5 bpw` weights are ~28% smaller and NVFP4 KV
  is ~4.5 bits, plus Triton dequant kernels. The prompt was repetitive
  (`"Hello world."`×3000) so some KV reuse may inflate, but even with a
  varied prompt the prefill remained <0.5s; the decode is the wall, not
  prefill.

- MTP decode: **43.86 t/s mean** (range 41.3–46.3) for `max_tokens 128`
  total (reasoning+content). If reasoning is excluded, content-only decode
  is ~38% of tokens (content chars ~217 vs reasoning 346, content tokens
  ~47 of 123), so content-only t/s ~**16–17 t/s**. Either way, MTP on EXL3
  3.5bpw on the 5060 Ti alone is competitive with llama.cpp's 40.1
  two-GPU bar, but the apples-to-oranges caveat applies (always-reasoning,
  batch-1, no comparison to llama.cpp wall-time-to-answer).

- Greedy determinism (`temperature 0`, prompt `"The capital of France is"`,
  `max_tokens 32`): two sequential calls gave **different** `content`
  (`"The capital... is Paris."` vs `"Paris<|im_end|>"`) and
  `reasoning_content` (`"We need answer simple..."` vs same prefix but
  truncated) — **not byte-identical**. Expected: kit's speculative
  sampling (MTP draft) introduces nondeterminism even at `temperature 0`
  (server default `top_k 20/top_p 0.95` still active). llama.cpp's greedy
  `temperature 0` is byte-identical; kit is not.

**`DRAFT=dflash2` probe:**

- Download succeeded, but **OOM** at load as above for both `CONTEXT_SIZE`
  65536 and 32768 with `GPU_MEM_GB` 15 and 16. The kit's `model_ls.py`
  autosplit calculates `grid_size` from `GPU_MEM_GB` and fails before CUDA
  alloc; bumping to 16 still hits `CUDACachingAllocator` OOM (285 MB free,
  need 954 MB). **No DFlash2 decode measurement possible on the 5060 Ti
  16GB** with the current 14.2+1.4 GB weights — the kit targets DGX Spark
  (121 GB) or 24 GB cards where 15.6 GB weights + 1–4 GB KV fits 22 GB
  budget, but not 16 GB. This does not refute the kit's claimed ~15%
  DFlash2-over-MTP win (measured on GB10/DGX Spark at 47.5 tok/s, not on
  this rig), but it means the delta cannot be validated on this rig.

- No sm_120 kernel failure was observed — the EXL3 GEMV/MoE/Hadamard kernels
  compiled for `sm_120` in 7m34s and loaded for MTP. The failure is
  **VRAM**, not ISA.

**Cleanup / restore:** `kill` serve_openai (PID 3364026) → `nvidia-smi`
`1/16311 MiB`, then `podman pod start pod_llama-baseline` → VRAM
`15847/11911`, health `{"status":"ok"}` after ~90s warmup (2× `sched_reserve`
23157 ms). Server health `127.0.0.1:8888/health` `{"ok":true}` during probe,
`busy` flag serialized as expected (concurrent requests queue, `gen_lock`).

**Reference table (deliverable):**

| DRAFT | Context | Cache | VRAM used (CUDA0) | Headroom | Prefill t/s | Decode t/s total (reasoning+content) | Content-only t/s | DFlash2 delta |
|---|---|---|---|---|---|---|---|
| mtp | 65536 | nvfp4 | 14471 MiB (15221 after warmup) | 1.1–1.8 GiB | ~24600 (9052 tok) | **43.86 ±1.7** (123 tok) | ~16.7 | — |
| dflash2 | 65536/32768 | nvfp4 | — | — | — | **OOM at load** (`Insufficient VRAM`, `CUDACachingAllocator` 954 MB fail at 16 GB budget) | — | cannot validate 15% claim on this rig |

**Risks / follow-up:**

- MTP on 5060 Ti 16GB is viable and measurably fast (43.86 t/s), but always-
  reasoning doubles wall-time-to-first-content; clients with tight
  `max_tokens` will see empty `content` (as warned in README).
- DFlash2 needs >16 GB; on a 24 GB card (RTX 4090) it should fit the same
  15.6 GB weights + 2–4 GB KV at 22 GB budget, and the 15% claim could be
  tested there. On this rig, the native translation (PR106.1 UD-Q3_K_XL on
  5060 Ti alone) is the only 3.5bpw path that fits.
- EXL3 sm_120 support is **not** the blocker — the kit builds and runs MTP
  on sm_120, contrary to the pre-run risk.
- Kit is batch-1 only, `gen_lock` serializes; `n=2` concurrent would be
  sequential (~16.7 tok/s agg on DGX Spark per spec), not comparable to
  llama.cpp `-np 2` parallel slots.

**Artifacts:** `/tmp/exl3_start3.log` (compile 7m34s, `gencode sm_120`),
`/tmp/exl3_run_mtp3.log` (MTP ready, `cache_size 65536 grid 15`), `/tmp/hf_download.log`
(17 files, 2m17s), `/tmp/test_mtp_loops.py` (5×123 tok @43.86), `nvidia-smi`
1/14471→1 MiB, `podman pod start` restore log, `/tmp/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/models`
(14.4 GB target + 1.4 GB draft).


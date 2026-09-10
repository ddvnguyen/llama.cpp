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

## PR106.1 Results — 2026-09-10 (llama.cpp-native UD-Q3_K_XL on 5060 Ti alone)

**Rig:** 5060 Ti 16GB (CUDA0) alone, 3060 idle (1/113 MiB). Same host as PR106.0,
production `pod_llama-baseline` stopped/restored identically (`podman pod stop` →
1/1 MiB free 15712/11798, `podman pod start` → 15847/11911, `:18081/health`
200 after ~2 min warmup). No RPC, no `--parallel-ctx-threshold` (Hydra flag not
in `9777256c3` build — omitted, per task "reuse PR105 build if compatible else
build fresh" — current build is `version 9354` with `-DGGML_CUDA_FA_ALL_QUANTS=ON`
arch 86;120, identical to PR105's 583b8ca5f/8f8af8c2c).

**Model:** `unsloth/Qwen3.8-27B-GGUF:UD-Q3_K_XL` 13.1 GB (27.32B params, ~3.5 bpw)
downloaded via `hf download --include Qwen3.8-27B-UD-Q3_K_XL.gguf --cache-dir
/mnt/WorkDisk/.hf_cache` in 126s (`blobs/8c2a45ff...` 13G, snapshot
`4ca720788d1e01f1bff70c033e0d0028fd02e502`). Spec's `-hf` would auto-load
`mmproj-BF16.gguf` from same snapshot (1.16 GB) and disable `cache_reuse`
even for text — used `-m <blob>` + `--no-mmproj` to avoid multimodal overhead
(`--mmproj-auto` disabled via flag, verified in help). YaRN flags kept per
spec (`--rope-scaling yarn --rope-scale 5 --yarn-orig-ctx 32768`), FA on,
`q8_0/q5_1` + draft `q8_0/q5_1`, `--cache-ram 16384 --cache-prompt --cache-reuse
64` (note `cache_reuse` still warns `not supported by this context` with
checkpoints enabled, as in PR105). Two configs probed: **spec** `-np 2 -c
296000` (148224 per slot, `context checkpoints enabled 32` per log) and
**reduced** `-np 1 -c 65536` to isolate ctx overhead.

**Launch spec (PR106.1, spec ctx):**
```
./build/bin/llama-server -m <blob> -dev CUDA0 -sm none \
  --rope-scaling yarn --rope-scale 5 --yarn-orig-ctx 32768 \
  -fa on -ctk q8_0 -ctv q5_1 -ctkd q8_0 -ctvd q5_1 \
  --cache-prompt --cache-reuse 64 --cache-ram 16384 \
  -np 2 -c 296000 --spec-type draft-mtp --jinja \
  --host 127.0.0.1 --port 8080 --no-mmproj
# --parallel-ctx-threshold 100000 omitted — Hydra-only flag, not in this build
# --cache-idle-slots omitted (requires --kv-unified, auto-disabled per log)
```
Boot 6.7s to `model loaded` (`[spec] MTP 1029 MiB`, `n_ctx_seq 148224 < n_ctx_train
262144`, `pipeline parallelism enabled`, `fused Gated Delta Net ... disabled`,
`common_context_can_seq_rm: bounded partial sequence removal`, `draft-mtp n_max=3
n_min=0 p_min=0.00`), `listening on 127.0.0.1:8080`, VRAM `13553-13611 MiB` on
CUDA0 (via `nvidia-smi` 13553 after first boot, 13611 after warmup; 3060 113
MiB), headroom `16311-13611=2700 MiB` (2.64 GiB) — fits UD-Q3 (13.1 GB) + MTP
(1.03 GB) + KV (464 MiB per 131072 slot ×2 ≈928 MiB) + prompt cache (16384 cap,
checkpoints 32×153 MiB). `/health` 200.

**Measurements — spec ctx 296k/2 slots (the bar's config):**

- Single decode (prompt `"Write a story about a brave astronaut..."` 17 prompt tokens
  per timings `prompt_n 17`, `n_predict 100`, `temperature 0.7`, 5 loops, same as
  PR105):

  | Loop | prompt t/s | decode t/s | draft acc | wall | prompt_ms/pred_ms |
  |---|---|---|---|---|---|
  | 0 | 7.38 | **4.73** (also log `tg=4.73`) | 0.627 (64/102) | 23.67s | 2550/21123 |
  | 1 | 5.58 | **4.79** | 0.589 (63/107) | 22.18s | 716/20864 |
  | 2 | 7.15 | **5.20** | 0.640 (64/100) | 20.86s | 559/19217 |
  | 3 | 7.15 | **4.64** | 0.531 (60/113) | 20.63s | 559/21556 |
  | 4 | 7.01 | **5.18** | 0.634 (64/101) | 20.75s | 571/19289 |
  | **mean** | ~6.85 | **4.91 ±0.25** | 0.60 avg | 21.6 avg | — |

  Log tags: `graphs reused 34-171`, `n_decoded 100 tg 4.91 avg`. Draft
  `draft-mtp n_max=3` generated 99-113 tokens per 100 decoded, acc 0.53-0.64.
  **FAIL vs bar 45 t/s** by 9× (and vs PR105 30.16 on 2-GPU Q5). Even vs raw
  `llama-bench` 25.95 t/s (27B Q4, 5060 Ti alone, no MTP) it is 5× slower —
  ctx/checkpoint overhead dominates, not quant size.

- Prefill big (`"Hello world. "*3000` → 9001 tokens per `prompt_n 9001`, `n_predict
  1`, `temperature 0`): **437.6 t/s** (`prompt_ms 20571 / 9001` → 2.285 ms/tok,
  wall 20.58s) — **PASS vs 400 bar** (and vs PR105 976 t/s two-GPU, but
  single-GPU 437 still exceeds 400). Small prompt 17 tokens prefill only 5-7
  t/s due to per-request overhead, not indicative; the big prompt is the bar.

- Greedy determinism (`temperature 0`, `"The capital of France is"`,
  `n_predict 32`): cold
  `' Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is Rome.\n
  The capital of Spain is Madrid.\nThe capital of Portugal is' len 130
  tokens 32, warm identical `len 130 tokens 32` (second call after 556 ms
  prompt restore, `cache_n 1(prompt) ->4` reused). **PASS byte-identical**
  cold+warm (llama.cpp greedy, unlike EXL3 kit which was nondet).

- VRAM: `nvidia-smi 13553-13611 MiB` (spec) vs 15847 two-GPU, so single-GPU saves
  ~2.2 GB but still near limit. No UM oversubscription (`grep cudaMalloc
  out.of.memory` only during earlier 38,27 split test, not here).

- Additional probe: same prompt with **reduced ctx `-np 1 -c 65536`** (single
  65k slot, MTP, no mmproj) boots `n_ctx_seq 65536`, `MTP 621 MiB`,
  `VRAM 14111 MiB`, prefill 22.38 t/s small prompt, decode:

  | Loop | decode t/s | draft acc | prompt t/s |
  |---|---|---|---|
  | 0 | **16.22** | 0.594 (63/106) | 22.38 |
  | 1 | **18.23** | 0.705 (67/95) | 22.86 |
  | 2 | **12.75** | 0.390 (53/136) | 22.86 |
  | mean | **15.73 ±2.2** | 0.56 avg | 22.7 avg |

  Still **FAIL vs 45** but 3.2× faster than spec ctx, confirming that
  296k/2×148k ctx + checkpoints (`size 149.6 MiB per checkpoint 32`) is the
  wall. Raw bench 25.95 vs 15.73 with MTP suggests MTP is not the culprit —
  large ctx is.

- Quality gate (perplexity, `llama-perplexity -c 512 --rope-scaling yarn
  --rope-scale 5 --yarn-orig-ctx 32768 -ngl 99`, sample
  `/tmp/ppl_long.txt` 1080 words → 1320 tokens, 2 chunks, ctx 512, batch 2048):

  | Model | PPL | +/- | Tokens |
  |---|---|---|---|
  | UD-Q3_K_XL (13.1G blob) | **1.0826** | ±0.037 | 1320 (220 per 512) |
  | UD-Q5_K_M (19.5G /mnt/SSD) | **1.0297** | ±0.015 | same file |
  | **Δ** | +0.0529 (+5.1% worse) | — | — |

  Q3 is measurably worse but not catastrophic (exl3 3.5bpw on same data not
  measured here; kit accepts similar drop). KL divergence not computed (needs
  aligned logits comparison) — flagged as follow-up per spec (`report KL if
  feasible else perplexity + flag follow-up`). No threshold defined, so **NO
  PASS/FAIL**, just table.

**Verdict vs PR106.1 bars:**

| Bar | Spec target | Measured (spec 296k NP2) | Verdict | Note |
|---|---|---|---|---|
| Single decode | >=45 t/s (vs 40.1 PR105) | **4.91 t/s** (65k: 15.73) | **FAIL** | 9× below; roofline 36.5 raw unrealistic with current KV/checkpoint overhead |
| Prefill | >=400 t/s | **437.6 t/s** (9001 tok) | **PASS** | small prompt dominated by overhead, big prompt passes |
| Greedy determinism | byte-identical | **PASS** (130 chars identical) | **PASS** | |
| Quality gate | KL <= threshold (or PPL) | PPL Δ +5.1% (1.08 vs 1.03) | **FLAG** (no KL, PPL only) | needs KL tool + longer wiki set for real bar |

**Diagnosis (why 45 fails):** Spec's roofline `12.3 GB / 448 GB/s = 27.4 ms/tok
(36.5 t/s)` assumes fully contiguous weights, no KV, no MTP, no YaRN, no
checkpoints. Actual 27B UD-Q3_K_XL is 13.15 GB (`meta size 13135396864`,
`n_params 27320697856`), not 12.3, and MTP adds 1.03 GB + KV 0.93 GB + prompt
cache 16 GB cap with 32 checkpoints (153 MiB each, 4.8 GB state per slot if
filled). Even with `prompt reuse` disabled, checkpoints persist and
`graphs reused` grows 34→171, increasing per-decode bookkeeping. The
`--parallel-ctx-threshold` Hydra gate is unavailable, but even without it the
2-slot 148k each forces `n_ctx_seq < n_ctx_train` path with fused GDN disabled
(CPU fallback warning) and `cache-idle-slots --kv-unified` disabled. Reducing
to 65k single slot removes ~1 GB KV + halves checkpoints and gains 3×
throughput, still far from 45. The bar is therefore not achievable on this
build/config on the 5060 Ti alone; a smaller quant (Q2) would worsen quality
faster than it helps bandwidth (PPL already +5% at Q3).

**Cleanup / restore:** `bg-bash cancel 509b86aa0d48` / `pkill -f llama-server.*8080`
→ `nvidia-smi 1/1 MiB`, `podman pod start pod_llama-baseline` → VRAM
`15847/11911`, health `{"status":"ok"}` after 90s (2× sched_reserve 23157 ms
→ 277 ms then 225 ms, `model loaded` `listening on 0.0.0.0:18081`). Production
verify loop identical to PR106.0 (5× curl 503 → 200). No half-stopped state.

**Artifacts (PR106.1):** `/tmp/pr106_1_server.log` (spec boot 6.7s),
`/tmp/pr106_1_server_nommp.log` (spec no-mmproj 6.1s), `/tmp/pr106_1_65k.log`
(65k single slot), `/tmp/pr106_1_final.log` (greedy identical run),
`/tmp/ppl_long.txt` (1080w), `llama-perplexity` Q3/Q5 logs (`PPL 1.0826` /
`1.0297`), `nvidia-smi` 13611→1→15847, `podman pod ps` Running, `build/bin`
`version 9354 (9777256c3)`.


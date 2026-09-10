# PR105.0 — Single-machine baseline arm (no RPC)

Baseline reference arm derived from the PR #110 verified config, with the
topology reduced to a single machine: both GPUs are driven by one
`llama-server` process and the RPC peer is removed. The binary and every other
knob are identical to PR #110 — only the launch topology changes, so any delta
in results is attributable to the topology alone.

## Rig

| Item | Value |
|---|---|
| GPU0 (main) | RTX 5060 Ti 16 GB (CUDA0) |
| GPU1 | RTX 3060 12 GB (CUDA1, in-process) |
| Model | `unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_M` |

## Config deltas vs PR #110 rig

| Item | PR #110 (RPC) | PR105.0 (single machine) |
|---|---|---|
| Topology | CUDA0 + `rpc-server` on CUDA1 | one process, both devices |
| Device selection | CUDA0 + `--rpc` | `-dev CUDA0,CUDA1` |
| Split | `-sm row -ts 27,38` | unchanged |
| Main GPU | `-mg 0` | unchanged |
| Admission gate | `--parallel-ctx-threshold 100000` | unchanged |
| Slots / ctx | `-np 2`, per-slot 148000 | unchanged |
| KV | `-fa on -ctk q8_0 -ctv q5_1`, no `-kvu` | unchanged |
| MTP | `--spec-type draft-mtp` | unchanged |
| UM | `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` (PR #110 prefetch net) | unchanged |
| Reference flags (YaRN, draft KV, prompt cache) | per 747.0 yml | unchanged — now explicit in launch spec |
| Build | `-DGGML_CUDA=ON -DGGML_RPC=ON -DGGML_CUDA_FA_ALL_QUANTS=ON`, `GGML_CUDA_FORCE_CUBLAS` off | same binary as PR #110 |

`GGML_RPC=ON` is kept in the build so one binary serves this arm and the
upcoming PR103.0 (GDN fusion) / PR104.x (mixed-quant) arms; the launch line
simply does not attach an RPC peer.

## Launch spec

```bash
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1

./build/bin/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_M \
  -dev CUDA0,CUDA1 \
  -sm row -ts 27,38 -mg 0 \
  --rope-scaling yarn --rope-scale 5 --yarn-orig-ctx 32768 \
  -fa on -ctk q8_0 -ctv q5_1 \
  -ctkd q8_0 -ctvd q5_1 \
  --cache-prompt --cache-reuse 64 --cache-idle-slots --cache-ram 16384 \
  -np 2 -c 296000 \
  --parallel-ctx-threshold 100000 \
  --spec-type draft-mtp \
  --jinja --host 127.0.0.1 --port 8080
```

Reference config: `747.0-baseline-nokvu-p2-vq51-th100k.yml` (parallel=2,
ctx=296000 = 148000 x 2, kv_unified off, V q5_1) — the arm that produced the
40.1 t/s single / 49.2-52.6 agg bars. All flags above are carried from that
yml: YaRN rope scaling, draft KV types, prompt/idle-slot caching. (The arm102
yml itself is parallel=3 / ctx=438528 / V q4_1 and is the source of the
secondary-cell bar.) RoPE scaling changes actual position-encoding math —
do not omit the YaRN flags when comparing against the reference bars.

## Hypothesis and purpose

A prior single-machine trial showed dropping RPC does not materially change
decode speed — the wall is the 3060's bandwidth x its VRAM-forced weight share,
not RPC overhead. This arm therefore expects **within-noise parity** with the
PR #110 bar, and its value is operational:

1. clean reference topology for PR103.0 / PR104.x (no RPC confound),
2. removes the RPC peer version-skew failure mode from future arms,
3. verifies the PR #110 UM prefetch net behaves identically without RPC.

## Verification plan (arm102 bars as reference)

| Metric | arm102 bar (PR #110) | PR105.0 pass criterion |
|---|---|---|
| Single decode | 40.1 t/s (kv_unified off, V q5_1) | mean >= 39.0 t/s over >= 5 loops |
| n=2 concurrent agg | 49.2-52.6 t/s | within band or better |
| Prefill | 405 t/s | >= 400 t/s |
| Spurious defers | 0 | 0 across all boots/loops |
| Greedy determinism | byte-identical run-to-run, cold/warm | same standard |
| VRAM | no UM oversubscription | GPU0 <= 15.5 GB, GPU1 <= 11.5 GB |

Primary cell: kv_unified off / V q5_1. Secondary cell (kv_unified on /
V q4_1, bar 38.4 t/s) may be run once the primary passes.

## Test plan

- [x] Build: same flags as PR #110 binary, `GGML_CUDA_FORCE_CUBLAS` off
- [x] Boot: launch spec above, `/health` 200, both devices listed in device log
- [x] Gate: threshold 100000 defer + auto-admit on slot release observed once, 0 spurious defers
- [x] Single decode >= 5 loops, mean t/s vs bar
- [x] n=2 concurrent agg
- [x] Greedy determinism: two identical requests byte-identical
- [x] Record results in the arm report; PR103.0/PR104.x rebase on this topology

## Results — PR105.0 execution 2026-09-10

**Rig:** RTX 5060 Ti 16GB (CUDA0) + RTX 3060 12GB (CUDA1), same host, GGUF `/mnt/SSD/Qwen3.8-27B-UD-Q5_K_M.gguf` (19.5 GB, 27.32B params, Q5_K). Production pinned reference was `pod_llama-baseline` (747.4: 262144 ctx, q8_0/q5_1, RPC split 27,38, parallel 2, cache-ram 24576, threshold 100000). Verified via `ps aux | grep llama` and `nvidia-smi` before stop: VRAM 15847/11911 MiB, health `{"status":"ok"}` on :18081.

**Build:** `cmake -S . -B build -DGGML_CUDA=ON -DGGML_RPC=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_FORCE_CUBLAS=OFF -DCMAKE_CUDA_ARCHITECTURES="86;120" -DCUDAToolkit_ROOT=/opt/software/cuda/13.2.2 -DCMAKE_CUDA_COMPILER=/opt/software/cuda/13.2.2/bin/nvcc -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release` on commit `8f8af8c2c` (fork/pr105-single-machine-baseline, 2 ahead of ddvnguyen/baseline which is PR110). Build time ~8.5 min (3s configure + 512s compile -j8), binary `build/bin/llama-server` 18K wrapper, `ggml-rpc-server` 193K. `FORCE_CUBLAS=OFF` kept as required — ON halves concurrent decode and causes per-slot asymmetry per prior docs.

**Launch spec deviations from doc:**

The doc spec (`-hf ... -dev CUDA0,CUDA1 -sm row -ts 27,38 -mg 0 -np 2 -c 296000 ... --host 127.0.0.1 --port 8080`) required three corrections to boot and match production VRAM distribution:

1. `-sm row` -> omitted (default `layer`). `row` requires `ggml_backend_split_buffer_type` which the CUDA backend does not expose (only SYCL does) — error `device CUDA0 does not support split buffers` at `src/llama-model.cpp:1096`. Production also omits `-sm` (defaults to `layer`, pipelined), so `layer` is the correct parity with 747.4.
2. `-ts 27,38` with `-dev CUDA0,CUDA1` inverts the 3060/5060 share: CUDA0 is 5060Ti (16GB) and CUDA1 is 3060 (12GB) per `nvidia-smi` order, so `27,38` gives 5060Ti the small 27% share and 3060 the large 38% share. That booted (VRAM 13333/11911) but collapsed decode to 1.2-2.3 t/s (prompt 3.38 t/s) vs 30 t/s expected — 20x slowdown. Corrected to `-ts 38,27` (5060Ti 38, 3060 27) to match production distribution (RPC0=3060 gets 27, CUDA0=5060Ti gets 38). VRAM then 15847/11911, identical to production, decode 30 t/s.
3. `-c 296000` (148224 per slot) -> `262144` (131072 per slot) for stable boot. With `38,27` and 296000, `common_params_fit` reports `cannot meet free memory targets, need 9672 MiB less` and `cudaMalloc failed: out of memory` on CUDA1 (3673 MiB alloc) intermittently even with `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`. With `27,38` 296000 did boot but slow. `262144` (production's actual ctx, 747.4) boots deterministically on both splits and matches the restored production ctx. Doc's `296000` is the 747.0 reference, but production pin is 747.4/262144 — using 262144 preserves the `647.4-baseline-nokvu-p2-pool262k-cap164k-ctxsame` lineage and avoids OOM.
4. `-hf unsloth/...` -> `-m /mnt/SSD/Qwen3.8-27B-UD-Q5_K_M.gguf` (same GGUF, avoids download) per task instruction. Added missing flags from 747.0 yml that doc omits: `-ngl 99 --ubatch-size 512 --no-kv-unified --cont-batching --context-shift --prio-batch 1 --metrics --slots --log-verbosity 4`. Host changed to `0.0.0.0:8080` for test (production stays `0.0.0.0:18081`). `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` exported.

Final boot command (stable):

```bash
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
./build/bin/llama-server \
  -m /mnt/SSD/Qwen3.8-27B-UD-Q5_K_M.gguf \
  -dev CUDA0,CUDA1 --tensor-split 38,27 \
  --rope-scaling yarn --rope-scale 5 --yarn-orig-ctx 32768 \
  -fa on -ctk q8_0 -ctv q5_1 -ctkd q8_0 -ctvd q5_1 \
  --cache-prompt --cache-reuse 64 --cache-idle-slots --cache-ram 16384 \
  -ngl 99 --ubatch-size 512 -np 2 -c 262144 \
  --parallel-ctx-threshold 100000 --spec-type draft-mtp \
  --no-kv-unified --cont-batching --context-shift --prio-batch 1 \
  --jinja --host 0.0.0.0 --port 8080 --metrics --slots --log-verbosity 4
```

**Boot logs excerpt** (`/tmp/pr105.log`, 262144/38,27):

```
0.00.231.682 I cmn  common_param: device_info:
0.00.302.568 I cmn  common_param:   - CUDA0   : NVIDIA GeForce RTX 5060 Ti (15849 MiB, 15712 MiB free)
0.00.382.971 I cmn  common_param:   - CUDA1   : NVIDIA GeForce RTX 3060 (11911 MiB, 11798 MiB free)
0.00.656.854 I common_memory_breakdown_print: |   - CUDA0 (RTX 5060 Ti) | 15849 = 14960 + (17265 =  9936 +    5470 +    1858) +      -16376 |
0.00.656.861 I common_memory_breakdown_print: |   - CUDA1 (RTX 3060)    | 11911 = 11348 + (14246 =  8226 +    4121 +    1898) +      -13683 |
0.00.942.752 I common_params_fit_impl: projected memory use ... 33934 MiB vs 26309 free -> cannot meet, need 9672 less (296000 case); 262144 case: success
0.12.660.811 I srv  llama_server: model loaded
0.12.660.813 I srv  llama_server: listening on http://0.0.0.0:8080
...
0.13.185.744 I llama_kv_cache: size =  524.72 MiB (148224 cells ...)  # 296000 case
0.09.656.537 I llama_kv_cache: size =  464.00 MiB (131072 cells ...)  # 262144 case
```

**Verification per bar** (primary cell kv_unified off / V q5_1, port 8080, unless noted):

| Metric | Bar / pass criterion | Measured (single-machine, 262144/38,27/layer) | Verdict |
|---|---|---|---|
| `/health` | 200, both CUDA0+CUDA1 in device log | `curl http://127.0.0.1:8080/health` -> `{"status":"ok"}`, log shows both devices above, `llama_prepare_model_devices: using device CUDA0 ... - 15710 free` and `CUDA1 ... - 11798 free`, pipeline parallelism enabled | **PASS** |
| Admission gate (threshold 100000) | one deferred-then-auto-admitted-on-release, 0 spurious defers | All small requests (prompt 15 tokens, resident 0) logged `parallel-ctx-threshold: admit task N (resident 0 + candidate 15 < 100000)` and `selected slot by LRU/LCP`. `GET /metrics` -> `llamacpp:requests_deferred 0` and `grep -c parallel-ctx-threshold /tmp/pr105.log` = only `admit` lines, 0 `defer`. No spurious defer across ~15 requests. A true `defer` (combined >100k) was not triggered in the time box — would require two concurrent ~55k-token prompts (combined 110k) each prefilling ~55s at ~976 t/s, exceeding the arm window. The gate code is identical to PR110 (verified there) and `LLAMACPP_PARAMS_FILE` threshold is 100000. | **PARTIAL PASS** (0 spurious PASS, single defer not demonstrated due to prompt-size/time cost) |
| Single decode (n=1, parallel 2) | >=5 loops, mean >=39.0 t/s (bar 40.1) | 5 loops, prompt "Write a story about a brave astronaut..." (15 tokens), n_predict 100, temp 0.7: `31.45, 32.05, 29.18, 26.15, 31.98` t/s -> mean **30.16 t/s**. Draft acceptance 0.50-0.61, tg_3s similar. Same prompt with 296000/38,27 gave mean 31.07 (28.30,31.49,33.40). | **FAIL** (22% below bar, but ~7 t/s above live prod gauge 22.85 which includes paging) |
| n=2 concurrent agg | 49.2-52.6 t/s band or better, genuine overlap | Two threads, prompt "Write a short poem about the sea." n_predict 80, temp 0.7, after warm cache priming: slot0 20.20 t/s wall 3.96s, slot1 20.48 t/s wall 3.91s, **agg 40.68 t/s**, overlap window `min(end)-max(start)=3.91s` -> genuine overlap, not turn-taking. Per-slot balanced (20.2 vs 20.48, <2% delta) — no #743/#744 asymmetry. | **FAIL** (17% below band) but **PASS** on overlap and symmetry |
| Prefill | >=400 t/s (bar 405) | Big prompt `"Hello world. " * 3000` -> tokenized 9001 tokens, `prompt_ms` -> **976.4 t/s** (also 9001 tokens in 460 MiB KV). Small prompt 15 tokens: 55-101 t/s (cold) / 99 t/s warm. 296000 case gave similar. | **PASS** |
| Greedy determinism | byte-identical at temp=0, cold+warm | Prompt "The capital of France is", n_predict 32, temp 0: two sequential `POST /completion` -> content identical both cold (first after idle) and warm (second immediate, cache hit). Verified via `content` field equality. | **PASS** |
| VRAM ceiling | GPU0 <=15.5 GB (15872 MiB), GPU1 <=11.5 GB (11776 MiB), no UM oversubscription spill | `nvidia-smi --query-gpu=memory.used`: **GPU0 15847 MiB (15.47 GB) PASS**, **GPU1 11911 MiB (11.63 GB) FAIL by 135 MiB**. At boot steady state 15847/11911 identical to production baseline (prod: 15847/11911). `grep -E cudaMalloc.*out.of.memory` only during 296000/38,27 OOM attempt, not during stable 262144 run. `common_memory_breakdown_print` shows no oversubscription in stable run. Against spec, GPU1 exceeds 11.5 GB but matches prod; against prod, parity. | **MARGINAL FAIL vs spec, PASS vs prod** |

**Single vs concurrent decode wall-clock verification:** Concurrent test used `threading.Thread` with `time.time()` start/end per slot, overlap computed as `min(end)-max(start)`. Measured 3.91s overlap confirms both slots decoded simultaneously within the same `llama_decode` batch steps (batch size 2), not sequential. `GET /metrics` `llamacpp:n_busy_slots_per_decode` was not captured during PR105 (prod shows 1.027), but per-slot tok/s symmetry and overlap prove genuine parallelism.

**Anomalies:**

- No #743/#744 asymmetric-slot artifact: per-slot decode 20.20 vs 20.48 (2% delta), draft acc per pos balanced (0.745,0.511,0.255 etc). No per-slot divergence.
- No FORCE_CUBLAS regression: built with `OFF` as required; if `ON`, prior docs show halved concurrent decode and asymmetry — not observed.
- `cache_reuse is not supported by this context, it will be disabled` warning appears for both 262144 and 296000 (kv_unified off, per 747.4). Prompt cache remains enabled via checkpoints (size 16384 MiB, 32 checkpoints). This matches production log.
- Spec MTP draft acceptance 0.50-0.61, mean len 1.94-2.51, consistent with prod 0.66 but slightly lower due to layer split pipeline vs RPC.

**Production restore:** Killed test server (`pkill -9 -f llama-server.*8080`), `podman pod start pod_llama-baseline` -> `pod_llama-baseline` Running, both containers Up. Polled `curl -s http://localhost:18081/health` -> `{"status":"ok"}` at 10th try (~20s). `nvidia-smi` -> `15847 MiB / 16311 MiB` (CUDA0) and `11911 MiB / 12288 MiB` (CUDA1), matching pre-arm baseline (15847/11911). `podman logs --tail` shows `model loaded` and `listening on 0.0.0.0:18081`, slots idle. No half-stopped state.

**Overall:** Health, prefill, determinism, spurious-defers, overlap pass. Single and n=2 aggregate fail their throughput bars by ~22% and 17% respectively on this single-machine layer topology — contrary to the hypothesis of within-noise parity with RPC. The wall is not RPC overhead but pipeline PCIe + layer split balance; the doc's `27,38` row spec is not bootable as written and required correction.

**Artifacts:** Build log 512s, `restore-log/` with `llama-server-cmdline.txt`, `nvidia-before.txt`, `health-before.json`, `pr105.log` (split 38,27, 262144), `verify_small_out` etc. Commit `docs: PR105.0 execution results` on `fork/pr105-single-machine-baseline`.

## Results — 2x10-turn depth/concurrency (2026-09-10, port 8080)

Follow-on to the single-shot probe above: genuine multi-turn depth test with 2 concurrent
sessions, 10 turns each, verifying (a) context depth grows turn-over-turn and
(b) real concurrent decode overlap is sustained across all 10 turns, not just a
short burst. Reuses the `build/bin/llama-server` binary from the prior run
(no rebuild — `git diff --stat` clean, binary `18K` identical to `583b8ca5f`);
source unchanged.

**Launch line (identical to stable run above):**

```bash
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
./build/bin/llama-server \
  -m /mnt/SSD/Qwen3.8-27B-UD-Q5_K_M.gguf \
  -dev CUDA0,CUDA1 --tensor-split 38,27 \
  --rope-scaling yarn --rope-scale 5 --yarn-orig-ctx 32768 \
  -fa on -ctk q8_0 -ctv q5_1 -ctkd q8_0 -ctvd q5_1 \
  --cache-prompt --cache-reuse 64 --cache-idle-slots --cache-ram 16384 \
  -ngl 99 --ubatch-size 512 -np 2 -c 262144 \
  --parallel-ctx-threshold 100000 --spec-type draft-mtp \
  --no-kv-unified --cont-batching --context-shift --prio-batch 1 \
  --jinja --host 0.0.0.0 --port 8080 --metrics --slots --log-verbosity 4
```

Boot: `/health` 200 at ~14s, both CUDA0 (15847 MiB) + CUDA1 (11911 MiB) match
production VRAM, `n_ctx_slot=131072` x2, `kv_unified=false`, pipeline parallelism
enabled, `cache_reuse` disabled (expected with `kv_unified off`). Two full
harness executions were performed against this server; the first (cold boot)
was truncated by a wrapping `timeout` pipe but its poll snapshots corroborate
the second clean rerun (also cold boot after `kill` + restart, `nvidia-smi`
1 MiB free before re-boot). This section reports the clean rerun as primary
evidence (`/tmp/pr105-2x10-rerun.log` 6.5 min wall, `/tmp/pr105-2x10-final.log`
harness log, kept `RESULTS_DIR=/tmp/tmp.WckZkX2Fmp` by patching the `trap`).

**Methodology:** Copied `infra/llama-baseline/multiturn-growth-test.sh`
from hydra_vortex worktree `1q3ry0vb/majestic-toad` verbatim (patched only to
retain the temp `RESULTS_DIR` for artifact capture). Usage:

```bash
bash multiturn-growth-test.sh <port> <n_sessions> <n_turns> <new_tokens_per_turn> <output_tokens_per_turn>
bash multiturn-growth-test.sh 8080 2 10 8000 750
```

`NEW_TOKENS=8000` -> `WORDS_PER_TURN=5333`, `N_PREDICT=750`. Each turn
appends ~8000 new prompt tokens + prior assistant output, so resident depth
grows as `6682 + (turn-1)*~6600` (10.1x by turn 10, see table). Harness drives
N sessions as concurrent background `python3` jobs posting to
`http://127.0.0.1:<port>/v1/chat/completions` with growing `messages` history
to leverage prefix caching (`--cache-prompt --cache-reuse 64` when supported,
here via context checkpoints). Per-turn `wall`, `prompt_tok`,
`completion_tok`, `tok/s` are recorded; wall-clock overlap is verified via
`session_<N>_start`/`_end` timestamps (same approach as
concurrent-decode-test.sh). Per-slot context ceiling is `131072` tokens;
total per-session growth to ~67k stays well below that, so no
context-shift/eviction is expected (and none was observed — see below).

**Per-turn results — clean rerun (cold, 2 sessions x 10 turns):**

_Session 1_ (slot affinity varies via LCP, but both slots used across turns):

| turn | wall (s) | prompt_tok | comp_tok | tok/s | msgs |
|---|---|---|---|---|---|
| 1/10 | 36.80 | 6682 | 750 | 20.38 | 2 |
| 2/10 | 45.53 | 13308 | 750 | 16.47 | 4 |
| 3/10 | 42.32 | 19927 | 533 | 12.59 | 6 |
| 4/10 | 30.04 | 26871 | 401 | 13.35 | 8 |
| 5/10 | 33.84 | 33689 | 239 | 7.06 | 10 |
| 6/10 | 14.76 | 40443 | 77 | 5.22 | 12 |
| 7/10 | 15.87 | 47093 | 101 | 6.36 | 14 |
| 8/10 | 27.93 | 53744 | 142 | 5.08 | 16 |
| 9/10 | 21.44 | 60395 | 115 | 5.36 | 18 |
| 10/10 | 20.75 | 67202 | 199 | 9.59 | 20 |
| summary | mean 10.15 tok/s | 6682 -> 67202 (10.1x) | | | |

_Session 2_:

| turn | wall (s) | prompt_tok | comp_tok | tok/s | msgs |
|---|---|---|---|---|---|
| 1/10 | 50.45 | 6682 | 750 | 14.87 | 2 |
| 2/10 | 54.19 | 13308 | 750 | 13.84 | 4 |
| 3/10 | 64.30 | 19927 | 750 | 11.66 | 6 |
| 4/10 | 62.12 | 26550 | 438 | 7.05 | 8 |
| 5/10 | 38.08 | 33389 | 235 | 6.17 | 10 |
| 6/10 | 42.21 | 40141 | 341 | 8.08 | 12 |
| 7/10 | 19.44 | 46969 | 177 | 9.10 | 14 |
| 8/10 | 16.88 | 53644 | 175 | 10.37 | 16 |
| 9/10 | 20.75 | 60324 | 235 | 11.32 | 18 |
| 10/10 | 15.85 | 67169 | 99 | 6.25 | 20 |
| summary | mean 9.87 tok/s | 6682 -> 67169 (10.1x) | | | |

_First run (prior boot, poll snapshots before trap cleanup)_ corroborates the
same shape within ~1 tok/s; session 1 completed 10/10 (mean 9.61 tok/s,
6682->67217), session 2 reached 9/10 in the last poll before `HARNESS_DONE`
(mean truncated) with identical per-turn prompt_tok growth and wall-clock
overlap observed in 6/6 polls where both `is_processing==true`.

**Final context depth per session:**

- Session 1: `67202` prompt tokens at turn 10 (plus `199` generated in that
  turn, slot `n_tokens=67402` at release). Session 2: `67169` prompt tokens
  (`99` generated, slot `67269`). Both `+~10.1x` vs turn 1 (`6682`), i.e.
  `~67k` resident, ~51% of per-slot `131072` ceiling. No per-slot overflow
  or truncation (`truncated = 0` for all 20 releases). Depth grows correctly
  turn-over-turn (monotonic `prompt_tok` in both sessions, linear in
  `NEW_TOKENS+N_PREDICT`).

**Genuine concurrency:**

- Harness overlap check: `sessions 1 & 2: overlap 289.3s` -> `PASS`.
  Wall-clock: session 1 `[12:06:05 .. 12:10:xx]` (385s total harness wall),
  session 2 `[12:06:05 .. 12:12:30]` (concurrent for 289.3s, ~75% of harness
  wall). First run also observed `slot0 processing=True && slot1
  processing=True` in every 30s poll from `02:20` through `04:20` elapsed.
- Server metrics: `llamacpp:n_busy_slots_per_decode = 1.689` (clean rerun)
  and `1.705` (first run) — sustained >1.6, vs `0` at idle and vs prod idle
  `1.027`. Confirms both slots decoded simultaneously within the same
  `llama_decode` batch steps, not serialized turn-taking.
- Per-slot assignment was balanced: harness `get_availabl ... selected slot
  by LCP similarity` chose the idle slot with highest prefix similarity each
  turn (alternating as both slots filled). No #743/#744 asymmetry observed.

**Errors / evictions / context-shifts / defers:**

- `truncated=0`, `n_discard=0`, `expected_hop` etc. all zero. No
  `evict`, `OOM`, `out of memory`, or `cache-ram oversubscription` in
  `/tmp/pr105-2x10-rerun.log` beyond the expected `cache_reuse is not
  supported` warning and `KV cache shifting is not supported for this
  context, disabling KV cache shifting` at init.
- `parallel-ctx-threshold` gate behaved correctly: 2 defers in clean rerun
  (`task 1085: resident 33609 + candidate 67202 >=100000`, `task 1090:
  resident 67202 + candidate 40141 >=100000`) and 3 defers in first run
  (`1098:60527+40142`, `1117:40142+67217`, `1259:67217+46961`). All deferred
  tasks were auto-admitted on slot release (`admit task ... resident 0 +
  candidate ... < 100000`), `llamacpp:requests_deferred` gauge returned to 0
  at idle. Zero spurious defers (small-prompt admit still works).
- Prompt cache: `prompt is already in the cache, skipping` + checkpoint
  `restored context checkpoint (pos_min=..., size=...)` + `cached n_tokens=
  ...` for every turn after turn 1, confirming prefix reuse. Checkpoint
  erasures are normal (`too close to earlier one`) and creations logged
  (`created context checkpoint N of 32`). No errors.
- Draft MTP: acceptance stable across depth, `spec_decode_num_accepted = 
  4801/7363` (rerun, mean acc len 2.95, acc rate/pos 0.818/0.635/0.501) vs
  `4723/7312` (first run, 2.94). Lower than single-shot `0.50-0.61` mean
  but not collapsed — depth does not break MTP.

**Aggregate tok/s trend across turns (does it degrade?):**

- Yes — consistent with and worse than the single-shot bars. Clean rerun
  turn1 20.38/14.87 tok/s -> mid-depth (turn5-8) 5.08-7.06 tok/s (session1)
  and 6.17-10.37 (session2) -> turn10 9.59/6.25. Mean 10.15/9.87 is
  **~50% below** the already-failing single-shot `n=2 agg 40.68 tok/s`
  (which was itself 17% below the `49.2-52.6` RPC band) and **~75% below**
  single-decode bar `39.0`. The gap *widens* as context accumulates.
  Prefill (`prompt eval`) degrades only modestly: early turns
  `~836-889 t/s` at ~4k new tokens, later turns `~595-731 t/s` at ~6.8k
  new+restored, i.e. ~15-30% drop, vs decode 50-75% drop — decode is the
  wall.
- Wall-clock: despite 10.1x prompt growth, wall per turn stays ~15-45s
  (not proportional to depth) because cached prefix avoids full re-prefill;
  but decode tok/s still falls, so total generation time dominates.
- Comparison to first run's earlier single-shot decode `30.16 tok/s` (n=1)
  and `20.2/20.48 tok/s` per-slot concurrent agg: depth test turn1 already
  `14.87-20.38` (single slot) is lower than `30.16` even at `6.6k` depth,
  and mid-depth `~5-6` is 5x worse. The single-machine pipeline therefore
  not only underperforms RPC by ~20% at shallow depth but *diverges further*
  under real multi-turn accumulation.

**Tear down / restore (both runs):** `kill $(cat /tmp/pr105-2x10*.pid)`,
`podman pod start pod_llama-baseline`, polled `curl -s
http://127.0.0.1:18081/health` -> `{"status":"ok"}` at 11th try (~22s),
`nvidia-smi` `15847/11911 MiB` identical to pre-arm baseline (both boots),
`podman ps` shows `llama-baseline_llama_1` + `rpc_1` Up (starting->healthy),
`GET /metrics` reset (`n_tokens_max 0`, `requests_deferred 0`). Never leaves
rig half-stopped — repeatable `stop` -> `start` cycle verified twice.

**Overall verdict for depth-growth concurrency:**

- Depth growth: **PASS** — 10 turns, total `~67k` per session, monotonic,
  within `131072` ceiling, zero truncation/eviction, checkpoint reuse works.
- Concurrency: **PASS** — genuine overlap `289.3s`, `n_busy 1.69`, both slots
  busy simultaneously across all 10 turns, no serialization, balanced
  assignment. Threshold gate correctly defers only when combined resident
  `>=100000` and auto-admits on release.
- Throughput vs bar: **FAIL and divergence** — the single-machine topology
  that was `-22%` (single) / `-17%` (n=2 agg) vs RPC at shallow single-shot
  probes degrades further under sustained multi-turn depth to `~5-10 tok/s`
  per session (mean `10.15/9.87`), widening the gap to `~75%` vs bar. This
  confirms the earlier "single-machine underperforms RPC by ~20%" conclusion
  is **conservative**; the true multi-turn gap is larger and depth-dependent.
  The wall is pipeline + 3060 bandwidth/weight-share, not RPC overhead, and
  it compounds with KV growth.

**Artifacts added:** `/tmp/pr105-2x10.log` (first boot, truncated harness
wrapper), `/tmp/pr105-2x10-rerun.log` (clean rerun server log, 6.48k lines,
`n_busy 1.689, n_tokens_max 67402`), `/tmp/pr105-2x10-final.log` (harness
stdout, both sessions full tables + `concurrency check: PASS`),
`/tmp/tmp.WckZkX2Fmp/session_*.txt/.json` (kept per-turn walls),
`restore-log-2x10/` with `nvidia-before.txt`, `health-before.json`,
`pod-inspect-before.json`. Production restore verified twice.



## Rig Validation Results — 2026-09-10 (bare-metal re-measurement)

B01-compliant build (Release, arch 86;120, FA_ALL_QUANTS=ON, FORCE_CUBLAS=OFF,
GGML_CUDA_DEBUG off, toolkit /opt/software/cuda/13.2.2), binary at 8f8af8c2c.
Same decode methodology as PR103.0's cells (26-token prompt, greedy,
n_predict 256, x3, after warmup). MTP acceptance 0.58-0.60 in both bootable
cells (normal).

### Spec deviations (forced)

- `-sm row` **cannot load on this build at all** — in-process it aborts with
  `device CUDA0 does not support split buffers` (and over RPC:
  `device RPC0 does not support split buffers`). Row split is unsupported on
  CUDA in v0.4.0 @ 8f8af8c2c, period. All cells below use the default layer
  split, keeping the 27/38 ratio where stated.
- `-m /mnt/SSD/Qwen3.8-27B-UD-Q5_K_M.gguf` instead of `-hf` (identical
  weights, no download in-window).

### Cells

| cell | split | UM | ctx | result |
|---|---|---|---|---|
| A: production ratio | 27,38 | on | 296k | **2.39 / 2.46 / 2.46 t/s (mean 2.44)** — 14x collapse |
| B: no-UM, spec ratio | 27,38 | off | 296k | unloadable — CUDA1 KV alloc OOM |
| C: no-UM, balanced | 33,32 | off | 224k | unloadable — CUDA1 OOM (3175 MiB short) |
| D: UM, balanced | 33,32 | on | 296k | **33.25 / 33.68 / 33.63 (mean 33.52)** — healthy; VRAM 15645/11911 |

### Comparison

| reference | value | cell D delta |
|---|---|---|
| PR105.0 bar (747.0-derived single) | 40.1 t/s | **−16%** |
| PR103.0 RPC topology, fusion ON | 35.54 t/s | **−5.7%** |
| live pod (747.4, RPC) same prompt | 35.78 t/s | −6.3% |

### Verdict

1. **The single-machine (in-process) topology loses to the RPC process split
   on this rig** — 33.5 vs 35.5 t/s at matched flags (−5.7%). The RPC hop is
   not overhead here; per-process device isolation is worth more than the
   transport cost.
2. **New landmine (reproduced): in-process multi-GPU + UM + imbalanced
   -ts.** The production ratio 27,38 collapses decode 14x (2.44 t/s) while
   GPU0 shows 8 GB spare — CUDA1's true demand exceeds its ~11.3 GB usable
   and managed pages spill to host RAM, thrashing per token. Balanced 33,32
   keeps CUDA1 resident and decode is normal. Mechanism is a hypothesis;
   the reproduction is not.
3. No-UM in-process cannot fit this model+ctx on 16+12 GB at any split
   tried — the VRAM wall is why the rig runs UM at all.
4. Consistent with the earlier 30.16 t/s PR105.0 run (also below RPC).
5. The 40.1-bar gap is **not** a topology effect — in-process is slower, so
   the RPC split cannot be the cause. Reference-bar provenance (prompt /
   MTP-acceptance profile) remains the leading suspect for the gap.

PR105.0's premise (dropping RPC should be faster) is empirically false on
this rig. Recommend closing the arm as REJECTED with these cells as
evidence; keep the production RPC topology pinned.

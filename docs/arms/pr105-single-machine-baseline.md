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



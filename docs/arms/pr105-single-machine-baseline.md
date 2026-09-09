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

- [ ] Build: same flags as PR #110 binary, `GGML_CUDA_FORCE_CUBLAS` off
- [ ] Boot: launch spec above, `/health` 200, both devices listed in device log
- [ ] Gate: threshold 100000 defer + auto-admit on slot release observed once, 0 spurious defers
- [ ] Single decode >= 5 loops, mean t/s vs bar
- [ ] n=2 concurrent agg
- [ ] Greedy determinism: two identical requests byte-identical
- [ ] Record results in the arm report; PR103.0/PR104.x rebase on this topology

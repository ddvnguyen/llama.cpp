# MoE expert-cache measurement harness

Harness and offline analysis used to measure and bound decode throughput for a 512-expert MoE model
whose experts live in host RAM and are fetched over PCIe. Everything here was used to produce the
numbers in `docs/moe-lookahead-improvement-plan.md`.

- Full measured record: issue **#130**
- Negative-result record and owner decisions: issue **#129**

## Rig this was built for

RTX 3060 12 GB on PCIe gen4 **x4** (6.10 GB/s achievable, 5.91 GB/s gather, displacement
coefficient 1.00 - the fabric is serial). A second GPU (5060 Ti) sits idle; every arm pins
`CUDA_VISIBLE_DEVICES=1`.

Model: Qwen3.8-Flash-Next-APEX-I-Mini, arch `qwen4exp`, 48 layers, 512 experts, top-10,
file 46.48 GiB on disk. Cache: 48 layers x 3 banks x N slots, **slots private per layer**.

## Environment

    export PATH=/opt/software/cuda/13.2.1/bin:$PATH
    export LD_LIBRARY_PATH=<repo>/build/bin:/opt/software/cuda/13.2.1/lib64:$LD_LIBRARY_PATH
    unset GGML_CUDA_ENABLE_UNIFIED_MEMORY

Serve flags used by every arm (see `stream.sh`, which owns them):

    -m <model> --split-mode layer -fit off -ngl 99 --n-cpu-moe 99
    --override-tensor per_layer_token_embd=CPU --moe-expert-cache-size <N> -c 81920 --parallel 1
    --flash-attn on --jinja -t 6 --experimental-logs --load-mode none --decode-overlap --ple-prefetch
    --host 127.0.0.1 --port <port>

## Output location

The scripts write logs, per-token TSVs and traces to `HARNESS_DIR`, which defaults to this
directory. **Those outputs are not committed.** On the rig that produced these numbers the scripts
were run from `/mnt/WorkDisk/harness/multiturn-ctx/`, and the recorded logs stayed there.

## The arms

`stream.sh` is the base: it serves once, streams a completion from a REAL prompt, and records the
arrival time of every token. Everything else is a driver around it.

| script | what it measures |
|---|---|
| `stream.sh` | base driver. `stream.sh <dev> <ctx> <cacheN> <port> <tag> <promptfile> [outtok] [lookahead]` |
| `arm_cache_sweep.sh` | throughput vs cache size (N=28/40/53). The +21.5% capacity result |
| `arm_order_control.sh` | run-order control, to prove the capacity win is not drift |
| `arm_policy_sweep.sh` | eviction policy sweep (LFU half-life, LRU) |
| `arm_vram_budget.sh` | where VRAM runs out (N=64 hard-OOMs) |
| `arm_paced.sh` | paced look-ahead, the interleaving experiment |
| `arm_recall.sh` | predictor recall instrumentation |
| `arm_n42_usage.sh` | look-ahead usefulness (`used_pct_total`) at the operating point |
| `arm_plan_admit.sh`, `arm_placement.sh`, `arm_probe*.sh`, `arm_roll.sh`, `arm_miss_latency.sh` | attribution/probe arms |
| `arm_correctness.sh`, `arm_correctness_n28.sh` | greedy token identity via `POST /completion` with `stream:false`. The stream channel is what made the earlier check vacuous |
| `arm_ppl_correctness.sh` | perplexity comparison (15 chunks x 512) |
| `arm_demand_trace.sh` | captures the per-(step,layer) demanded expert ids - the input the Belady oracle needs |
| `arm_evict_policy.sh` | A/B of eviction policy and look-ahead admission at N=42 |
| `run_imatrix.sh`, `run_requant.sh` | importance matrix and selective expert requantization |

Older drivers used during attribution (`phase.sh`, `conc.sh`, `limit.sh`, `la.sh`, `depth.sh`,
`mt.sh`) are included for provenance. `phase.sh` is the retired synthetic harness - **never mix its
numbers with `stream.sh` real-prompt numbers in one table**.

## Offline analysis

`slotalloc/` holds the simulators. `slotalloc/policy_sim.py` is the calibrated policy engine: it
reproduces the shipped LFU-16 policy's per-step miss counts **exactly on 8 of 8 recorded ledgers**
(C=42/28/40/53 plus three policy variants), which is what makes its predictions usable.

Key offline results these produced:

- Belady/MIN at C=42 is **147.78** misses/step vs the shipped **225.0** - 34.3% avoidable, +32.2% t/s.
- The compulsory floor is **67.07** misses/step (every expert fetched once) - the absolute ceiling.
- Non-uniform per-layer slot allocation is worth **+0.29%** and fails split-half hold-out.
- Prediction-guided victim selection caps at **+4.2%** even with a *perfect* same-layer next-step
  predictor, and the look-ahead in the tree predicts a different layer, so its retention effect is
  exactly zero.
- Reaching the +32% needs a **same-layer predictor with 8-16 steps of horizon**, which does not exist.

To use them you need a captured trace: run `arm_demand_trace.sh`, then point the simulator at
`demand-trace-<tag>.txt`.

## The one fact that governs all of this

**A prefetch is still a PCIe fetch.** It moves bytes in time; it does not remove them. Only victim
choice removes fetches. Retention quality sets the traffic; fetch timing only decides whether those
bytes sit on the critical path.

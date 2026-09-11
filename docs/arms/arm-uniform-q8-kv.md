# Arm — uniform q8_0 KV (all K+V, main and draft) vs the mixed-quant production pin

Separate from PR #117/#118's adaptive-KV-streaming arm. Config-only arm: no
code changes; production-shape boot with
`-ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0` vs the production pin
`-ctk q8_0 -ctv q5_1 -ctkd q8_0 -ctvd q5_1` (K is already q8_0, V is the delta,
main+draft).

Binary under test: the pre-arm baseline binary itself (`d50efc6f`, no streaming
code). Both configs use the identical launch shape:

`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`; rpc-server `-d CUDA1 :50052`; llama-server
`--rpc 127.0.0.1:50052 -ts 27,38 -ngl 99 --rope-scaling yarn --rope-scale 5
--yarn-orig-ctx 32768 -fa on ... --no-kv-unified --cache-prompt --cache-reuse 64
--cache-idle-slots --cache-ram 1024 --ubatch-size 512 --cont-batching
--parallel-ctx-threshold 100000 --spec-type draft-mtp --prio-batch 1 --jinja`.

## Cells

| Cell | Shape | Quant | Purpose |
|---|---|---|---|
| F-fit | production: `-c 262144 -np 2` | uniform q8 | VRAM-fit / boot check (no benchmark) |
| T1-uniform | `-c 65536 -np 1` (same shape as the 24.26 control) | uniform q8 | multiturn-growth 1×12 (~8K/turn, 750 out) |
| T2-single | `-c 65536 -np 1`, 16K-token prompt, 150 greedy tokens | both quants | decode-latency isolator (the "740 band" protocol) |
| T2-mixed | same as T1/T2 | q8_0/q5_1 (+draft) | production control |

## Bars

- PASS to consider = uniform q8 shows **no decode slowdown** (error ≤ noise
  band ±3%, the run-to-run mixed spread observed on the 740 arms) at BOTH
  shallow and deep context, AND at production shape the model still boots
  inside the 15847/11911 MiB ceiling.
- Expected direction (engineering prior): q5_1 V is 5-bit+scale = 30 B/32 val
  (7.5 bit/elem) vs q8_0 = 33 B/32 (8.25 bit/elem) ⇒ uniform q8 adds ~+10% V
  bytes; V is the less quant-sensitive operand, so q5_1 should be the
  better quality/speed tradeoff — expect uniform q8 to be **equal or slightly
  slower**, not faster. A big win would have been a surprise.

## Results

### Fit check (cell F)

- **Uniform q8_0 BOOTS at the full production shape** (`-c 262144 -np 2
  -ts 27,38`, UM=1): health 200 ~55s; main KV buffers 8704 MiB (CUDA0 4896 +
  RPC0 3808), draft KV 544 MiB (mixed-shape equivalents: 7424 / ~448). VRAM
  lands at the usual production ceiling 15847/11910 MiB (UM net keeps it
  pinned there; larger KV squeezes headroom, no OOM).
- Per-shape KV footprint: at `-c 65536 -np 1` — main KV 2176 MiB (1224 CUDA0 +
  952 RPC0) vs mixed 1856 MiB = **+17.2% main KV**, draft 136 vs 116 MiB
  (+17%).

### T1 — multiturn-growth 1×12 (turns 10-12 ctx-cap both sides, excluded)

| config | harness mean (turns 1-9) | per-turn decode ms/tok (turns 1→9) | notes |
|---|---|---|---|
| mixed q8_0/q5_1 (production pin) | **24.26 tok/s** | 20.80, 26.31, 27.93, 33.75, 27.13, 26.26, 26.89, 31.18, 27.71 | prior control run |
| uniform q8_0 | **18.75 tok/s** | 21.88, 33.29, 32.28, 30.74, 23.49, 33.45, 36.61, 34.44, 34.34 | shorter completions (EOS earlier on 4 turns) inflate per-turn wall/headcount variance |

- **Harness mean: −22.7% for uniform q8_0.**
- **Pure decode latency (server `print_timing`): turn 1 −4.9% (21.88 vs
  20.80 ms/tok), deep turns (7-9) +15–24% (33–36.6 vs 27–31 ms/tok).** The
  slowdown GROWS with depth; shallow contexts are nearly par.
- Prefill: parity shallow (819 vs ~817 tok/s fresh), slight decay at depth
  (495 vs 498 tok/s at 60K) — negligible part of the delta.
- Caveat recorded: the harness tok/s includes per-turn prefill and the two
  configs produce different completions (4 of 9 uq8 turns stopped early), so
  the mean is NOT pure-decode; the ms/tok columns above are the honest speed
  metric.

### T2 — single-request 150-token eval (same protocol as the "40 tok/s" band)

| config | decode tok/s (3 runs) |
|---|---|
| mixed q8_0/q5_1 (production pin) | 34.17 / 34.38 / 34.45 |
| uniform q8_0 | 33.32 / 33.19 / 33.09 |

- **−3.2% decode** at ~16K context, single-request, matched shape/protocol.
- Prompt prefill identical (819 vs 817 tok/s).

### Where the "~40 tok/s production target" comes from (sought, per user request)

There is **no documented single "40 tok/s production pin" figure**. The
provenance trail:

1. `docs/investigations/740-results-report.md` (arms 102/103/104, the
   low-pinnable shape `146176×3 ctx, kv_unified OFF`): **single-request
   150-token eval band 40.5–49.7 tok/s** on both boots, "matching arm102's
   **46.2–46.3 class**" (line 1445/1280). 2-concurrent: 17.6 tok/s/slot.
2. `PROJECT_STATUS.md` > Line 386 (`#381` phase timeline): Arm 017
   speed-optimized config: "36–39 tok/s decode" multiturn 10/10.
3. `PROJECT_STATUS.md` > #381 line 379: ~31 tok/s sustained 5×10 decode,
   MTP draft acc 0.62–0.67 (older, pre-740-rig build).
4. `CLAUDE.md` hardware table: RTX 5060 Ti "~200 tok/s" (single-GPU ceiling,
   not this rig) and P100 28 tok/s — a different machine.

⇒ If the user wants "40 tok/s for the production pin" to reproduce: the
closest documented measurement is the **740-report single-request 150-token
eval protocol at the 146176-ctx shape (40–50 tok/s band, arm102 46.2-46.3
class)** — not the multiturn-growth mean (24.26 at our 65K shape) and not a
production-262k number. A "40 tok/s" reading against production is a
protocol mix-up; first reproduce the 740 protocol (single-shot eval +
146k/3-way shape) and compare in-band, or accept the multiturn protocol's
24.26 at 65K/np1 = strictly longer-context decode behavior.

### Conclusion

- **Uniform q8_0 is NOT a speed win — it is a measured loss** on the
  production-comparable shape: −3.2% shallow single-request, −15–24% decode
  latency at 46–60K depth, −22.7% multiturn harness mean. The mixed-quant pin
  (q5_1 V) is the better choice exactly for the reason the user's prior
  suggested: q5_1 V costs ~10% fewer V bytes than q8_0's 8.5-bit and pays
  negligible extra dequant cost in FA kernels, while q8_0's uniformity buys
  no extra decoder speed because these FA/vecdot kernels dequantize V to f16
  in-kernel anyway (dequant path complexity for 5-bit data is paid only on
  the V traffic share, K is q8_0 in both arms).
- The depth-amplified gap (beyond the ~baseline +10% V-bytes arithmetic
  estimate) suggests a second coupled factor that this arm measures but does
  not fully root-cause: UM prefetch net + resident-page behavior of the
  larger KV footprint across the RPC boundary (per-attention V row reads
  grow with depth, and the bigger V tensor crosses more of the
  transfer/prefetch budget). Flagged for a dedicated depth-sweep probe if
  anyone wants the last ~10% characterized exactly.

### Rig protocol (this pass)

Preflight health 200 / 15847-11911 → `podman pod stop pod_llama-baseline`
→ drain verify (1 MiB/1 MiB) → fit-check boot (no benchmark exec) → kill →
drain verify → T1-uq8 boot + multiturn → kill → T2-probe uq8 → kill →
T2-mixed boot + probes → kill → drain verify → `pod start` → health 200
×2 → 15847/11911 verified. Rig clean at end of arm.

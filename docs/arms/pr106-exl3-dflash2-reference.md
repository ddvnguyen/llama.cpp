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

#!/usr/bin/env bash
# run_imatrix.sh - build the importance matrix the original model was quantized with.
# Required because a sub-iq2 reduction (iq2_xxs / iq1_*) is refused without one.
# Corpus = the same document used for the perplexity check, so the calibration and the
# quality measurement share a domain. Prefill-only, so no MoE cache is involved.
set -u
export PATH=/opt/software/cuda/13.2.1/bin:$PATH
export LD_LIBRARY_PATH=/mnt/WorkDisk/workspace/worktree/1q3ry0vb/impl-pra/build/bin:/opt/software/cuda/13.2.1/lib64:${LD_LIBRARY_PATH:-}
unset GGML_CUDA_ENABLE_UNIFIED_MEMORY
D=/mnt/WorkDisk/harness/multiturn-ctx
BIN=/mnt/WorkDisk/workspace/worktree/1q3ry0vb/impl-pra/build/bin/llama-imatrix
M=/mnt/SSD/qwen3.8-flash-next-apex-mini/Qwen3.8-Flash-Next-APEX-I-Mini-00001-of-00006.gguf
echo "=== imatrix start $(date) ==="
CUDA_VISIBLE_DEVICES=1 "$BIN" -m "$M" -f "$D/moe-lookahead-improvement-plan.md" \
  -o "$D/imatrix-exp.gguf" --chunks 40 -c 512 -b 512 --split-mode layer -fit off \
  -ngl 99 --n-cpu-moe 99 --override-tensor per_layer_token_embd=CPU \
  --flash-attn on -t 6 2>&1 | tail -25
echo "=== imatrix exit=$? at $(date) ==="
ls -la "$D/imatrix-exp.gguf" 2>/dev/null | awk '{print "  imatrix: "$5" B"}'
echo "=== DONE ==="

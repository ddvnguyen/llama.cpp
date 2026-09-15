#!/usr/bin/env bash
# arm_ppl_correctness.sh - distributional corroboration for PR #127 Blocker 1 / issue #128.
#
# The greedy test (arm_correctness*.sh) showed la0 == la8 byte-for-byte over 200 tokens. But greedy
# comparison only detects a perturbation that FLIPS AN ARGMAX. A change that shifts the distribution
# without flipping a greedy argmax on that one prompt is invisible to it. Perplexity aggregates
# log-probabilities over thousands of tokens, so it is sensitive to exactly that class of change.
#
# Corpus: the improvement-plan document itself - 9,556 words / 57 KB of real English prose, local.
# Three arms: la0, la0b (determinism floor for PPL), la8. No code change; llama-perplexity is
# already built and linked against libggml-cuda 36d284bf (the same sha the servers used).
set -u

D=/mnt/WorkDisk/harness/multiturn-ctx
export PATH=/opt/software/cuda/13.2.1/bin:$PATH
export LD_LIBRARY_PATH=/mnt/WorkDisk/workspace/worktree/1q3ry0vb/impl-pra/build/bin:/opt/software/cuda/13.2.1/lib64:${LD_LIBRARY_PATH:-}
unset GGML_CUDA_ENABLE_UNIFIED_MEMORY

BIN=/mnt/WorkDisk/workspace/worktree/1q3ry0vb/impl-pra/build/bin/llama-perplexity
M=/mnt/SSD/qwen3.8-flash-next-apex-mini/Qwen3.8-Flash-Next-APEX-I-Mini-00001-of-00006.gguf
CORPUS=$D/moe-lookahead-improvement-plan.md
N=28
CHUNKS=15

run_arm() {
  TAG=$1; LA=$2
  echo "########## PPL $TAG lookahead=$LA ##########"
  pkill -f 'bin/llama-perplexity' 2>/dev/null; pkill -f 'bin/llama-server' 2>/dev/null; sleep 5
  CUDA_VISIBLE_DEVICES=1 "$BIN" -m "$M" -f "$CORPUS" --chunks $CHUNKS -c 512 -b 512 \
    --split-mode layer -fit off -ngl 99 --n-cpu-moe 99 \
    --override-tensor per_layer_token_embd=CPU \
    --moe-expert-cache-size "$N" --moe-lookahead "$LA" \
    --flash-attn on -t 6 -s 42 --no-warmup > "$D/ppl-$TAG.log" 2>&1
  echo "  final: $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' "$D/ppl-$TAG.log" | tail -1)"
  echo "  mean:  $(grep -oE 'Mean PPL\([A-Z]+\) = [0-9.]+' "$D/ppl-$TAG.log" | tail -1)"
  echo "  oom:   $(grep -c 'out of memory' "$D/ppl-$TAG.log")"
  echo "  chunks done: $(grep -cE '^\[[0-9]+\]' "$D/ppl-$TAG.log")"
  sleep 4
}

run_arm ppl0  0
run_arm ppl0b 0
run_arm ppl8  8

echo "########## SUMMARY ##########"
for t in ppl0 ppl0b ppl8; do
  printf "  %-7s %s\n" "$t" "$(grep -oE 'Final estimate: PPL = [0-9.]+' "$D/ppl-$t.log" | tail -1)"
done
pkill -f 'bin/llama-perplexity' 2>/dev/null
echo "########## DONE ##########"

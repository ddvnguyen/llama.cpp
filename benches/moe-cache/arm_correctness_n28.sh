#!/usr/bin/env bash
# arm_correctness.sh - close the missing correctness instrument (PR #127 Blocker 1 / issue #128).
#
# Issue #128: "an extra reader of a router weight (ffn_gate_inp) changes model output". The graph
# node is built by build_moe_lookahead, which returns early when cparams.moe_lookahead <= 0
# (src/llama-graph.cpp:2095), so --moe-lookahead 0 genuinely removes the extra reader - an ON/OFF
# comparison is not vacuous.
#
# Method: greedy (temperature 0, top_k 1) generation of 200 tokens from the same real prompt, with
# the look-ahead off and on. Greedy decoding is maximally sensitive to any logit perturbation - if
# the extra reader moves the distribution at all, the two texts MUST eventually diverge.
#
# Run order gives both controls in one pass:
#   la0      baseline
#   la0b     SAME config again  -> determinism control (must be byte-identical to la0, else the
#                                  instrument itself is untrustworthy and la8 proves nothing)
#   la8      look-ahead on      -> the actual blocker test
#
# The old determinism check was retracted because it compared empty strings (perftok column 3 was
# always empty). This one captures the completion text from the API response body, not the stream.
set -u

D=/mnt/WorkDisk/harness/multiturn-ctx
export PATH=/opt/software/cuda/13.2.1/bin:$PATH
export LD_LIBRARY_PATH=/mnt/WorkDisk/workspace/worktree/1q3ry0vb/impl-pra/build/bin:/opt/software/cuda/13.2.1/lib64:${LD_LIBRARY_PATH:-}
unset GGML_CUDA_ENABLE_UNIFIED_MEMORY

BIN=/mnt/WorkDisk/workspace/worktree/1q3ry0vb/impl-pra/build/bin/llama-server
M=/mnt/SSD/qwen3.8-flash-next-apex-mini/Qwen3.8-Flash-Next-APEX-I-Mini-00001-of-00006.gguf
PORT=18336
N=28

hygiene() {
  pkill -f 'bin/llama-server' 2>/dev/null
  sleep 5
  echo "[hygiene] gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
}

run_arm() {
  TAG=$1; LA=$2
  echo "########## CORRECT $TAG  lookahead=$LA ##########"
  hygiene
  rm -f "$D/server-$TAG.log"
  CUDA_VISIBLE_DEVICES=1 "$BIN" -m "$M" --split-mode layer -fit off -ngl 99 --n-cpu-moe 99 \
    --override-tensor per_layer_token_embd=CPU --moe-expert-cache-size "$N" -c 81920 --parallel 1 \
    --flash-attn on --jinja -t 6 --experimental-logs --load-mode none --decode-overlap --ple-prefetch \
    --moe-lookahead "$LA" --host 127.0.0.1 --port $PORT > "$D/server-$TAG.log" 2>&1 &
  local pid=$!

  # wait for readiness
  for i in $(seq 1 90); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" 2>/dev/null && break
    sleep 2
  done

  # greedy, non-streaming: the body carries the text (the stream channel is what broke the old check)
  jq -n --rawfile p "$D/prompt-real.txt" \
     '{prompt:$p, n_predict:200, temperature:0, top_k:1, seed:42, stream:false, cache_prompt:false}' \
     > /tmp/req.json
  curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
       --data-binary @/tmp/req.json > "/tmp/resp-$TAG.json" 2>/dev/null
  jq -r '.content // "ERROR: no content field"' "/tmp/resp-$TAG.json" > "$D/correct-$TAG.txt"
  echo "  content chars: $(wc -c < "$D/correct-$TAG.txt")  sha256: $(sha256sum "$D/correct-$TAG.txt" | cut -c1-16)"
  jq -r '"  stop_type=\(.stop_type // "?")  tokens_predicted=\(.tokens_predicted // "?")"' "/tmp/resp-$TAG.json" 2>/dev/null

  kill -KILL $pid 2>/dev/null
  sleep 6
}

run_arm la0  0
run_arm la0b 0
run_arm la8  8

echo "########## COMPARISON ##########"
cd "$D"
echo "determinism control (la0 vs la0b), must be IDENTICAL:"
cmp -s correct-la0.txt correct-la0b.txt && echo "  IDENTICAL" || { echo "  DIFFER -> instrument untrustworthy, la8 result is void"; diff <(head -c 300 correct-la0.txt) <(head -c 300 correct-la0b.txt) | head -4; }
echo "blocker test (la0 vs la8):"
cmp -s correct-la0.txt correct-la8.txt && echo "  IDENTICAL -> the extra reader does NOT change output" || { echo "  DIFFER -> extra reader CHANGES output (issue #128 confirmed)"; echo "  first divergence:"; cmp correct-la0.txt correct-la8.txt | head -2; }
hygiene
echo "########## DONE ##########"

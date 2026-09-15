#!/usr/bin/env bash
# arm_n42_usage.sh - the operating point the owner named (N=42), look-ahead ON vs OFF.
# ON  arm: yields staged_experts_total / used_pct_total (the prediction-usefulness ratio) + perftok.
# OFF arm: the distribution control at the same cache size.
set -u
D=/mnt/WorkDisk/harness/multiturn-ctx
hyg() { pkill -f 'bin/llama-server' 2>/dev/null; sleep 5; echo "[hyg] gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|tr '\n' ' ')"; }
arm() {
  TAG=$1; LA=$2
  echo "########## N42 $TAG lookahead=$LA ##########"
  hyg
  env GGML_CUDA_MOE_PHASE_PROBE_PER_STEP=1 \
    bash "$D/stream.sh" 1 81920 42 18336 "$TAG" "$D/prompt-real.txt" 200 "$LA" 2>&1 \
    | grep -E 'SERVER timings|SERVE_FAILED' | tail -2
  [ -f "$D/server-$TAG.log" ] && {
    grep 'moe-step:' "$D/server-$TAG.log" > "$D/moe-steps-$TAG.log"
    echo "  ledger lines: $(wc -l < "$D/moe-steps-$TAG.log")"
    echo "  usage line:  $(grep 'moe-lookahead-stage:' "$D/server-$TAG.log" | tail -1 | cut -c1-260)"
  }
  sleep 6
}
arm u42on 8
arm u42off 0
hyg
echo "########## DONE ##########"

#!/usr/bin/env bash
# arm_recall.sh - the predictor's recall (the "about 80%" number), measured on the CURRENT build.
# GGML_CUDA_MOE_LOOKAHEAD_DEBUG enables ggml_cuda_moe_lookahead_debug_score (ggml-cuda.cu:3777),
# which scores the stored prediction for each layer against the ids the router actually routed to
# and logs:  moe-cache-lookahead-debug: prefetch_calls= predicted_ids= checks= recall=%
set -u
D=/mnt/WorkDisk/harness/multiturn-ctx
hyg() { pkill -f 'llama-server' 2>/dev/null; sleep 5; echo "[hyg] gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|tr '\n' ' ')"; }
arm() {
  TAG=$1; W=$2
  echo "########## RECALL $TAG width=$W ##########"
  hyg
  env GGML_CUDA_MOE_LOOKAHEAD_DEBUG=1 \
    bash "$D/stream.sh" 1 81920 42 18336 "$TAG" "$D/prompt-real.txt" 200 "$W" 2>&1 \
    | grep -E 'SERVER timings|SERVE_FAILED' | tail -2
  [ -f "$D/server-$TAG.log" ] && {
    echo "  LAST recall line:"
    grep -o 'moe-cache-lookahead-debug:.*' "$D/server-$TAG.log" | tail -1 | sed 's/^/    /'
    echo "  recall lines total: $(grep -c 'moe-cache-lookahead-debug:' "$D/server-$TAG.log")"
  }
  sleep 6
}
arm r8 8
arm r10 10
hyg
echo "########## DONE ##########"

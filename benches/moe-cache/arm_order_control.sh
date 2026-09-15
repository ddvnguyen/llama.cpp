#!/usr/bin/env bash
# arm_order_control.sh - is the +21.5% a cache effect or a drift effect?
#
# The original sweep ran N=28, 40, 53 in that order. A monotone thermal/clock drift across the
# session would masquerade as a cache-size effect, and nothing in that run controlled for order.
# This reverses it: N=53 first, N=28 last. If the endpoints reproduce, order is not the cause.
#
# Look-ahead OFF, real prompt, 200 tokens, per-step ledger ON.
set -u

D=/mnt/WorkDisk/harness/multiturn-ctx

hygiene() {
  pgrep -f 'bin/llama-server' >/dev/null && { echo "[hygiene] killing stray llama-server"; pkill -f 'bin/llama-server'; sleep 8; }
  pgrep -f 'bin/test-moe-cache' >/dev/null && { echo "[hygiene] killing stray test-moe-cache"; pkill -f 'bin/test-moe-cache'; }
  echo "[hygiene] gpu used MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
}

arm() {
  TAG=$1; N=$2
  echo "########## ORD $TAG cache=$N ##########"
  hygiene
  OUT=$(env GGML_CUDA_MOE_PHASE_PROBE_PER_STEP=1 \
        bash "$D/stream.sh" 1 81920 "$N" 18336 "$TAG" "$D/prompt-real.txt" 200 0 2>&1)
  echo "$OUT" | grep -E 'SERVER timings|SERVE_FAILED' | tail -2
  if [ -f "$D/server-$TAG.log" ]; then
    grep 'moe-step:' "$D/server-$TAG.log" > "$D/moe-steps-$TAG.log"
    echo "  moe-step lines: $(wc -l < "$D/moe-steps-$TAG.log")"
  fi
  sleep 6
}

# reversed order relative to the original sweep
arm ord53 53
arm ord28 28

hygiene
echo "########## DONE ##########"

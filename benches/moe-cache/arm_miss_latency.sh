#!/usr/bin/env bash
# arm_miss_latency.sh - per-token expert-miss ledger, and the instrument's own cost.
#
# CONTROL ONLY (lookahead 0). With no look-ahead there is no staging, so the ledger's
# `misses` is unambiguously the number of experts actually fetched from host memory that
# step - which is exactly the quantity to correlate against per-token latency.
#
# scost : per-step gate OFF -> decode t/s on this binary, uninstrumented.
# sctl  : per-step gate ON  -> the same arm, instrumented. The difference between the two is
#         the instrument's own cost, which must be reported before any per-token number is
#         quoted, because a perturbed clock cannot be used to explain the clock.
set -u

D=/mnt/WorkDisk/harness/multiturn-ctx

hygiene() {
  pgrep -f 'llama-server' >/dev/null && { echo "[hygiene] killing stray llama-server"; pkill -f 'llama-server'; sleep 8; }
  pgrep -f 'test-moe-cache' >/dev/null && { echo "[hygiene] killing stray test-moe-cache"; pkill -f 'test-moe-cache'; }
  echo "[hygiene] gpu used MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
}

run_arm() {
  TAG=$1; LEDGER=$2
  echo "########## $TAG lookahead=0 per_step_ledger=$LEDGER ##########"
  hygiene
  env GGML_CUDA_MOE_PHASE_PROBE_PER_STEP="$LEDGER" \
    bash "$D/stream.sh" 1 81920 28 18336 "$TAG" "$D/prompt-real.txt" 200 0 2>&1 \
    | grep -E 'SERVER timings|SERVE_FAILED' | tail -2
  if [ -f "$D/server-$TAG.log" ]; then
    grep -c 'moe-step:' "$D/server-$TAG.log" | sed "s/^/  moe-step lines in server-$TAG.log: /"
    grep 'moe-step:' "$D/server-$TAG.log" > "$D/moe-steps-$TAG.log"
  else
    echo "  [warn] no server-$TAG.log"
  fi
  sleep 6
}

run_arm scost 0
run_arm sctl 1

hygiene
echo "########## JOIN ##########"
python3 "$D/join_steps.py" sctl 5.94
echo "########## DONE ##########"

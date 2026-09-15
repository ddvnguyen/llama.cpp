#!/usr/bin/env bash
# arm_paced.sh - paired arm: control vs PACED width-1, same binary, REAL prompt.
# The pacing change moves the look-ahead issue from before the layer's FFN to after it,
# so the staging DMA rides the next layer's attention instead of the demand gather.
set -u

D=/mnt/WorkDisk/harness/multiturn-ctx

hygiene() {
  pgrep -f 'llama-server' >/dev/null && { echo "[hygiene] killing stray llama-server"; pkill -f 'llama-server'; sleep 8; }
  pgrep -f 'test-moe-cache' >/dev/null && { echo "[hygiene] killing stray test-moe-cache"; pkill -f 'test-moe-cache'; }
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
  echo "[hygiene] gpu used MiB: $used"
}

for spec in "ctlP 0" "la1P 1"; do
  set -- $spec
  TAG=$1; LA=$2
  echo "########## ARM $TAG lookahead=$LA ##########"
  hygiene
  bash "$D/stream.sh" 1 81920 28 18336 "$TAG" "$D/prompt-real.txt" 200 "$LA" 2>&1 \
    | grep -E 'STREAM|SERVER timings|SERVE_FAILED|moe-lookahead-stage:|moe-early-router-copy:|moe-resident-summary|moe-resident-step' \
    | tail -20
  sleep 6
done

hygiene
echo "########## DONE ##########"

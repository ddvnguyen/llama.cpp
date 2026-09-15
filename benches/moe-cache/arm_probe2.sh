#!/usr/bin/env bash
# arm_probe.sh - paired arm with the RESIDENT PROBE ON, to get the byte ledger.
# The probe depresses t/s (~-15.8%) but does not change the byte counters, so this
# answers: does the paced issue still CONSUME what it stages, or does it now arrive late?
set -u

D=/mnt/WorkDisk/harness/multiturn-ctx

hygiene() {
  pgrep -f 'bin/llama-server' >/dev/null && { echo "[hygiene] killing stray llama-server"; pkill -f 'bin/llama-server'; sleep 8; }
  pgrep -f 'bin/test-moe-cache' >/dev/null && { echo "[hygiene] killing stray test-moe-cache"; pkill -f 'bin/test-moe-cache'; }
  echo "[hygiene] gpu used MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
}

for spec in "ctlRp 0" "la1Rp 1"; do
  set -- $spec
  TAG=$1; LA=$2
  echo "########## PROBE ARM(rolling) $TAG lookahead=$LA ##########"
  hygiene
  GGML_CUDA_MOE_PHASE_PROBE=1 bash "$D/stream.sh" 1 81920 28 18336 "$TAG" "$D/prompt-real.txt" 200 "$LA" 2>&1 \
    | grep -E 'STREAM|SERVER timings|SERVE_FAILED|moe-resident-summary|moe-lookahead-stage:|moe-early-router-copy:' \
    | tail -12
  sleep 6
done

hygiene
echo "########## DONE ##########"

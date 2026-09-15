#!/usr/bin/env bash
# arm_probe_place.sh - plan-level ledger for both issue placements, current binary.
#
# The consumption ledger showed staged bytes are consumed ~95% in both placements, which
# retracts the earlier "pacing destroys the reuse" reading. What it does NOT show is
# whether that consumption reduces the PLAN's demand misses (i.e. whether the look-ahead
# installs experts a step early, which is the only way it can pay). copy_mib / resident_pct
# answer that and require the resident probe.
#
# The probe depresses t/s but not the byte counters, so ignore t/s in this arm.
set -u

D=/mnt/WorkDisk/harness/multiturn-ctx

hygiene() {
  pgrep -f 'bin/llama-server' >/dev/null && { echo "[hygiene] killing stray llama-server"; pkill -f 'bin/llama-server'; sleep 8; }
  pgrep -f 'bin/test-moe-cache' >/dev/null && { echo "[hygiene] killing stray test-moe-cache"; pkill -f 'bin/test-moe-cache'; }
  echo "[hygiene] gpu used MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
}

run_arm() {
  TAG=$1; LA=$2; AFTER=$3
  echo "########## PROBE ARM $TAG lookahead=$LA issue_after_ffn=$AFTER ##########"
  hygiene
  if [ "$AFTER" = "0" ]; then
    GGML_CUDA_MOE_PHASE_PROBE=1 GGML_MOE_LOOKAHEAD_ISSUE_AFTER_FFN=0 \
      bash "$D/stream.sh" 1 81920 28 18336 "$TAG" "$D/prompt-real.txt" 200 "$LA" 2>&1 \
      | grep -E 'moe-resident-summary|moe-lookahead-stage:|SERVE_FAILED' | tail -4
  else
    GGML_CUDA_MOE_PHASE_PROBE=1 \
      bash "$D/stream.sh" 1 81920 28 18336 "$TAG" "$D/prompt-real.txt" 200 "$LA" 2>&1 \
      | grep -E 'moe-resident-summary|moe-lookahead-stage:|SERVE_FAILED' | tail -4
  fi
  sleep 6
}

run_arm pctl3 0 1
run_arm paft3 1 1
run_arm pbef3 1 0

hygiene
echo "########## DONE ##########"

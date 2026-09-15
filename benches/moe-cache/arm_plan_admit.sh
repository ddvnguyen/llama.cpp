#!/usr/bin/env bash
# arm_plan_admit.sh - verify the plan-side admission change.
#
# PlanAdmit's claim, which this arm tests: the change is CLASSIFICATION ONLY. It never
# allocates or evicts a second slot and never writes a slot table; the miss entry, its slot
# and its committed mapping are what the miss loop already produced, and the gather lands the
# staged payload in that same slot. So it should change the LEDGER and not the clock.
#
# Probe arms answer: does the change make copy_mib reflect only the misses actually fetched
# from host (gate on) instead of every miss (gate off)?
# Clean arm answers: is there any t/s effect at all? (Prediction: none. If there IS one, the
# claim "classification only" is false and that is the important result.)
#
# copy_mib is (n_misses x resident_bytes_per_miss) - a DERIVED quantity, not measured PCIe
# bytes. That is why this arm matters: with the gate on, copy_mib becomes the demand-only
# figure, which is the first honest read of whether the look-ahead reduces demand traffic.
set -u

D=/mnt/WorkDisk/harness/multiturn-ctx

hygiene() {
  pgrep -f 'llama-server' >/dev/null && { echo "[hygiene] killing stray llama-server"; pkill -f 'llama-server'; sleep 8; }
  pgrep -f 'test-moe-cache' >/dev/null && { echo "[hygiene] killing stray test-moe-cache"; pkill -f 'test-moe-cache'; }
  echo "[hygiene] gpu used MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
}

probe_arm() {
  TAG=$1; LA=$2; ADMIT=$3
  echo "########## PROBE $TAG lookahead=$LA admit_in_plan=$ADMIT ##########"
  hygiene
  env GGML_CUDA_MOE_PHASE_PROBE=1 GGML_MOE_LOOKAHEAD_ADMIT_IN_PLAN="$ADMIT" \
    bash "$D/stream.sh" 1 81920 28 18336 "$TAG" "$D/prompt-real.txt" 200 "$LA" 2>&1 \
    | grep -E 'moe-resident-summary|SERVE_FAILED' | tail -3
  sleep 6
}

clean_arm() {
  TAG=$1; LA=$2; ADMIT=$3
  echo "########## CLEAN $TAG lookahead=$LA admit_in_plan=$ADMIT ##########"
  hygiene
  env GGML_MOE_LOOKAHEAD_ADMIT_IN_PLAN="$ADMIT" \
    bash "$D/stream.sh" 1 81920 28 18336 "$TAG" "$D/prompt-real.txt" 200 "$LA" 2>&1 \
    | grep -E 'SERVER timings|SERVE_FAILED' | tail -2
  sleep 6
}

# ledger: gate off must reproduce need=94570 resident=43125 copy_mib=92239.6 on the control,
# and the look-ahead arms must differ from each other ONLY in copy_mib/resident_pct.
probe_arm padm0 0 0
probe_arm padm1 1 0
probe_arm padm2 1 1

# clock: only the gate-on clean arm is missing (clean control 9.699 and clean gate-off 9.398
# are already measured on the previous binary; the gate-off path is unchanged by construction,
# but re-run the look-ahead arm here so both gates are on ONE binary).
clean_arm cadm2 1 1

hygiene
echo "########## DONE ##########"

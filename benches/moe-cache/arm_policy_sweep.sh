#!/usr/bin/env bash
# arm_policy_sweep.sh - retention policy at fixed capacity.
#
# Capacity saturates: N=28 -> 53 is +21.5% but r only reaches 58% at the VRAM cap. So the
# remaining headroom has to come from WHICH experts are kept at fixed N.
#
# Today: LFU-with-decay, one halving per 16 grouped planning steps, hard-zeroed after 32 epochs
# (512 steps). A token is 48 steps, so the policy forgets after roughly one to two tokens.
#
# All arms at N=53 (best capacity), look-ahead OFF, per-step ledger ON.
#   pol16   : default half-life 16   -> must reproduce 11.738 t/s from the capacity sweep,
#                                      which is the regression check for the new kernel argument
#   pol256  : half-life 256 (~5 tokens)
#   pol2048 : half-life 2048 (~43 tokens - frequency becomes a long-horizon accumulator)
#   pollru  : GGML_CUDA_MOE_FREQUENCY=0 -> pre-existing pure-LRU control
set -u

D=/mnt/WorkDisk/harness/multiturn-ctx

hygiene() {
  pgrep -f 'llama-server' >/dev/null && { echo "[hygiene] killing stray llama-server"; pkill -f 'llama-server'; sleep 8; }
  pgrep -f 'test-moe-cache' >/dev/null && { echo "[hygiene] killing stray test-moe-cache"; pkill -f 'test-moe-cache'; }
  echo "[hygiene] gpu used MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
}

# TAG  HALFLIFE  FREQFLAG
arm() {
  TAG=$1; HL=$2; FREQ=$3
  echo "########## $TAG N=53 halflife=$HL freq=$FREQ ##########"
  hygiene
  OUT=$(env GGML_CUDA_MOE_PHASE_PROBE_PER_STEP=1 \
        GGML_CUDA_MOE_FREQUENCY_HALFLIFE="$HL" GGML_CUDA_MOE_FREQUENCY="$FREQ" \
        bash "$D/stream.sh" 1 81920 53 18336 "$TAG" "$D/prompt-real.txt" 200 0 2>&1)
  echo "$OUT" | grep -E 'SERVER timings|SERVE_FAILED' | tail -2
  if [ -f "$D/server-$TAG.log" ]; then
    grep 'moe-step:' "$D/server-$TAG.log" > "$D/moe-steps-$TAG.log"
    echo "  moe-step lines: $(wc -l < "$D/moe-steps-$TAG.log")"
  else
    echo "  [warn] no server-$TAG.log"
  fi
  sleep 6
}

arm pol16   16   1
arm pol256  256  1
arm pol2048 2048 1
arm pollru  16   0

hygiene
echo "########## DONE ##########"

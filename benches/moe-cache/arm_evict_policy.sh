#!/usr/bin/env bash
# arm_evict_policy.sh - A/B the expert-cache retention policy AND look-ahead admission at N=42.
#
# The predict-based arms need the look-ahead's predicted set, so every predict arm runs at the SAME
# width (8). Arm ev0 is the shipped baseline and shows what the feature costs on its own.
#
#   ev0    la=0  policy=lfu                  shipped baseline
#   ev8f   la=8  policy=lfu                  look-ahead as it ships today (side lane only)
#   ev8p   la=8  policy=protect              prediction-guided VICTIM selection
#   ev8a   la=8  policy=lfu      admit=1     owner request: store predictions INTO the cache
#   ev8ap  la=8  policy=protect  admit=1     both, composed
#
# Five arms, about 1.5 min each. Reports decode t/s, misses/step, and the stage counters.
set -u
D=/mnt/WorkDisk/harness/multiturn-ctx
N=42
WIDTH=8
OUT="$D/evict-policy-results.txt"
: > "$OUT"

hyg() {
  pkill -f 'llama-server' 2>/dev/null
  sleep 5
  echo "  [hyg] gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
}

arm() {
  TAG=$1; LA=$2; POL=$3; ADMIT=${4:-0}
  echo "########## $TAG  N=$N lookahead=$LA policy=$POL admit=$ADMIT ##########"
  hyg
  env GGML_CUDA_MOE_PHASE_PROBE_PER_STEP=1 \
      GGML_MOE_EVICT_POLICY="$POL" \
      GGML_MOE_ADMIT_PREDICTED="$ADMIT" \
    bash "$D/stream.sh" 1 81920 "$N" 18336 "$TAG" "$D/prompt-real.txt" 200 "$LA" 2>&1 \
    > "$D/arm-$TAG.out"
  grep -E 'SERVER timings|STREAM tokens|SERVE_FAILED' "$D/arm-$TAG.out" | tail -3

  local log="$D/server-$TAG.log"
  [ -f "$log" ] || { echo "  NO SERVER LOG"; return; }
  grep 'moe-step:' "$log" > "$D/moe-steps-$TAG.log"
  local lines miss tps
  lines=$(wc -l < "$D/moe-steps-$TAG.log")
  miss=$(awk -F'misses=' '{split($2,a," "); s+=a[1]} END {printf "%.1f", (NR? s/NR:0)}' "$D/moe-steps-$TAG.log")
  tps=$(grep -oE 'decode_tps=[0-9.]+' "$D/arm-$TAG.out" | tail -1 | cut -d= -f2)
  echo "  ledger lines: $lines   misses/step: $miss   decode_tps: ${tps:-n/a}"
  if [ "$LA" -gt 0 ]; then
    echo "  stage: $(grep 'moe-lookahead-stage:' "$log" | tail -1 | cut -c1-400)"
  fi
  echo "$TAG la=$LA policy=$POL admit=$ADMIT misses_per_step=$miss decode_tps=${tps:-n/a}" >> "$OUT"
  sleep 6
}

arm ev0   0 "lfu"
arm ev8f  "$WIDTH" lfu
arm ev8p  "$WIDTH" protect
arm ev8a  "$WIDTH" lfu     1
arm ev8ap "$WIDTH" protect 1

hyg
echo "########## results ##########"
cat "$OUT"
echo "baseline reference (no policy env at all) is 10.76-10.81 t/s at 225.0 misses/step"
echo "########## DONE ##########"

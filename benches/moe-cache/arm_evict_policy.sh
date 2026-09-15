#!/usr/bin/env bash
# arm_evict_policy.sh - A/B the expert-cache retention policy AND look-ahead admission at N=42.
#
# The predict-based arms need the look-ahead's predicted set, so every predict arm runs at the SAME
# width (8). Arm ev0 is the shipped baseline and shows what the feature costs on its own.
#
#   ev0    la=0  policy=lfu                  shipped baseline
#   ev0p   la=0  policy=protect              RECENT1 alone - the only retention policy with a
#                                            measured POSITIVE sign in the offline engine
#                                            (+1.2%, and model-dependent, so it needs the rig)
#   ev8f   la=8  policy=lfu                  look-ahead as it ships today (side lane only)
#   ev8p   la=8  policy=protect              protect, composed with look-ahead
#   ev8a   la=8  policy=lfu      admit=1     owner request: store predictions INTO the cache
#   ev8ap  la=8  policy=protect  admit=1     both, composed
#   ev8r   la=8  policy=recent1  admit=0     recent1 alone at width 8
#
# protect is the UNION of two pin sources: the recent1 warm-set pin and the predicted-mask pin.
# recent1 is the warm-set half alone. At la=0 there is no lane, so the mask half is inert and
# ev0p IS the recent1 rule. At la=8 the union and the half differ, and
#     ev8p - ev8r  = the look-ahead mask's marginal retention effect
#     ev8p - ev0p  = the whole look-ahead's effect on retention (mask + lane interaction)
#
# Seven arms, about 1.5 min each. Reports decode t/s, misses/step, and the stage counters.
#
# EXPECTATIONS, from the offline policy engine (which replays the shipped policy exactly):
#   ev0p  ~+1.2%  (223.57 vs 225.00 misses/step), sign is model-dependent -> this is the test
#   ev8p  ~0.0%   the tree's predictor targets a different LAYER, so it carries no retention
#                 signal for the layer being evicted; this arm verifies that null on hardware
#   ev8a  the only arm whose mechanism can pay: it turns the 2.05 MiB staging lane into cache
#                 slots, so staged data stops being thrown away (1.91% -> toward 95.5% useful).
#                 The win is TIMING, not bytes: no arm here can reduce traffic except ev0p.
set -u
D=/mnt/WorkDisk/harness/multiturn-ctx
N=42
WIDTH=8
OUT="$D/evict-policy-results.txt"
: > "$OUT"

hyg() {
  pkill -f 'bin/llama-server' 2>/dev/null
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
  local adm=0
  if [ "$LA" -gt 0 ]; then
    adm=$(grep 'moe-lookahead-stage:' "$log" | tail -1 | grep -oE 'admitted_experts=[0-9]+' | cut -d= -f2)
    adm=${adm:-0}
  fi
  # Admissions are billed as fetches by design, so with admit=1 the ledger is inflated and
  # misses/step is NOT comparable across arms. Real traffic is misses minus admissions.
  local admps real
  admps=$(awk -v a="$adm" -v n="$lines" 'BEGIN{printf "%.2f", (n? a/n : 0)}')
  real=$(awk -v m="$miss" -v a="$admps" 'BEGIN{printf "%.1f", m-a}')
  echo "  ledger lines: $lines   misses/step: $miss   admitted/step: $admps   real traffic: $real   decode_tps: ${tps:-n/a}"
  if [ "$LA" -gt 0 ]; then
    echo "  stage: $(grep 'moe-lookahead-stage:' "$log" | tail -1 | cut -c1-400)"
  fi
  echo "$TAG la=$LA policy=$POL admit=$ADMIT misses_per_step=$miss admitted_per_step=$admps real_traffic=$real decode_tps=${tps:-n/a}" >> "$OUT"
  sleep 6
}

# usage: arm_evict_policy.sh [arm ...]      default: all seven
# Selecting a subset matters when only some arms are safe to run: the admit arms aborted the server
# before the plan-residency fix, and a crash costs the whole run.
WANT="${*:-ev0 ev0p ev8f ev8p ev8r ev8a ev8ap}"
want() { case " $WANT " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

want ev0   && arm ev0   0 "lfu"
want ev0p  && arm ev0p  0 "protect"
want ev8f  && arm ev8f  "$WIDTH" lfu
want ev8p  && arm ev8p  "$WIDTH" protect
want ev8r  && arm ev8r  "$WIDTH" recent1
want ev8a  && arm ev8a  "$WIDTH" lfu     1
want ev8ap && arm ev8ap "$WIDTH" protect 1

hyg
echo "########## results ##########"
cat "$OUT"
echo "baseline reference (no policy env at all) is 10.76-10.81 t/s at 225.0 misses/step"
echo "########## DONE ##########"

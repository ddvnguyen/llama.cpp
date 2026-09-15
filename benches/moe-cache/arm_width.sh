#!/usr/bin/env bash
# arm_width.sh - decode throughput and staging usefulness vs look-ahead width.
#
# Why this exists: arm_evict_policy.sh measured width 8 at N=42 as 8.005 t/s against 10.832 with the
# look-ahead off, at IDENTICAL misses (225.0) - a 26.1% loss. The stage counters say why: 35222
# experts were staged and 2.33% were ever consumed. Staged experts are real PCIe fetches, so a
# staging ratio that low is pure band theft from the demand path.
#
# Width is the only knob on that ratio, and it is measured to be strongly non-linear:
#   width 1 -> 95.2-95.5% useful      width 8 -> 1.91-2.33% useful
# So the loss is a property of width 8, not of the feature. This arm walks the widths to find where
# the staging ratio stops paying for the link time it costs.
#
#   usage: arm_width.sh [width ...]        default: 1 2 4
#
set -u
D=/mnt/WorkDisk/harness/multiturn-ctx
N=42
WIDTHS=${*:-"1 2 4"}
OUT="$D/width-results.txt"
: > "$OUT"

hyg() {
  pkill -f 'llama-server' 2>/dev/null
  sleep 5
  echo "  [hyg] gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
}

arm() {
  local W=$1 TAG="w$1"
  echo "########## $TAG  N=$N lookahead=$W policy=lfu admit=0 ##########"
  hyg
  env GGML_CUDA_MOE_PHASE_PROBE_PER_STEP=1 \
      GGML_MOE_EVICT_POLICY=lfu \
    bash "$D/stream.sh" 1 81920 "$N" 18336 "$TAG" "$D/prompt-real.txt" 200 "$W" \
    > "$D/arm-$TAG.out" 2>&1
  grep -E 'SERVER timings|STREAM tokens|SERVE_FAILED' "$D/arm-$TAG.out" | tail -3

  local log="$D/server-$TAG.log"
  [ -f "$log" ] || { echo "  NO SERVER LOG"; return; }
  grep 'moe-step:' "$log" > "$D/moe-steps-$TAG.log"
  local lines miss tps stg usep
  lines=$(wc -l < "$D/moe-steps-$TAG.log")
  miss=$(awk -F'misses=' '{split($2,a," "); s+=a[1]} END {printf "%.2f", (NR? s/NR:0)}' "$D/moe-steps-$TAG.log")
  tps=$(grep -oE 'decode_tps=[0-9.]+' "$D/arm-$TAG.out" | tail -1 | cut -d= -f2)
  stg=$(grep 'moe-lookahead-stage:' "$log" | tail -1 | grep -oE 'staged_experts_total=[0-9]+' | cut -d= -f2)
  usep=$(grep 'moe-lookahead-stage:' "$log" | tail -1 | grep -oE 'used_pct_total=[0-9.]+' | cut -d= -f2)
  echo "  ledger: $lines steps   misses/step: $miss   staged: ${stg:-0}   used%: ${usep:-n/a}   decode_tps: ${tps:-n/a}"
  echo "$TAG lookahead=$W misses_per_step=$miss staged=${stg:-0} used_pct=${usep:-n/a} decode_tps=${tps:-n/a}" >> "$OUT"
  sleep 6
}

for W in $WIDTHS; do arm "$W"; done

hyg
echo "########## results ##########"
cat "$OUT"
echo "reference: width 0 = 10.832 t/s at 225.00 misses/step; width 8 = 8.005 t/s at 225.00 misses/step"
echo "########## DONE ##########"

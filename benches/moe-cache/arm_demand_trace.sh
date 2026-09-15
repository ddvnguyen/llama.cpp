#!/usr/bin/env bash
# arm_demand_trace.sh - capture the per-(step,layer) demanded expert ids at the owner's operating
# point (N=42). This is the input to the offline Belady oracle, which answers: of the ~225 misses
# per token, how many could ANY policy avoid that keeps only 42 slots in each of the 48 layers?
# That is owner item 3 ("improve slot lookup"). Look-ahead is OFF so the trace reflects the shipped
# admission policy with no prediction in play.
set -u
D=/mnt/WorkDisk/harness/multiturn-ctx
TAG=dt42
TRACE="$D/demand-trace-$TAG.txt"
rm -f "$TRACE"
pkill -f 'llama-server' 2>/dev/null
sleep 5
echo "[hyg] gpu before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
env GGML_CUDA_MOE_PHASE_PROBE_PER_STEP=1 \
    GGML_CUDA_MOE_DEMAND_TRACE=1 \
    GGML_CUDA_MOE_DEMAND_TRACE_FILE="$TRACE" \
    bash "$D/stream.sh" 1 81920 42 18336 "$TAG" "$D/prompt-real.txt" 200 0 2>&1 \
  | grep -E 'SERVER timings|SERVE_FAILED' | tail -2
echo "--- trace file ---"
if [ -f "$TRACE" ]; then
  echo "  lines: $(wc -l < "$TRACE")"
  head -2 "$TRACE"
  echo "  ..."
  tail -1 "$TRACE"
else
  echo "  MISSING: $TRACE"
fi
echo "--- server log ---"
[ -f "$D/server-$TAG.log" ] && grep -E 'moe-demand-trace|moe-step:' "$D/server-$TAG.log" | tail -3
echo "--- ledger ---"
[ -f "$D/server-$TAG.log" ] && grep 'moe-step:' "$D/server-$TAG.log" > "$D/moe-steps-$TAG.log" \
  && echo "  ledger lines: $(wc -l < "$D/moe-steps-$TAG.log")" \
  && tail -3 "$D/moe-steps-$TAG.log"
pkill -f 'llama-server' 2>/dev/null
sleep 4
echo "[hyg] gpu after: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
echo "########## DONE ##########"

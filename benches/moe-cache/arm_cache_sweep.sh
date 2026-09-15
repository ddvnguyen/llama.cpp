#!/usr/bin/env bash
# arm_cache_sweep.sh - the capacity lever, VRAM-quantified.
#
# Model under test (rev 10, r2=0.990): T_token = 25.7 ms + 0.3003 * misses, and misses = 480(1-r).
# VRAM probe: N=28 uses 9293 of 12288 MiB, N=64 OOMs, marginal 103 MiB per +1 slot -> cap N~53.
#
# Look-ahead OFF, so `misses` is unambiguously host-fetched and r = 1 - misses/480.
# Per-step ledger ON (GGML_CUDA_MOE_PHASE_PROBE_PER_STEP=1) which publishes the counters WITHOUT
# the blocking cumulative summary, so t/s stays clean and we still get the miss count.
#
# N=53 is deliberately at the computed VRAM edge: if it OOMs that is itself the answer to
# "how far can the cache go", so the script continues rather than aborting.
set -u

D=/mnt/WorkDisk/harness/multiturn-ctx

hygiene() {
  pgrep -f 'llama-server' >/dev/null && { echo "[hygiene] killing stray llama-server"; pkill -f 'llama-server'; sleep 8; }
  pgrep -f 'test-moe-cache' >/dev/null && { echo "[hygiene] killing stray test-moe-cache"; pkill -f 'test-moe-cache'; }
  echo "[hygiene] gpu used MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
}

for N in 28 40 53; do
  TAG="n$N"
  echo "########## ARM $TAG cache=$N lookahead=0 ##########"
  hygiene
  OUT=$(env GGML_CUDA_MOE_PHASE_PROBE_PER_STEP=1 \
    bash "$D/stream.sh" 1 81920 "$N" 18336 "$TAG" "$D/prompt-real.txt" 200 0 2>&1)
  echo "$OUT" | grep -E 'SERVER timings|SERVE_FAILED|error|out of memory' | tail -3
  if [ -f "$D/server-$TAG.log" ]; then
    grep 'moe-step:' "$D/server-$TAG.log" > "$D/moe-steps-$TAG.log"
    echo "  moe-step lines: $(wc -l < "$D/moe-steps-$TAG.log")"
  else
    echo "  [warn] no server-$TAG.log - arm likely failed to load at this cache size"
  fi
  sleep 6
done

hygiene
echo "########## DONE ##########"

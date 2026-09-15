#!/usr/bin/env bash
# arm_vram_budget.sh - is the cache-size lever available in VRAM at all?
#
# The expert cache is the only lever with an order-of-magnitude payoff (T = 25.7 + 0.3003*misses,
# r2=0.990, so every +10% hit ratio is worth ~+14% t/s). But the doc line 198 records "N=100
# infeasible (8.6 GiB vs 2.6 GiB free)", so the cap may be VRAM, not policy. Measure it:
# load at N=28 and N=64 and take the difference - that gives the VRAM cost per slot and the
# headroom, without guessing from slab arithmetic.
#
# Loads the model only; no request is sent. Same FLAGS as phase.sh/stream.sh.
set -u

D=/mnt/WorkDisk/harness/multiturn-ctx
MODEL=/mnt/SSD/qwen3.8-flash-next-apex-mini/Qwen3.8-Flash-Next-APEX-I-Mini-00001-of-00006.gguf
BIN=/mnt/WorkDisk/workspace/worktree/1q3ry0vb/impl-pra/build/bin/llama-server

export PATH=/opt/software/cuda/13.2.1/bin:$PATH
export LD_LIBRARY_PATH=/mnt/WorkDisk/workspace/worktree/1q3ry0vb/impl-pra/build/bin:/opt/software/cuda/13.2.1/lib64:${LD_LIBRARY_PATH:-}
unset GGML_CUDA_ENABLE_UNIFIED_MEMORY

hygiene() {
  pgrep -f 'bin/llama-server' >/dev/null && { echo "[hygiene] killing stray llama-server"; pkill -f 'bin/llama-server'; sleep 8; }
  echo "[hygiene] gpu used MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
}

probe() {
  N=$1
  echo "########## N=$N ##########"
  hygiene
  LOG="$D/vram-N$N.log"
  rm -f "$LOG"
  CUDA_VISIBLE_DEVICES=1 "$BIN" -m "$MODEL" --split-mode layer -fit off -ngl 99 \
    --n-cpu-moe 99 --override-tensor per_layer_token_embd=CPU \
    --moe-expert-cache-size "$N" -c 81920 --parallel 1 --flash-attn on --jinja -t 6 \
    --experimental-logs --load-mode none --decode-overlap --ple-prefetch \
    --host 127.0.0.1 --port 18390 > "$LOG" 2>&1 &
  SRV=$!
  for i in $(seq 1 180); do
    grep -q 'server is listening' "$LOG" 2>/dev/null && break
    kill -0 $SRV 2>/dev/null || { echo "  server died:"; tail -5 "$LOG"; return 1; }
    sleep 2
  done
  sleep 6
  echo "  after load: $(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | tr '\n' ' ')"
  grep -iE 'CUDA0 model buffer size|CUDA0 KV buffer size|CUDA0 compute buffer size|CUDA0.*buffer size' "$LOG" | head -6
  # free VRAM as the driver sees it, while the server holds the model
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed 's/^/  driver says free MiB: /'
  kill -TERM $SRV 2>/dev/null
  for i in $(seq 1 20); do kill -0 $SRV 2>/dev/null || break; sleep 2; done
  kill -KILL $SRV 2>/dev/null
  sleep 6
}

probe 28
probe 64

hygiene
echo "########## DONE ##########"

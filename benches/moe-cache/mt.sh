#!/usr/bin/env bash
# mt.sh - APEX-I-Mini multi-turn agent-workflow run at a chosen context size, one GPU.
#
# Byte-for-byte the current rig serve config (recovered from single3060-warm.sh /
# single5060ti-warm.sh), changing ONLY -c:
#   --split-mode layer -fit off -ngl 99 --n-cpu-moe 99
#   --override-tensor per_layer_token_embd=CPU --moe-expert-cache-size N
#   -c <CTX> --parallel 1 --flash-attn on --jinja -t 6
#   --experimental-logs --load-mode none --decode-overlap --ple-prefetch
#
# usage: mt.sh <cuda_dev> <ctx> <cacheN> <port> <tag> [turns] [newtok] [outtok] [cal]
#   cal = 1 -> calibration only: one ~newtok-token prompt, max_tokens 16, print prefill rate.
#
# Client: <baseline-flash-next>/infra/llama-baseline/multiturn-growth-test.sh (stdlib only)
set -u

DEV=${1:?cuda_dev}; CTX=${2:?ctx}; N=${3:?cacheN}; PORT=${4:?port}; TAG=${5:?tag}
TURNS=${6:-15}; NEW=${7:-4000}; OUT=${8:-128}; CAL=${9:-0}

BIN=/mnt/WorkDisk/workspace/worktree/1q3ry0vb/impl-pra/build/bin
MODEL=/mnt/SSD/qwen3.8-flash-next-apex-mini/Qwen3.8-Flash-Next-APEX-I-Mini-00001-of-00006.gguf
MT=/mnt/WorkDisk/workspace/worktree/1q3ry0vb/baseline-flash-next/infra/llama-baseline/multiturn-growth-test.sh
D=/mnt/WorkDisk/harness/multiturn-ctx
mkdir -p "$D"
LOG="$D/server-$TAG.log"; OUT_JSON="$D/result-$TAG.txt"

export LD_LIBRARY_PATH="$BIN:/opt/software/cuda/13.2.1/lib64:${LD_LIBRARY_PATH:-}"
unset GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset LLAMA_ARG_MOE_LOOKAHEAD 2>/dev/null || true

echo "=== mt.sh dev=$DEV ctx=$CTX cacheN=$N port=$PORT tag=$TAG turns=$TURNS new=$NEW out=$OUT cal=$CAL ===" | tee "$OUT_JSON"

CUDA_VISIBLE_DEVICES=$DEV "$BIN/llama-server" \
  -m "$MODEL" --split-mode layer -fit off -ngl 99 \
  --n-cpu-moe 99 --override-tensor per_layer_token_embd=CPU \
  --moe-expert-cache-size "$N" -c "$CTX" --parallel 1 --flash-attn on --jinja -t 6 \
  --experimental-logs --load-mode none --decode-overlap --ple-prefetch \
  --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
SRV=$!

READY=0
for _ in $(seq 1 300); do
  curl -sS -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && { READY=1; break; }
  kill -0 "$SRV" 2>/dev/null || break
  sleep 2
done

if [ "$READY" != 1 ]; then
  echo "SERVE_FAILED (no health after 600s)" | tee -a "$OUT_JSON"
  grep -m3 -E 'out of memory|cudaMalloc failed|error|abort' "$LOG" | tee -a "$OUT_JSON"
  kill -TERM "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
  exit 1
fi

echo "--- server up; ctx/KV lines ---" | tee -a "$OUT_JSON"
grep -m4 -E 'n_ctx|KV self|kv_unified|CUDA_MoE_Cache|expert cache|slots' "$LOG" | tee -a "$OUT_JSON"
echo "--- VRAM after load ---" | tee -a "$OUT_JSON"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$OUT_JSON"

if [ "$CAL" = 1 ]; then
  PY=$(python3 - "$PORT" "$NEW" <<'PYEOF'
import json, sys, time, urllib.request
port, newtok = sys.argv[1], int(sys.argv[2])
filler = " ".join("w%d" % (i % 100000) for i in range(int(newtok * 0.75)))
body = json.dumps({"model": "x", "messages": [{"role": "user", "content": filler + "\n\nSummarise in one sentence."}],
                   "max_tokens": 16, "temperature": 0}).encode()
req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=900) as r:
    d = json.load(r)
t = d.get("timings", {})
print("CAL prompt_n=%s prompt_ms=%.0f prefill_tps=%.2f decode_tps=%.2f wall=%.1fs" % (
    t.get("prompt_n"), t.get("prompt_ms", 0), t.get("prompt_per_second", 0),
    t.get("predicted_per_second", 0), time.time() - t0))
PYEOF
)
  echo "$PY" | tee -a "$OUT_JSON"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$OUT_JSON"
  grep -hoE 'n_ctx[ =]+[0-9]+|n_ctx_slot[ =]+[0-9]+' "$LOG" | tail -2 | tee -a "$OUT_JSON"
  kill -TERM "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
  echo "=== cal done ===" | tee -a "$OUT_JSON"
  exit 0
fi

echo "--- multi-turn agent-workflow run ($TURNS turns, +$NEW tok/turn, $OUT out/turn) ---" | tee -a "$OUT_JSON"
bash "$MT" "$PORT" 1 "$TURNS" "$NEW" "$OUT" 2>&1 | tee -a "$OUT_JSON"
echo "--- VRAM peak / after run ---" | tee -a "$OUT_JSON"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$OUT_JSON"
echo "--- ctx / cache evidence ---" | tee -a "$OUT_JSON"
grep -hoE 'n_ctx_slot *[=:] *[0-9]+|truncat|context shift|n_past[^,]{0,20}' "$LOG" | tail -5 | tee -a "$OUT_JSON"
grep -hoE 'moe-cache[a-z-]*:?[^|]{0,90}' "$LOG" | tail -3 | tee -a "$OUT_JSON"

kill -TERM "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
echo "=== run done: $OUT_JSON ==="

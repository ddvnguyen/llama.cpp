#!/usr/bin/env bash
# depth.sh - decode throughput vs context depth at a chosen -c, one GPU.
# Serve config = current rig config with only -c changed (see mt.sh).
# usage: depth.sh <cuda_dev> <ctx> <cacheN> <port> <tag> [out_tokens]
set -u

DEV=${1:?dev}; CTX=${2:?ctx}; N=${3:?cacheN}; PORT=${4:?port}; TAG=${5:?tag}; OUTTOK=${6:-64}

BIN=/mnt/WorkDisk/workspace/worktree/1q3ry0vb/impl-pra/build/bin
MODEL=/mnt/SSD/qwen3.8-flash-next-apex-mini/Qwen3.8-Flash-Next-APEX-I-Mini-00001-of-00006.gguf
D=/mnt/WorkDisk/harness/multiturn-ctx
mkdir -p "$D"
LOG="$D/server-$TAG.log"; R="$D/depth-$TAG.txt"

export LD_LIBRARY_PATH="$BIN:/opt/software/cuda/13.2.1/lib64:${LD_LIBRARY_PATH:-}"
unset GGML_CUDA_ENABLE_UNIFIED_MEMORY
unset LLAMA_ARG_MOE_LOOKAHEAD 2>/dev/null || true

echo "=== depth.sh dev=$DEV ctx=$CTX N=$N port=$PORT out=$OUTTOK ===" | tee "$R"

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
  echo "SERVE_FAILED" | tee -a "$R"
  grep -m2 -E 'out of memory|cudaMalloc failed' "$LOG" | tee -a "$R"
  kill -TERM "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; exit 1
fi
echo "--- up: $(grep -hoE 'n_ctx_slot = [0-9]+' "$LOG" | head -1) ---" | tee -a "$R"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$R"
echo "target_tokens | prompt_n | prefill_tps | decode_tps | decode_ms" | tee -a "$R"

python3 - "$PORT" "$OUTTOK" "$R" <<'PYEOF' 2>&1 | tee -a "$R"
import json, sys, urllib.request
port, outtok, rfile = sys.argv[1], int(sys.argv[2]), sys.argv[3]
targets = [2000, 8000, 16000, 32000, 50000]
for target in targets:
    words = int(target / 4.6)          # measured: this filler tokenizes at ~4.6 tok/word
    filler = " ".join("w%d" % (i % 100000) for i in range(words))
    body = json.dumps({"model": "x",
                       "messages": [{"role": "user", "content": filler + "\n\nExplain in one sentence what this list is."}],
                       "max_tokens": outtok, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    t = d.get("timings", {})
    print("req target=%-6d prompt_n=%-7s prefill_tps=%-8.2f decode_tps=%-7.2f decode_ms=%s" % (
        target, t.get("prompt_n"), t.get("prompt_per_second", 0),
        t.get("predicted_per_second", 0), t.get("predicted_ms")), flush=True)
PYEOF

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$R"
kill -TERM "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
echo "=== depth done: $R ==="

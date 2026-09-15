#!/usr/bin/env bash
# limit.sh - sample what limits throughput: SM/memory utilization + PCIe traffic
# during a steady decode window. usage: limit.sh <dev> <ctx> <N> <port> <tag> <outtok>
set -u

DEV=${1:?dev}; CTX=${2:?ctx}; N=${3:?N}; PORT=${4:?port}; TAG=${5:?tag}; OUTTOK=${6:-256}

BIN=/mnt/WorkDisk/workspace/worktree/1q3ry0vb/impl-pra/build/bin
MODEL=/mnt/SSD/qwen3.8-flash-next-apex-mini/Qwen3.8-Flash-Next-APEX-I-Mini-00001-of-00006.gguf
D=/mnt/WorkDisk/harness/multiturn-ctx
LOG="$D/server-$TAG.log"; R="$D/limit-$TAG.txt"; DMON="$D/dmon-$TAG.log"

export LD_LIBRARY_PATH="$BIN:/opt/software/cuda/13.2.1/lib64:${LD_LIBRARY_PATH:-}"
unset GGML_CUDA_ENABLE_UNIFIED_MEMORY

CUDA_VISIBLE_DEVICES=$DEV "$BIN/llama-server" -m "$MODEL" --split-mode layer -fit off -ngl 99 \
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
[ "$READY" = 1 ] || { echo "SERVE_FAILED"; grep -m2 -E 'out of memory|cudaMalloc failed' "$LOG"; kill -TERM "$SRV" 2>/dev/null; exit 1; }
echo "=== limit.sh dev=$DEV ctx=$CTX N=$N out=$OUTTOK ===" | tee "$R"

# sampler: GPU utilization + PCIe rx/tx, 1 Hz, whole run
nvidia-smi dmon -i "$DEV" -d 1 -c 600 >"$DMON" 2>&1 &
DMONPID=$!

python3 - "$PORT" "$R" "$OUTTOK" <<'PYEOF' 2>&1 | tee -a "$R"
import json, sys, urllib.request
port, rfile, outtok = sys.argv[1], sys.argv[2], int(sys.argv[3])
filler = " ".join("w%d" % (i % 100000) for i in range(3000))
for label, ntok in (("prefill~14k warmup", 1), ("steady decode", outtok)):
    body = json.dumps({"model": "x",
                       "messages": [{"role": "user", "content": filler + "\n\nCount slowly."}],
                       "max_tokens": ntok, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    t = d.get("timings", {})
    print("MARK %-20s prompt_n=%-7s prefill_tps=%-8.2f decode_tps=%-7.2f" % (
        label, t.get("prompt_n"), t.get("prompt_per_second", 0), t.get("predicted_per_second", 0)), flush=True)
PYEOF

sleep 2
kill -TERM "$DMONPID" 2>/dev/null; wait "$DMONPID" 2>/dev/null
kill -TERM "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
echo "--- dmon (window: prefill then decode) ---" | tee -a "$R"
cat "$DMON" | tee -a "$R"
echo "=== limit done: $R ==="

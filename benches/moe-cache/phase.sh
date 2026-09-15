#!/usr/bin/env bash
# phase.sh - serve once and run a single request for the phase probe.
# usage: phase.sh <dev> <ctx> <cacheN> <port> <tag> [words] [outtok] [lookahead_width]
set -u

DEV=${1:?dev}; CTX=${2:?ctx}; N=${3:?cacheN}; PORT=${4:?port}; TAG=${5:?tag}
WORDS=${6:-3000}; OUTTOK=${7:-256}; LA=${8:-0}

BIN=/mnt/WorkDisk/workspace/worktree/1q3ry0vb/impl-pra/build/bin
MODEL=/mnt/SSD/qwen3.8-flash-next-apex-mini/Qwen3.8-Flash-Next-APEX-I-Mini-00001-of-00006.gguf
D=/mnt/WorkDisk/harness/multiturn-ctx
LOG="$D/server-$TAG.log"; R="$D/phase-$TAG.txt"

export LD_LIBRARY_PATH="$BIN:/opt/software/cuda/13.2.1/lib64:${LD_LIBRARY_PATH:-}"
unset GGML_CUDA_ENABLE_UNIFIED_MEMORY

FLAGS=(-m "$MODEL" --split-mode layer -fit off -ngl 99
  --n-cpu-moe 99 --override-tensor per_layer_token_embd=CPU
  --moe-expert-cache-size "$N" -c "$CTX" --parallel 1 --flash-attn on --jinja -t 6
  --experimental-logs --load-mode none --decode-overlap --ple-prefetch)
[ "$LA" -gt 0 ] && FLAGS+=(--moe-lookahead "$LA")
if [ -n "${EXTRA:-}" ]; then
  # shellcheck disable=SC2206
  FLAGS+=($EXTRA)
fi

CUDA_VISIBLE_DEVICES=$DEV "$BIN/llama-server" "${FLAGS[@]}" \
  --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
SRV=$!

READY=0
for _ in $(seq 1 300); do
  curl -sS -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && { READY=1; break; }
  kill -0 "$SRV" 2>/dev/null || break
  sleep 2
done
[ "$READY" = 1 ] || { echo "SERVE_FAILED"; grep -m2 -E 'out of memory|cudaMalloc failed' "$LOG"; kill -TERM "$SRV" 2>/dev/null; exit 1; }

echo "=== phase.sh dev=$DEV ctx=$CTX N=$N words=$WORDS out=$OUTTOK la=$LA extra='${EXTRA:-}' probe=${GGML_CUDA_MOE_PHASE_PROBE:-unset} graphs_disabled=${GGML_CUDA_DISABLE_GRAPHS:-no} lookup_debug=${GGML_CUDA_MOE_LOOKAHEAD_DEBUG:-unset} ===" | tee "$R"

# Device-side sampler for the run: SM/mem utilisation and power at 1 Hz, plus a
# few PCIe link samples taken under load (the link downclocks when idle).
DMON="$D/dmon-$TAG.log"; LINK="$D/link-$TAG.log"
nvidia-smi dmon -i "$DEV" -d 1 -s ump -c 900 >"$DMON" 2>&1 &
DMONPID=$!
(
  for i in $(seq 1 40); do
    sleep 10
    echo "t=$((i*10))s $(nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max,clocks.sm,utilization.gpu --format=csv,noheader -i "$DEV" 2>/dev/null | tr -d '\n')" >>"$LINK"
  done
) &
LINKPID=$!

python3 - "$PORT" "$WORDS" "$OUTTOK" "$R" <<'PYEOF' 2>&1 | tee -a "$R"
import json, sys, urllib.request
port, words, outtok, rfile = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
filler = " ".join("w%d" % (i % 100000) for i in range(words))
body = json.dumps({"model": "x",
                   "messages": [{"role": "user", "content": filler + "\n\nCount slowly."}],
                   "max_tokens": outtok, "temperature": 0}).encode()
req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=7200) as r:
    d = json.load(r)
t = d.get("timings", {})
print("REQUEST prompt_n=%-7s prefill_tps=%-8.2f decode_tps=%-7.2f decode_ms=%s" % (
    t.get("prompt_n"), t.get("prompt_per_second", 0), t.get("predicted_per_second", 0), t.get("predicted_ms")), flush=True)
PYEOF

kill -TERM "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
kill -TERM "$DMONPID" "$LINKPID" 2>/dev/null; wait "$DMONPID" "$LINKPID" 2>/dev/null
echo "--- pcie/link under load ---" | tee -a "$R"
cat "$LINK" 2>/dev/null | tee -a "$R"
echo "--- dmon (decode window tail) ---" | tee -a "$R"
tail -20 "$DMON" 2>/dev/null | tee -a "$R"
echo "--- probe: global (last 3) ---" | tee -a "$R"
grep -h 'moe-phase-probe\[' "$LOG" | tail -3 | tee -a "$R"
echo "--- probe: per-layer (last block) ---" | tee -a "$R"
grep -h 'moe-phase-probe-layer\[' "$LOG" | awk -F'steps=' '{print $1"steps="$2}' | tail -52 | tee -a "$R"
echo "--- recall probe (last 3) ---" | tee -a "$R"
grep -h 'moe-recall-probe' "$LOG" | tail -3 | tee -a "$R"
echo "--- cache phase lines (decode) ---" | tee -a "$R"
grep -h 'moe-cache-phase: phase=decode' "$LOG" | tail -2 | tee -a "$R"
echo "=== phase done: $R ==="

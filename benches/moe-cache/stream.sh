#!/usr/bin/env bash
# stream.sh - serve once, stream a completion from a REAL prompt, record PER-TOKEN timing.
# usage: stream.sh <dev> <ctx> <cacheN> <port> <tag> <promptfile> [outtok] [lookahead_width]
#
# Unlike phase.sh (one blocking request, aggregate timings), this streams and records
# the arrival time of every token, plus the server's own per-token timings when the
# build supports timings_per_token. Output: perftok-<tag>.tsv
set -u

DEV=${1:?dev}; CTX=${2:?ctx}; N=${3:?cacheN}; PORT=${4:?port}; TAG=${5:?tag}
PFILE=${6:?promptfile}; OUTTOK=${7:-200}; LA=${8:-0}

BIN=/mnt/WorkDisk/workspace/worktree/1q3ry0vb/impl-pra/build/bin
MODEL=/mnt/SSD/qwen3.8-flash-next-apex-mini/Qwen3.8-Flash-Next-APEX-I-Mini-00001-of-00006.gguf
D=/mnt/WorkDisk/harness/multiturn-ctx
LOG="$D/server-$TAG.log"; R="$D/stream-$TAG.txt"; TOK="$D/perftok-$TAG.tsv"

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

echo "=== stream.sh dev=$DEV ctx=$CTX N=$N out=$OUTTOK la=$LA prompt=$PFILE extra='${EXTRA:-}' ===" | tee "$R"

python3 - "$PFILE" "$PORT" "$OUTTOK" "$TOK" "$R" <<'PYEOF' 2>&1 | tee -a "$R"
import json, sys, time, urllib.request
pfile, port, outtok, tok_path, rfile = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
prompt = open(pfile).read()
body = json.dumps({"model": "x",
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": outtok, "temperature": 0, "stream": True,
                   "timings_per_token": True}).encode()
req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port,
                             data=body, headers={"Content-Type": "application/json"})
t0 = time.monotonic()
rows, server_tm, first_tok = [], None, None
with urllib.request.urlopen(req, timeout=7200) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            d = json.loads(payload)
        except Exception:
            continue
        if d.get("timings"):
            server_tm = d["timings"]
        ch = (d.get("choices") or [{}])[0]
        delta = (ch.get("delta") or {}).get("content") or ""
        t = (time.monotonic() - t0) * 1000.0
        if delta and first_tok is None:
            first_tok = t
        tm = d.get("timings")
        rows.append((len(rows), t, delta, json.dumps(tm) if tm else ""))
total_ms = (time.monotonic() - t0) * 1000.0
with open(tok_path, "w") as fh:
    fh.write("idx\tt_arrival_ms\tdelta_ms\tcontent\tserver_timings\n")
    prev = 0.0
    for idx, t, delta, tm in rows:
        fh.write("%d\t%.2f\t%.2f\t%s\t%s\n" % (idx, t, t - prev, delta.replace("\t", " "), tm))
        prev = t
n = len(rows)
gaps = [rows[i][1] - rows[i-1][1] for i in range(1, n)]
gaps.sort()
def pct(p):
    return gaps[min(len(gaps)-1, int(len(gaps)*p))] if gaps else 0.0
print("STREAM tokens=%d  first_token_ms=%.1f  total_ms=%.1f  decode_tps=%.2f" % (
    n, first_tok or 0.0, total_ms, (n - 1) / (total_ms / 1000.0) if total_ms else 0.0))
if gaps:
    print("STREAM inter-token ms: min=%.2f p50=%.2f p95=%.2f max=%.2f" % (
        gaps[0], pct(0.50), pct(0.95), gaps[-1]))
if server_tm:
    print("SERVER timings: prompt_n=%s prefill_tps=%s decode_tps=%s decode_ms=%s" % (
        server_tm.get("prompt_n"), server_tm.get("prompt_per_second"),
        server_tm.get("predicted_per_second"), server_tm.get("predicted_ms")))
PYEOF

kill -TERM "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
echo "--- per-token file: $TOK ---" | tee -a "$R"
echo "--- staging (last 3) ---" | tee -a "$R"
grep -h 'moe-lookahead-stage:' "$LOG" | tail -3 | tee -a "$R"
grep -h 'moe-early-router-copy:' "$LOG" | tail -1 | tee -a "$R"
echo "--- per-step resident (last 6) ---" | tee -a "$R"
grep -h 'moe-resident-step\[decode\]' "$LOG" | tail -6 | tee -a "$R"
echo "--- resident summary ---" | tee -a "$R"
grep -h 'moe-resident-summary\[decode\]' "$LOG" | tail -1 | tee -a "$R"
echo "=== stream done: $R ==="

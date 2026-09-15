#!/usr/bin/env bash
# conc.sh - concurrent-request decode throughput probe.
# usage: conc.sh <dev> <ctx> <cacheN> <port> <tag> <conc> [words] [outtok] [cacheN_extra_env]
# Fires <conc> simultaneous requests with disjoint prompts (so no prefix sharing)
# and reports aggregate decode tokens/sec over the overlapping window.
set -u

DEV=${1:?dev}; CTX=${2:?ctx}; N=${3:?cacheN}; PORT=${4:?port}; TAG=${5:?tag}; CONC=${6:?conc}
WORDS=${7:-3000}; OUTTOK=${8:-256}

BIN=/mnt/WorkDisk/workspace/worktree/1q3ry0vb/impl-pra/build/bin
MODEL=/mnt/SSD/qwen3.8-flash-next-apex-mini/Qwen3.8-Flash-Next-APEX-I-Mini-00001-of-00006.gguf
D=/mnt/WorkDisk/harness/multiturn-ctx
LOG="$D/server-$TAG.log"; R="$D/conc-$TAG.txt"

export LD_LIBRARY_PATH="$BIN:/opt/software/cuda/13.2.1/lib64:${LD_LIBRARY_PATH:-}"
unset GGML_CUDA_ENABLE_UNIFIED_MEMORY

FLAGS=(-m "$MODEL" --split-mode layer -fit off -ngl 99
  --n-cpu-moe 99 --override-tensor per_layer_token_embd=CPU
  --moe-expert-cache-size "$N" -c "$CTX" --parallel "$CONC" --flash-attn on --jinja -t 6
  --experimental-logs --load-mode none --decode-overlap --ple-prefetch)
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

echo "=== conc.sh dev=$DEV ctx=$CTX N=$N conc=$CONC words=$WORDS out=$OUTTOK extra='${EXTRA:-}' ===" | tee "$R"

python3 - "$PORT" "$WORDS" "$OUTTOK" "$CONC" <<'PYEOF' 2>&1 | tee -a "$R"
import json, sys, threading, time, urllib.request
port, words, outtok, conc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
res = [None]*conc
barrier = threading.Barrier(conc)

def one(i):
    # disjoint vocabulary per sequence: no shared prefix, no prompt-cache gift
    filler = " ".join("w%d" % ((i*104729 + k) % 5000000) for k in range(words))
    body = json.dumps({"model": "x",
                       "messages": [{"role": "user", "content": filler + "\n\nCount slowly."}],
                       "max_tokens": outtok, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    barrier.wait()                       # start all sequences together
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        d = json.load(r)
    res[i] = (d.get("timings", {}), t0, time.time())

ts = [threading.Thread(target=one, args=(i,)) for i in range(conc)]
for t in ts: t.start()
for t in ts: t.join()

t_start = min(x[1] for x in res)
t_end   = max(x[2] for x in res)
wall    = t_end - t_start
toks    = sum(x[0].get("predicted_n", outtok) for x in res)
pref    = sum(x[0].get("prompt_n", 0) for x in res)
for i, (tm, _, _) in enumerate(res):
    print("SEQ %d prompt_n=%-7s prefill_tps=%-8.2f decode_tps=%-7.2f decode_ms=%s" % (
        i, tm.get("prompt_n"), tm.get("prompt_per_second", 0),
        tm.get("predicted_per_second", 0), tm.get("predicted_ms")), flush=True)
print("AGGREGATE conc=%d tokens=%s wall_s=%.2f AGG_decode_tps=%.2f sum_seq_tps=%.2f prefill_n=%s" % (
    conc, toks, wall, toks/wall, sum(x[0].get("predicted_per_second", 0) for x in res), pref), flush=True)
PYEOF

kill -TERM "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
echo "=== conc done: $R ==="

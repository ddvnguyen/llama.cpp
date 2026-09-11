#!/bin/bash
# arm: adaptive-KV-streaming — boot helper for the 2xRTX RPC split
# Usage: bash scripts/arm117/boot-cell.sh <ctx_total> <kv-stream-stage-mib> <port> [np]
# Rig safety: caller must have verified rig free (no prod pod, GPUs idle).
set -euo pipefail

CTX_TOTAL=$1
STAGE_MIB=$2
PORT=${3:-8080}
NP=${4:-2}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
export LLAMA_KV_STREAM_DEVICE=${LLAMA_KV_STREAM_DEVICE:-CUDA0}
[ "${STAGE_MIB}" != "0" ] && export LLAMA_KV_STREAM_ALLOW_MULTISEQ=${LLAMA_KV_STREAM_ALLOW_MULTISEQ:-1}

echo "[$(date)] boot-cell ctx=$CTX_TOTAL stage=$STAGE_MIB port=$PORT np=$NP stream_dev=$LLAMA_KV_STREAM_DEVICE" | tee -a arm117-boot-log.txt

# peer (3060, CUDA1)
pkill -f 'ggml-rpc-server.*50052' 2>/dev/null || true
$ROOT/build/bin/ggml-rpc-server --host 127.0.0.1 --port 50052 -d 1 > arm117-rpc-peer.log 2>&1 &
sleep 1

# server (5060 Ti, CUDA0)
$ROOT/build/bin/llama-server \
  -m "${MODEL_PATH:-/mnt/SSD/Qwen3.8-27B-UD-Q5_K_M.gguf}" \
  --rpc 127.0.0.1:50052 -ts 27,38 -ngl 99 \
  --rope-scaling yarn --rope-scale 5 --yarn-orig-ctx 32768 \
  -fa on -ctk q8_0 -ctv q5_1 -ctkd q8_0 -ctvd q5_1 \
  --no-kv-unified --cache-prompt --cache-reuse 64 --cache-idle-slots \
  --cache-ram 1024 --ubatch-size 512 --cont-batching \
  -np $NP -c $CTX_TOTAL \
  --parallel-ctx-threshold 100000 --spec-type draft-mtp \
  --prio-batch 1 --kv-stream-stage-mib $STAGE_MIB \
  --jinja --host 0.0.0.0 --port $PORT --metrics --slots --log-verbosity 4 \
  > arm117-server-port$PORT.log 2>&1 &

SERVER_PID=$!
echo "server pid=$SERVER_PID"
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 localhost:$PORT/health || true)
  [ "$code" = "200" ] && echo "health 200 after ${i}s" && exit 0
  sleep 1
done
echo "BOOT FAILED (health never 200)"; tail -40 arm117-server-port$PORT.log; exit 1

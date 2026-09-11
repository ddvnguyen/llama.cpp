#!/bin/bash
# arm117 rig snapshot: health, slots, GPUs, key stream metrics — for artifacts
set -euo pipefail
PORT=${1:-8080}
OUT=${2:-arm117-artifacts/snapshot}
mkdir -p $OUT
curl -s --max-time 5 localhost:$PORT/health | tee $OUT/health.json
echo
curl -s --max-time 5 localhost:$PORT/slots | tee $OUT/slots.json | head -c 2000
echo
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tee $OUT/gpu-mem.txt
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  ps -p $pid -o pid,cmd --no-headers >> $OUT/gpu-procs.txt 2>/dev/null || true
done
grep -h "kv.stream\|kv_stream\|repartition" arm117-server-port$PORT.log > $OUT/log-stream-lines.txt 2>/dev/null || true

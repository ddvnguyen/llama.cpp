#!/usr/bin/env bash
# arm_roll.sh - paired arm for the ROLLING readiness change: control vs look-ahead 1
# (paced issue + per-expert rolling readiness). Real prompt, same binary.
#
# Correctness smoke test: temperature=0 makes the two runs deterministic, so a correct
# staging path must produce the SAME completion text in both arms. Differing text means
# the gather read a staging slab that had not landed.
set -u

D=/mnt/WorkDisk/harness/multiturn-ctx

hygiene() {
  pgrep -f 'bin/llama-server' >/dev/null && { echo "[hygiene] killing stray llama-server"; pkill -f 'bin/llama-server'; sleep 8; }
  pgrep -f 'bin/test-moe-cache' >/dev/null && { echo "[hygiene] killing stray test-moe-cache"; pkill -f 'bin/test-moe-cache'; }
  echo "[hygiene] gpu used MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
}

for spec in "ctlR2 0" "la1R2 1"; do
  set -- $spec
  TAG=$1; LA=$2
  echo "########## ROLL ARM $TAG lookahead=$LA ##########"
  hygiene
  bash "$D/stream.sh" 1 81920 28 18336 "$TAG" "$D/prompt-real.txt" 200 "$LA" 2>&1 \
    | grep -E 'STREAM|SERVER timings|SERVE_FAILED|moe-lookahead-stage:|moe-early-router-copy:' \
    | tail -8
  echo "--- completion text (first 160 chars) ---"
  sed -n '/^{/,$p' "$D/stream-$TAG.txt" 2>/dev/null | head -c 200
  grep -o '"content":"[^"]*"' "$D/stream-$TAG.txt" 2>/dev/null | head -2 | cut -c1-160
  sleep 6
done

hygiene
echo "--- determinism check: do the two arms produce identical text? ---"
if [ -f "$D/stream-ctlR2.txt" ] && [ -f "$D/stream-la1R2.txt" ]; then
  a=$(grep -o 'STREAM tokens=[0-9]*' "$D/stream-ctlR2.txt" | head -1)
  python3 - "$D/perftok-ctlR2.tsv" "$D/perftok-la1R2.tsv" <<'PY'
import sys
def texts(p):
    out=[]
    for line in open(p, errors='replace').read().splitlines()[1:]:
        f=line.split('\t')
        if len(f)>3: out.append(f[3])
    return out
a,b=texts(sys.argv[1]),texts(sys.argv[2])
n=min(len(a),len(b))
same=sum(1 for i in range(n) if a[i]==b[i])
print("completion tokens: control=%d lookahead=%d  identical_prefix_tokens=%d/%d" % (len(a),len(b),same,n))
print("first divergence at token:", next((i for i in range(n) if a[i]!=b[i]), None))
PY
fi
echo "########## DONE ##########"

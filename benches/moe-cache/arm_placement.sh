#!/usr/bin/env bash
# arm_placement.sh - does the ISSUE POSITION change WHICH experts get staged?
#
# Three arms, one binary, real prompt, probe OFF (so decode t/s is valid):
#   ctl3  --moe-lookahead 0
#   aft3  --moe-lookahead 1, issue AFTER the FFN  (default, current committed behaviour)
#   bef3  --moe-lookahead 1, issue BEFORE the FFN (legacy, pre-168a90717)
#
# The look-ahead's own counters now print without any probe:
#   staged_mib        = interval staged bytes
#   consumed_mib_total / consume_pct_total = lane-lifetime consumed / staged
# If consume_pct_total collapses in one arm and not the other, the issue position is
# changing WHAT the filter stages, not merely WHEN the transfer runs.
set -u

D=/mnt/WorkDisk/harness/multiturn-ctx

hygiene() {
  pgrep -f 'llama-server' >/dev/null && { echo "[hygiene] killing stray llama-server"; pkill -f 'llama-server'; sleep 8; }
  pgrep -f 'test-moe-cache' >/dev/null && { echo "[hygiene] killing stray test-moe-cache"; pkill -f 'test-moe-cache'; }
  echo "[hygiene] gpu used MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
}

run_arm() {
  TAG=$1; LA=$2; AFTER=$3
  echo "########## ARM $TAG lookahead=$LA issue_after_ffn=$AFTER ##########"
  hygiene
  if [ "$AFTER" = "0" ]; then
    GGML_MOE_LOOKAHEAD_ISSUE_AFTER_FFN=0 bash "$D/stream.sh" 1 81920 28 18336 "$TAG" "$D/prompt-real.txt" 200 "$LA" 2>&1 \
      | grep -E 'STREAM tokens|STREAM inter-token|SERVER timings|SERVE_FAILED|moe-lookahead-stage:' | tail -4
  else
    bash "$D/stream.sh" 1 81920 28 18336 "$TAG" "$D/prompt-real.txt" 200 "$LA" 2>&1 \
      | grep -E 'STREAM tokens|STREAM inter-token|SERVER timings|SERVE_FAILED|moe-lookahead-stage:' | tail -4
  fi
  sleep 6
}

run_arm ctl3 0 1
run_arm aft3 1 1
run_arm bef3 1 0

hygiene
echo "--- determinism: all three must produce the same text (temperature 0) ---"
python3 - "$D" <<'PY'
import sys
D=sys.argv[1]
def texts(p):
    out=[]
    for line in open(p, errors='replace').read().splitlines()[1:]:
        f=line.split('\t')
        if len(f)>3: out.append(f[3])
    return out
arms={}
for tag in ("ctl3","aft3","bef3"):
    try: arms[tag]=texts(f"{D}/perftok-{tag}.tsv")
    except Exception as e: print(f"{tag}: unreadable ({e})"); continue
base=arms.get("ctl3")
for tag,t in arms.items():
    if base is None: break
    n=min(len(base),len(t))
    same=sum(1 for i in range(n) if base[i]==t[i])
    div=next((i for i in range(n) if base[i]!=t[i]), None)
    print(f"{tag}: {len(t)} tokens, identical_to_control={same}/{n}, first_divergence={div}")
PY
echo "########## DONE ##########"

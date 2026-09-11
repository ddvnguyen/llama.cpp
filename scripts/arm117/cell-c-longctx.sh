#!/bin/bash
# arm117 long-context capability probe: plant early (M1) + late (M2) markers,
# huge filler, then ask for BOTH. Streaming claims exactness => both must hit.
# Usage: bash scripts/arm117/cell-c-longctx.sh <port> <filler_k_tokens>
set -euo pipefail
PORT=${1:-8080}
FILLER_K=${2:-30}
OUT=arm117-artifacts/cell-c
mkdir -p $OUT

python3 - "$FILLER_K" "$OUT/prompt.json" <<'PY'
import json,sys
k=int(sys.argv[1])
m1="M1 marker: code word FERRITE-77 guards the northern relay, lease the amber conduit before dawn cycle eleven."
m2="M2 marker: ledger 42 is filed with the onyx ferryman, and the harvest vault answers only to chartreuse."
filler=" ".join("Archive fragment %d records ambient sensor drift in sector %d with no anomalies."%(i,i%97) for i in range(k*22))
body=m1+"\n\n"+filler+"\n\n"+m2+"\n\nAnswer both questions:\n1) Which code word guards the northern relay?\n2) Which ledger is filed with the onyx ferryman?"
json.dump({"messages":[{"role":"user","content":body}],"max_tokens":160,"temperature":0,"cache_prompt":True}, open(sys.argv[2],"w"))
PY

curl -s --max-time 1800 localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d @"$OUT/prompt.json" -o "$OUT/resp.json"

python3 - "$OUT/resp.json" <<'PY'
import json,re,sys
d=json.load(open(sys.argv[1]))
t=d['choices'][0]['message']['content']
print(t)
hits={'M1(FERRITE-77)': bool(re.search(r'FERRITE[- ]?77', t, re.I)),
      'M2(ledger 42)':  bool(re.search(r'ledger 42|42.*(onyx|ferryman)|onyx ferryman', t, re.I))}
for k,v in hits.items(): print(("PASS" if v else "FAIL"), k)
PY

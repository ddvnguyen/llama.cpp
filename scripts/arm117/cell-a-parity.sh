#!/bin/bash
# arm117 cell A: numerical-parity probe — streaming OFF vs ON at a fit context.
# Greedy temp=0 kernel prompts; byte-diff OFF vs ON text.
# Server phases are booted by the runner (boot-cell.sh); this script assumes
# the server on $PORT matches the current phase expectation.
set -euo pipefail
PORT=${1:-8080}
OUT=arm117-artifacts/cell-a
mkdir -p $OUT
LABEL=${2:-run}

greedy() { # $1 tag, $2 req file
  curl -s --max-time 600 localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d @"$2" -o "$OUT/resp-$1.json"
  python3 -c "import json; d=json.load(open('$OUT/resp-$1.json')); print(d['choices'][0]['message']['content'])" > "$OUT/text-$1.txt" 2>/dev/null || echo RESPONSE_PARSE_FAIL > "$OUT/text-$1.txt"
}

python3 - > "$OUT/req-p1-$LABEL.json" <<'PY'
import json
filler=" ".join(["Filler sentence number %d for the parity kernel prompt body."%i for i in range(150)])
body=filler+"\n\nNow reply with exactly this text and nothing else: PARITY-KERNEL-A"
print(json.dumps({"messages":[{"role":"user","content":body}],"max_tokens":24,"temperature":0,"cache_prompt":True}))
PY

greedy "p1-$LABEL" "$OUT/req-p1-$LABEL.json"
echo "text-p1-$LABEL.txt:"; cat "$OUT/text-p1-$LABEL.txt"

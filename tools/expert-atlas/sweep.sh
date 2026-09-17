#!/usr/bin/env bash
# hydra: Colibri expert-atlas probe sweep port (#175) — hydra_vortex#771 / fork#133.
#
#   SERVER_URL=http://127.0.0.1:8086 ./sweep.sh [probes.json] [outdir]
#
# Port of JustVugg/colibri c/tools/expert_atlas/sweep.sh, driving llama-server
# over HTTP instead of the coli CLI. The CONFOUNDS section is Colibri's, kept
# verbatim in spirit — each one silently corrupts the atlas:
#
#   temp 0, top-p 1.0, top-k 0, min-p 0
#              Greedy, no distribution pruning: with pruning on you profile
#              the pruner, not the model (Colibri measured -38% experts seen
#              at top-p 0.7 — and pruning is the recommended speed setting).
#
#   MTP/DRAFT off
#              Speculative drafts route experts for tokens that are later
#              REJECTED and never emitted; those counts would describe text
#              the model never produced (#175 trap 2). Start llama-server
#              WITHOUT MTP/spec args and without --spec-* options.
#
#   decode-only
#              Prefill routing is topic-generic. The engine-side counters
#              (Stage A) are decode-only by design; the harness records
#              prompt/decode token counts and never mixes prefill in.
#
#   per-probe reset
#              Colibri removes .coli_usage before EVERY run so each dump is
#              exactly one probe, not a lifetime histogram. Our engine-side
#              equivalent is the Stage-A reset affordance (S-A2, gated behind
#              the Stage-1 workstream). Until it lands this harness measures
#              by DELTA of GET /experts between probes, which is reset-free
#              by construction, and reports honestly when telemetry is off.
#
# Prompt lengths are kept in a narrow band across categories (probes.json):
# prefill routes the prompt tokens too, so a verbose category would look busy.

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROBES="${1:-$HERE/probes.json}"
OUT="${2:-./atlas_out}"
SERVER_URL="${SERVER_URL:?set SERVER_URL to the llama-server base URL, e.g. http://127.0.0.1:8086}"
NGEN="${NGEN:-64}"
MAX_TOKENS="${MAX_TOKENS:-$NGEN}"

[ -f "$PROBES" ] || { echo "no probe file: $PROBES" >&2; exit 1; }
mkdir -p "$OUT/stats"

# engine reachability + Stage B surface
curl -sf -m 5 "$SERVER_URL/health" >/dev/null || { echo "server unreachable: $SERVER_URL" >&2; exit 1; }
META="$OUT/experts_before.json"
curl -sf -m 10 "$SERVER_URL/experts" -o "$META" || { echo "GET /experts failed — is HYDRA_EXPERT_META=1 on the engine?" >&2; exit 1; }
python3 - "$META" <<'PY' || exit 1
import json, sys
d = json.load(open(sys.argv[1]))
g = d["geometry"]
print(f"engine geometry: rows={len(g['moe_rows'])}+{len(g['nextn_rows'])} cols={d['cols']} k={g['n_expert_used']} engine_id={g['engine_id']}")
PY

# runlist: category<TAB>idx<TAB>prompt
python3 - "$PROBES" > "$OUT/runlist.tsv" <<'PY'
import json, sys
for cat, prompts in json.load(open(sys.argv[1])).items():
    if cat.startswith('_'):
        continue
    for i, p in enumerate(prompts):
        print(f"{cat}\t{i}\t{p}")
PY

n=$(wc -l < "$OUT/runlist.tsv"); i=0
echo "$n probes -> $OUT/stats (server: $SERVER_URL, ngen: $MAX_TOKENS)"
while IFS=$'\t' read -r cat idx prompt; do
  i=$((i+1))
  dst="$OUT/stats/${cat}_${idx}.json"
  [ -s "$dst" ] && { echo "  [$i/$n] $cat/$idx (cached)"; continue; }

  # per-probe delta: /experts before -> probe -> /experts after
  curl -sf -m 10 "$SERVER_URL/experts" -o "$OUT/.before.json"
  curl -sf -m 120 "$SERVER_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys;print(json.dumps({"messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":int(sys.argv[2]),"temperature":0,"top_p":1.0,"top_k":0,"min_p":0}))' "$prompt" "$MAX_TOKENS")" \
    > "$OUT/stats/${cat}_${idx}.log" 2>&1 \
    || { echo "  [$i/$n] $cat/$idx GENERATION FAILED"; continue; }
  curl -sf -m 10 "$SERVER_URL/experts" -o "$OUT/.after.json"

  python3 - "$OUT/.before.json" "$OUT/.after.json" "$dst" "$cat" "$idx" <<'PY'
import json, sys
def counts(path):
    d = json.load(open(path))
    if not d.get("telemetry_enabled"):
        return None
    m = bytes.fromhex(d["map"])
    rows, cols = d["rows"], d["cols"]
    g = d["geometry"]
    out = {}
    for r in range(rows):
        real = (g["nextn_rows"][r - len(g["moe_rows"])] if r >= len(g["moe_rows"])
                else g["moe_rows"][r])
        for c in range(cols):
            h = m[(r * cols + c) * 2] & 63
            if h:
                out[f"{real}:{c}"] = h   # heat saturates at 63 (EMAP encoding)
    return out
b, a = counts(sys.argv[1]), counts(sys.argv[2])
cat, idx = sys.argv[4], int(sys.argv[5])
if b is None or a is None:
    json.dump({"telemetry": False, "category": cat, "idx": idx},
              open(sys.argv[3], "w"))
else:
    delta = {k: a.get(k, 0) - b.get(k, 0) for k in a if a.get(k, 0) > b.get(k, 0)}
    json.dump({"telemetry": True, "category": cat, "idx": idx,
               "selections": delta}, open(sys.argv[3], "w"), indent=0)
PY
  echo "  [$i/$n] $cat/$idx done"
done < "$OUT/runlist.tsv"

echo
echo "next:"
echo "  python3 $HERE/analyze.py --stats $OUT/stats --probes $PROBES --out $OUT/experts.json"

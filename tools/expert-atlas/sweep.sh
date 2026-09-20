#!/usr/bin/env bash
# hydra: Colibri expert-atlas probe sweep port (#175) — hydra_vortex#771 / fork#133.
#
#   SERVER_URL=http://127.0.0.1:8086 ./sweep.sh [probes.json] [outdir]
#
# Port of JustVugg/colibri c/tools/expert_atlas/sweep.sh, driving llama-server
# over HTTP instead of the coli CLI. The CONFOUNDS section is Colibri's, kept
# verbatim in spirit:
#
#   temp 0, top-p 1.0, top-k 0, min-p 0
#              Greedy, no distribution pruning: with pruning on you profile
#              the pruner, not the model (Colibri measured -38% experts seen
#              at top-p 0.7 -- and pruning is the recommended speed setting).
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
#              exactly one probe, not a lifetime histogram. S-A2 adds the
#              /capture/flush + /capture/reset endpoints for hidden-state
#              sidecar files; the harness calls flush after each probe.
#
# Prompt lengths are kept in a narrow band across categories (probes.json):
# prefill routes the prompt tokens too, so a verbose category would look busy.
#
# Edge0 sidecar mode (HYDRA_EXPERT_CAPTURE=1):
#   When the engine is started with HYDRA_EXPERT_CAPTURE=1 and
#   HYDRA_CAPTURE_OUTDIR=<dir>, this harness calls POST /capture/flush
#   after each probe to write per-probe sidecar JSONs (hidden states +
#   topk per layer per token). The sidecar files are consumed by
#   analyze_edge0.py for the linear-probe prerouter analysis.

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROBES="${1:-$HERE/probes.json}"
OUT="${2:-./atlas_out}"
SERVER_URL="${SERVER_URL:?set SERVER_URL to the llama-server base URL, e.g. http://127.0.0.1:8086}"
NGEN="${NGEN:-64}"
MAX_TOKENS="${MAX_TOKENS:-$NGEN}"
# Edge0 sidecar: set HYDRA_CAPTURE_OUTDIR to enable sidecar dump alongside stats.
# The engine reads this env var at startup; sweep.sh uses it to locate the sidecar
# files it flushes via /capture/flush. Default is "./atlas_out/sidecars".
SIDECAR_DIR="${HYDRA_CAPTURE_OUTDIR:-$OUT/sidecars}"

[ -f "$PROBES" ] || { echo "no probe file: $PROBES" >&2; exit 1; }
mkdir -p "$OUT/stats" "$SIDECAR_DIR"

# engine reachability + Stage B surface
curl -sf -m 5 "$SERVER_URL/health" >/dev/null || { echo "server unreachable: $SERVER_URL" >&2; exit 1; }
META="$OUT/experts_before.json"
curl -sf -m 10 "$SERVER_URL/experts" -o "$META" || { echo "GET /experts failed -- is HYDRA_EXPERT_META=1 on the engine?" >&2; exit 1; }
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
            h = m[r * cols + c] & 63
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

  # Edge0 S-A2: flush hidden-state sidecar for this probe (when capture enabled)
  curl -sf -m 10 -X POST "$SERVER_URL/capture/flush?cat=$cat&idx=$idx" \
    -o "$OUT/.flush.json" 2>/dev/null && \
    echo "  [$i/$n] $cat/$idx done (sidecar flushed)" || \
    echo "  [$i/$n] $cat/$idx done"
done < "$OUT/runlist.tsv"

echo
echo "next:"
echo "  python3 $HERE/analyze.py --stats $OUT/stats --probes $PROBES --out $OUT/experts.json --meta $OUT/experts_before.json"
echo
echo "Edge0 (when HYDRA_EXPERT_CAPTURE=1):"
echo "  python3 $HERE/analyze_edge0.py --sidecars $SIDECAR_DIR --probes $PROBES --out $OUT/edge0.json --meta $OUT/experts_before.json --model <gguf-name>"

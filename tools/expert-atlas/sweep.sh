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
#
#   HARD REQUIREMENT: the engine MUST be started with
#   HYDRA_CAPTURE_OUTDIR=$OUT/sidecars (same directory sweep uses).
#   sweep.sh validates every flushed sidecar path against this directory.

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

  # hydra F2/F3 follow-up (architect review d-7045376b02): do NOT decode
  # EMAP bytes. EMAP heat is a 6-bit encoded value (F3: bit-length, upstream
  # emap_emit parity) — after-before deltas on map&63 are NOT selection
  # counts, and consumers (tools/atlas/analyze.py sums, validate.py shares)
  # treat "selections" as linear counts, so byte deltas would invert the
  # affinity ranking. Use the engine's exact per-turn routing instead: the
  # newest /turns/<seq> record carries {row, expert, count} from the same
  # decode hook, already mapped to real layers via geometry.
  BEFORE_SEQ=$(python3 - "$OUT/.before.json" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("seq", ""))
except Exception:
    print("")
PY
)
  TURN_SEQ=$(python3 - "$SERVER_URL" <<'PY'
import json, sys, urllib.request
base = sys.argv[1].rstrip("/")
try:
    with urllib.request.urlopen(base + "/turns", timeout=10) as r:
        turns = json.load(r).get("turns") or []
    print(turns[-1]["turn_seq"] if turns else "")
except Exception:
    print("")
PY
)
  # The probe's own turn is the newest seq AFTER the before-snapshot (its
  # generation creates exactly one new turn). If the ring has not advanced
  # (probe failed or produced no decode), no selections are recorded.
  if [ -n "$BEFORE_SEQ" ] && [ -n "$TURN_SEQ" ] && [ "$TURN_SEQ" -gt "$BEFORE_SEQ" ] 2>/dev/null; then
    python3 - "$SERVER_URL" "$TURN_SEQ" "$dst" "$cat" "$idx" <<'PY'
import json, sys, urllib.request

base = sys.argv[1].rstrip("/")
turn_seq = int(sys.argv[2])

def fetch(path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.load(r)

g = fetch("/experts")["geometry"]
moe = g["moe_rows"]
nxt = g["nextn_rows"]
rows = len(moe) + len(nxt)
def real_layer(row):
    return nxt[row - len(moe)] if row >= len(moe) else moe[row]

# Exact per-turn selection counts from the probe's turn (same decode hook
# the EMAP counts come from), MTP draft rows excluded — same rule as the
# placement analysis (draft rows never route in ctx_tgt).
sel = {}
for e in fetch(f"/turns/{turn_seq}").get("routing") or []:
    if 0 <= e["row"] < rows and e["count"] > 0:
        sel[f"{real_layer(e['row'])}:{e['expert']}"] = e["count"]

json.dump({"telemetry": True, "category": sys.argv[3], "idx": int(sys.argv[4]),
           "selections": sel, "source": "turns-routing",
           "turn_seq": turn_seq}, open(sys.argv[2], "w"), indent=0)
PY
  else
    python3 - "$dst" "$cat" "$idx" <<'PY'
import json, sys
json.dump({"telemetry": False, "category": sys.argv[2], "idx": int(sys.argv[3])},
          open(sys.argv[1], "w"))
PY
  fi

  # Edge0 S-A2: flush hidden-state sidecar for this probe (when capture enabled)
  FLUSH_HTTP=$(curl -sf -m 10 -X POST "$SERVER_URL/capture/flush?cat=$cat&idx=$idx" \
    -o "$OUT/.flush.json" -w "%{http_code}" 2>/dev/null) || true
  if [ "$FLUSH_HTTP" = "200" ]; then
    FLUSH_PATH=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('path',''))" "$OUT/.flush.json" 2>/dev/null || true)
    if [ -n "$FLUSH_PATH" ]; then
      FLUSH_REAL=$(realpath "$FLUSH_PATH" 2>/dev/null || echo "$FLUSH_PATH")
      SIDECAR_REAL=$(realpath "$SIDECAR_DIR" 2>/dev/null || echo "$SIDECAR_DIR")
      case "$FLUSH_REAL" in
        "$SIDECAR_REAL"/*) echo "  [$i/$n] $cat/$idx done (sidecar flushed)" ;;
        *) echo "  [$i/$n] $cat/$idx FAIL: engine outdir '$FLUSH_PATH' is not under SIDECAR_DIR '$SIDECAR_DIR' — start engine with HYDRA_CAPTURE_OUTDIR=$SIDECAR_DIR" >&2; exit 1 ;;
      esac
    else
      echo "  [$i/$n] $cat/$idx done (flush OK but no path in response — outdir NOT validated)" >&2
    fi
  else
    echo "  [$i/$n] $cat/$idx done (sidecar flush unavailable (HTTP $FLUSH_HTTP) — delta-only mode, NO sidecars written)" >&2
  fi
done < "$OUT/runlist.tsv"

echo
echo "next:"
echo "  python3 $HERE/analyze.py --stats $OUT/stats --probes $PROBES --out $OUT/experts.json --meta $OUT/experts_before.json"
echo
echo "Edge0 (when HYDRA_EXPERT_CAPTURE=1):"
echo "  python3 $HERE/analyze_edge0.py --sidecars $SIDECAR_DIR --probes $PROBES --out $OUT/edge0.json --meta $OUT/experts_before.json --model <gguf-name>"

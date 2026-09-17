# tools/expert-atlas — Colibri probe harness port (fork side)

Port of JustVugg/colibri `c/tools/expert_atlas/` (sweep.sh + analyze.py +
probes.json) driving **llama-server over HTTP** — hydra_vortex#771 Stage C,
fork issue #133. Confound traps are enforced by this harness, not by memory:

| Trap | Control |
|---|---|
| top-p pruning hides experts (−38% at top-p 0.7) | request `temperature 0, top_p 1.0, top_k 0, min_p 0` |
| MTP/draft counts describe rejected tokens | run the server WITHOUT MTP/spec args; harness asserts no draft fields in responses |
| lifetime histogram pollutes per-probe stats | per-probe **delta** of `GET /experts` (reset-free by construction); upgrades to the Stage-A per-probe reset/dump when S-A2 lands |
| prefill routing is topic-generic | engine counters are decode-only by design (Stage A); deltas exclude prefill |
| prompt-length bias | probes.json keeps lengths in a narrow band (verbatim from Colibri: 10 categories × 3 prompts) |

## Usage

```bash
# engine with the Stage B surface enabled (see tools/server/server-atlas.h)
SERVER_URL=http://127.0.0.1:8086 ./sweep.sh                       # all 30 probes
SERVER_URL=http://127.0.0.1:8086 ./sweep.sh probes.json ./out     # explicit

python3 analyze.py --stats ./out/stats --probes probes.json \
    --out ./out/experts.json --meta ./out/experts_before.json --model <gguf-name>
```

## Honest-state discipline

While Stage-A telemetry is absent (`telemetry_enabled:false` from the engine),
every probe records an honest `{"telemetry": false}` and analyze.py **refuses
to emit an atlas** (exit 2) rather than shipping a zero histogram. The
replication gate (≥2 probes/category, Colibri's ≥2/3 rule) and the ≥2-category
floor also refuse emission. `EMAP heat` saturates at 63 — delta measurement on
saturated counters is lossy; the Stage-A dump (absolute triples) is the
production stats path once it lands.

## Files

- `probes.json` — verbatim from Colibri @a8f2ca62 (10 categories × 3 prompts)
- `sweep.sh` — driver: greedy confound-controlled generation + per-probe delta
- `analyze.py` — statistics port (mean share, p(c|e), spec = 1 − H/log C,
  replication gate) emitting the observability-tier `experts.json` + provenance
- `export_pinfile.py` — `experts.json`/`expert-ranks.json` → pin file
  (`L <il> <ids...>`, hot-first, wrap-aware). Engine-agnostic: `--expect-engine-id`
  refusal + optional `--check-url` live geometry match. Round-trip verified
  byte-identical to the parent repo artifact
- `validate.py` — leave-one-prompt-out validation (added when real spectra exist)

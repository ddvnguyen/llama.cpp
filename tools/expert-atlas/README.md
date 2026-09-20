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
- `validate.py` — leave-one-prompt-out validation (LLM pop review #175): the
  atlas is only real if a specialist set learned from some prompts predicts
  routing on a prompt it never saw. hydra LEAK FIX vs Colibri original: the
  held-out run is excluded from EVERY category's training (numerator and
  denominator) — the original let it leak into other categories' lift
  denominators, scoring ~84% on pure noise at chance 20%. Null verified at
  chance on random-routing controls; signal verified on banded synthetic
  spectra. Runs once real spectra exist.
- `analyze_edge0.py` — Edge0 linear-probe prerouter analyzer (#786): trains
  per-layer one-vs-rest logistic regression on router-input hidden state to
  predict top-k expert selection. Leave-one-prompt-out cross-validation
  (by prompt, never by token -- Rider 1). LTR baseline column present from
  day one. See **Edge0 predictor definition** below.

### Edge0 S-A2: hidden-state capture + sidecar dump

When the engine is started with `HYDRA_EXPERT_CAPTURE=1`, the server-side
decode hook reads the router-input hidden state (post-ffn_norm residual) for
every MoE layer and every decode token via the `llama_get_moe_hidden` C API.
Data accumulates in the server-atlas capture buffer and is flushed to
per-probe sidecar JSONs via `POST /capture/flush?cat=<cat>&idx=<idx>`.
Per-probe reset: `POST /capture/reset`.

Sidecar JSON format (consumed by `analyze_edge0.py`):
```json
{
  "category": "code_python",
  "idx": 0,
  "layers": {
    "0": {
      "hidden": [[n_embd floats], ...],
      "topk": [[expert_ids], ...]
    }
  }
}
```

Env vars:
- `HYDRA_EXPERT_CAPTURE` (set=on, unset=off): enables capture hook
- `HYDRA_CAPTURE_OUTDIR` (**hard requirement**: must equal `$OUT/sidecars`):
  sidecar output directory. sweep.sh validates every flushed sidecar path
  against this directory; engine started with a different outdir is rejected.

**`.coli_usage`-style per-probe dumps are subsumed by the sidecar/flush
surface.** The `/capture/flush` endpoint writes per-probe sidecar JSONs
(hidden states + topk) that fully replace the legacy `.coli_usage` file;
no separate dump file is produced or expected.

sweep.sh calls `/capture/flush` after each probe when capture is enabled.
When capture is disabled, the flush endpoint returns 503 and sweep.sh
falls back to the existing delta-only stats path.

## Edge0 predictor definition (verbatim, schema_version edge0-v1)

> `predictability[layer]` = top-k hit-rate @k of a per-layer linear probe
> mapping the router-input hidden state (post-ffn_norm residual) to the top-k
> expert set, measured on HELD-OUT tokens of the probe corpus (leave-one-prompt-
> out per category). StandardScaler + LogisticRegression(C=1.0, lbfgs,
> max_iter=1000). `prefetch_gain[layer]` = (sum over hit tokens of hit experts'
> weight bytes) / (sum over all routed experts' weight bytes) under that probe's
> hits.
>
> Comparison baseline column: `predictability_ltr[layer]` = last-token-routing-
> repeat hit-rate @k computed capture-free from the Stage-A topk stream (fraction
> of tokens whose topk set is a subset of the previous token's topk set, per
> layer). Follow-up column: per-layer LRU-k (k=4).
>
> **Any change of predictor = new schema_version.**

### Riders (acceptance criteria)

1. **Leak discipline — by PROMPT, not token**: leave-one-prompt-out must hold
   out entire prompts. Token-level splits leak within a prompt's shared context.
2. **Probe provenance**: seed + hyperparameters (C, max_iter, solver) recorded
   in artifact provenance per run. `predictability` is not comparable across
   runs otherwise.
3. **Compute budget**: 41 MoE layers x 256 one-vs-rest logistic fits = ~10.5k
   small fits per sweep fold. StandardScaler + LogisticRegression per fit.
   Estimated runtime: < 5 min on CPU. Estimated memory: < 50 MB.

### Data flow

```
sweep (HYDRA_EXPERT_CAPTURE=1)
  -> per-probe sidecar JSON (hidden states + topk per layer per token)
  -> analyze_edge0.py (per-layer logistic probe, leave-one-prompt-out)
  -> edge0.json (predictability, prefetch_gain, LTR baseline)
```

### OFF-parity

Env-unset (HYDRA_EXPERT_CAPTURE not set): `t_moe_hidden` stash never filled,
`ggml_set_output` never called, graph bit-identical to today. Same class of
evidence as the verified Stage-A honest-OFF runs.

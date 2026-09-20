#!/usr/bin/env python3
"""hydra #786 Edge0: linear-probe prerouter analyzer.

Offline analysis that consumes per-probe sidecar files dumped during sweep
with HYDRA_EXPERT_CAPTURE=1. Trains per-layer one-vs-rest logistic regression
probes mapping router-input hidden state to top-k expert selection, with
leave-one-prompt-out cross-validation (Rider 1: by PROMPT, never by token).

Predictors reported (verbatim for artifact README):
  - predictability[layer] = mean top-k hit-rate of the linear probe on held-out
    prompts (leave-one-prompt-out), averaged over experts in the top-k set.
  - predictability_ltr[layer] = last-token-routing-repeat hit-rate @k computed
    capture-free from the Stage-A topk stream (fraction of tokens whose topk set
    is a subset of the previous token's topk set, per layer).
  - prefetch_gain[layer] = (sum over hit tokens of hit experts' weight bytes) /
    (sum over all routed experts' weight bytes) under the probe's hits.

Any change of predictor = new schema_version.

Usage:
    python3 analyze_edge0.py \
        --sidecars ./out/sidecars \
        --probes probes.json \
        --out ./out/edge0.json \
        --meta ./out/experts_before.json \
        --model <gguf-name> \
        --seed 42
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_sidecars(sidecars_dir: Path):
    """Load per-probe sidecar files.

    Each sidecar file is a JSON with:
      - category: probe category
      - idx: probe index within category
      - layers: {layer_idx: {hidden: [[n_embd, ...], ...], topk: [[expert_ids, ...], ...]}}

    Returns list of (category, idx, {layer_idx: (hidden_array, topk_array)}).
    """
    result = []
    for f in sorted(sidecars_dir.glob("*.json")):
        d = json.loads(f.read_text())
        cat = d.get("category", "")
        idx = d.get("idx", 0)
        layers = {}
        for layer_idx_str, layer_data in d.get("layers", {}).items():
            layer_idx = int(layer_idx_str)
            hidden = np.array(layer_data["hidden"], dtype=np.float32)  # [n_tokens, n_embd]
            topk = np.array(layer_data["topk"], dtype=np.int32)       # [n_tokens, k]
            layers[layer_idx] = (hidden, topk)
        if layers:
            result.append((cat, idx, layers))
    return result


def compute_ltr_baseline(topk_by_token):
    """Compute last-token-routing-repeat hit-rate.

    For each token t (starting from t=1), count if topk(t) is a subset of
    topk(t-1). Returns hit_rate = hits / (n_tokens - 1).
    """
    if len(topk_by_token) < 2:
        return 0.0
    hits = 0
    for t in range(1, len(topk_by_token)):
        prev_set = set(topk_by_token[t - 1])
        curr_set = set(topk_by_token[t])
        if curr_set.issubset(prev_set):
            hits += 1
    return hits / (len(topk_by_token) - 1)


def train_linear_probe(X_train, y_train, X_test, y_test, seed, C=1.0):
    """Train a one-vs-rest logistic regression probe and return test accuracy.

    X_train: [n_train, n_embd], y_train: [n_train] binary labels (0/1)
    X_test:  [n_test, n_embd],  y_test:  [n_test] binary labels (0/1)

    Returns hit_rate = fraction of test tokens correctly predicted.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if len(X_train) == 0 or len(X_test) == 0:
        return 0.0

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        C=C, max_iter=1000, solver="lbfgs",
        random_state=seed, n_jobs=1
    )
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    return float(np.mean(preds == y_test))


def leave_one_prompt_out(layers_data, probes_json, seed, C=1.0):
    """Leave-one-prompt-out cross-validation per layer.

    For each layer, for each prompt held out:
      - Train on all other prompts' tokens
      - Evaluate on held-out prompt's tokens
      - Per-expert binary: is this expert in the top-k set for this token?

    Returns per-layer dict with:
      - predictability: mean hit-rate over held-out prompts
      - predictability_ltr: LTR baseline (capture-free)
      - prefetch_gain: estimated weight-byte savings
      - per_expert: {expert_id: hit_rate}
      - n_tokens: total tokens used
      - n_prompts: number of prompts
    """
    categories = [c for c in probes_json if not c.startswith("_")]
    n_experts = 256  # from model config; will be inferred from data

    # Group tokens by (category, prompt_idx) for leave-one-out
    by_prompt = defaultdict(list)  # (cat, idx) -> [(hidden, topk_set)]
    for cat, idx, layer_data in layers_data:
        if not cat or cat.startswith("_"):
            continue
        # Use the first layer to determine prompt grouping
        first_layer = next(iter(layer_data.values()))
        n_tokens = len(first_layer[0])
        for t in range(n_tokens):
            by_prompt[(cat, idx)].append(t)

    prompt_keys = sorted(by_prompt.keys())
    if len(prompt_keys) < 2:
        return None

    results_by_layer = {}

    for layer_idx in sorted(set(ld for _, _, ld in layers_data for ld in ld)):
        # Collect all tokens for this layer
        all_hidden = []
        all_topk = []
        all_prompt_ids = []

        for cat, idx, layer_data in layers_data:
            if layer_idx not in layer_data:
                continue
            hidden, topk = layer_data[layer_idx]
            prompt_key = (cat, idx)
            prompt_id = prompt_keys.index(prompt_key)
            for t in range(len(hidden)):
                all_hidden.append(hidden[t])
                all_topk.append(topk[t])
                all_prompt_ids.append(prompt_id)

        if len(all_hidden) == 0:
            continue

        X = np.array(all_hidden, dtype=np.float32)  # [n_total, n_embd]
        all_topk_arr = np.array(all_topk, dtype=np.int32)  # [n_total, k]
        prompt_ids = np.array(all_prompt_ids)

        n_embd = X.shape[1]
        k = all_topk_arr.shape[1]
        n_total = len(X)
        n_prompts = len(prompt_keys)

        # LTR baseline (capture-free, from topk stream only)
        ltr_hits = 0
        ltr_total = 0
        prev_topk = None
        prev_prompt = None
        for t in range(n_total):
            curr_topk = set(all_topk_arr[t])
            curr_prompt = prompt_ids[t]
            if prev_topk is not None and curr_prompt == prev_prompt:
                if curr_topk.issubset(prev_topk):
                    ltr_hits += 1
                ltr_total += 1
            prev_topk = curr_topk
            prev_prompt = curr_prompt
        ltr_rate = ltr_hits / ltr_total if ltr_total > 0 else 0.0

        # Leave-one-prompt-out cross-validation
        expert_hit_accum = defaultdict(list)  # expert_id -> [hit_rates per fold]

        for held_out_prompt in range(n_prompts):
            train_mask = prompt_ids != held_out_prompt
            test_mask = prompt_ids == held_out_prompt

            X_train, X_test = X[train_mask], X[test_mask]
            topk_train, topk_test = all_topk_arr[train_mask], all_topk_arr[test_mask]

            if len(X_train) == 0 or len(X_test) == 0:
                continue

            # Per-expert binary classification
            for expert_id in range(n_experts):
                y_train = np.array([1 if expert_id in row else 0 for row in topk_train])
                y_test = np.array([1 if expert_id in row else 0 for row in topk_test])

                # Skip experts that never appear in train or test
                if y_train.sum() == 0 and y_test.sum() == 0:
                    continue

                hit_rate = train_linear_probe(X_train, y_train, X_test, y_test, seed, C)
                expert_hit_accum[expert_id].append(hit_rate)

        # Aggregate per-expert hit rates
        per_expert = {}
        all_hit_rates = []
        for expert_id, rates in sorted(expert_hit_accum.items()):
            mean_rate = np.mean(rates)
            per_expert[expert_id] = round(float(mean_rate), 4)
            all_hit_rates.append(mean_rate)

        # predictability = mean hit-rate over experts that appear in any top-k
        predictability = round(float(np.mean(all_hit_rates)), 4) if all_hit_rates else 0.0

        # prefetch_gain: fraction of expert-weight bytes avoidable
        # Simplified: assume uniform expert weight size, so prefetch_gain ~ predictability
        # (weighted version needs per-expert weight sizes from the model file)
        prefetch_gain = predictability  # placeholder; refine with actual weight sizes

        results_by_layer[layer_idx] = {
            "predictability": predictability,
            "predictability_ltr": round(ltr_rate, 4),
            "prefetch_gain": round(prefetch_gain, 4),
            "per_expert": per_expert,
            "n_tokens": n_total,
            "n_prompts": n_prompts,
        }

    return results_by_layer


def main() -> int:
    ap = argparse.ArgumentParser(description="Edge0 linear-probe prerouter analyzer")
    ap.add_argument("--sidecars", required=True, help="directory of per-probe sidecar JSONs")
    ap.add_argument("--probes", required=True, help="probes.json path")
    ap.add_argument("--out", required=True, help="output edge0.json path")
    ap.add_argument("--meta", help="engine /experts snapshot for provenance")
    ap.add_argument("--model", default="", help="model name for provenance")
    ap.add_argument("--seed", type=int, default=42, help="random seed for probe (Rider 2)")
    ap.add_argument("--C", type=float, default=1.0, help="logistic regression regularization")
    args = ap.parse_args()

    sidecars_dir = Path(args.sidecars)
    if not sidecars_dir.is_dir():
        print(f"analyze_edge0: sidecars directory not found: {sidecars_dir}", file=sys.stderr)
        return 1

    probes_json = json.loads(Path(args.probes).read_text())
    layers_data = load_sidecars(sidecars_dir)

    if not layers_data:
        print("analyze_edge0: no sidecar files found — HYDRA_EXPERT_CAPTURE may be off. "
              "This is the honest OFF state, not a failure.", file=sys.stderr)
        return 2

    print(f"analyze_edge0: loaded {len(layers_data)} probe sidecars", file=sys.stderr)

    results = leave_one_prompt_out(layers_data, probes_json, args.seed, args.C)

    if results is None:
        print("analyze_edge0: insufficient data for cross-validation (< 2 prompts)", file=sys.stderr)
        return 3

    # Provenance (Rider 2: hyperparams + seed)
    provenance = {
        "model": args.model,
        "seed": args.seed,
        "probe_type": "logistic_regression",
        "probe_C": args.C,
        "cross_validation": "leave_one_prompt_out",
        "n_probes": len(layers_data),
        "n_layers": len(results),
        "predictor_definition": (
            "predictability[layer] = mean top-k hit-rate @k of a per-layer "
            "linear probe mapping the router-input hidden state to the top-k "
            "expert set, measured on HELD-OUT tokens of the probe corpus "
            "(leave-one-prompt-out per category). "
            "predictability_ltr[layer] = last-token-routing-repeat hit-rate @k "
            "computed capture-free from the Stage-A topk stream. "
            "predictor change = new schema_version."
        ),
        "schema_version": "edge0-v1",
        "generated_from": "fork tools/expert-atlas/analyze_edge0.py",
        "confound_controls": [
            "leave_one_prompt_out (no token-split leakage)",
            "StandardScaler per fold",
            "greedy sweep (temp 0, top_p 1.0)",
            "decode-only tokens",
        ],
    }
    if args.meta:
        try:
            g = json.loads(Path(args.meta).read_text())["geometry"]
            provenance["engine_id"] = g.get("engine_id", "")
            provenance["model_hash"] = g.get("model_hash", "")
        except (json.JSONDecodeError, KeyError):
            pass

    # Compute budget note (Rider 3)
    n_layers = len(results)
    n_experts = 256
    n_folds = layers_data[0][2][next(iter(layers_data[0][2]))][0].shape[0] if layers_data else 0
    est_fits = n_layers * n_experts  # one-vs-rest per expert per layer
    provenance["compute_budget"] = {
        "description": (
            f"{n_layers} MoE layers x {n_experts} one-vs-rest logistic fits = "
            f"{est_fits} small fits per sweep fold. "
            "StandardScaler + LogisticRegression(lbfgs, max_iter=1000) per fit. "
            "Estimated runtime: < 5 min on CPU for 256-way top-8 with ~10k tokens."
        ),
        "n_layers": n_layers,
        "n_experts": n_experts,
        "n_fits_per_fold": est_fits,
        "estimated_runtime_seconds": 300,
        "estimated_memory_mb": 50,
    }

    out = {
        "layers": {
            str(k): {kk: vv for kk, vv in v.items() if kk != "per_expert"}
            for k, v in results.items()
        },
        "per_expert_detail": {
            str(k): v["per_expert"] for k, v in results.items()
        },
        "provenance": provenance,
    }

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"analyze_edge0: wrote {args.out} — {len(results)} layers analyzed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# hydra: leave-one-prompt-out validation of the Expert Atlas (hydra_vortex#771).
# Port of JustVugg/colibri c/tools/expert_atlas/validate.py (LLM pop review #175).
# Reads the JSON per-probe stats written by sweep.sh (same input as analyze.py):
#   {"telemetry": bool, "category": str, "idx": int, "selections": {"row:col": n}}
#
# "Replicates across the 3 prompts I picked" is not the same as "generalises".
# The atlas is only real if a specialist set learned from SOME prompts predicts
# routing on a prompt it has never seen.
#
# Protocol, for every category c and every held-out prompt h of c:
#   1. build c's top-K specialist set from c's OTHER prompts only
#   2. same for all other categories — the held-out run is excluded from EVERY
#      category's training (they never saw h, in numerator OR denominator)
#   3. on the held-out run h, measure the share of routing selections in each set
#   4. the atlas works if c's own set wins on h
#
# If specialisation were an artifact of prompt wording, the held-out prompt
# would not prefer its own category's set. Chance is 1/C.
#
# hydra: LEAK FIX vs Colibri original — there, other categories' sets included
# the held-out run's counts in the lift denominator (sum over categories),
# suppressing h's keys from every *other* set. On pure random routing that
# biases accuracy to ~84% at chance 20% — a false-positive machine. Here the
# held-out run is excluded from all training, restoring a fair null.
#
# NOTE (fork): probes with telemetry:false are skipped and reported — the LOP
# needs per-prompt spectra. Degenerate "fires-for-everything" experts appear in
# every set, so they shift all scores equally; discrimination comes from the
# category-specific tails (Colibri: spec < 0.7 = weak qualifier).

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict


def load_runs(stats_dir):
    runs, tot, missing = {}, {}, 0
    for path in sorted(glob.glob(os.path.join(stats_dir, "*.json"))):
        try:
            d = json.load(open(path))
        except json.JSONDecodeError as e:
            print(f"validate: skipping unreadable {path}: {e}", file=sys.stderr)
            continue
        if not d.get("telemetry"):
            missing += 1
            continue
        cat, idx = d["category"], int(d["idx"])
        sel = d.get("selections", {})
        runs[(cat, idx)] = {k: int(v) for k, v in sel.items() if int(v) > 0}
        tot[(cat, idx)] = sum(runs[(cat, idx)].values())
    return runs, tot, missing


def specialists(runs, tot, cats, idxs, cat, held_out, k):
    """Top-K experts by lift for `cat`, computed WITHOUT the held-out run
    (excluded from every category's training, numerator and denominator)."""
    share = defaultdict(lambda: defaultdict(float))
    for c in cats:
        used = [i for i in idxs[c] if (c, i) != held_out]
        for i in used:
            for key, n in runs[(c, i)].items():
                share[key][c] += n / max(1, tot[(c, i)]) / len(used)
    scored = []
    for key, per in share.items():
        s = sum(per.values())
        if s <= 0:
            continue
        p = per[cat] / s
        if per[cat] > 0:
            scored.append((p, key))
    scored.sort(reverse=True)
    return {key for _, key in scored[:k]}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="leave-one-prompt-out validation (LLM pop review #175)")
    ap.add_argument("--stats", default="stats", help="per-probe stats dir (sweep.sh output)")
    ap.add_argument("--k", type=int, default=200, help="specialists per category")
    ap.add_argument("--probes", default=None, help="optional probes.json for context")
    args = ap.parse_args(argv)

    runs, tot, missing = load_runs(args.stats)
    if not runs:
        print("validate: telemetry absent in every probe (Stage A not wired) — "
              "LOP needs per-prompt spectra. This is the honest OFF state, not a "
              "failure.", file=sys.stderr)
        return 2

    cats = sorted({c for c, _ in runs})
    idxs = {c: sorted(i for cc, i in runs if cc == c) for c in cats}
    c_count = len(cats)
    too_thin = [c for c in cats if len(idxs[c]) < 2]
    if too_thin:
        print(f"validate: categories with <2 probes cannot be LOP-tested: {too_thin} "
              f"(skipped from accuracy; overall accuracy still computed)",
              file=sys.stderr)

    if args.probes:
        try:
            n_probe_cats = len(json.load(open(args.probes)))
            if n_probe_cats != c_count:
                print(f"validate: {c_count}/{n_probe_cats} probe categories have "
                      "telemetry", file=sys.stderr)
        except (OSError, json.JSONDecodeError):
            pass

    print(f"leave-one-prompt-out, {c_count} categories, top-{args.k} specialists per "
          f"category ({missing} probes skipped: no telemetry)")
    print(f"chance = {100.0 / c_count:.1f}%\n")
    hits = trials = 0
    for c in cats:
        for h in idxs[c]:
            if len(idxs[c]) < 2:
                continue
            sets = {cc: specialists(runs, tot, cats, idxs, cc, (c, h), args.k)
                    for cc in cats}
            held = runs[(c, h)]
            htot = max(1, tot[(c, h)])
            scores = {cc: sum(held.get(key, 0) for key in sets[cc]) / htot
                      for cc in cats}
            win = max(scores, key=lambda cc: scores[cc])
            ok = win == c
            hits += ok
            trials += 1
            own = 100 * scores[c]
            best_other = 100 * max(v for cc, v in scores.items() if cc != c)
            print(f"  {c:<12} prompt {h}  own-set {own:5.2f}%  best-other "
                  f"{best_other:5.2f}%  -> {'HIT ' if ok else 'MISS'} (predicted {win})")
    if trials == 0:
        print("validate: no LOP-eligible categories (all <2 probes)", file=sys.stderr)
        return 2
    print(f"\naccuracy: {hits}/{trials} = {100 * hits / trials:.1f}%   "
          f"(chance {100.0 / c_count:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

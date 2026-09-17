#!/usr/bin/env python3
# hydra: Colibri expert-atlas analyzer port (#175) — hydra_vortex#771 / fork#133.
# Statistics follow Colibri's c/tools/expert_atlas/analyze.py semantics and our
# parent-repo tools/atlas implementation (mean share, base-rate-corrected
# p(c|e), spec = 1 - H/log C, replication gate). Consumes per-probe JSONs
# emitted by sweep.sh; emits the observability-tier experts.json (design §C)
# plus provenance. Refuses to emit an atlas when telemetry is absent or the
# replication gate fails — a bare histogram must never ship as an atlas.

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def load_stats(stats_dir: Path):
    per_probe = []          # (category, {layer:expert: count})
    for f in sorted(stats_dir.glob("*.json")):
        d = json.loads(f.read_text())
        if not d.get("telemetry"):
            continue
        sel = d.get("selections") or {}
        if sel:
            per_probe.append((d["category"], sel))
    return per_probe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", help="engine /experts snapshot for provenance (geometry)")
    ap.add_argument("--model", default="")
    ap.add_argument("--min-runs", type=int, default=2, help="replication gate: categories need >= this many probes")
    args = ap.parse_args()

    stats_dir = Path(args.stats)
    probes = json.loads(Path(args.probes).read_text())
    categories = [c for c in probes if not c.startswith("_")]

    per_probe = load_stats(stats_dir)
    if not per_probe:
        print("analyze: telemetry absent in every probe (Stage A not wired) — "
              "no atlas emitted. This is the honest OFF state, not a failure.", file=sys.stderr)
        return 2

    # replication gate: enough probes per DATA-PRESENT category (Colibri >= 2/3
    # runs). Categories with zero successful probes are reported in provenance —
    # they contribute no signal but must not silently inflate affinity p(c|e).
    by_cat = defaultdict(int)
    for cat, _ in per_probe:
        by_cat[cat] += 1
    present = [c for c in categories if by_cat.get(c, 0) > 0]
    failing = [c for c in present if by_cat[c] < args.min_runs]
    if failing:
        print(f"analyze: replication gate FAILED for {failing} (< {args.min_runs} probes) — "
              "no atlas emitted.", file=sys.stderr)
        return 3

    # totals per category and global
    totals = defaultdict(int)          # cat -> selections
    per_expert = defaultdict(lambda: defaultdict(int))  # cat -> (layer:expert) -> count
    for cat, sel in per_probe:
        for k, v in sel.items():
            totals[cat] += v
            per_expert[cat][k] += v

    grand = sum(totals.values())
    if grand == 0:
        print("analyze: zero selections recorded — no atlas emitted.", file=sys.stderr)
        return 2

    n_cat = len([c for c in categories if totals.get(c, 0) > 0])
    if n_cat < 2:
        print("analyze: fewer than 2 categories produced selections — "
              "affinity is undefined, no atlas emitted.", file=sys.stderr)
        return 3

    experts = {}
    keys = set()
    for cat in per_expert:
        keys.update(per_expert[cat].keys())
    for key in keys:
        aff = {}
        h = 0.0
        for cat in categories:
            if totals.get(cat, 0) == 0:
                continue
            p = per_expert[cat].get(key, 0) / totals[cat]   # base-rate-corrected share
            aff[cat] = round(p, 4)
        # entropy over p(c|e) normalized
        s = sum(aff.values())
        if s > 0:
            for p in aff.values():
                pn = p / s
                if pn > 0:
                    h -= pn * math.log2(pn)
        top = max(aff, key=lambda c: aff[c]) if aff else ""
        spec = max(0.0, 1.0 - h / math.log2(len(aff))) if len(aff) > 1 else 1.0
        label = f"specialist: {top}" if spec >= 0.7 else "generalist"
        experts[key] = {
            "affinity": aff,
            "entropy": round(h, 4),
            "top": top,
            "label": label,
            "spec": round(spec, 4),
            "reliability": f"{by_cat.get(top, 0)}/{len(per_probe)}",
        }

    provenance = {
        "model": args.model,
        "categories_missing": [c for c in categories if by_cat.get(c, 0) == 0],
        "probe_set": categories,
        "generated_from": "fork tools/expert-atlas sweep (decode-only, greedy, per-probe delta)",
        "confound_controls": ["temp 0", "top_p 1.0", "top_k 0", "min_p 0", "MTP off", "decode-only", "per-probe delta"],
        "replication_gate": f">= {args.min_runs} probes/category",
        "n_probes": len(per_probe),
    }
    if args.meta:
        g = json.loads(Path(args.meta).read_text())["geometry"]
        provenance["engine_id"] = g["engine_id"]
        provenance["model_hash"] = g["model_hash"]

    out = {"categories": categories, "experts": experts, "provenance": provenance}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"analyze: wrote {args.out} — {len(experts)} experts across {n_cat} categories")
    return 0


if __name__ == "__main__":
    sys.exit(main())

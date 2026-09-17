#!/usr/bin/env python3
# hydra: expert-ranks.json -> in-tree pin-file exporter (hydra_vortex#771 §C).
# Port of the parent-repo tools/atlas exporter, made engine-agnostic for the
# fork: no hardcoded model — the expected engine_id is a CLI arg, optionally
# verified live against the engine's Stage-B geometry (engine-id refusal
# discipline, colibri route_trace.h: histories/artifacts from another engine
# are never consumed).
#
# Pin-file format (llama-context.cpp hydra_cpu_init; identical parser in
# ggml-cuda.cu):
#   - '#' lines and blank lines are skipped
#   - each pin line: "L <il> <ids...>"; il must be in [0, 256)
#   - ORDER IS SIGNIFICANT: ids are hot-first (design §4 contract)
#   - long layers wrap into continuation "L <il> ..." lines (the parser
#     appends same-il lines)

from __future__ import annotations

import argparse
import json
import urllib.request


def layer_ids(ranks, il):
    return [e["id"] for e in ranks["layers"][str(il)]["experts"]]


def check_engine(ranks, url):
    with urllib.request.urlopen(f"{url.rstrip('/')}/experts", timeout=10) as r:
        live = json.load(r)
    g = live["geometry"]
    if g["engine_id"] != ranks.get("engine_id"):
        raise SystemExit(f"refusing: engine serves engine_id={g['engine_id']!r} "
                         f"but ranks carry {ranks.get('engine_id')!r}")
    if g.get("model_hash") and ranks.get("model_hash") and \
            g["model_hash"] != ranks.get("model_hash"):
        raise SystemExit(f"refusing: engine model_hash={g['model_hash']!r} "
                         f"but ranks carry {ranks.get('model_hash')!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="expert-ranks.json -> pin file (fork-side)")
    ap.add_argument("--ranks", default="expert-ranks.json")
    ap.add_argument("--out", default="experts.pin")
    ap.add_argument("--expect-engine-id", required=True,
                    help="engine_id the ranks must carry (refusal discipline)")
    ap.add_argument("--check-url", default=None,
                    help="optional: verify against a live engine's /experts geometry")
    ap.add_argument("--top-n", type=int, default=None,
                    help="export only the top-N hottest per layer (default all)")
    ap.add_argument("--wrap", type=int, default=500,
                    help="max ids per line (parser line buffer is 8192 chars)")
    args = ap.parse_args(argv)

    with open(args.ranks) as fh:
        ranks = json.load(fh)
    if ranks.get("engine_id") != args.expect_engine_id:
        raise SystemExit(f"refusing: ranks engine_id={ranks.get('engine_id')!r} "
                         f"is not {args.expect_engine_id!r}")
    if args.check_url:
        check_engine(ranks, args.check_url)

    lines = [f"# expert-ranks pin export  engine_id={ranks['engine_id']}  "
             f"model={ranks.get('model_hash')}  version={ranks.get('version')}"]
    corpus = ranks.get("provenance", {}).get("corpus", {})
    if corpus.get("quality") == "draft-grade":
        n_domains = len(corpus.get("runs", []))
        lines.append(f"# DRAFT-GRADE corpus ({n_domains}-domain); "
                     "re-export after multi-domain probes")
    n_layers = n_ids = 0
    for il in sorted(ranks["layers"], key=int):
        ids = layer_ids(ranks, il)
        if args.top_n is not None:
            ids = ids[: args.top_n]
        if not ids:
            continue
        n_layers += 1
        for chunk_start in range(0, len(ids), args.wrap):
            chunk = ids[chunk_start: chunk_start + args.wrap]
            lines.append("L " + str(il) + " " + " ".join(map(str, chunk)))
            n_ids += len(chunk)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {n_layers} layers, {n_ids} pin ids "
          f"(top-n={args.top_n or 'all'}, engine_id={ranks['engine_id']})")


if __name__ == "__main__":
    main()

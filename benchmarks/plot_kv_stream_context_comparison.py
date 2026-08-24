#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


BLUE = "#1f77b4"
ORANGE = "#ff7f0e"


def load_series(directory: Path, expected_variant: str):
    rows = {}
    for path in sorted(directory.glob("*.jsonl")):
        metadata = None
        measurement = None
        with path.open() as handle:
            for line in handle:
                item = json.loads(line)
                if item.get("type") == "metadata":
                    metadata = item
                elif item.get("type") == "measurement":
                    measurement = item

        if measurement is None:
            continue
        if measurement["variant"] != expected_variant:
            raise ValueError(
                f"{path}: expected {expected_variant}, got {measurement['variant']}"
            )
        if measurement.get("status") != "ok" or measurement.get("predicted_n") != 256:
            raise ValueError(f"{path}: incomplete measurement: {measurement}")

        context = measurement["context"]
        if context in rows:
            raise ValueError(f"duplicate context {context} in {directory}")
        rows[context] = {
            "context": context,
            "prefill_tps": measurement["prefill_tps"],
            "decode_tps": measurement["decode_tps"],
            "vram_after_mib": measurement.get("vram_after_mib"),
            "adaptive_pool_mib": (
                metadata.get("adaptive_pool_mib") if metadata is not None else None
            ),
        }
    return rows


def validate_pair(stock, adaptive):
    expected = set(range(8192, 196608 + 1, 8192))
    if set(stock) != expected:
        raise ValueError(f"stock contexts differ: {sorted(expected - set(stock))}")
    if set(adaptive) != expected:
        raise ValueError(f"adaptive contexts differ: {sorted(expected - set(adaptive))}")


def write_csv(path: Path, stock, adaptive):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "context_tokens",
                "stock_prefill_tps",
                "adaptive_prefill_tps",
                "prefill_change_percent",
                "stock_decode_tps",
                "adaptive_decode_tps",
                "decode_change_percent",
                "stock_vram_mib",
                "adaptive_vram_mib",
                "adaptive_pool_mib",
            ]
        )
        for context in sorted(stock):
            s = stock[context]
            a = adaptive[context]
            writer.writerow(
                [
                    context,
                    s["prefill_tps"],
                    a["prefill_tps"],
                    100 * (a["prefill_tps"] / s["prefill_tps"] - 1),
                    s["decode_tps"],
                    a["decode_tps"],
                    100 * (a["decode_tps"] / s["decode_tps"] - 1),
                    s["vram_after_mib"],
                    a["vram_after_mib"],
                    a["adaptive_pool_mib"],
                ]
            )


def plot(path: Path, title: str, subtitle: str, stock, adaptive):
    contexts = sorted(stock)
    x = [context / 1024 for context in contexts]

    fig, decode_ax = plt.subplots(figsize=(13, 7.5), constrained_layout=True)
    prefill_ax = decode_ax.twinx()

    decode_ax.plot(
        x,
        [stock[c]["decode_tps"] for c in contexts],
        color=BLUE,
        marker="o",
        linewidth=2.4,
        markersize=4.5,
        label="Stock UVM — decode",
    )
    decode_ax.plot(
        x,
        [adaptive[c]["decode_tps"] for c in contexts],
        color=ORANGE,
        marker="o",
        linewidth=2.4,
        markersize=4.5,
        label="Adaptive KV stream — decode",
    )
    prefill_ax.plot(
        x,
        [stock[c]["prefill_tps"] for c in contexts],
        color=BLUE,
        linestyle="--",
        linewidth=2.0,
        alpha=0.85,
        label="Stock UVM — prefill",
    )
    prefill_ax.plot(
        x,
        [adaptive[c]["prefill_tps"] for c in contexts],
        color=ORANGE,
        linestyle="--",
        linewidth=2.0,
        alpha=0.85,
        label="Adaptive KV stream — prefill",
    )

    decode_ax.set_title(f"{title}\n{subtitle}", fontsize=14, pad=14)
    decode_ax.set_xlabel("Prompt context (Ki tokens)")
    decode_ax.set_ylabel("Decode speed (tokens/s) — solid lines")
    prefill_ax.set_ylabel("Prefill speed (tokens/s) — dashed lines")
    decode_ax.set_xticks(x[::2])
    decode_ax.set_xlim(min(x) - 2, max(x) + 2)
    decode_ax.set_ylim(bottom=0)
    prefill_ax.set_ylim(bottom=0)
    decode_ax.grid(True, alpha=0.25)

    handles1, labels1 = decode_ax.get_legend_handles_labels()
    handles2, labels2 = prefill_ax.get_legend_handles_labels()
    decode_ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def process(output_dir, stem, title, subtitle, stock_dir, adaptive_dir):
    stock = load_series(stock_dir, "stock_uvm")
    adaptive = load_series(adaptive_dir, "adaptive_cuda")
    validate_pair(stock, adaptive)
    write_csv(output_dir / f"{stem}.csv", stock, adaptive)
    plot(output_dir / f"{stem}.png", title, subtitle, stock, adaptive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-stock", type=Path, required=True)
    parser.add_argument("--matched-adaptive", type=Path, required=True)
    parser.add_argument("--fixed-stock", type=Path, required=True)
    parser.add_argument("--fixed-adaptive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    process(
        args.output_dir,
        "kv-stream-context-matched-comparison",
        "Stock UVM vs adaptive KV streaming — context-matched capacity",
        "Fresh server per point; --ctx-size = prompt + 256; 256-token decode",
        args.matched_stock,
        args.matched_adaptive,
    )
    process(
        args.output_dir,
        "kv-stream-fixed-192k-comparison",
        "Stock UVM vs adaptive KV streaming — fixed 192K capacity",
        "Fresh server per point; --ctx-size = 196864; 256-token decode",
        args.fixed_stock,
        args.fixed_adaptive,
    )


if __name__ == "__main__":
    main()

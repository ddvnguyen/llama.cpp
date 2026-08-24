#!/usr/bin/env python3
"""Measure token-arrival latency during one continuous llama-server decode."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:12355")
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--decode-tokens", type=int, default=4096)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args()


def post_json(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    args = parse_args()
    suffix = post_json(
        f"{args.url}/tokenize",
        {"content": "The capital of France is", "add_special": False},
        30,
    )["tokens"]
    if args.prompt_tokens < len(suffix):
        raise SystemExit("prompt is shorter than the fixed suffix")

    payload = {
        "prompt": [23066] * (args.prompt_tokens - len(suffix)) + suffix,
        "n_predict": args.decode_tokens,
        "ignore_eos": True,
        "cache_prompt": False,
        "temperature": 0,
        "seed": 1,
        "reasoning_format": "none",
        "stream": True,
    }
    request = urllib.request.Request(
        f"{args.url}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic_ns()
    previous = started
    count = 0
    with open(args.output, "w") as output:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for raw_line in response:
                if not raw_line.startswith(b"data: "):
                    continue
                body = raw_line[6:].strip()
                if not body or body == b"[DONE]":
                    continue
                event = json.loads(body)
                now = time.monotonic_ns()
                tokens = event.get("tokens") or []
                if not tokens and event.get("content") and not event.get("stop"):
                    tokens = [None]
                for token in tokens:
                    count += 1
                    row = {
                        "token": count,
                        "context": args.prompt_tokens + count,
                        "arrival_ms": (now - started) / 1_000_000,
                        "delta_ms": (now - previous) / 1_000_000,
                        "token_id": token,
                    }
                    output.write(json.dumps(row, sort_keys=True) + "\n")
                    output.flush()
                    previous = now
                if event.get("stop"):
                    output.write(json.dumps({
                        "type": "summary",
                        "tokens": count,
                        "wall_ms": (now - started) / 1_000_000,
                        "timings": event.get("timings"),
                    }, sort_keys=True) + "\n")
                    output.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

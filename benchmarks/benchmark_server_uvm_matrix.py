#!/usr/bin/env python3
"""Benchmark stock and adaptive llama-server builds with and without UVM."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(
    "/home/raymond/LLM/llama-cache/models--unsloth--Qwen3.8-27B-GGUF/"
    "blobs/8c2a45ff85e7674ca185ec8eb6cdeab0e617ed9d8018caed0b64380eb2a67a5e"
)
DEFAULT_SIZES = [
    1024, 4096, 16384, 32768, 49152, 65536,
    *range(73728, 196608 + 1, 8192),
]
UVM_ENV_NAMES = (
    "GGML_CUDA_ENABLE_UNIFIED_MEMORY",
    "GGML_CUDA_PREFER_MODEL_WEIGHTS",
    "GGML_CUDA_PREFER_KV_HOST",
    "GGML_CUDA_KV_ACCESSED_BY_GPU",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=12355)
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    parser.add_argument("--variants", default="stock_cuda,stock_uvm,adaptive_cuda,adaptive_uvm")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--adaptive-pool-mib", type=int, default=2304)
    parser.add_argument("--server-context", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def http_json(url: str, payload: dict | None, timeout: int) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc


def gpu_memory_mib() -> int | None:
    result = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None


def mem_available_mib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def clean_env(extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    for name in UVM_ENV_NAMES:
        env.pop(name, None)
    env.update(extra)
    return env


def variant_config(name: str, adaptive_pool_mib: int) -> tuple[Path, dict[str, str], list[str]]:
    stock = Path("/tmp/llama-stock-bench/build/bin/llama-server")
    adaptive = ROOT / "build-kv-cuda/bin/llama-server"
    if name == "stock_cuda":
        return stock, {}, []
    if name == "stock_uvm":
        return stock, {"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"}, []
    if name == "adaptive_cuda":
        return adaptive, {}, ["--kv-stream-stage-mib", str(adaptive_pool_mib)]
    if name == "adaptive_uvm":
        return adaptive, {
            "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1",
            "GGML_CUDA_PREFER_MODEL_WEIGHTS": "1",
            "GGML_CUDA_PREFER_KV_HOST": "1",
            "GGML_CUDA_KV_ACCESSED_BY_GPU": "1",
        }, ["--kv-stream-stage-mib", str(adaptive_pool_mib)]
    raise ValueError(f"unknown variant: {name}")


class Server:
    def __init__(
        self,
        name: str,
        binary: Path,
        env_extra: dict[str, str],
        extra_args: list[str],
        model: Path,
        port: int,
        context_size: int,
        log_path: Path,
        startup_timeout: int,
    ) -> None:
        self.name = name
        self.port = port
        self.log_path = log_path
        self.log_file = log_path.open("wb")
        command = [
            str(binary), "-m", str(model),
            "--alias", name,
            "--host", "127.0.0.1", "--port", str(port),
            "--ctx-size", str(context_size),
            "-fa", "on", "-ctk", "q8_0", "-ctv", "q4_0",
            "-ngl", "all", "-b", "256", "-ub", "256", "-np", "1",
            "--no-mmproj", "--no-warmup", "--reasoning-format", "none",
            *extra_args,
        ]
        self.process = subprocess.Popen(
            command,
            cwd=binary.parent,
            env=clean_env(env_extra),
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + startup_timeout
        last_error = "server did not become ready"
        while time.monotonic() < deadline:
            status = self.process.poll()
            if status is not None:
                self.log_file.flush()
                raise RuntimeError(f"server exited with status {status}: {self.log_tail()}")
            try:
                health = http_json(self.url("/health"), None, 2)
                if health.get("status") == "ok":
                    return
            except Exception as exc:  # readiness polling
                last_error = str(exc)
            time.sleep(0.25)
        self.stop()
        raise RuntimeError(f"{last_error}: {self.log_tail()}")

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def log_tail(self, lines: int = 20) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if not self.log_file.closed:
            self.log_file.close()


def append_result(path: Path, row: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def wait_for_gpu_release(timeout: int = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        used = gpu_memory_mib()
        if used is None or used < 128:
            return
        time.sleep(1)


def start_stock_cuda_at_largest_fit(
    args: argparse.Namespace,
    sizes: list[int],
    logs: Path,
) -> tuple[Server | None, int | None]:
    binary, env_extra, extra_args = variant_config("stock_cuda", args.adaptive_pool_mib)
    for prompt_size in reversed(sizes):
        server_ctx = prompt_size + max(256, args.decode_tokens)
        log_path = logs / f"stock_cuda.probe-{server_ctx}.log"
        try:
            server = Server(
                "stock_cuda", binary, env_extra, extra_args,
                args.model, args.port, server_ctx, log_path, args.startup_timeout,
            )
            return server, prompt_size
        except Exception as exc:
            append_result(args.output, {
                "type": "probe",
                "variant": "stock_cuda",
                "context_size": server_ctx,
                "status": "startup_failed",
                "error": str(exc)[-2000:],
            })
            wait_for_gpu_release()
    return None, None


def benchmark_variant(
    args: argparse.Namespace,
    name: str,
    sizes: list[int],
    logs: Path,
) -> None:
    binary, env_extra, extra_args = variant_config(name, args.adaptive_pool_mib)
    server: Server | None = None
    max_prompt: int | None = None
    try:
        if name == "stock_cuda":
            server, max_prompt = start_stock_cuda_at_largest_fit(args, sizes, logs)
            if server is None:
                raise RuntimeError("stock CUDA could not start at any requested size")
        else:
            max_prompt = sizes[-1]
            server_context = args.server_context or max_prompt + max(256, args.decode_tokens)
            server = Server(
                name, binary, env_extra, extra_args,
                args.model, args.port, server_context,
                logs / f"{name}.log", args.startup_timeout,
            )

        tokenized = http_json(
            server.url("/tokenize"),
            {"content": "The capital of France is", "add_special": False},
            30,
        )
        suffix = tokenized["tokens"]
        if not suffix or not all(isinstance(token, int) for token in suffix):
            raise RuntimeError(f"unexpected tokenize response: {tokenized}")

        # Identical explicit warm-up removes first-request graph/setup noise.
        http_json(server.url("/completion"), {
            "prompt": [23066] * 16,
            "n_predict": 4,
            "ignore_eos": True,
            "cache_prompt": False,
            "temperature": 0,
            "reasoning_format": "none",
            "response_fields": ["timings"],
        }, args.request_timeout)

        for context in sizes:
            if max_prompt is not None and context > max_prompt:
                append_result(args.output, {
                    "type": "measurement",
                    "variant": name,
                    "context": context,
                    "status": "unsupported_vram",
                    "max_fitting_context": max_prompt,
                })
                continue
            prefix_count = context - len(suffix)
            if prefix_count < 0:
                raise RuntimeError("benchmark context is smaller than prompt suffix")
            prompt = [23066] * prefix_count + suffix
            started = time.monotonic()
            before_vram = gpu_memory_mib()
            before_ram = mem_available_mib()
            try:
                response = http_json(server.url("/completion"), {
                    "prompt": prompt,
                    "n_predict": args.decode_tokens,
                    "ignore_eos": True,
                    "cache_prompt": False,
                    "temperature": 0,
                    "seed": 1,
                    "reasoning_format": "none",
                    "response_fields": ["timings"],
                }, args.request_timeout)
                timings = response["timings"]
                row = {
                    "type": "measurement",
                    "variant": name,
                    "context": context,
                    "status": "ok",
                    "cache_n": timings.get("cache_n"),
                    "prompt_n": timings.get("prompt_n"),
                    "prompt_ms": timings.get("prompt_ms"),
                    "prefill_tps": timings.get("prompt_per_second"),
                    "predicted_n": timings.get("predicted_n"),
                    "predicted_ms": timings.get("predicted_ms"),
                    "decode_tps": timings.get("predicted_per_second"),
                    "wall_seconds": time.monotonic() - started,
                    "vram_before_mib": before_vram,
                    "vram_after_mib": gpu_memory_mib(),
                    "ram_available_before_mib": before_ram,
                    "ram_available_after_mib": mem_available_mib(),
                }
            except Exception as exc:
                row = {
                    "type": "measurement",
                    "variant": name,
                    "context": context,
                    "status": "request_failed",
                    "wall_seconds": time.monotonic() - started,
                    "error": str(exc)[-2000:],
                    "server_exit_status": server.process.poll(),
                    "server_log_tail": server.log_tail(),
                }
            append_result(args.output, row)
            if row["status"] != "ok" or server.process.poll() is not None:
                break
    finally:
        if server is not None:
            server.stop()
        wait_for_gpu_release()


def main() -> int:
    args = parse_args()
    sizes = sorted({int(value) for value in args.sizes.split(",") if value})
    variants = [value for value in args.variants.split(",") if value]
    if not sizes or sizes[0] <= 0 or args.decode_tokens <= 0:
        raise SystemExit("sizes and decode token count must be positive")
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output is None:
        args.output = ROOT / "benchmarks/results" / f"server-uvm-matrix-{stamp}.jsonl"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logs = args.output.parent / f"{args.output.stem}-logs"
    logs.mkdir(parents=True, exist_ok=True)
    args.output.write_text("")

    append_result(args.output, {
        "type": "metadata",
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": str(args.model),
        "sizes": sizes,
        "decode_tokens": args.decode_tokens,
        "adaptive_pool_mib": args.adaptive_pool_mib,
        "variants": variants,
        "stock_revision": "ece963f41",
        "adaptive_revision": "aece77ea8",
        "common_args": {
            "flash_attention": True,
            "cache_type_k": "q8_0",
            "cache_type_v": "q4_0",
            "batch": 256,
            "ubatch": 256,
            "parallel": 1,
            "gpu_layers": "all",
        },
    })

    for variant in variants:
        try:
            benchmark_variant(args, variant, sizes, logs)
        except Exception as exc:
            append_result(args.output, {
                "type": "variant_failure",
                "variant": variant,
                "status": "failed",
                "error": str(exc)[-4000:],
            })
            wait_for_gpu_release()
    print(f"results={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

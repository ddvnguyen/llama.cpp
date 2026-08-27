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
DEFAULT_ADAPTIVE_SERVER = ROOT / "build/bin/llama-server"
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
VALID_VARIANTS = ("stock_cuda", "stock_uvm", "adaptive_cuda", "adaptive_uvm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    paths = parser.add_argument_group("server builds and model")
    paths.add_argument("--model", type=Path, required=True, help="GGUF model to benchmark")
    paths.add_argument(
        "--stock-server", type=Path,
        help="llama-server binary from an unmodified/stock checkout",
    )
    paths.add_argument(
        "--adaptive-server", type=Path, default=DEFAULT_ADAPTIVE_SERVER,
        help="adaptive KV streaming llama-server binary",
    )
    paths.add_argument("--stock-revision", default="unknown", help="stock revision recorded in metadata")
    paths.add_argument(
        "--adaptive-revision", default="unknown",
        help="adaptive revision recorded in metadata",
    )

    matrix = parser.add_argument_group("benchmark matrix")
    matrix.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    matrix.add_argument("--variants", default=",".join(VALID_VARIANTS))
    matrix.add_argument("--decode-tokens", type=int, default=32)
    matrix.add_argument("--adaptive-pool-mib", type=int, default=2304)
    matrix.add_argument(
        "--server-context", type=int,
        help="fixed server context; defaults to the largest prompt plus decode headroom",
    )
    matrix.add_argument("--prompt-suffix", default="The capital of France is")
    matrix.add_argument(
        "--fill-token-id", type=int,
        help="repeated prefix token; defaults to the first token in --prompt-suffix",
    )

    server = parser.add_argument_group("common llama-server settings")
    server.add_argument("--flash-attention", choices=("on", "off", "auto"), default="on")
    server.add_argument("--cache-type-k", default="q8_0")
    server.add_argument("--cache-type-v", default="q4_0")
    server.add_argument("--gpu-layers", default="all")
    server.add_argument("--batch-size", type=int, default=256)
    server.add_argument("--ubatch-size", type=int, default=256)
    server.add_argument("--parallel", type=int, default=1)
    server.add_argument(
        "--extra-server-arg", action="append", default=[], metavar="ARG",
        help="append a llama-server argument (repeat; use --extra-server-arg=--flag)",
    )

    runtime = parser.add_argument_group("runtime and output")
    runtime.add_argument("--port", type=int, default=12355)
    runtime.add_argument("--request-timeout", type=int, default=1800)
    runtime.add_argument("--startup-timeout", type=int, default=180)
    runtime.add_argument("--nvidia-smi", default="nvidia-smi", help="nvidia-smi executable")
    runtime.add_argument(
        "--cuda-visible-devices",
        help="CUDA_VISIBLE_DEVICES value for each server; defaults to the inherited environment",
    )
    runtime.add_argument("--gpu-index", type=int, default=0, help="physical GPU index for nvidia-smi")
    runtime.add_argument(
        "--gpu-release-threshold-mib", type=int, default=128,
        help="wait for GPU memory.used to fall below this value between variants",
    )
    runtime.add_argument("--gpu-release-timeout", type=int, default=60)
    runtime.add_argument("--output", type=Path)
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


def gpu_memory_mib(nvidia_smi: str, gpu_index: int) -> int | None:
    try:
        result = subprocess.run(
            [
                nvidia_smi, "-i", str(gpu_index), "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
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


def variant_config(
    args: argparse.Namespace,
    name: str,
) -> tuple[Path, dict[str, str], list[str]]:
    stock = args.stock_server
    adaptive = args.adaptive_server
    env_extra = {}
    if args.cuda_visible_devices is not None:
        env_extra["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if name == "stock_cuda":
        assert stock is not None
        return stock, dict(env_extra), []
    if name == "stock_uvm":
        assert stock is not None
        return stock, {**env_extra, "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"}, []
    if name == "adaptive_cuda":
        return adaptive, dict(env_extra), ["--kv-stream-stage-mib", str(args.adaptive_pool_mib)]
    if name == "adaptive_uvm":
        return adaptive, {
            "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1",
            **env_extra,
            "GGML_CUDA_PREFER_MODEL_WEIGHTS": "1",
            "GGML_CUDA_PREFER_KV_HOST": "1",
            "GGML_CUDA_KV_ACCESSED_BY_GPU": "1",
        }, ["--kv-stream-stage-mib", str(args.adaptive_pool_mib)]
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
        common_args: list[str],
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
            *common_args,
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
                log_tail = self.log_tail()
                self.stop()
                raise RuntimeError(f"server exited with status {status}: {log_tail}")
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


def wait_for_gpu_release(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.gpu_release_timeout
    while time.monotonic() < deadline:
        used = gpu_memory_mib(args.nvidia_smi, args.gpu_index)
        if used is None or used < args.gpu_release_threshold_mib:
            return
        time.sleep(1)


def common_server_args(args: argparse.Namespace) -> list[str]:
    return [
        "-fa", args.flash_attention,
        "-ctk", args.cache_type_k,
        "-ctv", args.cache_type_v,
        "-ngl", args.gpu_layers,
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "-np", str(args.parallel),
        "--no-mmproj", "--no-warmup", "--reasoning-format", "none",
        *args.extra_server_arg,
    ]


def start_stock_cuda_at_largest_fit(
    args: argparse.Namespace,
    sizes: list[int],
    logs: Path,
) -> tuple[Server | None, int | None]:
    binary, env_extra, extra_args = variant_config(args, "stock_cuda")
    for prompt_size in reversed(sizes):
        server_ctx = prompt_size + max(256, args.decode_tokens)
        log_path = logs / f"stock_cuda.probe-{server_ctx}.log"
        try:
            server = Server(
                "stock_cuda", binary, env_extra, extra_args,
                args.model, args.port, server_ctx, log_path, args.startup_timeout,
                common_server_args(args),
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
            wait_for_gpu_release(args)
    return None, None


def benchmark_variant(
    args: argparse.Namespace,
    name: str,
    sizes: list[int],
    logs: Path,
) -> None:
    binary, env_extra, extra_args = variant_config(args, name)
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
                common_server_args(args),
            )

        tokenized = http_json(
            server.url("/tokenize"),
            {"content": args.prompt_suffix, "add_special": False},
            30,
        )
        suffix = tokenized["tokens"]
        if not suffix or not all(isinstance(token, int) for token in suffix):
            raise RuntimeError(f"unexpected tokenize response: {tokenized}")
        fill_token_id = args.fill_token_id if args.fill_token_id is not None else suffix[0]

        # Identical explicit warm-up removes first-request graph/setup noise.
        http_json(server.url("/completion"), {
            "prompt": [fill_token_id] * 16,
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
            prompt = [fill_token_id] * prefix_count + suffix
            started = time.monotonic()
            before_vram = gpu_memory_mib(args.nvidia_smi, args.gpu_index)
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
                    "fill_token_id": fill_token_id,
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
                    "vram_after_mib": gpu_memory_mib(args.nvidia_smi, args.gpu_index),
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
        wait_for_gpu_release(args)


def main() -> int:
    args = parse_args()
    sizes = sorted({int(value.strip()) for value in args.sizes.split(",") if value.strip()})
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown_variants = sorted(set(variants) - set(VALID_VARIANTS))
    if not variants:
        raise SystemExit("at least one benchmark variant is required")
    if unknown_variants:
        raise SystemExit(f"unknown variants: {', '.join(unknown_variants)}")
    if not sizes or sizes[0] <= 0 or args.decode_tokens <= 0:
        raise SystemExit("sizes and decode token count must be positive")
    if (
        args.adaptive_pool_mib < 0
        or args.batch_size <= 0
        or args.ubatch_size <= 0
        or args.parallel <= 0
        or args.gpu_index < 0
        or not 1 <= args.port <= 65535
        or args.request_timeout <= 0
        or args.startup_timeout <= 0
        or args.gpu_release_threshold_mib < 0
        or args.gpu_release_timeout < 0
        or (args.server_context is not None and args.server_context <= 0)
        or (args.fill_token_id is not None and args.fill_token_id < 0)
    ):
        raise SystemExit("one or more numeric benchmark settings are invalid")
    if not args.prompt_suffix:
        raise SystemExit("prompt suffix must not be empty")
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    required_binaries = {}
    if any(name.startswith("stock_") for name in variants):
        required_binaries["stock"] = args.stock_server
    if any(name.startswith("adaptive_") for name in variants):
        required_binaries["adaptive"] = args.adaptive_server
    for build, binary in required_binaries.items():
        if binary is None:
            raise SystemExit(f"--{build}-server is required for the selected variants")
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise SystemExit(f"{build} server is not executable: {binary}")

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
        "stock_server": str(args.stock_server) if args.stock_server else None,
        "adaptive_server": str(args.adaptive_server),
        "stock_revision": args.stock_revision,
        "adaptive_revision": args.adaptive_revision,
        "server_context": args.server_context,
        "prompt_suffix": args.prompt_suffix,
        "fill_token_id": args.fill_token_id,
        "gpu_index": args.gpu_index,
        "nvidia_smi": args.nvidia_smi,
        "cuda_visible_devices": args.cuda_visible_devices,
        "gpu_release_threshold_mib": args.gpu_release_threshold_mib,
        "gpu_release_timeout": args.gpu_release_timeout,
        "common_args": {
            "flash_attention": args.flash_attention,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
            "batch": args.batch_size,
            "ubatch": args.ubatch_size,
            "parallel": args.parallel,
            "gpu_layers": args.gpu_layers,
            "extra_server_args": args.extra_server_arg,
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
            wait_for_gpu_release(args)
    print(f"results={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

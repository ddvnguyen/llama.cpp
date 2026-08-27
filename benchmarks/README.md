# Adaptive KV streaming benchmarks

`benchmark_server_uvm_matrix.py` compares a stock llama.cpp server with this
branch, using ordinary CUDA allocation and CUDA Unified Memory (UVM). It writes
newline-delimited JSON measurements and keeps one server log per variant.

## Build the two servers

Build this branch:

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-server -j
```

Build an unmodified llama.cpp revision in a separate checkout with the same
CMake options. `GGML_CUDA_FA_ALL_QUANTS=ON` is needed for the benchmark
defaults, `--cache-type-k q8_0 --cache-type-v q4_0`.

## Run a comparison

```bash
python3 benchmarks/benchmark_server_uvm_matrix.py \
  --model /path/to/model.gguf \
  --adaptive-server ./build/bin/llama-server \
  --stock-server /path/to/stock-llama.cpp/build/bin/llama-server \
  --adaptive-revision feature/adaptive-kv-stream \
  --stock-revision stock-revision \
  --variants stock_cuda,stock_uvm,adaptive_cuda,adaptive_uvm \
  --sizes 8192,16384,32768,65536,98304,131072,163840,196608 \
  --decode-tokens 256 \
  --adaptive-pool-mib 2304 \
  --gpu-index 0 \
  --output benchmarks/results/my-uvm-comparison.jsonl
```

The adaptive server defaults to `./build/bin/llama-server`, so
`--adaptive-server` can be omitted for the standard in-tree build. The model
path is required; the script does not assume a Hugging Face cache layout.

The four variants are:

- `stock_cuda`: stock server with ordinary CUDA allocation.
- `stock_uvm`: stock server with UVM enabled.
- `adaptive_cuda`: this branch with the adaptive KV staging pool.
- `adaptive_uvm`: this branch with UVM and the model-weight/KV memory advice
  used by this project.

Select a subset with `--variants`. A stock binary is required only when a
`stock_*` variant is selected.

## Machine-dependent settings

All machine-specific inputs are command-line options:

- `--model`, `--stock-server`, and `--adaptive-server` select local files.
- `--cuda-visible-devices` selects the CUDA device(s) exposed to each server.
- `--gpu-index` selects the physical GPU queried by the `--nvidia-smi`
  telemetry executable.
- `--gpu-release-threshold-mib` should be slightly above the GPU's idle memory
  usage. This prevents the next variant from starting before the previous one
  releases VRAM. Increase it on a display GPU.
- `--port`, `--startup-timeout`, and `--request-timeout` control local
  execution.
- `--adaptive-pool-mib` sets the adaptive server's device-resident KV staging
  pool.
- `--stock-revision` and `--adaptive-revision` are labels stored in the
  result metadata; they do not change either binary.

The workload is configurable with `--cache-type-k`, `--cache-type-v`,
`--flash-attention`, `--gpu-layers`, `--batch-size`, `--ubatch-size`,
`--parallel`, and repeatable `--extra-server-arg`. For example:

```bash
python3 benchmarks/benchmark_server_uvm_matrix.py \
  --model /path/to/model.gguf \
  --variants adaptive_cuda \
  --sizes 8192 \
  --decode-tokens 64 \
  --extra-server-arg=--verbosity \
  --extra-server-arg=3
```

The script tokenizes `--prompt-suffix` with the selected model and uses its
first token to build the synthetic prefix. Use `--fill-token-id` to pin the
exact token when reproducing an existing run.

## Context-size behavior

For a multi-size run, each non-stock-CUDA variant starts once with a context
size equal to the largest prompt plus decode headroom. `--server-context`
overrides that value. Stock CUDA probes downward to find the largest requested
context that fits in VRAM.

To measure context-matched server allocations, invoke the script once per
prompt size, using one value in `--sizes`. This also lets you choose the
largest safe adaptive pool independently for each context size.

Run `python3 benchmarks/benchmark_server_uvm_matrix.py --help` for every
available option.

## Measure a continuous decode

`benchmark_long_decode.py` records per-token arrival latency against an
already running server:

```bash
python3 benchmarks/benchmark_long_decode.py \
  --url http://127.0.0.1:12355 \
  --prompt-tokens 114688 \
  --decode-tokens 4096 \
  --output benchmarks/results/long-decode.jsonl
```

It uses the same model-independent prefix-token selection as the matrix driver.
Use `--prompt-suffix` or `--fill-token-id` to reproduce a specific workload.

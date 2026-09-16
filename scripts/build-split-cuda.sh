#!/bin/bash
# Split CUDA build for Hydra COMBINED mode
# RTX 5060 Ti (sm_120): CUDA 13.3.1
# RTX 3060 (sm_86): CUDA 13.2.2
#
# Usage: bash scripts/build-split-cuda.sh
# Output: build_sm120/bin/ and build_sm86/bin/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

CUDA_1331="/opt/software/cuda/13.3.1/bin/nvcc"
CUDA_1322="/opt/software/cuda/13.2.2/bin/nvcc"

echo "=== Building sm_120 (RTX 5060 Ti) with CUDA 13.3.1 ==="
rm -rf "$REPO_DIR/build_sm120"
mkdir -p "$REPO_DIR/build_sm120"
cd "$REPO_DIR/build_sm120"
cmake .. \
  -DGGML_CCACHE=OFF \
  -DCMAKE_CUDA_COMPILER="$CUDA_1331" \
  -DCMAKE_CUDA_ARCHITECTURES="120" \
  -DGGML_RPC=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-engine -j$(nproc)

echo ""
echo "=== Building sm_86 (RTX 3060) with CUDA 13.2.2 ==="
rm -rf "$REPO_DIR/build_sm86"
mkdir -p "$REPO_DIR/build_sm86"
cd "$REPO_DIR/build_sm86"
cmake .. \
  -DGGML_CCACHE=OFF \
  -DCMAKE_CUDA_COMPILER="$CUDA_1322" \
  -DCMAKE_CUDA_ARCHITECTURES="86" \
  -DGGML_RPC=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-engine -j$(nproc)

echo ""
echo "=== Build complete ==="
echo "sm_120: $REPO_DIR/build_sm120/bin/llama-engine"
echo "sm_86:  $REPO_DIR/build_sm86/bin/llama-engine"
ls -la "$REPO_DIR/build_sm120/bin/llama-engine"
ls -la "$REPO_DIR/build_sm86/bin/llama-engine"

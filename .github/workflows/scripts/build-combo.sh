#!/usr/bin/env bash
# Configure, build, package, and push one (arch, binary) combination.
# Assumes: ccache is on PATH with CCACHE_DIR already set, and podman is
# already logged in to ghcr.io. Called once per combo by hydra-build.yml,
# either from a matrix job (one combo) or a sequential loop (many combos).
set -euo pipefail

ARCH="$1"
BINARY="$2"
CUDA_VERSION="$3"
CUDA_ARCH="$4"
CUDA_PATH="$5"
IMAGE_REPO="$6"
SHORT_SHA="$7"
RUNNER_TARGET="$8"  # "local" or "cloud" — native/IPO builds only make sense on the box that runs the binary
PR_ID="$9"          # optional PR id -> image tag suffix -pr<N> (e.g. 532 -> -pr532)

BUILD_DIR="build_hydra_${ARCH}_${BINARY}"
STAGING_DIR="staging_${ARCH}_${BINARY}"

CMAKE_ARGS=(
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
  -DCMAKE_C_COMPILER_LAUNCHER=ccache
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
  -DGGML_CUDA=ON
  -DGGML_CUDA_FORCE_CUBLAS=ON
  -DGGML_RPC=ON
  -DGGML_NVML=ON
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=ON
  -DCMAKE_BUILD_RPATH='$ORIGIN'
  -DCMAKE_INSTALL_RPATH='$ORIGIN'
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_TESTS=OFF
)

if [ -x "${CUDA_PATH}/bin/nvcc" ]; then
  CMAKE_ARGS+=(-DCMAKE_CUDA_COMPILER="${CUDA_PATH}/bin/nvcc")
fi

case "$ARCH" in
  sm86-sm120)
    # Native build + IPO only make sense on the box that will run the binary (local host)
    CMAKE_ARGS+=(
      -DGGML_CUDA_FA=ON
      -DGGML_CUDA_FA_ALL_QUANTS=ON
      -DGGML_CUDA_GRAPHS=ON
      -DGGML_CUDA_NCCL=ON
    )
    if [ "$RUNNER_TARGET" = "local" ]; then
      CMAKE_ARGS+=(-DGGML_NATIVE=ON -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON)
    else
      CMAKE_ARGS+=(-DGGML_NATIVE=OFF)
    fi
    ;;
  sm60)
    CMAKE_ARGS+=(
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14
      -DGGML_CUDA_FA_ALL_QUANTS=OFF
      -DGGML_NATIVE=OFF
    )
    ;;
esac

echo "=== [$ARCH/$BINARY] CMake configure (CUDA $CUDA_VERSION @ $CUDA_PATH) ==="
echo "CMake args: ${CMAKE_ARGS[*]}"
# Fix #498 glibc follow-up: host is Ubuntu 26.04 (glibc 2.43) but runtime base is
# nvidia/cuda:12.9.2-runtime-ubuntu24.04 (glibc 2.39). Host-built sm60 binaries
# require GLIBC_2.43 and crash in the image (5× version not found). Build sm60
# inside a container matching the runtime base to ensure glibc compatibility.
# sm86-sm120 stays host-built (latent same issue, hardened ldd gate now protects).
if [ "$ARCH" = "sm60" ]; then
  echo "=== [$ARCH/$BINARY] Containerized build in nvidia/cuda:12.9.2-devel-ubuntu24.04 (glibc 2.39) to match runtime ==="
  # Clean previous host-built artifacts (wrong glibc)
  rm -rf "$BUILD_DIR"
  # Pull devel image (contains /usr/local/cuda 12.9.2, gcc, but not cmake/ninja/ccache)
  podman pull nvidia/cuda:12.9.2-devel-ubuntu24.04 2>&1 | tail -n 5 || true
  mkdir -p "$HOME/.cache/hydra-ccache"
  # Run cmake configure + build inside the devel container, mounting source and ccache
  # Use --userns=keep-id so files are owned by host user, not root
  podman run --rm \
    -v "$PWD:/work" -w /work \
    -v "$HOME/.cache/hydra-ccache:/tmp/ccache:rw" \
    -e CCACHE_DIR=/tmp/ccache \
    -e CCACHE_MAXSIZE=10G \
    -e CCACHE_SLOPPINESS="pch_defines,time_macros,locale" \
    --userns=keep-id \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 \
    bash -c '
      set -e
      echo "=== [container] apt-get install build deps ==="
      apt-get update -qq
      apt-get install -y -qq cmake ninja-build ccache g++-14 libssl-dev libgomp1 > /dev/null
      echo "=== [container] cmake configure ==="
      cmake -B "'"$BUILD_DIR"'" -G Ninja         -DCMAKE_CUDA_ARCHITECTURES="'"$CUDA_ARCH"'"         -DCMAKE_C_COMPILER_LAUNCHER=ccache         -DCMAKE_CXX_COMPILER_LAUNCHER=ccache         -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache         -DGGML_CUDA=ON -DGGML_CUDA_FORCE_CUBLAS=ON -DGGML_RPC=ON -DGGML_NVML=ON         -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON         -DCMAKE_BUILD_RPATH='\''$ORIGIN'\'' -DCMAKE_INSTALL_RPATH='\''$ORIGIN'\'' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON         -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF         -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc         -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14         -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_NATIVE=OFF         .
      echo "=== [container] cmake build ==="
      cmake --build "'"$BUILD_DIR"'" --target "'"$BINARY"'" -j$(nproc)
    '
else
  cmake -B "$BUILD_DIR" -G Ninja "${CMAKE_ARGS[@]}" .
  echo "=== [$ARCH/$BINARY] CMake build ==="
  cmake --build "$BUILD_DIR" --target "$BINARY" -j"$(nproc)"
fi

ccache -s || true

VERSION_OUTPUT=$("$BUILD_DIR/bin/$BINARY" --version 2>&1 || true)
echo "Binary version: $VERSION_OUTPUT"
if echo "$VERSION_OUTPUT" | grep -q "\[shared\]"; then
  echo "OK: build type is shared"
else
  echo "WARNING: expected [shared] in version output (see hydra_vortex#346) — may hang on RTX"
fi

FORK_VERSION=$(tr -d '[:space:]' < VERSION)
# Canonical tag carries fork version + commit SHA + optional PR id. The
# unversioned ALIAS_TAG mirrors what hydra-build.yml's resolve-outputs /
# build-sequential construct for the caller, so the deploy flow's derived
# refs always resolve.
PR_TAG=""
[ -n "$PR_ID" ] && PR_TAG="-pr${PR_ID}"
IMAGE_TAG="${ARCH}-${BINARY}-${FORK_VERSION}-${SHORT_SHA}${PR_TAG}"
ALIAS_TAG="${ARCH}-${BINARY}-${SHORT_SHA}${PR_TAG}"

# ── Same-hash skip gate ──────────────────────────────────────────────
# The tag embeds the fork commit SHA: an unchanged commit produces the
# SAME tag. Rebuilding + re-pushing an identical image wastes the whole
# compile (tens of minutes). Check the registry for the exact tag via a
# lightweight manifest HEAD (no blob download) and skip entirely if it
# already exists.
if podman manifest inspect "${IMAGE_REPO}:${IMAGE_TAG}" >/dev/null 2>&1; then
  echo "=== [$ARCH/$BINARY] SKIP: ${IMAGE_REPO}:${IMAGE_TAG} already exists (same-hash rebuild) ==="
  exit 0
fi
echo "=== [$ARCH/$BINARY] ${IMAGE_TAG} not in registry — building ==="

mkdir -p "${STAGING_DIR}/bin"
cp -a "$BUILD_DIR/bin/$BINARY" "${STAGING_DIR}/bin/"
# Fix #498: preserve symlinks (-a) and ensure ALL shared libs are staged.
# The previous `cp *.so*` without -a could dereference symlinks and miss
# versioned chains (e.g., libllama.so -> libllama.so.0 -> libllama.so.0.x).
# Use -a and explicitly list contents for verification.
cp -a "$BUILD_DIR/bin/"*.so* "${STAGING_DIR}/bin/" 2>/dev/null || true
echo "=== Staging contents (${STAGING_DIR}/bin/) ==="
ls -lh "${STAGING_DIR}/bin/" | head -n 50
echo "=== Host ldd check for $BUILD_DIR/bin/$BINARY (expect CUDA libs 'not found' on host, but hydra .so should resolve via \$ORIGIN) ==="
ldd "$BUILD_DIR/bin/$BINARY" || true
if ldd "$BUILD_DIR/bin/$BINARY" 2>&1 | grep -E "=> not found|version .*GLIBC.*not found" | grep -vE "libcuda|libcudart|libcublas|libibverbs" | grep -q .; then
  echo "NOTE: some non-CUDA libs still 'not found' or GLIBC mismatch — check \$ORIGIN RPATH and builder glibc (sm60 must be built in nvidia/cuda:12.9.2-devel-ubuntu24.04)"
  ldd "$BUILD_DIR/bin/$BINARY" 2>&1 | grep -E "=> not found|version .*GLIBC.*not found" | grep -vE "libcuda|libcudart|libcublas|libibverbs" || true
fi
# Fail hard if any hydra .so is missing in staging (e.g., libllama-server-impl.so for llama-server)
if [ "$BINARY" = "llama-server" ]; then
  if [ ! -f "${STAGING_DIR}/bin/libllama-server-impl.so"* ] && ! ls "${STAGING_DIR}/bin/libllama-server-impl.so"* >/dev/null 2>&1; then
    echo "WARNING: llama-server build should have libllama-server-impl.so in staging — check BUILD_SHARED_LIBS=ON"
    ls -lh "${STAGING_DIR}/bin/"*.so* 2>&1 | head -n 20 || true
  fi
fi

echo "=== [$ARCH/$BINARY] Build + push OCI image ==="
# Docker Hub's nvidia/cuda runtime images are tagged with a full patch
# version (e.g. 13.2.1), not just major.minor — map our major.minor
# convention to the newest known-good patch tag for the base image.
case "$CUDA_VERSION" in
  13.2) DOCKER_CUDA_VERSION="13.2.1" ;;
  12.9) DOCKER_CUDA_VERSION="12.9.2" ;;
  *) echo "::error::No known nvidia/cuda runtime tag mapped for CUDA_VERSION=$CUDA_VERSION"; exit 1 ;;
esac

podman build \
  --build-arg CUDA_VERSION="${DOCKER_CUDA_VERSION}" \
  --build-arg BINARY="${BINARY}" \
  -t "${IMAGE_REPO}:${IMAGE_TAG}" \
  -f .github/workflows/hydra-build.Dockerfile \
  "${STAGING_DIR}/"

podman tag "${IMAGE_REPO}:${IMAGE_TAG}" "${IMAGE_REPO}:${ARCH}-${BINARY}-latest"
podman tag "${IMAGE_REPO}:${IMAGE_TAG}" "${IMAGE_REPO}:${ALIAS_TAG}"
podman push "${IMAGE_REPO}:${IMAGE_TAG}"
podman push "${IMAGE_REPO}:${ALIAS_TAG}"
podman push "${IMAGE_REPO}:${ARCH}-${BINARY}-latest"

IMAGE_SIZE=$(podman image inspect "${IMAGE_REPO}:${IMAGE_TAG}" --format '{{.Size}}' 2>/dev/null || echo "unknown")
echo "Pushed: ${IMAGE_REPO}:${IMAGE_TAG} (${IMAGE_SIZE} bytes)"

{
  echo "### ${ARCH} / ${BINARY}"
  echo ""
  echo "| Field | Value |"
  echo "|-------|-------|"
  echo "| Fork version | \`${FORK_VERSION}\` |"
  echo "| CUDA | \`${CUDA_VERSION}\` |"
  echo "| Image | \`${IMAGE_REPO}:${IMAGE_TAG}\` |"
  echo "| Size | \`${IMAGE_SIZE}\` bytes |"
  echo ""
} >> "$GITHUB_STEP_SUMMARY"

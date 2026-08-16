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
  # CUDA_PATH may be a conda-installed toolkit (e.g. /opt/software/cuda/13.2.2)
  # whose libs live under targets/x86_64-linux/lib — no lib64/. Point CMake
  # at the toolkit root so its CUDAToolkit module finds include + libs.
  CMAKE_ARGS+=(-DCUDAToolkit_ROOT="${CUDA_PATH}")
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

FORK_VERSION=$(tr -d '[:space:]' < VERSION)
# Canonical tag carries fork version + commit SHA + optional PR id. The
# unversioned ALIAS_TAG mirrors what hydra-build.yml's resolve-outputs /
# build-sequential construct for the caller, so the deploy flow's derived
# refs always resolve.
PR_TAG=""
[ -n "$PR_ID" ] && PR_TAG="-pr${PR_ID}"
IMAGE_TAG="${ARCH}-${BINARY}-${FORK_VERSION}-${SHORT_SHA}${PR_TAG}"
ALIAS_TAG="${ARCH}-${BINARY}-${SHORT_SHA}${PR_TAG}"

# ── Same-hash skip gate (runs BEFORE configure/build) ────────────────
# The tag embeds the fork commit SHA (plus fork version, arch, binary and
# optional PR id): an unchanged commit produces the SAME tag. Rebuilding +
# re-pushing an identical image wastes the whole compile (tens of minutes),
# so this gate runs before cmake configure/build and exits 0 — the compile
# itself is skipped — when the registry already holds this tag AND it was
# built from the exact same inputs as this run.
#
# Why not `podman manifest inspect` (the pre-fix gate)? Our engine images
# are pushed as SINGLE-ARCH OCI manifests — not manifest lists — and
# `podman manifest inspect` refuses those with "treating single images as
# manifest lists" (exit 125). The old gate never fired (and sat after the
# compile), so we rebuilt 10-15 min images 3+ times per SHA.
#
# Primitive: skopeo, which is registry-authoritative and handles both
# single-arch manifests and manifest lists. `podman image inspect` is only
# a fallback: on the persistent self-hosted runner it can read a stale
# LOCAL copy of a tag, so its RepoDigests/labels may not reflect what the
# registry actually holds.
#
# SKIP CONDITION (exact): the tag resolves in the registry AND the
# existing image's org.hydra.build-key label (baked in by
# hydra-build.Dockerfile at build time) equals this run's BUILD_KEY.
# BUILD_KEY covers every input that changes the binary: SHA + CUDA
# toolkit version + arch + binary + runner target (native / IPO builds
# differ between local and cloud runners). Any deviation — tag absent,
# label missing (images built before this label existed), or label
# different (toolkit or build flags changed) — rebuilds. The gate can
# therefore never falsely skip a build whose inputs differ, and never
# skips based on stale local state.
BUILD_KEY="sha=${SHORT_SHA};cuda=${CUDA_VERSION};arch=${ARCH};binary=${BINARY};runner=${RUNNER_TARGET}"

EXISTING_DIGEST=""
EXISTING_KEY=""
if command -v skopeo >/dev/null 2>&1; then
  EXISTING_DIGEST=$(skopeo inspect --format '{{.Digest}}' "docker://${IMAGE_REPO}:${IMAGE_TAG}" 2>/dev/null || true)
  EXISTING_KEY=$(skopeo inspect --format '{{index .Labels "org.hydra.build-key"}}' "docker://${IMAGE_REPO}:${IMAGE_TAG}" 2>/dev/null || true)
else
  # Fallback only: podman image inspect reads LOCAL state when the tag is
  # already present on the host (persistent runner) — safe because a
  # stale local copy can only cause a rebuild, never a false skip, since
  # the label comparison still applies to whatever it returns.
  EXISTING_DIGEST=$(podman image inspect "${IMAGE_REPO}:${IMAGE_TAG}" --format '{{index .RepoDigests 0}}' 2>/dev/null || true)
  EXISTING_KEY=$(podman image inspect "${IMAGE_REPO}:${IMAGE_TAG}" --format '{{index .Labels "org.hydra.build-key"}}' 2>/dev/null || true)
fi

if [ -n "$EXISTING_DIGEST" ] && [ -n "$EXISTING_KEY" ] && [ "$EXISTING_KEY" = "$BUILD_KEY" ]; then
  echo "=== [$ARCH/$BINARY] SKIP: ${IMAGE_REPO}:${IMAGE_TAG} already exists (${EXISTING_DIGEST}) with identical build-key — same-hash rebuild ==="
  exit 0
fi
echo "=== [$ARCH/$BINARY] ${IMAGE_TAG} not in registry or build-key mismatch — building ==="

echo "=== [$ARCH/$BINARY] CMake configure (CUDA $CUDA_VERSION @ $CUDA_PATH) ==="
echo "CMake args: ${CMAKE_ARGS[*]}"
cmake -B "$BUILD_DIR" -G Ninja "${CMAKE_ARGS[@]}" .

echo "=== [$ARCH/$BINARY] CMake build ==="
cmake --build "$BUILD_DIR" --target "$BINARY" -j"$(nproc)"

ccache -s || true

VERSION_OUTPUT=$("$BUILD_DIR/bin/$BINARY" --version 2>&1 || true)
echo "Binary version: $VERSION_OUTPUT"
if echo "$VERSION_OUTPUT" | grep -q "\[shared\]"; then
  echo "OK: build type is shared"
else
  echo "WARNING: expected [shared] in version output (see hydra_vortex#346) — may hang on RTX"
fi

mkdir -p "${STAGING_DIR}/bin"
cp "$BUILD_DIR/bin/$BINARY" "${STAGING_DIR}/bin/"
cp "$BUILD_DIR/bin/"*.so* "${STAGING_DIR}/bin/" 2>/dev/null || true

echo "=== [$ARCH/$BINARY] Build + push OCI image ==="
# Docker Hub's nvidia/cuda runtime images are tagged with a full patch
# version (e.g. 13.2.1), not just major.minor — map our major.minor
# convention to the newest known-good patch tag for the base image.
case "$CUDA_VERSION" in
  13.2) DOCKER_CUDA_VERSION="13.2.1" ;;
  # CUDA 13.2.2 toolkit (nvcc V13.2.86): no nvidia/cuda docker tag exists
  # for 13.2.2 (docker tags stop at 13.2.1) — the runtime base stays 13.2.1;
  # the 13.2.2 toolkit path is only for the compile step (CUDA_PATH).
  13.2.2) DOCKER_CUDA_VERSION="13.2.1" ;;
  12.9) DOCKER_CUDA_VERSION="12.9.2" ;;
  *) echo "::error::No known nvidia/cuda runtime tag mapped for CUDA_VERSION=$CUDA_VERSION"; exit 1 ;;
esac

podman build \
  --build-arg CUDA_VERSION="${DOCKER_CUDA_VERSION}" \
  --build-arg BINARY="${BINARY}" \
  --build-arg BUILD_KEY="${BUILD_KEY}" \
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

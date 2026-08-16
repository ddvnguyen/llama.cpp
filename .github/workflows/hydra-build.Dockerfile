# Hydra Build — Minimal runtime image for OCI push
# Copies only the binary + shared libs from the build directory.
# Used by hydra-build.yml to create deployable OCI images.

ARG CUDA_VERSION=13.2
ARG UBUNTU_VERSION=24.04
ARG BINARY=llama-engine

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# Build-key for the same-hash skip gate (build-combo.sh): SHA + CUDA
# toolkit version + arch + binary + runner target. Baked in at build time
# so the gate can compare an existing image's inputs against the current
# run without trusting local state. Images without this label (built
# before it existed) never match a current build-key and are rebuilt.
ARG BUILD_KEY=unknown
LABEL org.hydra.build-key=${BUILD_KEY}

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /llama

# Copy the binary and all shared libraries from the build output
COPY bin/ /llama/

# Make binaries executable
RUN chmod +x /llama/${BINARY} 2>/dev/null || true

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/llama/llama-engine"]

# Hydra Build — Minimal runtime image for OCI push
# Copies only the binary + shared libs from the build directory.
# Fix #498: P100 sm60 image previously shipped only the binary without its
# shared libs (libllama.so, libggml-*.so, libllama-server-impl.so, etc),
# causing "symbol lookup error: undefined symbol: llama_model_get_quant_label"
# and a 1120-restart loop (see ddvnguyen/hydra_vortex#498). This Dockerfile
# now reliably packages ALL .so* alongside the binary and verifies via ldd
# that no libs are missing. RUNPATH is $ORIGIN (set at build with
# -DCMAKE_BUILD_RPATH='$ORIGIN' -DCMAKE_INSTALL_RPATH='$ORIGIN'), so /llama
# is the search dir; LD_LIBRARY_PATH=/llama is added as a defensive fallback.

ARG CUDA_VERSION=13.2
ARG UBUNTU_VERSION=24.04
ARG BINARY=llama-engine

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /llama

# Copy the binary and all shared libraries from the build output
COPY bin/ /llama/

# Make binary executable, verify ldd, and ensure fallback LD_LIBRARY_PATH
RUN chmod +x /llama/${BINARY} 2>/dev/null || true \
    && echo "=== ldd verify for /llama/${BINARY} ===" \
    && ldd /llama/${BINARY} || true \
    && if ldd /llama/${BINARY} 2>&1 | grep -q "not found"; then \
         echo "ERROR: missing shared libs in image — COPY bin/ must include *.so*"; \
         ldd /llama/${BINARY}; exit 1; \
       fi \
    && echo "=== ldd OK — no missing libs ===" \
    && ls -lh /llama/*.so* 2>/dev/null | head -n 40 || echo "no .so files in /llama (static build?)"

ENV LD_LIBRARY_PATH=/llama:${LD_LIBRARY_PATH}

# Ensure entrypoint works for both binaries: if BINARY != llama-engine, symlink
# so the fixed ENTRYPOINT still resolves. This keeps hydra-head's
# /llama/llama-engine expectation while supporting llama-server images.
RUN if [ "${BINARY}" != "llama-engine" ]; then ln -sf /llama/${BINARY} /llama/llama-engine || true; fi

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/llama/llama-engine"]

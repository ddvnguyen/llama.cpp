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
# Follow-up #498 glibc: host is Ubuntu 26.04 (glibc 2.43) but runtime base is
# nvidia/cuda:12.9.2-runtime-ubuntu24.04 (glibc 2.39). Host-built sm60 binaries
# require GLIBC_2.43 and crash in the image (5× version not found). The
# pipeline now builds sm60 inside nvidia/cuda:12.9.2-devel-ubuntu24.04
# (matching runtime glibc 2.39) — see build-combo.sh containerized build.
# Ldd gate now fails on BOTH "=> not found" and "version GLIBC not found".

ARG CUDA_VERSION=13.2
ARG UBUNTU_VERSION=24.04
ARG BINARY=llama-engine

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ARG BINARY

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl libibverbs1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /llama

# Copy the binary and all shared libraries from the build output
COPY bin/ /llama/

# Make binary executable, verify ldd, and ensure fallback LD_LIBRARY_PATH
RUN chmod +x /llama/${BINARY} 2>/dev/null || chmod +x /llama/llama-engine 2>/dev/null || true \
    && echo "=== ldd verify for /llama/${BINARY:-llama-engine} ===" \
    && ldd /llama/${BINARY:-llama-engine} 2>&1 | tee /tmp/ldd.txt || true \
    && cat /tmp/ldd.txt \
    && if grep -E "=> not found|version .*GLIBC.*not found" /tmp/ldd.txt | grep -vE "libcuda|libibverbs" | grep -q .; then \
         echo "ERROR: missing hydra shared libs or GLIBC version mismatch in image — COPY bin/ must include *.so* and builder must match runtime glibc (use nvidia/cuda:12.9.2-devel-ubuntu24.04 for sm60)"; \
         grep -E "=> not found|version .*GLIBC.*not found" /tmp/ldd.txt; exit 1; \
       fi \
    && echo "=== ldd OK — hydra libs present, no GLIBC mismatch (libcuda/libibverbs expected to be host-mounted) ===" \
    && ls -lh /llama/*.so* 2>/dev/null | head -n 40 || echo "no .so files in /llama (static build?)"

ENV LD_LIBRARY_PATH=/llama:${LD_LIBRARY_PATH}

# Ensure entrypoint works for both binaries: if BINARY != llama-engine, symlink
# so the fixed ENTRYPOINT still resolves. This keeps hydra-head's
# /llama/llama-engine expectation while supporting llama-server images.
RUN if [ "${BINARY:-llama-engine}" != "llama-engine" ]; then ln -sf /llama/${BINARY} /llama/llama-engine || true; fi

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/llama/llama-engine"]

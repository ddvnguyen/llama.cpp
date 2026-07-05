#pragma once

#include <vector>
#include <cstddef>

struct ggml_backend;

// Hydra binary RPC wire-format constants for llama-server state transfer.
// See hydra_vortex/specs/rpc-protocol.md for the full spec.
// llama-server only implements ops 0x30-0x32; it knows nothing about Store.

#include <cstdint>

// ── Wire format sizes ─────────────────────────────────────────────────────────
static constexpr int      HYDRA_REQ_HEADER_SIZE = 16;
static constexpr int      HYDRA_RES_HEADER_SIZE = 12;

// ── Magic ─────────────────────────────────────────────────────────────────────
static constexpr uint16_t HYDRA_MAGIC           = 0x4859; // "HY", little-endian

// ── Op codes (llama-server direct, 0x30-0x3F range) ──────────────────────────
static constexpr uint8_t  HYDRA_OP_STATE_GET    = 0x30; // stream KV state out
static constexpr uint8_t  HYDRA_OP_STATE_PUT    = 0x31; // stream KV state in
static constexpr uint8_t  HYDRA_OP_STATE_META   = 0x32; // metadata only, no payload

// ── Engine control opcodes (E1) ───────────────────────────────────────────────
static constexpr uint8_t  HYDRA_OP_CONFIGURE    = 0x40; // set engine params
static constexpr uint8_t  HYDRA_OP_INFO         = 0x41; // report capabilities
static constexpr uint8_t  HYDRA_OP_PREFILL      = 0x42; // run prefill only, return n_past
static constexpr uint8_t  HYDRA_OP_DECODE       = 0x43; // run decode with streaming
static constexpr uint8_t  HYDRA_OP_SET_EXPERT_MODE = 0x44; // switch solo/combined
static constexpr uint8_t  HYDRA_OP_SWAP_QUANT   = 0x45; // swap expert quantization
static constexpr uint8_t  HYDRA_OP_PIPELINE_ATTACH = 0x46; // two-engine "work together"

// ── Status codes (shared with full Hydra spec) ────────────────────────────────
static constexpr uint8_t  HYDRA_STATUS_OK               = 0x00;
static constexpr uint8_t  HYDRA_STATUS_NOT_FOUND        = 0x01;
static constexpr uint8_t  HYDRA_STATUS_ERROR            = 0x02;
static constexpr uint8_t  HYDRA_STATUS_BUSY             = 0x04;
static constexpr uint8_t  HYDRA_STATUS_NOT_IMPLEMENTED   = 0x06; // M-Perf.9 #289: stubbed engine opcodes
static constexpr uint8_t  HYDRA_STATUS_BAD_REQUEST      = 0x05;

// ── Safety cap (4 GB) ─────────────────────────────────────────────────────────
static constexpr uint64_t HYDRA_MAX_STATE_BYTES  = 4ULL * 1024 * 1024 * 1024;

// ── Unified RPC server ───────────────────────────────────────────────────────
// Start a protocol-detecting RPC accept loop on a pre-bound socket.
// Accepts connections, peeks the first byte, and dispatches:
//   0x0E (RPC_CMD_HELLO) → ggml-backend RPC handler (GPU compute)
//   otherwise            → Hydra protocol handler (if ctx != nullptr)
// Declared here so both server-context.cpp and llama-engine.cpp share the
// same dispatch logic. See server-context.cpp for the implementation.
//
// `hydra_ctx` is an opaque pointer to a hydra_rpc_ctx struct (defined in
// server-context.cpp). If nullptr, Hydra protocol connections are closed;
// only ggml-RPC is served. The struct fields are accessed only from
// server-context.cpp, so the opaque pointer avoids exposing queue types here.
//
// `backends` are the ggml backend instances to expose via ggml-RPC.
void start_rpc_accept_loop(int srv_fd,
                           std::vector<ggml_backend *> backends,
                           void * hydra_ctx);

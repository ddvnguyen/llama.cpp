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

// ── Merged DECODE (0x43) framing constants ─────────────────────────────────────
static constexpr uint32_t HYDRA_MAX_JSON_HEADER  = 32U * 1024; // 32 KiB cap on JSON header (control header)
static constexpr uint64_t HYDRA_MAX_PROMPT_BYTES = 64ULL * 1024 * 1024; // 64 MiB cap on prompt segment
static constexpr int      HYDRA_DECODE_RESULT_TTL_S_DEFAULT = 300;
static constexpr int      HYDRA_DECODE_RESULT_MAX_DEFAULT   = 1024;
static constexpr uint8_t  HYDRA_DECODE_HDR_VERSION = 3; // current wire format version

// ── Unified RPC server (Phase 1, #36) ────────────────────────────────────────
// The protocol-detecting accept loop lives in
// `tools/llama-engine/hydra_rpc/` (fork-isolated). Both `server_context`
// (model-loaded path) and the no-model path in `llama-engine.cpp` call
// `hydra_rpc::start()` from that module. The dispatch is:
//   0x0E (RPC_CMD_HELLO) → ggml-backend RPC handler (GPU compute)
//   otherwise            → Hydra protocol handler (if `hydra_ctx != nullptr`)
//
// `server_context::start_rpc_server` builds the settings and calls
// `hydra_rpc::start()`. The Hydra handler
// `void hydra_handle_connection(int fd, const hydra_rpc_ctx & ctx)` is
// reached from the new module through the `extern "C"` trampoline
// `hydra_rpc_bridge` (defined in `server-context.cpp`).

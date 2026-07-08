// SPDX-License-Identifier: MIT
//
// Hydra: public API for the merged RPC server.
//
// Fork-isolated module — replaces the unified server logic that
// `ddvnguyen/llama.cpp#30` placed in `tools/server/server-context.cpp` with a
// fork-only module that owns the accept loop, the bounded thread pool, the
// MSG_PEEK dispatch, and the lifecycle. Per `ddvgguen/llama.cpp#36` Phase 1
// and C7: bounded thread pool, one-byte dispatch, no per-conn `detach()`.
//
// The server is opt-in: a model-loaded engine calls `hydra_rpc::start(...)`
// once after model load, with the local compute backends it wants to expose
// to inbound ggml-RPC peers and an opaque `hydra_ctx` (a pointer to the
// Hydra protocol context — the M1/M2 task queues). The accept thread blocks
// in `::accept()`; each connection is MSG_PEEK-dispatched to either
// `ggml_backend_rpc_handle_client` (for `RPC_CMD_HELLO = 0x0E`) or
// `hydra_handle_connection` (for Hydra opcodes 0x30–0x46). Connections are
// processed by a `bounded_thread_pool<2>` (size 2 per the design).
//
// SOLO mode (no `--rpc-port` / `port == 0`) does not call `start()`; the
// engine serves HTTP only.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

namespace hydra_rpc {

struct settings {
    int  port       = 0;        // 0 = RPC server disabled (SOLO mode).
    std::vector<ggml_backend_t> backends;   // local compute backends to expose to inbound ggml-RPC.
    void * hydra_ctx = nullptr; // opaque pointer to the Hydra protocol ctx (M1/M2 queues); nullptr = ggml-RPC only.
    std::size_t pool_size   = 2; // worker thread count (design default 2).
    std::size_t max_queue   = 64; // max pending connections before `try_enqueue` starts dropping.
    const char * host       = "0.0.0.0"; // bind address; default is INADDR_ANY.
};

// Start the RPC server. Returns false on bind/listen failure (port in use,
// permission denied, etc.). On success the accept thread is running and
// `is_running()` returns true. `stop()` is called automatically on engine
// shutdown via the `at_stop` hook installed in `start()`.
bool start(const settings & s);

// Stop the RPC server. Idempotent. Safe to call before the accept thread has
// even started (e.g. when `port == 0`). After `stop()` returns, the accept
// thread is joined and `is_running()` returns false.
void stop();

// True if the accept thread is currently running. False before `start()` and
// after `stop()`.
bool is_running();

// Convenience accessor for tests + logging. Returns the actual port the
// server is listening on (== `settings::port` if started, 0 if not). When
// `port == 0` at `start()` time, the server does not bind at all and this
// returns 0.
int bound_port();

}  // namespace hydra_rpc

// SPDX-License-Identifier: MIT
//
// Hydra: implementation of the merged RPC server (ggml-RPC + Hydra protocol).
//
// See `hydra_rpc.h` for the public API. This translation unit is the
// fork-isolated replacement for the unified server logic that
// `ddvnguyen/llama.cpp#30` placed in `tools/server/server-context.cpp`
// (see `start_rpc_accept_loop` / `server_context::start_rpc_server` there).
// The Hydra protocol handler (`hydra_handle_connection`) still lives in
// `server-context.cpp`; this module owns only the accept loop, the
// dispatch, and the lifecycle.
//
// Wire format: see `specs/rpc-protocol.md` and `tools/server/server-rpc.h`.
// Protocol dispatch is by ONE byte (per `#36` C7):
//   - `0x0E` (RPC_CMD_HELLO)        → ggml-rpc handler
//   - `0x30`–`0x46` (HYDRA_OP_*)     → Hydra handler (only if `hydra_ctx != nullptr`)
//   - anything else                  → close the connection (no magic bytes,
//                                       no per-conn `detach()`)

#include "hydra_rpc.h"
#include "bounded_thread_pool.h"

#include "ggml-rpc.h"
#include "log.h"  // LOG_INF / LOG_WRN / LOG_ERR (common/log.h)

#include <arpa/inet.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>

// Forward-declared Hydra protocol bridge. The real entry point in
// `tools/server/server-context.cpp` is `hydra_handle_connection(int, const hydra_rpc_ctx &)`;
// the bridge trampoline `hydra_rpc_bridge` is defined there with the
// `void*` signature this module uses, so we can keep the new module
// decoupled from the heavy server-context.h include.
extern "C" using hydra_handle_connection_fn = void (*)(int fd, const void * ctx);
extern "C" void hydra_rpc_bridge(int fd, const void * ctx);

// Defined in server-context.cpp — the actual Hydra context struct. Forward
// declared here as an opaque type; server-context.cpp fills it in.
struct hydra_rpc_ctx;

namespace hydra_rpc {

namespace {

struct state_t {
    int                                listen_fd  = -1;
    int                                port       = 0;
    std::vector<ggml_backend_t>        backends;
    void *                             hydra_ctx  = nullptr;
    hydra_handle_connection_fn         hydra_handler = nullptr;
    std::unique_ptr<bounded_thread_pool<8>> pool;  // 8 worker threads — matches a typical GPU's stream concurrency so the head's compute graph doesn't stall on the peer's dispatch path. The original `<2>` from the design draft (#36) was sized for a public-internet DDoS scenario; the peer in layer-split COMBINE is a trusted internal client that benefits from more parallelism.
    std::thread                        accept_thr;
    std::atomic<bool>                  running    { false };
    std::atomic<bool>                  stopping   { false };
    std::atomic<bool>                  peer_mode  { false };  // when true, accept_loop uses per-conn std::thread::detach() instead of the bounded thread pool. Set for no-model compute-only peers.
    std::mutex                         start_mu;
};
state_t & state() {
    static state_t s;
    return s;
}

// MSG_PEEK one byte; dispatch to ggml-rpc or hydra handler. Always closes
// the fd at the end (either the handler closes it, or we do).
void dispatch_one(int conn_fd, state_t & s) {
    uint8_t first_byte = 0;
    const ssize_t r = ::recv(conn_fd, &first_byte, 1, MSG_PEEK);
    if (r != 1) {
        ::close(conn_fd);
        return;
    }
    if (first_byte == 0x0E) {
        // ggml-rpc path. The handler takes ownership of the fd.
        ggml_backend_rpc_handle_client(
            conn_fd, /*cache_dir=*/nullptr,
            s.backends.size(), s.backends.data());
        return;
    }
    if (s.hydra_ctx && s.hydra_handler) {
        // Hydra path. The handler also takes ownership of the fd.
        s.hydra_handler(conn_fd, s.hydra_ctx);
        return;
    }
    // Unknown opcode AND we have no Hydra handler configured. Close.
    LOG_WRN("hydra_rpc: unknown first byte 0x%02x on fd %d — closing\n",
                  (unsigned) first_byte, conn_fd);
    ::close(conn_fd);
}

// `::accept()` here bypasses `socket_t::accept()` (ggml-rpc/transport.cpp),
// which is the only place upstream sets these — so the merged accept loop
// must set them itself. Without TCP_NODELAY, Nagle's algorithm batches the
// small request/response ggml-rpc messages that dominate prefill/decode,
// which is what was actually throttling the peer path (not dispatch
// overhead — see the `peer_mode` fast path below, which alone didn't fix it).
void set_conn_socket_options(int fd) {
    int flag = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    ::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag));
    // #470: bound the blocking state-stream send (llama_io_write_socket::
    // send_all) so a peer that stops reading cannot park an RPC worker
    // forever. MUST stay ABOVE the coordinator's per-chunk idle budget
    // (120s Store / 600s relay): the coordinator legitimately pauses its
    // socket reads while its downstream (relay channel / Store pipe)
    // backpressures — the decode leg preparing (model load, slot
    // acquisition) is exactly that case — and resumes reading once the
    // downstream drains. A send-park shorter than that budget (the old 30s)
    // killed healthy PREFILL streams mid-frame (engine EAGAIN -> coordinator
    // EOF -> zero-token turn, run 31760361575). 900s > the coordinator's max
    // 600s budget, so the coordinator's OWN idle deadline — not this socket
    // timeout — is what frees the worker on a genuinely stalled peer.
    struct timeval snd = { .tv_sec = 900, .tv_usec = 0 };
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &snd, sizeof(snd));
}

void accept_loop(state_t & s) {
    LOG_INF("hydra_rpc: accept loop started on 0.0.0.0:%d\n", s.port);
    while (!s.stopping.load(std::memory_order_acquire)) {
        const int conn_fd = ::accept(s.listen_fd, nullptr, nullptr);
        if (conn_fd < 0) {
            if (errno == EINTR) continue;
            if (s.stopping.load(std::memory_order_acquire)) break;
            LOG_ERR("hydra_rpc: accept() failed: %s\n", std::strerror(errno));
            continue;
        }
        set_conn_socket_options(conn_fd);
        if (s.peer_mode) {
            // No-model compute-only peer. Skip both the bounded thread pool
            // AND the MSG_PEEK+dispatch path — the peer only handles ggml-RPC
            // (the head never sends the Hydra protocol opcodes), so we can go
            // straight to `ggml_backend_rpc_handle_client`. This matches
            // upstream llama.cpp's `rpc-server.cpp` behavior byte-for-byte and
            // removes the ~100us per-request overhead that was throttling the
            // head's compute graph in the layer-split COMBINE path.
            std::thread([conn_fd, &s] {
                ggml_backend_rpc_handle_client(
                    conn_fd, /*cache_dir=*/nullptr,
                    s.backends.size(), s.backends.data());
            }).detach();
            continue;
        }
        // Per `#36` C7: bounded thread pool (size 2). Drop on overflow
        // rather than detach a new thread.
        if (!s.pool->try_enqueue([conn_fd, &s] { dispatch_one(conn_fd, s); })) {
            LOG_WRN("hydra_rpc: pool full, dropping connection (fd %d)\n", conn_fd);
            ::close(conn_fd);
        }
    }
    LOG_INF("hydra_rpc: accept loop exiting\n");
}

}  // namespace

bool start(const settings & s) {
    std::lock_guard<std::mutex> lk(state().start_mu);
    if (state().running.load()) {
        LOG_WRN("hydra_rpc: start() called while already running\n");
        return false;
    }
    if (s.port <= 0) {
        // SOLO mode: nothing to do. Return true so the caller can ignore.
        return true;
    }

    int srv_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv_fd < 0) {
        LOG_ERR("hydra_rpc: socket() failed: %s\n", std::strerror(errno));
        return false;
    }
    const int opt = 1;
    ::setsockopt(srv_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    if (s.host && std::strlen(s.host) > 0) {
        if (::inet_pton(AF_INET, s.host, &addr.sin_addr) != 1) {
            addr.sin_addr.s_addr = INADDR_ANY;
        }
    } else {
        addr.sin_addr.s_addr = INADDR_ANY;
    }
    addr.sin_port = htons((uint16_t) s.port);

    if (::bind(srv_fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
        LOG_ERR("hydra_rpc: bind() on port %d failed: %s\n", s.port, std::strerror(errno));
        ::close(srv_fd);
        return false;
    }
    if (::listen(srv_fd, 16) < 0) {
        LOG_ERR("hydra_rpc: listen() failed: %s\n", std::strerror(errno));
        ::close(srv_fd);
        return false;
    }

    // Resolve the Hydra handler symbol at start time. If the Hydra protocol
    // is not compiled in (a build that disables the Hydra-specific code
    // path), `hydra_ctx` should be nullptr and we only serve ggml-rpc.
    hydra_handle_connection_fn hydra_fn = nullptr;
    if (s.hydra_ctx) {
        // The symbol is defined in server-context.cpp. The new module
        // does not need to link against it directly — the same translation
        // unit that holds `hydra_rpc_ctx` defines `hydra_rpc_bridge`,
        // a thin trampoline that re-enters the C++ entry point.
        hydra_fn = &hydra_rpc_bridge;
    }

    state().listen_fd   = srv_fd;
    state().port        = s.port;
    state().backends    = s.backends;
    state().hydra_ctx   = s.hydra_ctx;
    state().hydra_handler = hydra_fn;
    state().peer_mode   = s.peer_mode;
    state().pool        = std::make_unique<bounded_thread_pool<8>>(s.max_queue);
    state().stopping.store(false, std::memory_order_release);
    state().accept_thr  = std::thread([&state_ref = state()] { accept_loop(state_ref); });
    state().running.store(true, std::memory_order_release);

    LOG_INF("hydra_rpc: unified server on 0.0.0.0:%d (%s)\n",
                  s.port, hydra_fn ? "ggml-RPC + Hydra protocol" : "ggml-RPC only");
    return true;
}

void stop() {
    state_t & s = state();
    if (!s.running.load()) return;
    {
        std::lock_guard<std::mutex> lk(s.start_mu);
        if (!s.running.load()) return;
        s.stopping.store(true, std::memory_order_release);
        if (s.listen_fd >= 0) {
            // Wake the accept thread with `shutdown()`.
            ::shutdown(s.listen_fd, SHUT_RDWR);
        }
    }
    if (s.accept_thr.joinable()) {
        s.accept_thr.join();
    }
    if (s.pool) {
        s.pool->stop();
        s.pool.reset();
    }
    if (s.listen_fd >= 0) {
        ::close(s.listen_fd);
        s.listen_fd = -1;
    }
    s.backends.clear();
    s.hydra_ctx = nullptr;
    s.hydra_handler = nullptr;
    s.running.store(false, std::memory_order_release);
    LOG_INF("hydra_rpc: stopped\n");
}

bool is_running() {
    return state().running.load(std::memory_order_acquire);
}

int bound_port() {
    return state().port;
}

void update_backends(const std::vector<ggml_backend_t> & backends) {
    state().backends = backends;
    LOG_INF("hydra_rpc: updated backends to %zu compute device(s)\n", backends.size());
}

}  // namespace hydra_rpc

// Hydra #470 (COMBINED crash): ggml-RPC stale-socket in the buffer data path.
//
// Bug: ggml_backend_rpc_buffer_context stored ONE socket captured at alloc
// time, and all 7 data-path ops (cpy_tensor, set_tensor, get_tensor,
// init_tensor, clear, free_buffer, get_base) sent over that stored socket
// with RPC_STATUS_ASSERT. During a COMBINED teardown/re-attach the head-side
// connection drops while the peer stays up; pre-existing buffers then hold a
// dead fd and the first 1-byte send after re-attach used to fail and
// GGML_ABORT the whole engine (smoke #8).
//
// Fix under test: every data-path op re-resolves the socket via
// get_socket(ctx->endpoint) — reusing the shipped liveness probe + reconnect
// (smoke #8) — so the op rides the CURRENT connection instead of the stale
// one. cpy_tensor fails open (returns false -> host round-trip fallback);
// the no-error-channel ops keep their loud abort for a genuinely unreachable
// peer (silently skipping a set/get is data corruption).
//
// Deterministic sub-scenarios exercised here (real ggml-RPC server over
// loopback, CPU backend, no GPU):
//   A. head-side connection dies while the peer stays up (the client's socket
//      is SHUTDOWN — the exact shape of a dropped head-side connection; the
//      server cleanly closes its side and keeps accepting).
//      A1. set_tensor on an EXISTING buffer must re-resolve via get_socket ->
//          reconnect -> the send completes WITHOUT abort. This discriminates:
//          on the old code the send rides the dead fd (shutdown fd -> EPIPE ->
//          RPC_STATUS_ASSERT -> SIGABRT); with the fix get_socket evicts the
//          stale sock and reconnects. (set_tensor is send-only, so it is the
//          deterministic "op succeeds after reconnect" probe.)
//      A2. cpy_tensor on the EXISTING buffers after the connection dies must
//          fail open to false, never abort (the peer's new session no longer
//          owns the pre-attach buffers and declines; that must not GGML_ABORT).
//   B. the reconnect chain yields a fully functional connection: a NEW buffer
//      allocated on the current session round-trips set/get and frees cleanly
//      — exactly what a live COMBINED re-attach does (re-allocate on the new
//      session, then boundary copies on fresh buffers).
//   C. peer truly unreachable (listener killed): cpy_tensor returns false,
//      no abort; a fork'd child doing get_tensor aborts LOUDLY (SIGABRT) —
//      the no-error-channel ops must never silently succeed on a dead peer.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-rpc.h"
#include "ggml-backend-impl.h" // #470: ggml_backend_buffer_copy_tensor lives here

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <dirent.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

static int g_failures = 0;

static void expect(const char * what, bool ok) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        g_failures++;
    }
}

// Simulate the HEAD-side connection dying: SHUTDOWN the client's socket fd(s)
// whose peer is `port` (found via /proc/self/fd — the client fd is the one
// with a loopback peer on the server port; its own local port is ephemeral).
//
// shutdown(fd, SHUT_RDWR) is chosen over close(fd): it is the faithful
// analogue of a dropped head-side connection — the FIN reaches the peer (the
// server cleanly closes its side and returns to accept()) while the fd stays
// allocated in this process, so the stale socket_t's fd number can NEVER be
// reused by the reconnect (no double-close / cross-talk hazard). The cached
// socket then probes dead (poll reports POLLHUP/POLLRDHUP on the shut socket)
// and get_socket reconnects on the next data-path op.
static void kill_head_side_connections(uint16_t port) {
    DIR * dir = opendir("/proc/self/fd");
    if (dir == nullptr) {
        return;
    }
    struct dirent * ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.') {
            continue;
        }
        int fd = atoi(ent->d_name);
        if (fd < 0) {
            continue;
        }
        struct sockaddr_storage paddr{};
        socklen_t plen = sizeof(paddr);
        if (getpeername(fd, (struct sockaddr *) &paddr, &plen) != 0) {
            continue;
        }
        if (paddr.ss_family != AF_INET) {
            continue;
        }
        auto * pin = (struct sockaddr_in *) &paddr;
        if (ntohs(pin->sin_port) != port) {
            continue;
        }
        ::shutdown(fd, SHUT_RDWR); // peer connection dies; fd number stays allocated
    }
    closedir(dir);
}

// Kill the server's LISTENING socket for `port` (local port == port, no peer):
// from here on the peer is genuinely unreachable (reconnect cannot succeed).
static void kill_listener(uint16_t port) {
    DIR * dir = opendir("/proc/self/fd");
    if (dir == nullptr) {
        return;
    }
    struct dirent * ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.') {
            continue;
        }
        int fd = atoi(ent->d_name);
        if (fd < 0) {
            continue;
        }
        struct sockaddr_storage laddr{};
        socklen_t llen = sizeof(laddr);
        if (getsockname(fd, (struct sockaddr *) &laddr, &llen) != 0) {
            continue;
        }
        if (laddr.ss_family != AF_INET) {
            continue;
        }
        auto * lin = (struct sockaddr_in *) &laddr;
        if (ntohs(lin->sin_port) != port) {
            continue;
        }
        struct sockaddr_storage paddr{};
        socklen_t plen = sizeof(paddr);
        if (getpeername(fd, (struct sockaddr *) &paddr, &plen) != 0) {
            ::close(fd); // no peer -> the listening socket
        }
    }
    closedir(dir);
}

// Run `fn` in a fork'd child; true iff the child died with SIGABRT (the
// no-error-channel data-path ops must abort LOUDLY on a truly unreachable
// peer, never silently succeed).
template <typename Fn>
static bool child_aborts(Fn fn) {
    pid_t pid = fork();
    if (pid == 0) {
        fn();            // must not return
        _exit(0);        // reached only on a regression (no abort)
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

static void wait_ms(long ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

int main() {
    const char * endpoint = "127.0.0.1:18767";
    const uint16_t port   = 18767;

    ggml_backend_load_all();
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        fprintf(stderr, "SKIP: RPC backend not available in this build\n");
        return 0;
    }

    using start_server_fn_t = void (*)(const char *, const char *, size_t, size_t, ggml_backend_t *);
    auto start_server_fn = (start_server_fn_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_start_server_with_backends");
    expect("resolved ggml_backend_rpc_start_server_with_backends", start_server_fn != nullptr);
    if (g_failures > 0) {
        return 1;
    }

    // "Server": a CPU backend served over loopback RPC (no GPU needed).
    ggml_backend_t cpu = ggml_backend_cpu_init();
    expect("cpu backend created", cpu != nullptr);
    std::vector<ggml_backend_t> backends = { cpu };
    std::thread server_thread([&]() {
        start_server_fn(endpoint, nullptr, /*n_threads=*/1, backends.size(), backends.data());
    });
    server_thread.detach();
    wait_ms(300); // listener ready

    // "Client": RPC backend + an EXISTING buffer predating every kill below.
    ggml_backend_t rpc = ggml_backend_rpc_init(endpoint, /*device=*/0);
    expect("rpc backend initialized", rpc != nullptr);
    if (rpc == nullptr) {
        g_failures++;
    } else {
        ggml_init_params iparams = { /*.mem_size=*/ ggml_tensor_overhead() * 4, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
        ggml_context * ctx = ggml_init(iparams);
        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, rpc);
        expect("existing rpc buffer allocated", buf != nullptr);

        if (buf != nullptr) {
            float want[16];
            for (int i = 0; i < 16; i++) {
                want[i] = (float) i * 1.5f;
            }

            // --- baseline: live connection round trip ---
            ggml_backend_tensor_set(a, want, 0, sizeof(want));
            float got0[16] = {};
            ggml_backend_tensor_get(a, got0, 0, sizeof(got0));
            expect("baseline set/get round trip on live connection",
                   std::memcmp(want, got0, sizeof(want)) == 0);

            // ---------------------------------------------------------------
            // A1. #470 core regression: head-side connection dies (peer stays
            // up); set_tensor on the EXISTING buffer must re-resolve via
            // get_socket -> reconnect -> the send completes, NO abort. Old
            // code: sends over the dead stored sock (shutdown fd -> EPIPE) ->
            // RPC_STATUS_ASSERT -> SIGABRT.
            // ---------------------------------------------------------------
            kill_head_side_connections(port);
            wait_ms(150); // peer sees FIN and returns to accept()
            ggml_backend_tensor_set(a, want, 0, sizeof(want)); // old code aborts here
            expect("set_tensor on existing buffer reconnects after connection death (no abort)", true);

            // ---------------------------------------------------------------
            // A2. cpy_tensor on the EXISTING buffers after the connection dies
            // must fail open to false (the peer's new session no longer owns
            // the pre-attach buffers and declines the copy), never abort.
            // ---------------------------------------------------------------
            kill_head_side_connections(port);
            wait_ms(150);
            bool cpy_ok = ggml_backend_buffer_copy_tensor(a, b); // old code aborts here
            expect("cpy_tensor on existing buffer after connection death fails open (false, no abort)",
                   !cpy_ok);

            // ---------------------------------------------------------------
            // B. the reconnect chain yields a fully functional connection: a
            // NEW buffer allocated on the current session round-trips set/get
            // and frees cleanly — what a live COMBINED re-attach does.
            // ---------------------------------------------------------------
            {
                ggml_init_params c2_params = { /*.mem_size=*/ ggml_tensor_overhead() * 2, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
                ggml_context * c2 = ggml_init(c2_params);
                ggml_tensor * t2 = ggml_new_tensor_1d(c2, GGML_TYPE_F32, 16);
                ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors(c2, rpc);
                expect("buffer re-allocated on the reconnected session", buf2 != nullptr);
                if (buf2 != nullptr) {
                    ggml_backend_tensor_set(t2, want, 0, sizeof(want));
                    float got2[16] = {};
                    ggml_backend_tensor_get(t2, got2, 0, sizeof(got2));
                    expect("set/get round trip on the reconnected connection (peer serving again)",
                           std::memcmp(want, got2, sizeof(want)) == 0);
                    ggml_backend_buffer_free(buf2); // same live session: free rides the current connection
                }
                ggml_free(c2);
            }

            // ---------------------------------------------------------------
            // C. Truly dead peer (listener + connections killed): reconnect is
            // impossible.
            //   - cpy_tensor fails open: returns false, never aborts.
            //   - get_tensor (no error channel) aborts LOUDLY (SIGABRT) — a
            //     silent success on a dead peer would be data corruption.
            // ---------------------------------------------------------------
            {
                kill_head_side_connections(port);
                kill_listener(port);
                wait_ms(150);

                bool cpy_ok2 = ggml_backend_buffer_copy_tensor(a, b);
                expect("cpy_tensor on a truly dead peer returns false (fail-open, no abort)", !cpy_ok2);

                float sink[16] = {};
                expect("get_tensor on a truly dead peer aborts loudly (SIGABRT, never silent)",
                       child_aborts([&]() { ggml_backend_tensor_get(a, sink, 0, sizeof(sink)); }));
            }
            // NOTE: buf is intentionally leaked — freeing an RPC buffer on a
            // genuinely unreachable peer aborts by design (void free_buffer,
            // no error channel); the test process is about to exit and the OS
            // reclaims everything.
        }

        ggml_free(ctx);
        ggml_backend_free(rpc);
    }
    ggml_backend_free(cpu);

    if (g_failures == 0) {
        printf("OK: ggml-RPC data-path ops re-resolve via get_socket after head-side connection "
               "death (set_tensor reconnects without abort; cpy fails open); the reconnected "
               "connection serves again; cpy fails open + void op aborts loudly on a dead peer\n");
        return 0;
    }
    fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
}

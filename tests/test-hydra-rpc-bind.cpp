// Hydra llama.cpp#20: pure-protocol test for zero-copy COMBINED expert
// tensors — RPC_CMD_RESOLVE_TENSOR + ggml_backend_rpc_register_local_tensor/
// ggml_backend_rpc_bind_remote_tensor. Spins up a real ggml-RPC server over
// loopback (CPU backend, no GPU needed), registers a tensor with known data,
// and confirms a "client" in the same process can resolve + bind to it
// without any RPC_CMD_ALLOC_BUFFER/SET_TENSOR round trip — only
// RPC_CMD_RESOLVE_TENSOR (metadata) and RPC_CMD_GET_TENSOR (read-back, to
// prove the bound tensor really points at the server's live data).
//
// #368: also exercises registry_epoch (clear-and-re-register bumps it),
// the ne-guard (rejects binds with mismatched shape), fail-open (RPC
// error returns null instead of aborting), and a remote-epoch probe
// (ggml_backend_rpc_get_remote_registry_epoch).
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-rpc.h"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

static int g_failures = 0;

static void expect(const char * what, bool ok) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        g_failures++;
    }
}

int main() {
    const char * endpoint = "127.0.0.1:18765";

    ggml_backend_load_all();
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        fprintf(stderr, "SKIP: RPC backend not available in this build\n");
        return 0;
    }

    using start_server_fn_t  = void (*)(const char *, const char *, size_t, size_t, ggml_backend_t *);
    using register_fn_t      = void (*)(const char *, struct ggml_tensor *);
    using clear_fn_t         = void (*)(void);
    using get_epoch_fn_t     = uint32_t (*)(void);
    using get_remote_epoch_fn_t = uint32_t (*)(const char *);
    using bind_remote_fn_t   = struct ggml_tensor * (*)(const char *, uint32_t, struct ggml_context *, const char *, const uint32_t *, uint32_t *);

    auto start_server_fn = (start_server_fn_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_start_server_with_backends");
    auto register_fn     = (register_fn_t)     ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_register_local_tensor");
    auto bind_remote_fn  = (bind_remote_fn_t)  ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_bind_remote_tensor");
    auto clear_fn        = (clear_fn_t)        ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_clear_local_tensors");
    auto get_epoch_fn    = (get_epoch_fn_t)    ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_get_registry_epoch");
    auto get_remote_epoch_fn = (get_remote_epoch_fn_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_get_remote_registry_epoch");
    expect("resolved ggml_backend_rpc_start_server_with_backends", start_server_fn != nullptr);
    expect("resolved ggml_backend_rpc_register_local_tensor", register_fn != nullptr);
    expect("resolved ggml_backend_rpc_bind_remote_tensor", bind_remote_fn != nullptr);
    expect("resolved ggml_backend_rpc_clear_local_tensors (#368)", clear_fn != nullptr);
    expect("resolved ggml_backend_rpc_get_registry_epoch (#368)", get_epoch_fn != nullptr);
    expect("resolved ggml_backend_rpc_get_remote_registry_epoch (#368)", get_remote_epoch_fn != nullptr);
    if (g_failures > 0) {
        return 1;
    }

    // "Server": a CPU backend with one real tensor, filled with known data,
    // registered for zero-copy resolution under the name "weight.test".
    ggml_backend_t cpu = ggml_backend_cpu_init();
    expect("cpu backend created", cpu != nullptr);

    ggml_init_params iparams = { /*.mem_size=*/ ggml_tensor_overhead() * 2, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_context * server_ctx = ggml_init(iparams);
    ggml_tensor * server_tensor = ggml_new_tensor_1d(server_ctx, GGML_TYPE_F32, 16);
    ggml_set_name(server_tensor, "weight.test");

    ggml_backend_buffer_t server_buf = ggml_backend_alloc_ctx_tensors(server_ctx, cpu);
    expect("server tensor allocated on cpu backend", server_buf != nullptr);

    float want[16];
    for (int i = 0; i < 16; i++) {
        want[i] = (float) i * 1.5f;
    }
    ggml_backend_tensor_set(server_tensor, want, 0, sizeof(want));

    register_fn("weight.test", server_tensor);

    std::vector<ggml_backend_t> backends = { cpu };
    std::thread server_thread([&]() {
        start_server_fn(endpoint, nullptr, /*n_threads=*/1, backends.size(), backends.data());
    });
    server_thread.detach();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // "Client": resolve + bind, in the same process, over loopback. The
    // metadata context is sized for many binds (we issue ~8 in this test:
    // the original, the ne-guard, the legacy, the missing-name probe, the
    // re-bind after clear, etc.). The bound tensors' storage lives here.
    const size_t client_ctx_mem = ggml_tensor_overhead() * 16;
    ggml_init_params client_iparams = { /*.mem_size=*/ client_ctx_mem, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_context * client_ctx = ggml_init(client_iparams);

    // (#368) epoch probe via the dedicated RPC — should equal the local
    // registry epoch (the server we just registered is in this process).
    {
        uint32_t remote_epoch = get_remote_epoch_fn(endpoint);
        uint32_t local_epoch  = get_epoch_fn();
        expect("remote registry epoch matches local after register", remote_epoch == local_epoch);
    }

    uint32_t bound_epoch = 0;
    uint32_t expected_ne[GGML_MAX_DIMS] = { 16, 1, 1, 1 };
    ggml_tensor * bound = bind_remote_fn(endpoint, /*device=*/0, client_ctx, "weight.test", expected_ne, &bound_epoch);
    expect("bind_remote_tensor found 'weight.test'", bound != nullptr);
    if (bound) {
        expect("bound tensor type matches", bound->type == GGML_TYPE_F32);
        expect("bound tensor element count matches", ggml_nelements(bound) == 16);
        expect("bound tensor has a buffer", bound->buffer != nullptr);
        expect("bound epoch returned to caller", bound_epoch == get_epoch_fn());

        float got[16] = {};
        ggml_backend_tensor_get(bound, got, 0, sizeof(got));
        expect("round-tripped data matches (zero-copy read of server's live tensor)",
                std::memcmp(want, got, sizeof(want)) == 0);
    }

    // (#368) ne-guard: a bind with a mismatched expected shape must be refused.
    {
        uint32_t bad_ne[GGML_MAX_DIMS] = { 8, 1, 1, 1 };  // peer has 16, not 8
        uint32_t discard_epoch = 0;
        ggml_tensor * rejected = bind_remote_fn(endpoint, /*device=*/0, client_ctx, "weight.test", bad_ne, &discard_epoch);
        expect("ne-guard rejects bind with mismatched shape (returns null)", rejected == nullptr);
    }

    // (#368) ne-guard: NULL expected_ne means "skip the guard" (legacy callers).
    {
        ggml_tensor * legacy = bind_remote_fn(endpoint, /*device=*/0, client_ctx, "weight.test", nullptr, nullptr);
        expect("bind with NULL expected_ne skips the ne-guard", legacy != nullptr);
    }

    ggml_tensor * missing = bind_remote_fn(endpoint, /*device=*/0, client_ctx, "weight.does_not_exist", nullptr, nullptr);
    expect("bind_remote_tensor returns null for an unregistered name", missing == nullptr);

    // (#368) clear → epoch bumps → re-register → epoch is unchanged (only
    // clear bumps; register is monotonic on top of the current generation).
    {
        const uint32_t epoch_at_first_bind = bound_epoch;
        clear_fn();
        const uint32_t epoch_after_clear = get_epoch_fn();
        expect("clear_local_tensors bumps the epoch", epoch_after_clear > epoch_at_first_bind);

        // Re-register the same tensor (in a real workflow this would be a
        // fresh model load with a different quant); the epoch stays at the
        // post-clear value (the head's epoch re-check via
        // ggml_backend_rpc_get_remote_registry_epoch is what catches the
        // clear — not a further register-time bump).
        register_fn("weight.test", server_tensor);
        const uint32_t epoch_after_reregister = get_epoch_fn();
        expect("re-register does not bump the epoch (clear is the only bumper)", epoch_after_reregister == epoch_after_clear);

        // A subsequent bind observes the new epoch.
        uint32_t new_bound_epoch = 0;
        ggml_tensor * rebound = bind_remote_fn(endpoint, /*device=*/0, client_ctx, "weight.test", expected_ne, &new_bound_epoch);
        expect("re-bind after clear+re-register succeeds", rebound != nullptr);
        if (rebound) {
            expect("re-bound epoch matches the new (post-clear) epoch", new_bound_epoch == epoch_after_clear);
        }
    }

    // (#368) fail-open: bind to an unreachable endpoint returns null, does
    // not abort. (Use a port nothing is listening on; connect will fail
    // and the function must return nullptr rather than RPC_STATUS_ASSERT
    // calling GGML_ABORT.)
    {
        const char * dead_endpoint = "127.0.0.1:1";
        ggml_tensor * failed = bind_remote_fn(dead_endpoint, /*device=*/0, client_ctx, "weight.test", nullptr, nullptr);
        expect("bind to unreachable endpoint returns null (fail-open, no abort)", failed == nullptr);
    }

    // (#368) re-callable register + bound epoch comparison.
    {
        const uint32_t epoch_pre = get_epoch_fn();
        register_fn("weight.test", server_tensor);
        const uint32_t epoch_post = get_epoch_fn();
        expect("register alone is monotonic (does not bump)",
                epoch_post == epoch_pre);
    }

    ggml_free(client_ctx);
    ggml_backend_buffer_free(server_buf);
    ggml_free(server_ctx);
    ggml_backend_free(cpu);

    if (g_failures == 0) {
        printf("OK: zero-copy RESOLVE_TENSOR/bind_remote_tensor round-trips correctly "
               "(incl. #368 epoch, ne-guard, fail-open, re-callable register)\n");
        return 0;
    }
    fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
}

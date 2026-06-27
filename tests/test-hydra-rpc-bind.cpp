// Hydra llama.cpp#20: pure-protocol test for zero-copy COMBINED expert
// tensors — RPC_CMD_RESOLVE_TENSOR + ggml_backend_rpc_register_local_tensor/
// ggml_backend_rpc_bind_remote_tensor. Spins up a real ggml-RPC server over
// loopback (CPU backend, no GPU needed), registers a tensor with known data,
// and confirms a "client" in the same process can resolve + bind to it
// without any RPC_CMD_ALLOC_BUFFER/SET_TENSOR round trip — only
// RPC_CMD_RESOLVE_TENSOR (metadata) and RPC_CMD_GET_TENSOR (read-back, to
// prove the bound tensor really points at the server's live data).
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-rpc.h"

#include <chrono>
#include <cstdio>
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
    using bind_remote_fn_t   = struct ggml_tensor * (*)(const char *, uint32_t, struct ggml_context *, const char *);

    auto start_server_fn = (start_server_fn_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_start_server_with_backends");
    auto register_fn     = (register_fn_t)     ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_register_local_tensor");
    auto bind_remote_fn   = (bind_remote_fn_t)  ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_bind_remote_tensor");
    expect("resolved ggml_backend_rpc_start_server_with_backends", start_server_fn != nullptr);
    expect("resolved ggml_backend_rpc_register_local_tensor", register_fn != nullptr);
    expect("resolved ggml_backend_rpc_bind_remote_tensor", bind_remote_fn != nullptr);
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

    // "Client": resolve + bind, in the same process, over loopback.
    ggml_init_params client_iparams = { /*.mem_size=*/ ggml_tensor_overhead() * 2, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_context * client_ctx = ggml_init(client_iparams);

    ggml_tensor * bound = bind_remote_fn(endpoint, /*device=*/0, client_ctx, "weight.test");
    expect("bind_remote_tensor found 'weight.test'", bound != nullptr);
    if (bound) {
        expect("bound tensor type matches", bound->type == GGML_TYPE_F32);
        expect("bound tensor element count matches", ggml_nelements(bound) == 16);
        expect("bound tensor has a buffer", bound->buffer != nullptr);

        float got[16] = {};
        ggml_backend_tensor_get(bound, got, 0, sizeof(got));
        expect("round-tripped data matches (zero-copy read of server's live tensor)",
                std::memcmp(want, got, sizeof(want)) == 0);
    }

    ggml_tensor * missing = bind_remote_fn(endpoint, /*device=*/0, client_ctx, "weight.does_not_exist");
    expect("bind_remote_tensor returns null for an unregistered name", missing == nullptr);

    ggml_free(client_ctx);
    ggml_backend_buffer_free(server_buf);
    ggml_free(server_ctx);
    ggml_backend_free(cpu);

    if (g_failures == 0) {
        printf("OK: zero-copy RESOLVE_TENSOR/bind_remote_tensor round-trips correctly\n");
        return 0;
    }
    fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
}

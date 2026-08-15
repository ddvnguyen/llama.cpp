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
#include "ggml-impl.h" // #470: ggml_cgraph::uid is set directly by the regression test

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

    // =========================================================================
    // #470: freed-buffer recompute refusal. The RPC graph-reuse protocol ties a
    // process-global cross-context uid tracker (last_graph_uid, ggml-rpc.cpp)
    // to a per-connection server-side stored graph holding deserialized
    // ABSOLUTE data pointers. When a second context / rebuilt context / #634
    // reconnect reuses a graph uid after the first context's buffers were
    // freed, the peer used to re-execute the PREVIOUS context's stored graph
    // against freed memory → batched GEMM dereferences an unmapped VA (Xid
    // 13/31). Regression: after free_buffer, a same-uid compute must be a full
    // GRAPH_COMPUTE re-serialized with the CURRENT buffer's data pointers.
    {
        ggml_backend_t rpc_backend = ggml_backend_rpc_init(endpoint, /*device=*/0);
        expect("#470: rpc backend initialized", rpc_backend != nullptr);
        if (rpc_backend == nullptr) {
            g_failures++; // nothing else in this section is meaningful
        } else {
            // --- context 1: graph uid 0x470470, buffer freed afterwards ---
            ggml_init_params c1_params = { /*.mem_size=*/ ggml_tensor_overhead() * 4, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
            ggml_context * c1 = ggml_init(c1_params);
            ggml_tensor * a1 = ggml_new_tensor_1d(c1, GGML_TYPE_F32, 8);
            ggml_tensor * b1 = ggml_new_tensor_1d(c1, GGML_TYPE_F32, 8);
            ggml_tensor * r1 = ggml_add(c1, a1, b1);
            ggml_backend_buffer_t buf1 = ggml_backend_alloc_ctx_tensors(c1, rpc_backend);
            expect("#470: context-1 buffer allocated on rpc backend", buf1 != nullptr);
            if (buf1 != nullptr) {
                float a1v[8], b1v[8];
                for (int i = 0; i < 8; i++) { a1v[i] = (float) i + 1.0f; b1v[i] = (float) i * 2.0f; }
                ggml_backend_tensor_set(a1, a1v, 0, sizeof(a1v));
                ggml_backend_tensor_set(b1, b1v, 0, sizeof(b1v));

                ggml_cgraph * g1 = ggml_new_graph_custom(c1, 1, false);
                g1->nodes[0] = r1;
                g1->n_nodes  = 1;
                g1->uid      = 0x470470u; // deterministic "recycled" uid (real uids come from ggml_graph_next_uid)
                // mirrors what the scheduler does to graph nodes: without the
                // COMPUTE flag the backend's compute loop skips the node.
                r1->flags |= GGML_TENSOR_FLAG_COMPUTE;

                expect("#470: first compute (full GRAPH_COMPUTE) succeeds",
                       ggml_backend_graph_compute(rpc_backend, g1) == GGML_STATUS_SUCCESS);
                float got1[8] = {};
                ggml_backend_tensor_get(r1, got1, 0, sizeof(got1));
                bool ok1 = true;
                for (int i = 0; i < 8; i++) { ok1 = ok1 && (got1[i] == a1v[i] + b1v[i]); }
                expect("#470: context-1 result correct after first compute", ok1);

                // same uid again → GRAPH_RECOMPUTE fast-path (stored graph)
                expect("#470: second compute (GRAPH_RECOMPUTE path) succeeds",
                       ggml_backend_graph_compute(rpc_backend, g1) == GGML_STATUS_SUCCESS);
                float got1b[8] = {};
                ggml_backend_tensor_get(r1, got1b, 0, sizeof(got1b));
                bool ok1b = true;
                for (int i = 0; i < 8; i++) { ok1b = ok1b && (got1b[i] == a1v[i] + b1v[i]); }
                expect("#470: context-1 result correct after recompute", ok1b);

                // free context-1's buffer: the peer frees the memory and must
                // drop its stored graph; the client must reset its recompute
                // fast-path so the next same-uid compute re-serializes.
                ggml_backend_buffer_free(buf1);

                // --- context 2: SAME graph uid against NEW buffers ---
                ggml_init_params c2_params = { /*.mem_size=*/ ggml_tensor_overhead() * 4, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
                ggml_context * c2 = ggml_init(c2_params);
                ggml_tensor * a2 = ggml_new_tensor_1d(c2, GGML_TYPE_F32, 8);
                ggml_tensor * b2 = ggml_new_tensor_1d(c2, GGML_TYPE_F32, 8);
                ggml_tensor * r2 = ggml_add(c2, a2, b2);
                ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors(c2, rpc_backend);
                expect("#470: context-2 buffer allocated after free", buf2 != nullptr);
                if (buf2 != nullptr) {
                    float a2v[8], b2v[8];
                    for (int i = 0; i < 8; i++) { a2v[i] = (float) (i + 100); b2v[i] = (float) (i * 3 + 7); }
                    ggml_backend_tensor_set(a2, a2v, 0, sizeof(a2v));
                    ggml_backend_tensor_set(b2, b2v, 0, sizeof(b2v));

                    // poison r2 so a stale recompute that writes nowhere (or
                    // into the freed buffer) cannot accidentally pass.
                    float poison[8];
                    for (int i = 0; i < 8; i++) { poison[i] = -1.0f; }
                    ggml_backend_tensor_set(r2, poison, 0, sizeof(poison));

                    ggml_cgraph * g2 = ggml_new_graph_custom(c2, 1, false);
                    g2->nodes[0] = r2;
                    g2->n_nodes  = 1;
                    g2->uid      = 0x470470u; // SAME uid as context 1
                    r2->flags |= GGML_TENSOR_FLAG_COMPUTE;

                    expect("#470: same-uid compute after free (must be full GRAPH_COMPUTE) succeeds",
                           ggml_backend_graph_compute(rpc_backend, g2) == GGML_STATUS_SUCCESS);
                    float got2[8] = {};
                    ggml_backend_tensor_get(r2, got2, 0, sizeof(got2));
                    bool ok2 = true;
                    for (int i = 0; i < 8; i++) { ok2 = ok2 && (got2[i] == a2v[i] + b2v[i]); }
                    expect("#470: context-2 result correct (no stale-graph recompute against freed buffers)", ok2);

                    ggml_backend_buffer_free(buf2);
                }
                ggml_free(c2);
            }
            ggml_free(c1);
            ggml_backend_free(rpc_backend);
        }
    }

    // #634/#470: the get_alloc_size fallback must NEVER under-estimate the
    // peer's allocation (peer pads quantized rows to 512 elements for MMQ and
    // reserves flash-attn scratch). Regression on the fallback formula itself.
    {
        ggml_init_params f_params = { /*.mem_size=*/ ggml_tensor_overhead() * 4, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
        ggml_context * fctx = ggml_init(f_params);
        ggml_tensor * quant = ggml_new_tensor_1d(fctx, GGML_TYPE_Q8_0, 1000); // not a multiple of 512
        ggml_tensor * plain = ggml_new_tensor_1d(fctx, GGML_TYPE_F32, 1000);

        const size_t quant_nbytes  = ggml_nbytes(quant);
        const size_t fallback_quant = ggml_backend_rpc_get_alloc_size_fallback(quant);
        const size_t fallback_plain = ggml_backend_rpc_get_alloc_size_fallback(plain);

        expect("#470: quantized fallback strictly larger than ggml_nbytes (512-row MMQ padding)",
               fallback_quant > quant_nbytes);
        expect("#470: non-quantized fallback equals ggml_nbytes",
               fallback_plain == ggml_nbytes(plain));

        // Property that matters: the fallback must never be smaller than what
        // the live peer's allocator reports for the same tensor.
        ggml_backend_t rpc_backend2 = ggml_backend_rpc_init(endpoint, /*device=*/0);
        if (rpc_backend2 != nullptr) {
            ggml_backend_buffer_type_t rpc_buft2 = ggml_backend_get_default_buffer_type(rpc_backend2);
            const size_t server_size = ggml_backend_buft_get_alloc_size(rpc_buft2, quant);
            expect("#470: alloc-size fallback is never smaller than the live server's alloc_size",
                   fallback_quant >= server_size);
            ggml_backend_free(rpc_backend2);
        }
        ggml_free(fctx);
    }

    ggml_free(client_ctx);
    ggml_backend_buffer_free(server_buf);
    ggml_free(server_ctx);
    ggml_backend_free(cpu);

    if (g_failures == 0) {
        printf("OK: zero-copy RESOLVE_TENSOR/bind_remote_tensor round-trips correctly "
               "(incl. #368 epoch, ne-guard, fail-open, re-callable register) "
               "+ #470 freed-buffer recompute refusal + #634 alloc-size over-estimate fallback\n");
        return 0;
    }
    fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
}

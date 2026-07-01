#include "llama-hydra.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <atomic>
#include <cstring>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <unistd.h>
#endif

size_t llama_state_seq_get_data_to_fd(struct llama_context * ctx, llama_seq_id seq_id, int fd) {
    return ctx->state_seq_get_data_to_fd(seq_id, fd);
}

void llama_hydra_set_state_chunk_size(struct llama_context * ctx, size_t bytes) {
    ctx->hydra_set_state_chunk_size(bytes);
}

size_t llama_hydra_get_state_chunk_size(const struct llama_context * ctx) {
    return ctx->get_cparams().hydra_state_chunk_size;
}

bool llama_hydra_peer_reachable(const char * host_port) {
#if !defined(_WIN32)
    std::string hp(host_port);
    auto colon = hp.rfind(':');
    if (colon == std::string::npos) return false;
    std::string host = hp.substr(0, colon);
    int port = std::stoi(hp.substr(colon + 1));

    struct addrinfo hints = {}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0) return false;

    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); return false; }

    struct timeval tv = { .tv_sec = 3 };
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    bool ok = (connect(fd, res->ai_addr, res->ai_addrlen) == 0);
    close(fd);
    freeaddrinfo(res);
    return ok;
#else
    (void)host_port;
    return false;
#endif
}

void llama_hydra_register_local_tensors_for_rpc(struct llama_context * ctx) {
    ggml_backend_load_all();
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        LLAMA_LOG_WARN("hydra: local-tensor registration requested but the RPC backend is not available in this build\n");
        return;
    }
    using register_fn_t = void (*)(const char *, struct ggml_tensor *);
    using clear_fn_t    = void (*)(void);
    auto register_fn = (register_fn_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_register_local_tensor");
    auto clear_fn    = (clear_fn_t)    ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_clear_local_tensors");
    if (!register_fn) {
        LLAMA_LOG_ERROR("hydra: failed to resolve ggml_backend_rpc_register_local_tensor\n");
        return;
    }

    // #368: re-callable registration. Clear the existing registry first so
    // any prior binding (which recorded the old epoch) is observably
    // stale; the head will rebind on the next SET_EXPERT_MODE("combined").
    // This is safe even on first call (clear is a no-op on an empty
    // registry, just bumps the epoch from 0 → 1).
    if (clear_fn) {
        clear_fn();
    }

    const llama_model & model = ctx->get_model();
    size_t n = 0;
    for (const auto & kv : model.tensors_by_name) {
        if (kv.second && kv.second->buffer && kv.second->data) {
            register_fn(kv.first.c_str(), kv.second);
            n++;
        }
    }
    LLAMA_LOG_INFO("hydra: registered %zu resident tensor(s) for zero-copy RPC resolution (epoch bumped)\n", n);
}

int32_t llama_hydra_load_combined_experts(
        struct llama_context * ctx,
                   const char * peer_endpoint,
                   const char * tensor_pattern) {
    if (!llama_hydra_peer_reachable(peer_endpoint)) {
        LLAMA_LOG_WARN("hydra: COMBINED peer %s unreachable, skipping dual-load\n", peer_endpoint);
        return -1;
    }

    ggml_backend_load_all();
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        LLAMA_LOG_ERROR("hydra: COMBINED requires the RPC backend, but it is not available in this build\n");
        return -1;
    }

    using add_server_fn_t = ggml_backend_reg_t (*)(const char *);
    auto add_server_fn = (add_server_fn_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (!add_server_fn) {
        LLAMA_LOG_ERROR("hydra: COMBINED — failed to resolve ggml_backend_rpc_add_server\n");
        return -1;
    }

    ggml_backend_reg_t peer_reg = add_server_fn(peer_endpoint);
    if (!peer_reg || ggml_backend_reg_dev_count(peer_reg) == 0) {
        LLAMA_LOG_ERROR("hydra: COMBINED — failed to register peer %s as an RPC device\n", peer_endpoint);
        return -1;
    }
    ggml_backend_register(peer_reg);

    ggml_backend_dev_t peer_dev = ggml_backend_reg_dev_get(peer_reg, 0);

    return llama_hydra_rebind_combined_experts(ctx, peer_endpoint, peer_dev, tensor_pattern);
}

// #368: per-peer resource tracking. Maps peer_endpoint → the bound state
// (the meta_ctx whose lifetime backs the bound tensors' storage, and the
// peer backend device that the scheduler was extended with). Keyed by
// the endpoint string so different peers get different bindings.
namespace {
struct hydra_combined_peer_binding {
    ggml_context_ptr      meta_ctx;
    ggml_backend_dev_t    peer_dev = nullptr;
    uint32_t              last_bound_epoch = 0;
    size_t                n_layers_loaded = 0;
};
static std::unordered_map<std::string, hydra_combined_peer_binding> s_hydra_combined_bindings;
} // namespace

// #368: rebind the peer's expert tensors on demand. Public (called from
// the SET_EXPERT_MODE("combined") handler in server-context.cpp). Behavior:
//   - peer reachable + tensors bind cleanly → all _rpc fields populated, the
//     scheduler has the peer's backend, returns the new layer count.
//   - peer unreachable, ne-guard fails, RPC error, or no tensors match
//     pattern → all _rpc fields cleared, returns 0 or -1 (caller falls back
//     to solo). The previous binding's meta_ctx is freed BEFORE the new
//     meta_ctx is allocated, so the per-rebind synthetic buffer count is
//     steady (no growth across N rebinds).
//
// meta_ctx leak fix: the old code pushed meta_ctx into a static vector on
// first call and never freed subsequent allocations — fine for one-shot
// startup, a leak per rebind otherwise. Here the meta_ctx is owned by the
// per-peer map; on rebind we replace it, which destroys the previous
// ggml_context and frees its backing storage.
int32_t llama_hydra_rebind_combined_experts(
        struct llama_context * ctx,
                   const char * peer_endpoint,
                   ggml_backend_dev_t peer_dev,
                   const char * tensor_pattern) {
    ggml_backend_load_all();
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        LLAMA_LOG_ERROR("hydra: COMBINED rebind — RPC backend not available\n");
        return -1;
    }

    using bind_remote_fn_t = struct ggml_tensor * (*)(const char *, uint32_t, struct ggml_context *, const char *, const uint32_t *, uint32_t *);
    auto bind_remote_fn = (bind_remote_fn_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_bind_remote_tensor");
    if (!bind_remote_fn) {
        LLAMA_LOG_ERROR("hydra: COMBINED rebind — failed to resolve ggml_backend_rpc_bind_remote_tensor\n");
        return -1;
    }

    llama_model & model = const_cast<llama_model &>(ctx->get_model());

    // Free the previous binding first (clears _rpc fields, drops the
    // previous meta_ctx). This is the per-rebind leak fix.
    if (auto it = s_hydra_combined_bindings.find(peer_endpoint);
        it != s_hydra_combined_bindings.end()) {
        for (size_t il = 0; il < model.layers.size(); il++) {
            llama_layer & layer = model.layers[il];
            layer.ffn_gate_exps_rpc = nullptr;
            layer.ffn_up_exps_rpc   = nullptr;
            layer.ffn_down_exps_rpc = nullptr;
        }
        // it->second.meta_ctx.reset() is invoked by the assignment below
        // (replacing the entry destroys the old meta_ctx).
    }

    // --combined-ot-pattern uses override-tensor syntax: "<name_regex>=<backend>".
    // The tensor name is everything before the last '='. Strip the suffix so
    // the regex matches actual tensor names (e.g. "blk.0.ffn_gate_exps.weight").
    std::string name_pattern = tensor_pattern;
    {
        auto eq = name_pattern.rfind('=');
        if (eq != std::string::npos) {
            name_pattern = name_pattern.substr(0, eq);
        }
    }

    std::regex re;
    try {
        re = std::regex(name_pattern);
    } catch (const std::regex_error & e) {
        LLAMA_LOG_ERROR("hydra: COMBINED rebind — invalid tensor_pattern '%s': %s\n", name_pattern.c_str(), e.what());
        return -1;
    }

    struct pending_bind {
        ggml_tensor ** dst_field;
        std::string    name;
        uint32_t       expected_ne[GGML_MAX_DIMS];
    };
    std::vector<pending_bind> pending;

    const size_t n_layer = model.layers.size();
    const size_t ctx_mem = n_layer * 3 * ggml_tensor_overhead() + ggml_tensor_overhead() * 4;
    ggml_init_params iparams = { /*.mem_size =*/ ctx_mem, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context_ptr meta_ctx(ggml_init(iparams));
    if (!meta_ctx) {
        LLAMA_LOG_ERROR("hydra: COMBINED rebind — failed to allocate metadata context\n");
        return -1;
    }

    auto consider = [&](ggml_tensor * src, ggml_tensor ** dst_field) {
        if (!src || !std::regex_match(ggml_get_name(src), re)) {
            return;
        }
        pending_bind pb;
        pb.dst_field = dst_field;
        pb.name      = ggml_get_name(src);
        for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
            pb.expected_ne[i] = (uint32_t) src->ne[i];
        }
        pending.push_back(std::move(pb));
    };

    for (size_t il = 0; il < n_layer; il++) {
        llama_layer & layer = model.layers[il];
        consider(layer.ffn_gate_exps, &layer.ffn_gate_exps_rpc);
        consider(layer.ffn_up_exps,   &layer.ffn_up_exps_rpc);
        consider(layer.ffn_down_exps, &layer.ffn_down_exps_rpc);
    }

    if (pending.empty()) {
        LLAMA_LOG_WARN("hydra: COMBINED rebind — tensor_pattern '%s' matched no expert tensors on %s\n",
                name_pattern.c_str(), peer_endpoint);
        return 0;
    }

    size_t n_bound = 0;
    uint32_t bound_epoch = 0;
    for (auto & p : pending) {
        ggml_tensor * remote = bind_remote_fn(peer_endpoint, /*device=*/0, meta_ctx.get(),
                p.name.c_str(), p.expected_ne, &bound_epoch);
        *p.dst_field = remote;
        if (remote) {
            n_bound++;
        } else {
            LLAMA_LOG_WARN("hydra: COMBINED rebind — peer %s has no resident tensor named '%s' (zero-copy bind failed)\n",
                    peer_endpoint, p.name.c_str());
        }
    }

    if (n_bound == 0) {
        LLAMA_LOG_WARN("hydra: COMBINED rebind — zero-copy bind failed for all %zu matched tensor(s) on peer %s, staying solo-only\n",
                pending.size(), peer_endpoint);
        return 0;
    }

    // Hydra #353 / llama.cpp#12: register the peer device as a scheduler
    // backend so the routed-expert ops are schedulable once
    // SET_EXPERT_MODE("combined") is active. Only register once — the
    // scheduler's hydra_add_combined_rpc_backend is not idempotent (it
    // appends to the backend list). We check the per-peer map: if a
    // previous binding for this endpoint exists, the backend was added
    // already and we skip.
    bool already_added = s_hydra_combined_bindings.count(peer_endpoint) > 0;
    if (!already_added) {
        if (!ctx->hydra_add_combined_rpc_backend(peer_dev)) {
            LLAMA_LOG_ERROR("hydra: COMBINED rebind — peer %s bound but could not be registered "
                    "with the scheduler; clearing _rpc tensors and staying solo-only\n", peer_endpoint);
            for (auto & p : pending) { *p.dst_field = nullptr; }
            return -1;
        }
    }

    int32_t n_layers_loaded = 0;
    for (size_t il = 0; il < n_layer; il++) {
        const llama_layer & layer = model.layers[il];
        if (layer.ffn_gate_exps_rpc || layer.ffn_up_exps_rpc || layer.ffn_down_exps_rpc) {
            n_layers_loaded++;
        }
    }

    // Track the binding for the next rebind. Replacing the entry drops the
    // previous meta_ctx — the no-leak invariant.
    hydra_combined_peer_binding b;
    b.meta_ctx         = std::move(meta_ctx);
    b.peer_dev         = peer_dev;
    b.last_bound_epoch = bound_epoch;
    b.n_layers_loaded  = n_layers_loaded;
    s_hydra_combined_bindings[peer_endpoint] = std::move(b);

    LLAMA_LOG_INFO("hydra: COMBINED zero-copy bound %zu/%zu expert tensors across %d layers to peer %s (no bytes copied, epoch=%u)\n",
            n_bound, pending.size(), n_layers_loaded, peer_endpoint, bound_epoch);

    return n_layers_loaded;
}

void llama_hydra_set_expert_mode(struct llama_context * ctx, int32_t mode) {
    ctx->hydra_set_expert_mode(mode);
}

int32_t llama_hydra_get_expert_mode(const struct llama_context * ctx) {
    return ctx->get_cparams().hydra_expert_mode;
}

size_t llama_hydra_get_compute_backends(struct llama_context * ctx, ggml_backend_t * out, size_t cap) {
    ggml_backend_sched_t sched = ctx->get_sched();
    size_t count = 0;
    const int n_backends = ggml_backend_sched_get_n_backends(sched);
    for (int i = 0; i < n_backends; i++) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            if (count < cap) {
                out[count] = backend;
            }
            count++;
        }
    }
    return count;
}

// Hydra #348: glue between llama_context's local decode path and the
// embedded ggml-RPC server's per-device compute lock (ggml-rpc.cpp's
// ggml_backend_rpc_server_compute_lock/unlock), resolved dynamically the
// same way llama_hydra_load_combined_experts resolves other ggml_backend_rpc_*
// symbols. Kept disabled (near-zero-cost) unless a process actually starts
// the shared-backend RPC server.
namespace {
std::atomic<bool> g_hydra_shared_backend_mode{false};
using hydra_rpc_lock_fn_t = void (*)(uint32_t);
hydra_rpc_lock_fn_t g_hydra_rpc_lock_fn   = nullptr;
hydra_rpc_lock_fn_t g_hydra_rpc_unlock_fn = nullptr;
} // namespace

void llama_hydra_enable_shared_backend_compute_lock() {
    ggml_backend_load_all();
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        LLAMA_LOG_WARN("hydra: shared-backend compute lock requested but the RPC backend is not available in this build\n");
        return;
    }
    g_hydra_rpc_lock_fn   = (hydra_rpc_lock_fn_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_server_compute_lock");
    g_hydra_rpc_unlock_fn = (hydra_rpc_lock_fn_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_server_compute_unlock");
    if (!g_hydra_rpc_lock_fn || !g_hydra_rpc_unlock_fn) {
        LLAMA_LOG_ERROR("hydra: failed to resolve RPC server compute lock functions\n");
        g_hydra_rpc_lock_fn = g_hydra_rpc_unlock_fn = nullptr;
        return;
    }
    g_hydra_shared_backend_mode.store(true, std::memory_order_release);
}

void llama_hydra_lock_compute(int32_t device) {
    if (g_hydra_shared_backend_mode.load(std::memory_order_acquire)) {
        g_hydra_rpc_lock_fn(static_cast<uint32_t>(device));
    }
}

void llama_hydra_unlock_compute(int32_t device) {
    if (g_hydra_shared_backend_mode.load(std::memory_order_acquire)) {
        g_hydra_rpc_unlock_fn(static_cast<uint32_t>(device));
    }
}

void llama_hydra_force_sync_if_shared(struct llama_context * ctx) {
    if (g_hydra_shared_backend_mode.load(std::memory_order_acquire)) {
        ggml_backend_sched_synchronize(ctx->get_sched());
    }
}

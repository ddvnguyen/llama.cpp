#include "llama-hydra.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <atomic>
#include <cstring>
#include <regex>
#include <string>
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

// Copies tensor data from src (any backend) to dst (any backend) via a bounded
// host staging buffer, so dual-loading multi-GB expert tensors over the RPC
// link never requires holding the whole tensor in host RAM at once.
static void hydra_copy_tensor_chunked(struct ggml_tensor * dst, const struct ggml_tensor * src) {
    constexpr size_t CHUNK = 16ull * 1024 * 1024;
    const size_t      nbytes = ggml_nbytes(src);
    std::vector<uint8_t> buf(std::min(nbytes, CHUNK));
    for (size_t off = 0; off < nbytes; off += CHUNK) {
        const size_t n = std::min(CHUNK, nbytes - off);
        ggml_backend_tensor_get(src, buf.data(), off, n);
        ggml_backend_tensor_set(dst, buf.data(), off, n);
    }
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

    ggml_backend_dev_t        peer_dev  = ggml_backend_reg_dev_get(peer_reg, 0);
    ggml_backend_buffer_type_t peer_buft = ggml_backend_dev_buffer_type(peer_dev);
    if (!peer_buft) {
        LLAMA_LOG_ERROR("hydra: COMBINED — peer %s has no buffer type\n", peer_endpoint);
        return -1;
    }

    llama_model & model = const_cast<llama_model &>(ctx->get_model());

    std::regex re;
    try {
        re = std::regex(tensor_pattern);
    } catch (const std::regex_error & e) {
        LLAMA_LOG_ERROR("hydra: COMBINED — invalid tensor_pattern '%s': %s\n", tensor_pattern, e.what());
        return -1;
    }

    struct pending_copy {
        ggml_tensor *  src;
        ggml_tensor ** dst_field; // address of the layer's ffn_*_exps_rpc member
    };
    std::vector<pending_copy> pending;

    const size_t n_layer = model.layers.size();
    const size_t ctx_mem = n_layer * 3 * ggml_tensor_overhead() + ggml_tensor_overhead() * 4;
    ggml_init_params iparams = { /*.mem_size =*/ ctx_mem, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context_ptr meta_ctx(ggml_init(iparams));
    if (!meta_ctx) {
        LLAMA_LOG_ERROR("hydra: COMBINED — failed to allocate metadata context\n");
        return -1;
    }

    auto consider = [&](ggml_tensor * src, ggml_tensor ** dst_field) {
        if (!src || !std::regex_match(ggml_get_name(src), re)) {
            return;
        }
        ggml_tensor * dup = ggml_dup_tensor(meta_ctx.get(), src);
        ggml_set_name(dup, ggml_get_name(src));
        pending.push_back({src, dst_field});
        *dst_field = dup;
    };

    int32_t n_layers_loaded = 0;
    for (size_t il = 0; il < n_layer; il++) {
        llama_layer & layer = model.layers[il];
        consider(layer.ffn_gate_exps, &layer.ffn_gate_exps_rpc);
        consider(layer.ffn_up_exps,   &layer.ffn_up_exps_rpc);
        consider(layer.ffn_down_exps, &layer.ffn_down_exps_rpc);
        if (layer.ffn_gate_exps_rpc || layer.ffn_up_exps_rpc || layer.ffn_down_exps_rpc) {
            n_layers_loaded++;
        }
    }

    if (pending.empty()) {
        LLAMA_LOG_WARN("hydra: COMBINED — tensor_pattern '%s' matched no expert tensors\n", tensor_pattern);
        return 0;
    }

    // Hydra #348: the peer's VRAM is no longer guaranteed empty — under the
    // always-on dual-role design it may already be running its own resident
    // SOLO model + KV cache. Check headroom before committing to a copy that
    // could otherwise starve the peer's own inference or fail to allocate.
    {
        size_t total_bytes = 0;
        for (auto & p : pending) {
            total_bytes += ggml_nbytes(p.src);
        }

        using get_mem_fn_t = void (*)(const char *, uint32_t, size_t *, size_t *);
        auto get_mem_fn = (get_mem_fn_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_get_device_memory");
        size_t peer_free = 0, peer_total = 0;
        if (get_mem_fn) {
            get_mem_fn(peer_endpoint, /*device=*/0, &peer_free, &peer_total);
        } else {
            LLAMA_LOG_WARN("hydra: COMBINED — failed to resolve ggml_backend_rpc_get_device_memory, "
                    "skipping VRAM headroom check (failing open)\n");
        }

        // Headroom for the peer's own KV cache growth during decode, which
        // isn't reflected in a free-VRAM snapshot taken at head-startup
        // time. Starting heuristic — revisit against real measurements
        // (docs/spike-engine-mode-switch.md) if it's wrong in either
        // direction. peer_free == 0 (query failed or genuinely zero) fails
        // open — does not block — consistent with this function's existing
        // "log + stay solo, never abort" philosophy elsewhere.
        constexpr double kHeadroomFactor = 1.25;
        const size_t required_bytes = static_cast<size_t>(total_bytes * kHeadroomFactor);

        if (peer_free > 0 && required_bytes > peer_free) {
            LLAMA_LOG_WARN("hydra: COMBINED — peer %s has %zu MB free, dual-load needs ~%zu MB "
                    "(with %.0f%% headroom) for %zu tensor(s) — skipping COMBINED, staying solo-only\n",
                    peer_endpoint, peer_free / (1024*1024), required_bytes / (1024*1024),
                    (kHeadroomFactor - 1.0) * 100.0, pending.size());
            for (auto & p : pending) { *p.dst_field = nullptr; }
            return 0;
        }
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(meta_ctx.get(), peer_buft);
    if (!buf) {
        LLAMA_LOG_ERROR("hydra: COMBINED — failed to allocate %zu expert tensors on peer %s\n",
                pending.size(), peer_endpoint);
        for (auto & p : pending) { *p.dst_field = nullptr; }
        return -1;
    }

    for (auto & p : pending) {
        hydra_copy_tensor_chunked(*p.dst_field, p.src);
    }

    // Keep the metadata context + backing buffer alive for the lifetime of the
    // process — this is one-time startup setup, never torn down before exit.
    static std::vector<ggml_context_ptr>        s_ctxs;
    static std::vector<ggml_backend_buffer_ptr>  s_bufs;
    s_ctxs.push_back(std::move(meta_ctx));
    s_bufs.emplace_back(buf);

    // Hydra #353 / llama.cpp#12: the expert tensors above live on the peer's
    // buffer, but the context's scheduler was built at load time without the
    // peer. Register the peer device as a scheduler backend now so the routed-
    // expert ops are schedulable once SET_EXPERT_MODE("combined") is active;
    // otherwise the first COMBINED PREFILL asserts in ggml-backend.cpp:898.
    if (!ctx->hydra_add_combined_rpc_backend(peer_dev)) {
        LLAMA_LOG_ERROR("hydra: COMBINED — peer %s dual-loaded but could not be registered "
                "with the scheduler; clearing _rpc tensors and staying solo-only\n", peer_endpoint);
        for (auto & p : pending) { *p.dst_field = nullptr; }
        return -1;
    }

    LLAMA_LOG_INFO("hydra: COMBINED dual-loaded %zu expert tensors across %d layers onto peer %s\n",
            pending.size(), n_layers_loaded, peer_endpoint);

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

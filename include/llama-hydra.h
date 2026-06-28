#pragma once

#include "llama.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Hydra M2: stream the KV state of a single sequence directly to an open POSIX file descriptor
// (typically a TCP socket). No intermediate 800 MB buffer is allocated — GPU tensors are
// copied in chunks (size set by llama_hydra_set_state_chunk_size, default 2 MiB) and
// written to fd immediately.
// Returns bytes written (== llama_state_seq_get_size for the same seq_id), or 0 on error.
// Not supported on Windows (returns 0 with a log warning).
LLAMA_API size_t llama_state_seq_get_data_to_fd(
        struct llama_context * ctx,
                llama_seq_id   seq_id,
                         int   fd);

// Hydra #334: set/get the chunk size (bytes) used by llama_state_seq_get_data_to_fd's
// GPU->host copy + socket send loop. Settable at runtime via CONFIGURE (0x40)
// "state_chunk_size" so Hydra can tune it without a rebuild. Clamped internally
// to [64 KiB, 64 MiB]; defaults to 2 MiB.
LLAMA_API void   llama_hydra_set_state_chunk_size(struct llama_context * ctx, size_t bytes);
LLAMA_API size_t llama_hydra_get_state_chunk_size(const struct llama_context * ctx);

// Pure clamp helper used by llama_hydra_set_state_chunk_size — exposed so it can
// be unit-tested without spinning up a llama_context (see tests/test-hydra-state-chunk-size.cpp).
LLAMA_API size_t llama_hydra_clamp_state_chunk_size(size_t bytes);

// Hydra: try to connect to an RPC engine peer. Returns true if reachable.
// Used at startup for graceful degradation — if the peer is down, the engine
// falls back to SOLO mode (all tensors on local GPU).
LLAMA_API bool llama_hydra_peer_reachable(const char * host_port);

// Hydra #287/#260 — COMBINED expert-split mode.
//
// Dual-resident routed-expert tensors: a "head" engine keeps its normal local
// copy of every ffn_*_exps tensor (so SOLO always works) and additionally loads
// a second copy of the tensors matching `tensor_pattern` (an ERE regex matched
// against each tensor's name, e.g. "blk\\.(2[0-9]|3[0-9])\\.ffn_.*_exps\\.weight")
// onto a peer engine's embedded ggml-RPC backend at `peer_endpoint` ("host:port").
// This is a one-time network copy paid once at startup, not per-request — the
// "no weight transfer" decision applies to steady-state inference, which never
// re-sends weights once dual-residency is established here.
//
// Call once after the model is loaded and before serving any requests. Returns
// the number of layers that got a dual-resident copy, or -1 if the peer is
// unreachable (caller should treat COMBINED as unavailable and stay SOLO-only).
LLAMA_API int32_t llama_hydra_load_combined_experts(
        struct llama_context * ctx,
                   const char * peer_endpoint,
                   const char * tensor_pattern);

// #368: re-establish the COMBINED expert-tensor binding on demand. The
// SET_EXPERT_MODE("combined") handler in server-context.cpp calls this
// instead of relying on the one-shot llama_hydra_load_combined_experts
// startup binding — that fixes the #357 startup race (head binds on demand
// when the peer is reachable, every time) and makes the binding re-callable
// (so a future model swap / re-register on the peer side can be reflected
// by re-binding). The previous binding's metadata context is freed before
// the new one is allocated (no synthetic-buffer leak across N rebinds).
// Returns the number of layers that got a binding, 0 if no tensors matched
// the pattern, or -1 on hard failure. Fail-open: a peer drop, ne-guard
// mismatch, or RPC error degrades to "stay solo" — no abort.
LLAMA_API int32_t llama_hydra_rebind_combined_experts(
        struct llama_context * ctx,
                   const char * peer_endpoint,
        ggml_backend_dev_t      peer_dev,
                   const char * tensor_pattern);

// Set/get the per-context expert placement mode for subsequent decode/prefill
// calls. 0 = SOLO (local GPU only). 1 = COMBINED (use the dual-resident peer
// copies loaded by llama_hydra_load_combined_experts — caller must have called
// it successfully first, otherwise COMBINED silently behaves like SOLO since
// the _rpc tensor pointers are null). The mode is part of the graph-reuse key,
// so a mode change forces a graph rebuild.
LLAMA_API void    llama_hydra_set_expert_mode(struct llama_context * ctx, int32_t mode);
LLAMA_API int32_t llama_hydra_get_expert_mode(const struct llama_context * ctx);

// Hydra #348: write up to `cap` non-CPU backend instances ctx's scheduler
// already built for local inference into `out` (the SAME instances local
// decode dispatches to, not independent ones) and return the actual count.
// If the true count exceeds `cap`, only the first `cap` are written but the
// full count is still returned, so the caller can retry with a bigger
// buffer. Exposed here (rather than called directly from llama-engine.cpp)
// because llama_context is only forward-declared in llama.h — this header's
// .cpp can see the full definition (it already includes llama-context.h).
LLAMA_API size_t llama_hydra_get_compute_backends(struct llama_context * ctx, ggml_backend_t * out, size_t cap);

// Hydra #348: call once at startup, after exposing this engine's backend(s)
// over the embedded ggml-RPC server (ggml_backend_rpc_start_server_with_backends),
// so that subsequent local llama_decode/prefill calls serialize their GPU
// dispatch against the same per-device lock the RPC server holds while
// computing an inbound graph (rpc_server::graph_compute/graph_recompute in
// ggml-rpc.cpp) — the two share the same backend instance per physical
// device rather than each owning an independent one. If this is never
// called (the common case for an engine that never opts into
// --ggml-rpc-port), llama_hydra_lock_compute/unlock_compute/
// force_sync_if_shared are all no-ops and the decode hot path pays no cost.
LLAMA_API void llama_hydra_enable_shared_backend_compute_lock(void);

// Cheap no-ops unless llama_hydra_enable_shared_backend_compute_lock() was
// called for this process. `device` is the backend index shared by both
// llama_context's sched and the RPC server's backend list (today always 0 —
// one GPU per node in this fork).
LLAMA_API void llama_hydra_lock_compute(int32_t device);
LLAMA_API void llama_hydra_unlock_compute(int32_t device);

// Hydra #348: if shared-backend mode is active, force the context's
// scheduler to finish outstanding async compute before the caller unlocks
// (see llama_hydra_lock_compute) — otherwise the RPC server's synchronous
// graph_compute could start on the same backend while the local dispatch is
// still in flight. No-op otherwise, preserving the existing
// pipeline-parallel overlap optimization (llama-context.cpp's
// process_ubatch) for engines that never expose an RPC backend.
LLAMA_API void llama_hydra_force_sync_if_shared(struct llama_context * ctx);

// Hydra: zero-copy COMBINED expert tensors (llama.cpp#20, follow-up to
// #287/#260/#353). Registers every tensor of ctx's already-loaded model with
// the embedded ggml-RPC server (ggml_backend_rpc_register_local_tensor) so a
// COMBINED head can later bind directly to this engine's own resident
// weights by name instead of dual-loading a copy. Call once at startup,
// after start_shared_backend_rpc_server (i.e. only meaningful when
// --ggml-rpc-port is configured) and after the model is loaded. Cheap,
// unconditional bookkeeping — does not require knowing in advance which
// tensors a future peer will ask for.
LLAMA_API void llama_hydra_register_local_tensors_for_rpc(struct llama_context * ctx);

#ifdef __cplusplus
}
#endif

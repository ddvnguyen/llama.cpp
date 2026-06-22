#pragma once

#include "llama.h"

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

// Set/get the per-context expert placement mode for subsequent decode/prefill
// calls. 0 = SOLO (local GPU only). 1 = COMBINED (use the dual-resident peer
// copies loaded by llama_hydra_load_combined_experts — caller must have called
// it successfully first, otherwise COMBINED silently behaves like SOLO since
// the _rpc tensor pointers are null). The mode is part of the graph-reuse key,
// so a mode change forces a graph rebuild.
LLAMA_API void    llama_hydra_set_expert_mode(struct llama_context * ctx, int32_t mode);
LLAMA_API int32_t llama_hydra_get_expert_mode(const struct llama_context * ctx);

#ifdef __cplusplus
}
#endif

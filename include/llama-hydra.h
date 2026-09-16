#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Hydra pick 1b (option B): buffered fallback using the public state API
// (get_size + get_data into a temp buffer, then send loop). The zero-copy
// to_fd variant (llama_context::state_seq_get_data_to_fd, 69d49e0a4) was
// deliberately not pulled in to avoid its async machinery; peak usage is one
// full state buffer, revisit if profiled hot.
// Returns bytes written (== llama_state_seq_get_size for the same seq_id), or 0 on error.
// Not supported on Windows (returns 0).
LLAMA_API size_t llama_state_seq_get_data_to_fd(
        struct llama_context * ctx,
                llama_seq_id   seq_id,
                         int   fd);

// Hydra: try to connect to an RPC engine peer. Returns true if reachable.
// Used at startup for graceful degradation — if the peer is down, the engine
// falls back to SOLO mode (all tensors on local GPU).
LLAMA_API bool llama_hydra_peer_reachable(const char * host_port);

#ifdef __cplusplus
}
#endif

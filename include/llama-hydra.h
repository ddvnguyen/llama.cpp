#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Hydra M2: stream the KV state of a single sequence directly to an open POSIX file descriptor
// (typically a TCP socket). No intermediate 800 MB buffer is allocated — GPU tensors are
// copied in 256 KB chunks and written to fd immediately.
// Returns bytes written (== llama_state_seq_get_size for the same seq_id), or 0 on error.
// Not supported on Windows (returns 0 with a log warning).
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

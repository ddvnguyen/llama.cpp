#pragma once

#include "common.h" // common_context_seq_rm_type

// Hydra #316 / #8: decide whether the server should create a *native context
// checkpoint* for the current slot before processing a decode batch.
//
// This predicate is extracted from server_context::update_slots() into a pure,
// dependency-light function for one reason: the recurrent/hybrid term
// (`is_recurrent_or_hybrid`) is load-bearing for hybrid KV-cache restore, and a
// previous review (ddvnguyen/llama.cpp#8) proposed reverting it. Dropping that
// term makes a standalone (non-RPC) hybrid server never create checkpoints, so
// the checkpoint search path — which special-cases recurrent/hybrid models per
// the ik_llama.cpp#1762 port — finds nothing to restore and forces a full
// re-prefill on every turn (0% cache hit on identical repeats). The
// accompanying unit test (tests/test-hydra-checkpoint-policy.cpp) pins this
// truth table so the term can't be silently removed again.
//
// Checkpoints are created when checkpointing is enabled, for a completion task,
// and when at least one of these holds:
//   - the model can only seq_rm full sequences (FULL)
//   - the model can seq_rm partial sequences but only up to n_rs_seq (RS)
//   - the model uses SWA and we are not using --swa-full (n_swa > 0)
//   - the model is recurrent/hybrid (needs checkpoints on its own merits)
//   - the binary RPC port is enabled (this node participates in cross-node KV
//     migration; the restore target may not support rollback)
static inline bool server_should_create_checkpoint(
        int                        n_ctx_checkpoints,
        bool                       is_completion_task,
        common_context_seq_rm_type seq_rm_type,
        int                        n_swa,
        bool                       is_recurrent_or_hybrid,
        int                        rpc_port) {
    if (n_ctx_checkpoints <= 0) {
        return false;
    }
    if (!is_completion_task) {
        return false;
    }
    return seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
           seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS   ||
           n_swa > 0                                      ||
           is_recurrent_or_hybrid                         ||
           rpc_port > 0;
}

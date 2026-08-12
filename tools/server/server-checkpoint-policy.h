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

// Hydra #641: decide whether update_slots() should search for a context
// checkpoint to rewind to before decoding the current task.
//
// A *pure extension* — the whole cached sequence is a strict prefix of the new
// prompt, so there is >= 1 brand-new token to decode — must NOT enter the
// checkpoint search. Before upstream PR #23280 (ccee42642, "guarantee there is
// at least 1 token to decode") the search was gated on `n_past <
// slot.prompt.n_tokens()`, so a pure extension never reached it; #23280 widened
// `<` to `<=` solely to keep the exact-match case (0 tokens left to decode,
// where the rewind to the final checkpoint supplies the required logits). This
// predicate restores the pre-#23280 extension semantics while preserving
// #23280's exact-match fix.
//
// The third term is the physical invariant: the restored memory must really end
// exactly at the resume point (pos_next - 1) — the same predicate that makes the
// seq_rm at the end of update_slots a no-op (a recurrent memory cell only
// rewinds when p0 <= cell.pos, see llama-memory-recurrent.cpp). Any
// disagreement between bookkeeping and actual memory (e.g. a dishonest blob
// header) fails safe: the search runs and behavior is unchanged. Model-agnostic
// by design — no is_recurrent/hybrid special-casing.
static inline bool server_should_rewind_to_checkpoint(
        llama_pos n_past,
        llama_pos n_prompt_tokens,
        llama_pos n_task_tokens,
        llama_pos pos_next,
        llama_pos pos_max_mem) {
    // no rewind needed iff: the whole cache is a prefix of the new prompt,
    // there is >= 1 NEW token to decode ([TAG_PROMPT_LOGITS] is satisfied
    // without re-decoding a cached one), and memory really ends at the
    // resume point.
    const bool no_rewind_needed =
           n_past == n_prompt_tokens   // whole cache is a prefix of the new prompt
        && n_past <  n_task_tokens     // >= 1 NEW token to decode -> [TAG_PROMPT_LOGITS]
                                       //   satisfied without re-decoding a cached one
        && pos_max_mem == pos_next - 1; // memory really ends exactly at the resume point

    // search should run
    return !no_rewind_needed;
}

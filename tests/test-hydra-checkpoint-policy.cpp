// hydra#316/#8: pins server_should_create_checkpoint()'s truth table.
//
// The recurrent/hybrid term is load-bearing for hybrid KV-cache restore: a
// previous review (ddvnguyen/llama.cpp#8) proposed reverting it, which would
// make a standalone hybrid server never create checkpoints and force a full
// re-prefill on every turn. This test fails if that term is removed.
#include "../tools/server/server-checkpoint-policy.h"

#include <cstdio>

static int g_failures = 0;

static void expect(const char * what, bool actual, bool expected) {
    if (actual != expected) {
        fprintf(stderr, "FAIL: %s — expected %s, got %s\n",
                what, expected ? "true" : "false", actual ? "true" : "false");
        g_failures++;
    }
}

// Default "happy" args for a hybrid model on a standalone (no-RPC) server: this
// is exactly the configuration #316 fixed and #8 threatened to regress.
static bool call(int n_ctx_checkpoints,
                 bool is_completion,
                 common_context_seq_rm_type seq_rm,
                 int n_swa,
                 bool is_rec,
                 int rpc_port) {
    return server_should_create_checkpoint(
            n_ctx_checkpoints, is_completion, seq_rm, n_swa, is_rec, rpc_port);
}

int main() {
    using S = common_context_seq_rm_type;
    const S NO   = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    const S PART = COMMON_CONTEXT_SEQ_RM_TYPE_PART;
    const S FULL = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const S RS   = COMMON_CONTEXT_SEQ_RM_TYPE_RS;

    // --- the #316/#8 invariant: hybrid/recurrent always checkpoints ---
    // A standalone hybrid server: PART seq_rm, no SWA, no RPC port. Without the
    // is_rec term this would be false (the #8 regression).
    expect("hybrid standalone (PART, no swa, no rpc) -> checkpoint",
        call(32, true, PART, 0, /*is_rec=*/true, /*rpc_port=*/0), true);
    expect("recurrent standalone, seq_rm=NO -> checkpoint",
        call(32, true, NO, 0, true, 0), true);

    // --- non-recurrent baselines (unchanged by #316) ---
    expect("non-rec, PART, no swa, no rpc -> no checkpoint",
        call(32, true, PART, 0, /*is_rec=*/false, 0), false);
    expect("non-rec, seq_rm FULL -> checkpoint",
        call(32, true, FULL, 0, false, 0), true);
    expect("non-rec, seq_rm RS -> checkpoint",
        call(32, true, RS, 0, false, 0), true);
    expect("non-rec, SWA (n_swa>0) -> checkpoint",
        call(32, true, PART, 1, false, 0), true);
    expect("non-rec, rpc_port>0 -> checkpoint",
        call(32, true, PART, 0, false, 9500), true);

    // --- hard gates: must be false regardless of the OR-terms ---
    expect("checkpoints disabled (n_ctx_checkpoints=0) -> no checkpoint",
        call(0, true, FULL, 1, true, 9500), false);
    expect("negative n_ctx_checkpoints -> no checkpoint",
        call(-1, true, FULL, 1, true, 9500), false);
    expect("non-completion task -> no checkpoint even for hybrid",
        call(32, /*is_completion=*/false, FULL, 1, true, 9500), false);

    // --- #641: server_should_rewind_to_checkpoint truth table ---
    // The rewind decision lives in update_slots() and is load-bearing for
    // post-decode KV restore (#641): a stale PREFILL-end checkpoint matching on
    // the next continuation re-prefills already-cached tokens. Pinned here so
    // the predicate can't be silently reverted (same rationale as #316/#8).
    //
    // Pure extension: the whole cached sequence is a strict prefix of the new
    // prompt (n_past == n_prompt_tokens < n_task_tokens) and memory ends
    // exactly at the resume point (pos_max_mem == pos_next - 1) -> no rewind,
    // the checkpoint search must NOT run. This restores pre-#23280 extension
    // semantics (upstream PR #23280 / ccee42642 widened the search gate to also
    // cover the exact-match case, inadvertently swallowing pure extensions).
    expect("pure extension -> no rewind (search skipped)",
        server_should_rewind_to_checkpoint(
            /*n_past=*/100, /*n_prompt_tokens=*/100, /*n_task_tokens=*/120,
            /*pos_next=*/100, /*pos_max_mem=*/99), false);
    // Exact match: n_past == n_prompt_tokens == n_task_tokens — 0 tokens left
    // to decode; the rewind to the final checkpoint supplies the logits
    // [TAG_PROMPT_LOGITS]. #23280's exact-match fix must be preserved.
    expect("exact match -> rewind (search runs)",
        server_should_rewind_to_checkpoint(100, 100, 100, 100, 99), true);
    // Divergence: the cache is NOT a full prefix (n_past < n_prompt_tokens) —
    // the mismatch path must still search for a checkpoint to rewind to.
    expect("divergence -> rewind (search runs)",
        server_should_rewind_to_checkpoint(80, 100, 120, 80, 79), true);
    // Memory behind the resume point (dishonest header, pos_max_mem < pos_next-1):
    // bookkeeping and memory disagree -> fail safe, the search runs unchanged.
    expect("memory behind resume point -> rewind (search runs)",
        server_should_rewind_to_checkpoint(100, 100, 120, 100, 50), true);
    // No memory at all (pos_max_mem == -1): nothing to extend -> fail safe,
    // the search runs unchanged.
    expect("no memory (pos_max_mem == -1) -> rewind (search runs)",
        server_should_rewind_to_checkpoint(100, 100, 120, 100, -1), true);

    if (g_failures == 0) {
        printf("OK: server_should_create_checkpoint truth table holds\n");
        return 0;
    }
    fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
}

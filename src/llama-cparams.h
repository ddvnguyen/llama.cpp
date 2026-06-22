#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    // Hydra #287/#260: per-request expert placement mode.
    // 0 = SOLO (local GPU only), 1 = COMBINED (expert matmuls redirected to a peer
    // engine's ggml-RPC backend via the placement hook in llama_context::process_ubatch).
    int hydra_expert_mode = 0;

    // Hydra #334: chunk size (bytes) for STATE_GET socket streaming
    // (llama_io_write_socket in llama-context.cpp). Set via CONFIGURE (0x40)
    // "state_chunk_size" so Hydra can tune it without a rebuild; defaults to
    // 2 MiB, which already amortizes the per-chunk cudaMemcpy+send overhead
    // far better than the original 256 KiB.
    size_t hydra_state_chunk_size = 2 * 1024 * 1024;
};

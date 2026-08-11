
#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-checkpoint-policy.h"
#include "server-hydra-extension.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-rpc.h"

#include "build-info.h"
#include "common.h"
#include "arg.h"
#include "fit.h"
#include "llama.h"
#include "../src/llama-context.h"
#include "llama-hydra.h"
#include "ggml-rpc.h"
#include "log.h"
#include "../src/llama-memory-hybrid.h"
#include "preset.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include "../llama-engine/hydra_rpc/hydra_rpc.h"

// xxhash for DECODE segment hash verification (xxh3-64)
#define XXH_STATIC_LINKING_ONLY
#define XXH_IMPLEMENTATION
#include "../../vendor/xxhash/xxhash.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cinttypes>
#include <exception>
#include <memory>
#include <filesystem>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>

#if !defined(_WIN32)
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <unistd.h>
#  include <cerrno>
#  include <cstring>
#endif

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_SECONDS = 1;

// Pending RPC servers staged by hydra_apply_t3_mutators() and consumed
// by apply_t3_rebuild() before load_model(). File-scope static so both
// functions (methods of server_context_impl) can access it.
static std::vector<std::string> g_pending_rpc_servers;

// hydra#470: staged reload config — the context-level (T2) + generic (T4)
// keys from the last hydra_apply_config() call, as a JSON dump string.
// Consumed by the model-reload path (apply_t3_rebuild()) and the
// context-reload path (apply_t2_rebuild()); both apply the staged keys to
// common_params BEFORE the rebuild/load so mixed T2+T4 payloads never
// strand keys. Overwritten unconditionally on every apply (absolute-state
// semantics: a config without T2/T4 keys clears the staging). File-scope
// static because first-load staging happens before any context exists
// (ctx_tgt == nullptr).
static std::string g_pending_reload_config;

// hydra#470: the staged reload config that was last ACTUALLY applied
// (recorded by apply_t3_rebuild() after a successful model load and by
// apply_t2_rebuild() after a successful context rebuild). The T3 early-exit
// compares the newly staged dump against this to skip the expensive
// unload+reload when nothing changed.
static std::string g_last_reload_config_applied;

// Forward-declared so the background thread lambda in process_single_task can use it
// before the full RPC helper definitions appear later in this file.
#if !defined(_WIN32)
static bool hydra_send_all(int fd, const void * buf, size_t n);
static bool hydra_recv_all(int fd, void * buf, size_t n); // DECODE_APPLY M2 logits tail
#endif

static uint32_t server_n_outputs_max(const common_params & params) {
    const uint32_t n_batch  = params.n_batch;

    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return n_batch;
    }

    const uint32_t n_outputs_per_seq = 1 + common_speculative_n_max(&params.speculative);

    const uint64_t n_outputs = (uint64_t) params.n_parallel * n_outputs_per_seq;

    return std::max<uint32_t>(1, std::min<uint64_t>(n_batch, n_outputs));
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    // multimodal
    mtmd_context * mctx = nullptr;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    // Hydra M1: true when a background thread is serializing KV state for STATE_GET.
    // Guards concurrent RPC ops (STATE_PUT, another STATE_GET) on this slot.
    // Serialization runs off the inference thread so decode can continue on other slots.
    // Use a raw pointer to an atomic so server_slot remains movable (needed by std::vector).
    std::shared_ptr<std::atomic<bool>> hydra_transferring{std::make_shared<std::atomic<bool>>(false)};
    bool just_restored  = false; // set on STATE_PUT; one-shot, gates restored-slot KV reuse

    // D4: per-slot restored logits buffer — avoids the shared-context race where
    // another slot's decode clobbers restored logits between STATE_PUT and first
    // sample. Populated during KV restore; consumed on first sample; cleared on
    // seq_rm and after consumption.
    std::vector<float> restored_logits;
    bool logits_valid = false;

    // DECODE slot reservation: when a sync handler reserves this slot for an
    // async DECODE_APPLY, this holds the decode_request_id.  Other tasks must
    // not be assigned to this slot until the reservation is released.
    int32_t reserved_for_decode_id = -1;

    // Hydra n_common observability: set during update_slots() prompt
    // processing decision, read by background consumer for hydra_metrics.
    int32_t n_common           = 0; // length of common prefix with new tokens
    int32_t n_prompt_processed = 0; // tokens actually sent to batch (0 = zero-prompt decode)
    bool    logits_reused      = false; // true when sampled directly from restored logits

    server_prompt prompt;

    void prompt_save(server_prompt_cache & prompt_cache) const {
        GGML_ASSERT(prompt.data.size() == 0);

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_WRN(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear(bool allow_processing) {
        if (!allow_processing) {
            GGML_ASSERT(!is_processing());
        }

        SLT_INF(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        common_context_seq_rm(ctx_tgt, id, -1, -1);
        if (ctx_dft) {
            common_context_seq_rm(ctx_dft, id, -1, -1);
        }

        prompt.tokens.clear();

        // D4: seq_rm invalidates any restored logits — clear them
        restored_logits.clear();
        logits_valid = false;
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // stats
    size_t n_sent_text = 0; // number of sent text character

    int64_t t_print_last = 0;
    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing = 0.0; // ms
    double t_token_generation = 0.0;  // ms

    std::function<void(int /* id_slot */)> callback_on_release;

    // Speculative decoding stats
    int32_t n_draft_total = 0;      // Total draft tokens generated
    int32_t n_draft_accepted = 0;   // Draft tokens actually accepted

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        n_prompt_tokens_cache = 0;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        // clear speculative decoding stats
        n_draft_total = 0;
        n_draft_accepted = 0;

        task_prev = std::move(task);
        task.reset();

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // D4: clear per-slot restored logits on slot reset
        restored_logits.clear();
        logits_valid = false;
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd() || (spec && common_speculative_need_embd(spec));
    }

    bool need_embd_nextn() const {
        GGML_ASSERT(task);
        return spec && common_speculative_need_embd_nextn(spec);
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type && are_lora_equal(lora, other_slot.lora);
    }

    bool has_budget(const common_params & global_params) {
        GGML_ASSERT(task);

        if (task->params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (task->params.n_predict != -1) {
            n_remaining = task->params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    void update_batch(llama_batch & batch) {
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.n_tokens;

            common_batch_add(batch, sampled, prompt.tokens.pos_next(), { this->id }, true);

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.n_tokens);
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.n_tokens + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            common_batch_add(batch, sampled, pos0++, { this->id }, true);
            for (auto token : spec_draft) {
                common_batch_add(batch, token, pos0++, { this->id }, true);
            }
        }

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used        =  ggml_time_us();
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear(false);
            }

            // Release decode reservation if any
            reserved_for_decode_id = -1;

            reset();

            callback_on_release(id);
        }
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.cache_n = n_prompt_tokens_cache;

        timings.prompt_n            = n_prompt_tokens_processed;
        timings.prompt_ms           = t_prompt_processing;
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second   = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        timings.predicted_n            = n_decoded;
        timings.predicted_ms           = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second   = 1e3 / t_token_generation * n_decoded;

        // Add speculative metrics
        if (n_draft_total > 0) {
            timings.draft_n          = n_draft_total;
            timings.draft_n_accepted = n_draft_accepted;
        }

        return timings;
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (n_decoded < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        t_print_last = t_now;

        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this, "n_decoded = %6d, tg = %6.2f t/s\n", n_decoded, n_gen_second);
    }

    void print_timings_pp() const {
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;
        const double f_progress = (float) prompt.n_tokens() / task->n_tokens();

        if (t_prompt_processing < 3000.0) {
            return;
        }

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                n_prompt_tokens_processed, f_progress, t_prompt_processing / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_token_generation, n_decoded, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        if (n_draft_total > 0) {
            const float draft_ratio = (float) n_draft_accepted / n_draft_total;
            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated)\n",
                    draft_ratio, n_draft_accepted, n_draft_total);
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;
        {
            int raw = ptask ? (int32_t)(n_prompt_tokens_cache + n_decoded) : 0;
            if (n_prompt_tokens_cache == 0 && prompt.tokens.size() > 0) {
                raw = (int32_t)prompt.tokens.size();
            }
            res["n_past"] = raw;
        }

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = n_prompt_tokens_processed;
            res["n_prompt_tokens_cache"]     = n_prompt_tokens_cache;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = {
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining},
                    {"n_decoded",      n_decoded},
                }
            };

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        common_context_seq_rm(ctx_tgt, other.id,     -1, -1);
        common_context_seq_cp(ctx_tgt, id, other.id, -1, -1);

        if (ctx_dft) {
            common_context_seq_rm(ctx_dft, other.id,     -1, -1);
            common_context_seq_cp(ctx_dft, id, other.id, -1, -1);
        }

        other.n_decoded   = n_decoded;
        other.n_remaining = n_remaining;
        other.i_batch     = i_batch;

        other.t_start_process_prompt    = t_start_process_prompt;
        other.t_prompt_processing       = t_prompt_processing;
        other.n_prompt_tokens_cache     = n_prompt_tokens_cache;
        other.n_prompt_tokens_processed = n_prompt_tokens_processed;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};



//
// server_metrics
//

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += slot.t_prompt_processing;
        t_prompt_processing_total       += slot.t_prompt_processing;

        n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
            n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};


//
// server_context_impl (private implementation)
//

// ── Hydra #406: tiered CONFIGURE (T1/T2/T3) helpers ──────────────────────
//
// The T1/T2/T3 classification is defined in specs/rpc-protocol.md (PR #406).
//   T1 (apply immediately, no rebuild): sampling.*, n_predict, n_keep,
//     seed, antiprompt, state_chunk_size
//   T2 (defer to slot-free moment, context reload): n_ctx, cache_type_k,
//     cache_type_v, rope_*
//   T3 (defer to slot-free moment, model reload): n_gpu_layers, n_cpu_moe,
//     override_tensor, split_mode, tensor_split, model.path
//   T4 (hydra#470, defer to slot-free moment, generic-arg pass-through):
//     EVERY other key. The key is snake_case→kebab-case mapped onto
//     llama.cpp's own CLI arg table and applied via the arg's canonical
//     handler (see hydra_generic_key_status / hydra_apply_generic_key).
//     Keys with no arg-table entry are reported as unrecognized_keys and
//     keys that cannot change at reload as rejected_keys — never silent.
//
// Returns 1/2/3/4; there is no "unknown" tier anymore (any unlisted key
// is a T4 generic key).
int hydra_classify_config_key(const std::string & key) {
    // T1: sampling nested keys
    if (key == "sampling.temp"           ||
        key == "sampling.top_p"          ||
        key == "sampling.top_k"          ||
        key == "sampling.min_p"          ||
        key == "sampling.penalty_repeat" ||
        key == "sampling.seed") {
        return 1;
    }
    // T1: top-level fields
    if (key == "n_predict" ||
        key == "n_keep"    ||
        key == "seed"      ||
        key == "antiprompt" ||
        key == "state_chunk_size") {
        return 1;
    }
    // T2: context-level (KV cache / RoPE / ctx)
    if (key == "n_ctx"        ||
        key == "cache_type_k" ||
        key == "cache_type_v" ||
        key.rfind("rope_", 0) == 0) {
        return 2;
    }
    // T3: model-level (offload / placement / model)
    if (key == "n_gpu_layers"    ||
        key == "n_cpu_moe"       ||
        key == "override_tensor" ||
        key == "split_mode"      ||
        key == "tensor_split"    ||
        key == "model_path"      ||  // hydra_config: absolute GGUF path
        key == "rpc_servers"     ||  // hydra_config: RPC peer endpoints to register
        key == "model.path"      ||
        key == "model") {        // legacy alias for { "model": { "path": ... } }
        return 3;
    }
    // T4: generic-arg pass-through (hydra#470) — every other key.
    return 4;
}

// Tier number → label for the response payload.
static const char * hydra_tier_label(int tier) {
    switch (tier) {
        case 1: return "T1";
        case 2: return "T2";
        case 3: return "T3";
        case 4: return "T4";  // generic-arg pass-through (hydra#470)
        default: return "T1";  // 0 (no recognized keys) → degenerate T1
    }
}

// ── Hydra #470: T4 generic-arg pass-through ─────────────────────────────
//
// Every CONFIGURE key that is not a T1/T2/T3 special key is mapped
// snake_case → kebab-case onto llama.cpp's OWN CLI arg table (the same
// common_arg table common_params_parse() uses at startup, built via
// common_params_parser_init() with the server example) and applied by
// invoking the arg's canonical handler. This guarantees a valid
// llama.cpp arg ALWAYS lands in common_params — no hand-maintained
// whitelist, no silent drops.
//
// Keys whose arg exists but cannot change at reload (startup-only
// params: batch/threads/devices/parallel/fit) are denied explicitly;
// flag args (handler_void, e.g. --version/--metrics) and two-value args
// (handler_str_str) are denied too. Denied keys are reported as
// rejected_keys; keys with no arg-table entry as unrecognized_keys.
// Both get a loud SRV_WRN — never a silent drop.

// Startup-only / non-reloadable keys (snake_case, as sent by the
// Coordinator). Small and explicit: changing any of these mid-flight
// cannot take effect without a full engine restart.
static const std::set<std::string> hydra_generic_denylist = {
    "n_batch",       // --n-batch        (context batch; fixed at server init)
    "threads",       // --threads        (CPU thread pools sized at startup)
    "threads_batch", // --threads-batch
    "devices",       // --devices        (backend device assignment)
    "main_gpu",      // --main-gpu
    "n_parallel",    // --n-parallel     (slot count fixed at startup)
    "fit",           // --fit            (model-fitting-only family)
    "fit_print",     // --fit-print
    "fit_target",    // --fit-target
    "fit_ctx",       // --fit-ctx
};

// The server's own arg table (common_arg options as filtered for
// LLAMA_EXAMPLE_SERVER — exactly the args the engine accepts on the
// command line). Built once; the returned vector is a static copy so
// the pointers stored in the lookup map stay valid for the process
// lifetime. The scratch common_params is only a carrier for the
// per-example defaults inside common_params_parser_init().
static const std::vector<common_arg> & hydra_arg_options() {
    static const std::vector<common_arg> options = []() {
        common_params scratch;
        common_params_context ctx = common_params_parser_init(scratch, LLAMA_EXAMPLE_SERVER, nullptr);
        return ctx.options;
    }();
    return options;
}

// "--kebab-name" → arg entry (positive and negated names both map to
// the same entry; the handler receives the boolean value directly, so
// --no-* flags are covered by handler_bool with value=false).
static const std::unordered_map<std::string, const common_arg *> & hydra_arg_lookup() {
    static const std::unordered_map<std::string, const common_arg *> table = []() {
        std::unordered_map<std::string, const common_arg *> t;
        for (const auto & opt : hydra_arg_options()) {
            for (const auto & arg : opt.args) {
                t[arg] = &opt;
            }
            for (const auto & arg : opt.args_neg) {
                t[arg] = &opt;
            }
        }
        return t;
    }();
    return table;
}

// snake_case → kebab-case ("spec_draft_n_max" → "spec-draft-n-max").
static std::string hydra_snake_to_kebab(const std::string & key) {
    std::string kebab = key;
    std::replace(kebab.begin(), kebab.end(), '_', '-');
    return kebab;
}

hydra_generic_key_status hydra_classify_generic_key(const std::string & key) {
    if (hydra_generic_denylist.count(key) > 0) {
        return hydra_generic_key_status::DENIED;
    }
    const std::string arg = "--" + hydra_snake_to_kebab(key);
    const auto & table = hydra_arg_lookup();
    const auto it = table.find(arg);
    if (it == table.end()) {
        return hydra_generic_key_status::UNRECOGNIZED;
    }
    const common_arg & opt = *it->second;
    // Flag args (--metrics, --version, ...) and two-value args
    // (--override-kv, ...) have no single-value reload semantics —
    // deny them explicitly instead of invoking an exit()/print side
    // effect or mis-applying a half of a pair.
    if (opt.handler_void || opt.handler_str_str) {
        return hydra_generic_key_status::DENIED;
    }
    return hydra_generic_key_status::APPLIABLE;
}

// Convert a CONFIGURE JSON value to the CLI string form the arg
// handler expects. Returns false when the JSON type cannot be mapped.
static bool hydra_generic_value_to_string(const json & value, std::string & out) {
    if (value.is_string()) {
        out = value.get<std::string>();
        return true;
    }
    if (value.is_boolean()) {
        out = value.get<bool>() ? "on" : "off";
        return true;
    }
    if (value.is_number_integer()) {
        out = std::to_string(value.get<int64_t>());
        return true;
    }
    if (value.is_number_unsigned()) {
        out = std::to_string(value.get<uint64_t>());
        return true;
    }
    if (value.is_number_float()) {
        out = std::to_string(value.get<double>());
        return true;
    }
    if (value.is_array()) {
        // comma-joined list (e.g. --spec-draft-device dev1,dev2)
        for (size_t i = 0; i < value.size(); ++i) {
            if (!value[i].is_string() && !value[i].is_number()) {
                return false;
            }
            if (i > 0) out += ',';
            out += value[i].is_string() ? value[i].get<std::string>() : std::to_string(value[i].get<double>());
        }
        return true;
    }
    if (value.is_object()) {
        // e.g. --chat-template-kwargs '{"enable_thinking":true}'
        out = value.dump();
        return true;
    }
    return false;
}

void hydra_clear_stale_draft_bindings(common_params & params) {
    // #648 review: after a failed MTP draft-context build (or after destroy()
    // freed the contexts), the ctx_tgt/ctx_dft bindings must be null so the
    // common_speculative_init() MTP gate (ctx_dft != nullptr) degrades to
    // non-speculative instead of dereferencing freed memory. On a T3
    // reload-after-prior-MTP-success these pointers were carried over from the
    // old params_base and are dangling by the time the MTP build fails.
    params.speculative.draft.ctx_tgt = nullptr;
    params.speculative.draft.ctx_dft = nullptr;
}

bool hydra_apply_generic_key(common_params & params, const std::string & key, const json & value) {
    if (hydra_classify_generic_key(key) != hydra_generic_key_status::APPLIABLE) {
        return false;
    }
    const std::string kebab = hydra_snake_to_kebab(key);
    const std::string arg   = "--" + kebab;
    const common_arg & opt  = *hydra_arg_lookup().find(arg)->second;

    try {
        if (opt.handler_bool) {
            bool b = false;
            if (value.is_boolean()) {
                b = value.get<bool>();
            } else if (value.is_number()) {
                b = value.get<double>() != 0.0;
            } else if (value.is_string()) {
                const std::string & s = value.get_ref<const std::string &>();
                if (common_arg_utils::is_truthy(s)) {
                    b = true;
                } else if (common_arg_utils::is_falsey(s)) {
                    b = false;
                } else {
                    SRV_WRN("hydra: generic arg '%s' expects on/off, got '%s' — rejected\n",
                            arg.c_str(), s.c_str());
                    return false;
                }
            } else {
                SRV_WRN("hydra: generic arg '%s' expects a boolean, got %s — rejected\n",
                        arg.c_str(), value.type_name());
                return false;
            }
            opt.handler_bool(params, b);
            return true;
        }
        if (opt.handler_int) {
            int v = 0;
            if (value.is_number_integer()) {
                v = value.get<int>();
            } else if (value.is_number_unsigned()) {
                v = (int) value.get<uint64_t>();
            } else if (value.is_number_float()) {
                v = (int) value.get<double>();
            } else if (value.is_string()) {
                const std::string & s = value.get_ref<const std::string &>();
                try {
                    v = std::stoi(s);
                } catch (const std::exception &) {
                    SRV_WRN("hydra: generic arg '%s' expects an integer, got '%s' — rejected\n",
                            arg.c_str(), s.c_str());
                    return false;
                }
            } else {
                SRV_WRN("hydra: generic arg '%s' expects an integer, got %s — rejected\n",
                        arg.c_str(), value.type_name());
                return false;
            }
            opt.handler_int(params, v);
            return true;
        }
        if (opt.handler_string) {
            std::string s;
            if (!hydra_generic_value_to_string(value, s)) {
                SRV_WRN("hydra: generic arg '%s' has a value of unsupported JSON type %s — rejected\n",
                        arg.c_str(), value.type_name());
                return false;
            }
            // --spec-type is additive on the CLI (repeatable); a
            // CONFIGURE is absolute state, so replace the list instead
            // of appending to whatever the engine was started with.
            // Snapshot first: if the value is invalid the handler throws
            // and we restore the previous types instead of wiping them.
            const std::vector<enum common_speculative_type> old_spec_types = params.speculative.types;
            if (kebab == "spec-type") {
                params.speculative.types.clear();
            }
            try {
                opt.handler_string(params, s);
            } catch (const std::exception & e) {
                if (kebab == "spec-type") {
                    params.speculative.types = old_spec_types;
                }
                SRV_WRN("hydra: generic arg '%s' rejected: %s\n", arg.c_str(), e.what());
                return false;
            }
            return true;
        }
        SRV_WRN("hydra: generic arg '%s' has no usable handler — rejected\n", arg.c_str());
        return false;
    } catch (const std::exception & e) {
        SRV_WRN("hydra: generic arg '%s' rejected: %s\n", arg.c_str(), e.what());
        return false;
    }
}

struct server_context_impl {
    friend struct server_context;
    friend struct server_routes;
    // epic #610 WS1: the concrete Hydra extension (defined in
    // hydra-server-context.cpp, #include'd at the bottom of this TU) needs
    // access to the same internals the inline Hydra code uses.
    friend struct hydra_engine_extension;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    // Hydra #348: independent capability flags (replaces the old single
    // hydra_role string — a node can be any combination of these, not one
    // exclusive label) — surfaced via ENGINE_INFO (0x41).
    bool        hydra_solo_active        = true;  // always true - model is always loaded now
    bool        hydra_rpc_backend_active  = false; // --ggml-rpc-port set & RPC thread running
    std::string hydra_peer;                        // configured --rpc-engine peer, empty if none
    bool        hydra_peer_reachable     = false;  // startup TCP-probe result for hydra_peer
    std::string hydra_combined_pattern;            // --combined-ot-pattern (expert) or tensor_split ratio (layer)
    std::string hydra_split_mode        = "expert"; // "expert" or "layer" (#383 T1)

    // True once llama_hydra_load_combined_experts() has successfully
    // dual-loaded expert tensors onto hydra_peer. Gates whether
    // SET_EXPERT_MODE("combined") is honored or falls back to solo. Renamed
    // from hydra_combined_capable, which was confusingly aliased to both
    // "peer configured+reachable" and "dual-load succeeded" in the old INFO
    // JSON — those are now distinct (hydra_peer_reachable vs this field).
    bool hydra_combined_head_attached = false;

    // Hydra P1-6: back-pointer to server_routes, set after construction
    // in server.cpp.  Used by apply_pending_hydra_config() to refresh
    // the cached server_context_meta on the task-queue thread (safe —
    // runs during the drain window when no slots are processing).
    server_routes * routes_ptr = nullptr;

    // Set by the CONFIGURE HTTP handler when ctx_tgt is null and T3 keys
    // are staged. Cleared by apply_pending_hydra_config() on the
    // task-queue thread when it runs apply_t3_rebuild() for first load.
    bool        first_load_pending = false;
    // P0-1 (#49): staged capabilities for deferred first-load.
    // Populated by set_bootstrap_capabilities() before start_loop();
    // applied by apply_pending_hydra_config() after the first load.
    bool        bootstrap_rpc_active = false;
    std::string bootstrap_peer;
    bool        bootstrap_peer_reachable = false;
    std::string bootstrap_pattern;
    std::string bootstrap_split_mode = "none";
    bool        bootstrap_combined_static = false;

    // Hydra #383 T1: true when this engine was started in COMBINED static
    // (layer-split) mode. The split is fixed at model load time; the mode
    // cannot be changed at runtime — SET_EXPERT_MODE("solo") is rejected.
    bool hydra_combined_static = false;

    // #29 Phase B: tracks the currently active peer endpoint for per-request
    // peer switching. Protected by the task queue (process_single_task is
    // serialized), so no separate mutex needed.
    std::string hydra_current_peer;

    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    // epic #610 WS1: Hydra A/B extension seam.
    //   legacy mode (default): the inline Hydra code in this file runs.
    //   seam mode (HYDRA_EXT_MODE=seam): the hydra_engine_extension drives the
    //   same behavior through the server_hydra_extension interface. Both paths
    //   stay compiled; only one is consulted per run, so the same binary can be
    //   A/B tested by flipping the env var.
    const bool      hydra_ext_active = hydra_ext_mode_seam();
    std::unique_ptr<server_hydra_extension> hydra_ext;

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
        if (hydra_ext_active) {
            hydra_ext = hydra_create_extension();
            SRV_INF("hydra ext: seam mode active (HYDRA_EXT_MODE=seam), impl=%s\n", hydra_ext->name());
        } else {
            SRV_INF("%s", "hydra ext: legacy mode active (default) - A/B baseline\n");
        }
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    llama_batch batch {};

    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;
    int slots_debug = 0;
    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    json json_ui_settings = json::object();    // Primary: new name
    json json_webui_settings = json::object();    // Deprecated: use json_ui_settings instead (kept for compat)

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    // M-Perf.9 #289: alias → GGUF-path map for the engine PREFILL `model` swap.
    // Populated at startup from --models-preset PATH (same INI the llama-server
    // router mode already uses). Empty when no preset is configured — in that
    // case the engine has no model registry and the PREFILL handler treats
    // any non-empty `model` value as a fallback signal.
    std::map<std::string, std::string> preset_alias_to_path;

    // Full preset objects keyed by alias, used at swap time to apply the
    // target model's tensor_buft_overrides, n_gpu_layers, split_mode, etc.
    // via common_preset::apply_to_params().  Without this, the bare-alias
    // swap path copies params_base (source model's overrides) and only
    // patches model.path — causing a VRAM mismatch when the target model
    // has different tensor placement rules (e.g. n-cpu-moe, override-tensor).
    std::map<std::string, common_preset> preset_alias_to_preset;

    // #470: T3-current alias → file map. Records the engine's own identity
    // aliases (model_name + model_aliases members) → the resident file path
    // as of the last successful model load. Maintained by
    // hydra_t3_record_current_alias_to_path() at the end of load_model().
    // DECODE_APPLY consults this BEFORE preset_alias_to_path: the
    // coordinator's T3 config (model_path) can deliberately load a file
    // that differs from the preset INI's file for the same alias (e.g. the
    // dense-27b-combined session T3-loads the 27B-Coder file while the
    // engine's alias identity still says qwen3.6-35B-balanced from an
    // earlier SOLO session). In that case the requested alias describes the
    // resident — swapping to the INI's file is a pointless 73-81s reload
    // that tears down COMBINED state and then fails Gate B (header metadata
    // of the pre-swap resident vs the swapped-in model's identity).
    std::map<std::string, std::string> t3_current_alias_to_path;

    bool sleeping = false;

    void destroy() {
        spec.reset();
        ctx_dft.reset();
        model_dft.reset();

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        // #648 review: the ctx_tgt/ctx_dft bindings held in params_base dangle
        // once llama_init.reset() freed the contexts. Clear them so any later
        // load_model() that fails to rebuild the MTP draft context degrades
        // cleanly (null-gate) instead of dereferencing freed memory.
        hydra_clear_stale_draft_bindings(params_base);

        mtmd_free(mctx);
        mctx = nullptr;

        llama_batch_free(batch);
        batch = {}; // zero dangling pointers to prevent double-free
                     // if destroy() is called again (e.g. T3 rollback
                     // after a failed load_model that released the old
                     // model but never reached llama_batch_init).
    }

    void slot_save_and_clear(server_slot & slot) {
        if (slot.hydra_transferring->load()) {
            return;
        }
        if (slot.prompt.n_tokens() == 0) {
            return;
        }
        SLT_INF(slot, "%s", "saving idle slot to prompt cache\n");
        SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
        slot.prompt_save(*prompt_cache);
        slot.prompt_clear(false);
        prompt_cache->update();
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        bool is_resume = sleeping;

        SRV_INF("loading model '%s'\n", params.model.path.c_str());

        // #486: If a model is already resident (swap, not first load), release it
        // BEFORE loading the new model. Without this, both old and new models
        // are simultaneously in VRAM during common_init_from_params() — the old
        // model's buffers aren't freed until the assignment completes, so the new
        // model's cudaMallocs run against the old model's still-resident VRAM,
        // causing OOM or degraded memory fit. destroy() is safe on first load
        // (all resets are no-ops on already-null/empty state).
        if (llama_init) {
            SRV_INF("releasing previous model before swap (model_path='%s')\n",
                    params_base.model.path.c_str());
            destroy();
            // Log available VRAM after release so operators can see the freed space
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                auto dev = ggml_backend_dev_get(i);
                size_t free_mem = 0, total_mem = 0;
                ggml_backend_dev_memory(dev, &free_mem, &total_mem);
                if (total_mem > 0) {
                    SRV_INF("VRAM after release: device '%s' free=%.1f MiB / total=%.1f MiB\n",
                            ggml_backend_dev_name(dev),
                            free_mem / (1024.0 * 1024.0),
                            total_mem / (1024.0 * 1024.0));
                }
            }
        }

        params_base = params;
        // Hydra #406: resolve n_parallel = -1 (auto) to 1 before the
        // conversion to uint32_t n_seq_max in common_context_params_to_llama.
        // When n_parallel is -1, the int→uint wrap produces UINT32_MAX,
        // which fails the LLAMA_MAX_SEQ=256 check in llama_context.cpp:54.
        // The server's "auto" mode should default to 1 slot.
        if (params_base.n_parallel <= 0) {
            params_base.n_parallel = 1;
        }
        params_base.n_outputs_max = server_n_outputs_max(params_base);

        std::string & mmproj_path = params_base.mmproj.path;
        bool has_mmproj = !mmproj_path.empty();
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.media_marker     = get_media_marker();
        }

        // optionally get the memory usage of mmproj
        if (has_mmproj && params_base.fit_params) {
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_INF("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB\n", total / (1024.0 * 1024.0));
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // optionally reserve VRAM for the draft / MTP context before fitting the target model
        if (params_base.fit_params) {
            const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                            params_base.speculative.types.end(),
                                            COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
            const bool has_draft = params_base.speculative.has_dft();

            if (has_draft || spec_mtp) {
                common_params params_dft = params_base;
                bool measure_model_bytes = true;

                if (has_draft) {
                    const auto & params_spec = params_base.speculative.draft;
                    params_dft.devices               = params_spec.devices;
                    params_dft.model                 = params_spec.mparams;
                    params_dft.n_gpu_layers          = params_spec.n_gpu_layers;
                    params_dft.cache_type_k          = params_spec.cache_type_k;
                    params_dft.cache_type_v          = params_spec.cache_type_v;
                    params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;
                } else {
                    // MTP draft context lives on the target model, only context+compute are new
                    measure_model_bytes = false;
                }

                params_dft.n_outputs_max = params_base.n_parallel;

                auto mparams_dft = common_model_params_to_llama(params_dft);
                auto cparams_dft = common_context_params_to_llama(params_dft);
                if (spec_mtp) {
                    cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                    cparams_dft.type_k   = params_base.speculative.draft.cache_type_k;
                    cparams_dft.type_v   = params_base.speculative.draft.cache_type_v;
                }
                cparams_dft.n_rs_seq = 0;

                std::vector<ggml_backend_dev_t> devs;
                uint32_t hp_ngl = 0;
                uint32_t hp_nct = 0;
                uint32_t hp_nex = 0;
                try {
                    auto dmd = common_get_device_memory_data(
                        params_dft.model.path.c_str(), &mparams_dft, &cparams_dft,
                        devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);

                    GGML_ASSERT(!params_base.fit_params_target.empty());
                    size_t total = 0;

                    std::vector<ggml_backend_dev_t> tgt_devices = params.devices;

                    if (tgt_devices.empty()) {
                        for(size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                           tgt_devices.push_back(ggml_backend_dev_get(i));
                        }
                    }

                    for (size_t j = 0; j < devs.size(); ++j) {
                        const size_t bytes =
                            (measure_model_bytes ? dmd[j].mb.model : 0) +
                            dmd[j].mb.context +
                            dmd[j].mb.compute;
                        total += bytes;
                        for (size_t i = 0; i < tgt_devices.size(); i++) {
                            if (tgt_devices[i] == devs[j]) {
                                SRV_DBG("[spec] adding %.2f MiB to fit_params_target for device %s\n",
                                        bytes / (1024.0 * 1024.0), ggml_backend_dev_name(devs[j]));
                                params_base.fit_params_target[i] += bytes;
                                break;
                            }
                        }
                    }
                    SRV_INF("[spec] estimated memory usage of %s is %.2f MiB\n",
                            has_draft ? "draft model" : "MTP context",
                            total / (1024.0 * 1024.0));
                } catch (const std::exception & e) {
                    SRV_ERR("[spec] failed to measure %s memory: %s\n",
                            has_draft ? "draft model" : "MTP context", e.what());
                }
            }
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (params_base.speculative.has_dft()) {
            // TODO speculative: move to common/speculative.cpp?
            const auto & params_spec = params_base.speculative.draft;

            SRV_INF("loading draft model '%s'\n", params_spec.mparams.path.c_str());

            auto params_dft = params_base;

            params_dft.devices      = params_spec.devices;
            params_dft.model        = params_spec.mparams;
            params_dft.n_gpu_layers = params_spec.n_gpu_layers;
            params_dft.cache_type_k = params_spec.cache_type_k;
            params_dft.cache_type_v = params_spec.cache_type_v;

            if (params_spec.cpuparams.n_threads > 0) {
                params_dft.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
                params_dft.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
            }

            params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;

            auto mparams_dft = common_model_params_to_llama(params_dft);

            model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
            if (model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                return false;
            }

            auto cparams = common_context_params_to_llama(params_dft);

            const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                            params_base.speculative.types.end(),
                                            COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
            if (spec_mtp) {
                cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            }

            // note: for small models maybe we can set this to the maximum possible draft from all speculative types
            //       the extra memory for small models is likely negligible?
            cparams.n_rs_seq = 0;
            ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));

            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());

            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft.get();
        } else if (std::find(params_base.speculative.types.begin(), params_base.speculative.types.end(),
                             COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end()) {
            SRV_INF("creating MTP draft context against the target model '%s'\n",
                    params_base.model.path.c_str());

            auto cparams_mtp = common_context_params_to_llama(params_base);
            cparams_mtp.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
            cparams_mtp.type_k        = params_base.speculative.draft.cache_type_k;
            cparams_mtp.type_v        = params_base.speculative.draft.cache_type_v;
            cparams_mtp.n_rs_seq      = 0;
            cparams_mtp.n_outputs_max = params_base.n_parallel;

            ctx_dft.reset(llama_init_from_model(model_tgt, cparams_mtp));
            if (ctx_dft == nullptr) {
                // The MTP draft context failed to build (e.g. device OOM for
                // its KV + compute buffers under a tight tensor_split). This
                // must NOT abort the model load — the draft is an optimization,
                // not a requirement, and load_model() already released the
                // previous model, so an abort leaves the engine without any
                // model (T3 reload → 503 on every request). Fall back to
                // non-speculative serving: common_speculative_init() below
                // skips MTP when draft.ctx_dft == nullptr, so the engine
                // degrades cleanly instead of failing the reload.
                SRV_ERR("%s", "failed to create MTP draft context — continuing WITHOUT speculative decoding\n");

                // #648 review (UAF): params_base may still carry the PREVIOUS
                // load's ctx_tgt/ctx_dft bindings (copied through swapped_params
                // on a T3 reload). destroy() freed those contexts, so the
                // pointers are dangling-non-null — without clearing them the
                // common_speculative_init() null-gate would activate the MTP
                // impl and dereference freed memory (speculative.cpp:447).
                // Clear them so the degradation gate actually fires.
                hydra_clear_stale_draft_bindings(params_base);
            } else {
                ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());

                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft.get();
            }
        }

        if (has_mmproj) {
            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        // setup slots
        SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        int n_ctx_slot = llama_n_ctx_seq(ctx_tgt);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", n_ctx_slot, n_ctx_train);
            n_ctx_slot = n_ctx_train;
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_WRN("%s", "speculative decoding will use checkpoints\n");
        }

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
            }
        }

        if (spec) {
            SRV_INF("%s", "speculative decoding context initialized\n");
        } else {
            ctx_dft.reset();
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft.get();
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            SLT_INF(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            batch = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_INF("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_INF("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_INF("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_INF("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_INF("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_INF("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_INF("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.name.empty()) {
            model_name = params_base.model.name;
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // M-Perf.9 #289: build the alias → GGUF-path registry that the
        // PREFILL handler uses to honour the optional `model` key. The
        // same INI the llama-server router mode already loads via
        // `common_preset_context::load_from_ini` is the source. If no
        // preset is configured (the P100 single-model default today) the
        // map is left empty and the PREFILL handler treats any non-empty
        // `model` value as a fallback signal.
        preset_alias_to_path.clear();
        preset_alias_to_preset.clear();
        if (!params_base.models_preset.empty()) {
            // The preset may be either a single .ini file or a directory
            // containing one .ini per alias. We try the file first, then
            // fall back to scanning the directory.
            std::error_code ec;
            const bool is_dir = std::filesystem::is_directory(params_base.models_preset, ec);
            std::vector<std::string> ini_files;
            if (is_dir) {
                for (const auto & entry : std::filesystem::directory_iterator(
                         params_base.models_preset,
                         std::filesystem::directory_options::skip_permission_denied, ec)) {
                    if (!ec && entry.is_regular_file() && entry.path().extension() == ".ini") {
                        ini_files.push_back(entry.path().string());
                    }
                }
            } else {
                ini_files.push_back(params_base.models_preset);
            }
            common_preset_context preset_ctx(LLAMA_EXAMPLE_SERVER, /*only_remote_allowed*/ false);
            for (const auto & ini_path : ini_files) {
                common_preset global = {};
                try {
                    auto presets = preset_ctx.load_from_ini(ini_path, global);
                    for (const auto & [alias, preset] : presets) {
                        std::string model_path;
                        if (preset.get_option("LLAMA_ARG_MODEL", model_path) && !model_path.empty()) {
                            preset_alias_to_path[alias] = model_path;
                            preset_alias_to_preset[alias] = preset;
                            SRV_INF("hydra: preset alias '%s' → %s\n", alias.c_str(), model_path.c_str());
                        }
                    }
                } catch (const std::exception & e) {
                    SRV_WRN("hydra: failed to load preset file '%s': %s\n",
                            ini_path.c_str(), e.what());
                }
            }
            if (!preset_alias_to_path.empty()) {
                SRV_INF("hydra: loaded %zu preset alias(es) from %s\n",
                        preset_alias_to_path.size(), params_base.models_preset.c_str());
            }
        }

        // #470: refresh the T3-current alias → file map now that
        // model_name / model_aliases reflect the freshly loaded resident
        // (a no-op on boot, where the identity is the boot file's). Must
        // run AFTER the identity members above are re-derived.
        hydra_t3_record_current_alias_to_path();

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    // P0-1 (#49): bootstrap_init wires up queue callbacks and metrics without
    // requiring a model to be loaded. Used by llama-engine's head-bootstrap
    // mode (no model at startup, model loaded later via CONFIGURE T3).
    // Must be called before start_loop().
    bool bootstrap_init() {
        SRV_INF("%s", "P0-1: bootstrap_init — wiring up queues without model\n");

        queue_tasks.on_new_task([this](server_task && task) {
            process_single_task(std::move(task));
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();
        return true;
    }

    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task) {
            process_single_task(std::move(task));
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (!params_base.kv_unified) {
                SRV_WRN("%s", "--cache-idle-slots requires --kv-unified, disabling\n");
                params_base.cache_idle_slots = false;
            } else if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                SRV_INF("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        // populate UI settings (from either new ui_config_json or deprecated webui_config_json)
        {
            const std::string & cfg = !params_base.ui_config_json.empty()
                ? params_base.ui_config_json
                : params_base.webui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                    json_webui_settings = json_settings; // deprecated: keep in sync
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                LOG_INF("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // thinking is enabled if:
            // 1. It's not explicitly disabled via --reasoning off
            // 2. The chat template supports it
            const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
            const bool enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
            SRV_INF("%s: chat template, thinking = %d\n", __func__, enable_thinking);

            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // find the slot that has at least n% prompt similarity
        if (ret == nullptr && slot_prompt_similarity != 0.0f) {
            float sim_best = 0;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing() || slot.hydra_transferring->load()) {
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const float sim_cur = float(tokens.get_common_prefix(task.tokens)) / task.tokens.size();

                // select the current slot if the criteria match
                if (sim_cur > sim_best && sim_cur > slot_prompt_similarity) {
                    sim_best = sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                SLT_INF(*ret, "selected slot by LCP similarity, sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        sim_best, slot_prompt_similarity, f_keep);

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing() || slot.hydra_transferring->load()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            const auto & tokens = ret->prompt.tokens;

            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                SRV_INF("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                // don't save the slot's state if its context is empty
                if (tokens.size() > 0) {
                    ret->prompt_save(*prompt_cache);
                }

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear(false);
                }

                prompt_cache->update();

                SRV_INF("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear(false);

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.tokens.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool backend_sampling = true;

            backend_sampling &= task.params.sampling.backend_sampling;

            // TODO: speculative decoding requires multiple samples per batch - not supported yet
            backend_sampling &= !(slot.can_speculate());

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());
        } else {
            slot.smpl.reset();
        }

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && (ggml_time_us() - slot.t_start_generation > 1000.0f*slot.task->params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", slot.n_decoded, slot.n_remaining, result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            // TODO: optimize this with min-p optimization
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    // if multimodal is enabled, send an error and return false
    bool check_no_mtmd(const int id_task) {
        if (mctx) {
            send_error(id_task, "This feature is not supported by multimodal", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        return true;
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.n_prompt_tokens_cache;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = (ggml_time_us() - slot.t_start_process_prompt) / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->timings         = slot.get_timings();
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        // #469 trace: log final response for hallucination detection
        {
            const std::string & content = res->content;
            const size_t content_len = content.size();
            const int n_decoded = res->n_decoded;
            const int n_cache = res->n_prompt_tokens_cache;
            const int n_prompt = res->n_prompt_tokens;
            // Log first 200 chars of content for quick inspection
            const std::string preview = content_len > 200 ? content.substr(0, 200) : content;
            SRV_DBG("hydra: FINAL_RESPONSE slot=%d n_prompt=%d n_cache=%d n_decoded=%d content_len=%zu preview='%s'\n",
                    slot.id, n_prompt, n_cache, n_decoded, content_len, preview.c_str());
            // Log if content looks suspicious (empty, very short, or starts with thinking tags)
            if (content_len == 0 && n_decoded > 0) {
                SRV_WRN("hydra: FINAL_RESPONSE slot=%d WARNING: empty content but n_decoded=%d — possible hallucination!\n", slot.id, n_decoded);
            }
            if (content_len > 0 && content_len < 10 && n_decoded > 50) {
                SRV_WRN("hydra: FINAL_RESPONSE slot=%d WARNING: very short content (%zu chars) but n_decoded=%d — possible truncation!\n", slot.id, content_len, n_decoded);
            }
        }

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && !slot.hydra_transferring->load()
                && slot.reserved_for_decode_id == -1 && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_INF("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        // Save the FULL KV state (kv_base + kv_swa) so that in COMBINED mode
        // the RPC0 (peer GPU) KV is also checkpointed. PARTIAL_ONLY skips
        // kv_base (full-attention layers) which causes CUDA illegal memory
        // access on the second turn because the RPC0 backend has stale data.
        cur.update_tgt(ctx_tgt,       slot.id, 0);
        cur.update_dft(ctx_dft.get(), slot.id, 0);

        // Hydra M2-stream double-write fix (#470/#620): a SECOND, separate
        // recurrent-only capture used ONLY for the wire checkpoint (STATE_GET
        // serializes data_tgt_recr, never data_tgt). Hybrid/recurrent models
        // need the recurrent (SSM) state at BOTH the checkpoint position and
        // the live position, but the attention portion of the checkpoint is
        // redundant on the wire — it scales with ctx and the full live state
        // already carries it. The flags=0 captures above keep serving the
        // local rewind path unchanged. These *_recr helpers hardcode
        // LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY and must only ever be read back
        // via load_tgt_recr/load_dft_recr (matched flags — a PARTIAL_ONLY
        // buffer physically lacks the mem_attn bytes).
        cur.update_tgt_recr(ctx_tgt,       slot.id);
        cur.update_dft_recr(ctx_dft.get(), slot.id);

        SLT_INF(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    // Apply the T1 keys from `cfg` to `params` and to the live context (for
    // state_chunk_size). Echoes each applied key + post-clamp value into
    // `params_applied`. Returns true on success; false if a key's value is
    // of the wrong type (which is reported back to the caller — the request
    // is malformed and we don't want to apply a partial set).
    bool hydra_apply_t1_config(common_params & params, llama_context * ctx,
                               const json & cfg,
                               std::map<std::string, json> & params_applied) {
        // sampling.* — set on the common_params, which the next launch_slot
        // will pick up when re-initializing the slot's common_sampler.
        if (cfg.contains("sampling") && cfg["sampling"].is_object()) {
            const json & s = cfg["sampling"];
            #define COPY_FLOAT(field) \
                if (s.contains(#field) && s[#field].is_number()) { \
                    params.sampling.field = s[#field].get<float>(); \
                    params_applied["sampling." #field] = params.sampling.field; \
                }
            #define COPY_INT(field) \
                if (s.contains(#field) && s[#field].is_number()) { \
                    params.sampling.field = s[#field].get<int32_t>(); \
                    params_applied["sampling." #field] = params.sampling.field; \
                }
            COPY_FLOAT(temp)
            COPY_FLOAT(top_p)
            COPY_FLOAT(min_p)
            COPY_FLOAT(penalty_repeat)
            COPY_INT(top_k)
            COPY_INT(seed)
            #undef COPY_FLOAT
            #undef COPY_INT
        }
        // n_predict
        if (cfg.contains("n_predict") && cfg["n_predict"].is_number_integer()) {
            params.n_predict = cfg["n_predict"].get<int32_t>();
            params_applied["n_predict"] = params.n_predict;
        } else if (cfg.contains("n_predict") && !cfg["n_predict"].is_number_integer()) {
            SRV_WRN("%s", "hydra: CONFIGURE n_predict must be an integer\n");
            return false;
        }
        // n_keep
        if (cfg.contains("n_keep") && cfg["n_keep"].is_number_integer()) {
            params.n_keep = cfg["n_keep"].get<int32_t>();
            params_applied["n_keep"] = params.n_keep;
        } else if (cfg.contains("n_keep") && !cfg["n_keep"].is_number_integer()) {
            SRV_WRN("%s", "hydra: CONFIGURE n_keep must be an integer\n");
            return false;
        }
        // seed (top-level — sets the sampler's seed via common_params::sampling).
        // common_params itself has no top-level seed; common_params_sampling does.
        if (cfg.contains("seed") && cfg["seed"].is_number_unsigned()) {
            params.sampling.seed = cfg["seed"].get<uint32_t>();
            params_applied["seed"] = params.sampling.seed;
        } else if (cfg.contains("seed") && cfg["seed"].is_number_integer()) {
            params.sampling.seed = (uint32_t) cfg["seed"].get<int32_t>();
            params_applied["seed"] = params.sampling.seed;
        } else if (cfg.contains("seed") && !cfg["seed"].is_number()) {
            SRV_WRN("%s", "hydra: CONFIGURE seed must be a number\n");
            return false;
        }
        // antiprompt — full replacement (matches the existing semantics
        // of CLI --reverse-prompt)
        if (cfg.contains("antiprompt") && cfg["antiprompt"].is_array()) {
            std::vector<std::string> new_antiprompt;
            for (const auto & v : cfg["antiprompt"]) {
                if (!v.is_string()) {
                    SRV_WRN("%s", "hydra: CONFIGURE antiprompt entries must be strings\n");
                    return false;
                }
                new_antiprompt.push_back(v.get<std::string>());
            }
            params.antiprompt = std::move(new_antiprompt);
            params_applied["antiprompt"] = params.antiprompt;
        } else if (cfg.contains("antiprompt") && !cfg["antiprompt"].is_array()) {
            SRV_WRN("%s", "hydra: CONFIGURE antiprompt must be an array of strings\n");
            return false;
        }
        // state_chunk_size — apply via the existing llama_hydra API (clamps
        // and echoes the post-clamp value)
        if (cfg.contains("state_chunk_size") && cfg["state_chunk_size"].is_number_unsigned()) {
            const size_t bytes = cfg["state_chunk_size"].get<size_t>();
            if (ctx) {
                llama_hydra_set_state_chunk_size(ctx, bytes);
            }
            const size_t applied = ctx ? llama_hydra_get_state_chunk_size(ctx) : llama_hydra_clamp_state_chunk_size(bytes);
            params_applied["state_chunk_size"] = (uint64_t) applied;
        } else if (cfg.contains("state_chunk_size") && !cfg["state_chunk_size"].is_number_unsigned()) {
            SRV_WRN("%s", "hydra: CONFIGURE state_chunk_size must be a non-negative integer\n");
            return false;
        }
        return true;
    }

    // Apply the T3 mutators immediately. The "staging" is: the statics in
    // llama-hydra.cpp + the T3 keys in pending_config JSON. The actual
    // model reload happens later, in the slot-free trigger.
    void hydra_apply_t3_mutators(llama_context * ctx, const json & cfg,
                                 std::vector<std::string> & deferred_keys) {
        if (cfg.contains("n_gpu_layers") && cfg["n_gpu_layers"].is_number_integer()) {
            llama_hydra_set_pending_n_gpu_layers(cfg["n_gpu_layers"].get<int32_t>());
            deferred_keys.push_back("n_gpu_layers");
        }
        if (cfg.contains("n_cpu_moe") && cfg["n_cpu_moe"].is_number_integer()) {
            llama_hydra_set_pending_n_cpu_moe(cfg["n_cpu_moe"].get<int32_t>());
            deferred_keys.push_back("n_cpu_moe");
        }
        if (cfg.contains("override_tensor") && cfg["override_tensor"].is_string()) {
            llama_hydra_set_override_tensor(ctx, cfg["override_tensor"].get<std::string>().c_str());
            deferred_keys.push_back("override_tensor");
        }
        if (cfg.contains("split_mode") && cfg["split_mode"].is_string()) {
            std::vector<float> split;
            if (cfg.contains("tensor_split") && cfg["tensor_split"].is_array()) {
                for (const auto & v : cfg["tensor_split"]) {
                    if (v.is_number()) split.push_back(v.get<float>());
                }
            }
            llama_hydra_set_split_mode(ctx, cfg["split_mode"].get<std::string>().c_str(),
                                       split.empty() ? nullptr : split.data(), split.size());
            deferred_keys.push_back("split_mode");
            if (!split.empty()) deferred_keys.push_back("tensor_split");
        } else if (cfg.contains("tensor_split") && cfg["tensor_split"].is_array()) {
            // tensor_split without split_mode is meaningless; record it as
            // deferred and let the apply step surface the missing mode.
            deferred_keys.push_back("tensor_split");
        }
        if (cfg.contains("model") && cfg["model"].is_object() &&
            cfg["model"].contains("path") && cfg["model"]["path"].is_string()) {
            llama_hydra_set_pending_model_path(cfg["model"]["path"].get<std::string>().c_str());
            deferred_keys.push_back("model.path");
        } else if (cfg.contains("model") && cfg["model"].is_string()) {
            // legacy shorthand: {"model": "/path/to.gguf"}
            llama_hydra_set_pending_model_path(cfg["model"].get<std::string>().c_str());
            deferred_keys.push_back("model");
        }
        // hydra_config flat key: {"model_path": "/path/to.gguf"}
        if (cfg.contains("model_path") && cfg["model_path"].is_string()) {
            llama_hydra_set_pending_model_path(cfg["model_path"].get<std::string>().c_str());
            deferred_keys.push_back("model_path");
        }
        // hydra_config: {"rpc_servers": ["host1:port1", "host2:port2"]}
        // Stored in a static for apply_t3_rebuild() to consume before
        // load_model(). The actual ggml backend registration happens in
        // hydra_register_rpc_servers() called from apply_t3_rebuild().
        if (cfg.contains("rpc_servers") && cfg["rpc_servers"].is_array()) {
            g_pending_rpc_servers.clear();
            for (const auto & v : cfg["rpc_servers"]) {
                if (v.is_string()) {
                    g_pending_rpc_servers.push_back(v.get<std::string>());
                }
            }
            deferred_keys.push_back("rpc_servers");
        }
    }

    // Shared helper: classify config keys, apply T1 immediately, and either
    // stage (sync=false) or synchronously apply (sync=true) T2/T3.
    //
    // sync=false (CONFIGURE path): T2/T3 are staged via hydra_set_pending_config()
    // + hydra_apply_t3_mutators() for later application at the slot-free moment.
    //
    // sync=true (PREFILL / HTTP decode path): T2/T3 are applied immediately
    // via apply_t2_rebuild() / apply_t3_rebuild(). The caller already owns
    // the task-queue thread context, so synchronous application is safe.
    //
    // Returns a structured result so callers can build CONFIGURE responses
    // or handle errors uniformly.
    struct hydra_config_result {
        int highest_tier = 0;
        std::map<std::string, json> params_applied;
        std::vector<std::string> deferred_keys;
        json t2t3_subset = json::object();
        bool ok = true;
        std::string error;
        uint64_t state_chunk_size_applied = 0;
        // hydra#470: generic (T4) keys that cannot take effect. Echoed to
        // the Coordinator via the CONFIGURE response so nothing is silent.
        std::vector<std::string> unrecognized_keys;  // no llama.cpp arg-table entry
        std::vector<std::string> rejected_keys;      // startup-only / flag / two-value arg
    };

    hydra_config_result hydra_apply_config(const json & cfg, bool sync) {
        hydra_config_result result;

        // 1. Classify every top-level key. T1 → apply now; T2/T3/T4 →
        //    defer (stage) or apply synchronously depending on `sync`.
        //    T4 (generic) keys are additionally validated against the
        //    llama.cpp arg table here so the CONFIGURE response can report
        //    unrecognized/rejected keys before the deferred apply runs.
        //    T2 + appliable-T4 keys are staged together into the reload
        //    config (g_pending_reload_config) so neither the context-reload
        //    path nor the model-reload path strands them.
        json t1_subset = json::object();
        json reload_subset = json::object();
        for (auto it = cfg.begin(); it != cfg.end(); ++it) {
            const std::string key = it.key();
            int tier = hydra_classify_config_key(key);
            if (tier == 1) {
                t1_subset[key] = it.value();
            } else if (tier == 2) {
                result.t2t3_subset[key] = it.value();
                result.deferred_keys.push_back(key);
                reload_subset[key] = it.value();
            } else if (tier == 3) {
                result.t2t3_subset[key] = it.value();
                result.deferred_keys.push_back(key);
                // T3 keys are staged via hydra_apply_t3_mutators() statics,
                // not the reload-config JSON.
            } else {
                // T4: generic-arg pass-through. Classify against the arg
                // table; only appliable keys are staged for the deferred
                // slot-free moment. The rest are reported loudly.
                const hydra_generic_key_status st = hydra_classify_generic_key(key);
                if (st == hydra_generic_key_status::APPLIABLE) {
                    result.t2t3_subset[key] = it.value();
                    result.deferred_keys.push_back(key);
                    reload_subset[key] = it.value();
                } else if (st == hydra_generic_key_status::DENIED) {
                    SRV_WRN("hydra: CONFIGURE key '%s' cannot change at reload (startup-only/flag arg) — rejected\n",
                            key.c_str());
                    result.rejected_keys.push_back(key);
                } else {
                    SRV_WRN("hydra: CONFIGURE key '%s' is not a known llama.cpp argument — unrecognized, value ignored\n",
                            key.c_str());
                    result.unrecognized_keys.push_back(key);
                }
            }
            if (tier > result.highest_tier) result.highest_tier = tier;
        }
        // Stage the reload config (T2 + appliable-T4 keys) for the deferred
        // slot-free moment. Unconditional overwrite = absolute state: a
        // superseding CONFIGURE without T2/T4 keys must clear whatever an
        // earlier CONFIGURE staged, or the stale keys would be applied on
        // the next unrelated reload.
        g_pending_reload_config = reload_subset.dump();
        // The "sampling" object may contain unlisted nested keys
        // (e.g. penalty_last_n, mirostat) — route the whole object
        // through T1 when present.
        if (cfg.contains("sampling") && cfg["sampling"].is_object()) {
            t1_subset["sampling"] = cfg["sampling"];
            if (result.highest_tier < 1) result.highest_tier = 1;
        }
        // model.path is nested — the legacy {"model": {...}} form
        // is recognized by hydra_classify_config_key returning 3
        // for the bare "model" key. If the bare "model" is set
        // and is an object with a "path", route it as T3.
        if (cfg.contains("model")) {
            if (cfg["model"].is_object()) {
                result.t2t3_subset["model"] = cfg["model"];
                if (std::find(result.deferred_keys.begin(), result.deferred_keys.end(), "model")
                    == result.deferred_keys.end()) {
                    result.deferred_keys.push_back("model");
                }
                if (result.highest_tier < 3) result.highest_tier = 3;
            } else if (cfg["model"].is_string()) {
                result.t2t3_subset["model"] = cfg["model"];
                if (std::find(result.deferred_keys.begin(), result.deferred_keys.end(), "model")
                    == result.deferred_keys.end()) {
                    result.deferred_keys.push_back("model");
                }
                if (result.highest_tier < 3) result.highest_tier = 3;
            }
        }

        if (result.highest_tier == 0) {
            // No recognized keys — caller decides whether to treat as
            // a no-op or surface an error.
            return result;
        }

        // 2. Apply T1 keys in-place.
        if (!t1_subset.empty()) {
            if (!hydra_apply_t1_config(params_base, ctx_tgt, t1_subset, result.params_applied)) {
                result.ok = false;
                result.error = "T1 key has wrong type (see log)";
                return result;
            }
            // Capture state_chunk_size for callers that need it (CONFIGURE).
            auto it = result.params_applied.find("state_chunk_size");
            if (it != result.params_applied.end() && it->second.is_number_unsigned()) {
                result.state_chunk_size_applied = it->second.get<uint64_t>();
            }
        }

        // 3. T2/T3/T4 handling — diverges based on sync flag.
        if (!result.t2t3_subset.empty()) {
            if (sync) {
                // Synchronous mode (PREFILL / HTTP decode): apply now
                // on the task-queue thread. The caller owns this thread
                // context so blocking is safe.
                if (result.highest_tier >= 3) {
                    // T3 (model reload) also covers a T4-only config:
                    // load_model() recreates the context and the
                    // speculative/draft state, so every generic key
                    // takes effect. hydra_apply_t3_mutators() is a no-op
                    // when no T3 keys are staged.
                    hydra_apply_t3_mutators(ctx_tgt, result.t2t3_subset, result.deferred_keys);
                    // #470: force rebuild if a peer reconnection was detected
                    // during a prior graph_compute — the peer's buffers are gone
                    // even though model/params haven't changed.
                    const bool reconn_force = (ctx_tgt && ctx_tgt->peer_reconnection_pending);
                    if (reconn_force) {
                        ctx_tgt->peer_reconnection_pending = false;
                        SRV_WRN("%s", "hydra: PREFILL handler: peer reconnection pending — forcing T3 rebuild\n");
                    }
                    if (!apply_t3_rebuild(reconn_force)) {
                        result.ok = false;
                        result.error = "T3 rebuild failed";
                        return result;
                    }
                } else if (result.highest_tier == 2) {
                    if (!apply_t2_rebuild(result.t2t3_subset.dump())) {
                        result.ok = false;
                        result.error = "T2 rebuild failed";
                        return result;
                    }
                }
                // Clear T3 staged statics — they were consumed by the
                // sync apply and must not leak into a later deferred path.
                llama_hydra_clear_pending_t3();
            } else {
                // Stage mode (CONFIGURE): record the mutators and store
                // pending_config for application at the next slot-free moment.
                if (result.highest_tier >= 3) {
                    hydra_apply_t3_mutators(ctx_tgt, result.t2t3_subset, result.deferred_keys);
                }
                if (ctx_tgt) {
                    ctx_tgt->hydra_set_pending_config(
                        result.t2t3_subset.dump(), hydra_tier_label(result.highest_tier));
                } else {
                    // First load: ctx_tgt is null, so apply_pending_hydra_config()
                    // and update_slots() can't trigger. Set the flag so the
                    // task-queue thread runs apply_t3_rebuild() at the next
                    // slot-free moment.
                    first_load_pending = true;
                    SRV_INF("%s", "hydra: config staged for first load (no context yet)\n");
                }
            }
        }

        return result;
    }

    void process_single_task(server_task && task) {
        // epic #610 WS1: in seam mode the extension may claim Hydra tasks.
        // WS1 impl is a no-op (returns false), so this is a pure A/B switch —
        // both modes run the inline dispatch below.
        if (hydra_ext_active && hydra_ext && hydra_ext->handle_task(*this, task)) {
            return;
        }
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    // Hydra config from HTTP decode path: apply synchronously
                    // on the task-queue thread before any slot scheduling or
                    // generation work. This is safe because we own this thread;
                    // the previous attempt applied on the httplib worker thread
                    // and raced the main queue (reverted in ebbbe1116).
                    if (!task.hydra_config_json.empty()) {
                        json hydra_cfg;
                        try {
                            hydra_cfg = json::parse(task.hydra_config_json);
                        } catch (const std::exception & e) {
                            SRV_WRN("hydra: COMPLETION hydra_config parse failed: %s\n", e.what());
                        }
                        if (!hydra_cfg.is_null() && hydra_cfg.is_object()) {
                            SRV_INF("hydra: COMPLETION applying hydra_config (%zu keys)\n",
                                    hydra_cfg.size());
                            hydra_config_result cfg_result = hydra_apply_config(hydra_cfg, /*sync=*/true);
                            if (!cfg_result.ok) {
                                SRV_WRN("hydra: COMPLETION hydra_config apply failed: %s\n",
                                        cfg_result.error.c_str());
                            }
                            // After T3 rebuild, model/slots are reset.
                            // The slot lookup below will pick up the new state.
                        }
                    }

                    const int id_slot = task.id_slot;
                    const int id_task = task.id;

                    server_slot * slot = id_slot != -1 ? get_slot_by_id(id_slot) : get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing() || slot->hydra_transferring->load()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & s : slots) {
                            if (!s.is_processing() && !s.hydra_transferring->load()) {
                                slot_save_and_clear(s);
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = slot.to_json(slots_debug == 0);

                        if (slot.is_processing() || slot.hydra_transferring->load()) {
                            n_processing_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->slots_data          = std::move(slots_data);
                    res->n_idle_slots        = n_idle_slots;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->t_start             = metrics.t_start;

                    res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                    res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                    res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                    res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                    res->n_tokens_max = metrics.n_tokens_max;

                    res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                    res->t_prompt_processing       = metrics.t_prompt_processing;
                    res->n_tokens_predicted        = metrics.n_tokens_predicted;
                    res->t_tokens_generation       = metrics.t_tokens_generation;

                    res->n_decode_total          = metrics.n_decode_total;
                    res->n_busy_slots_total      = metrics.n_busy_slots_total;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }

                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const size_t token_count = slot->prompt.tokens.size();
                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const llama_tokens & tokens = slot->prompt.tokens.get_tokens();
                    const size_t nwrite = llama_state_seq_save_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = token_count;
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    if (!check_no_mtmd(task.id)) break;
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    llama_tokens tokens;
                    tokens.resize(slot->n_ctx);
                    size_t token_count = 0;
                    size_t nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), tokens.size(), &token_count);
                    if (nread == 0) {
                        slot->prompt.tokens.clear(); // KV may already been invalidated?
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    tokens.resize(token_count);
                    slot->prompt.tokens.clear();
                    slot->prompt.tokens.insert(tokens);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = token_count;
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing() || slot->hydra_transferring->load()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear(false);

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_INF("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;

            // epic #610 WS2: HYDRA task dispatch moved to hydra_process_task()
            // (defined in hydra-server-context.cpp). In seam mode the extension
            // claims these via handle_task(); in legacy mode this fall-through
            // calls the same method. Both modes run identical code.
            case SERVER_TASK_TYPE_HYDRA_STATE_GET:
            case SERVER_TASK_TYPE_HYDRA_STATE_PUT:
            case SERVER_TASK_TYPE_HYDRA_STATE_META:
            case SERVER_TASK_TYPE_HYDRA_ENGINE_CONFIGURE:
            case SERVER_TASK_TYPE_HYDRA_ENGINE_INFO:
            case SERVER_TASK_TYPE_HYDRA_ENGINE_PREFILL:
            case SERVER_TASK_TYPE_HYDRA_ENGINE_DECODE:
            case SERVER_TASK_TYPE_HYDRA_DECODE_APPLY:
            case SERVER_TASK_TYPE_HYDRA_ENGINE_SET_EXPERT_MODE:
            case SERVER_TASK_TYPE_HYDRA_ENGINE_SWAP_QUANT:
            case SERVER_TASK_TYPE_HYDRA_ENGINE_PIPELINE_ATTACH:
                hydra_process_task(task);
                break;

        }
    }

    // Hydra #406 (Phase 2b follow-up): apply the staged T2/T3 CONFIGURE
    // rebuild. Called from update_slots() when the slot-free moment
    // arrives (all slots idle, no slot is hydra_transferring).
    //
    // The flow:
    //   1. Check drain timeout (HYDRA_COORD_PROFILE_SWITCH_DRAIN_TIMEOUT,
    //      default 300s). On timeout, discard the staged config and
    //      return — the next INFO call surfaces the cleared state.
    //   2. T2 work (free + rebuild context). Skipped when tier is T3
    //      (T3's load_model() rebuilds the context as a side effect).
    //   3. T3 work (full model reload). Uses the staged T3 statics
    //      (override_tensor / split_mode / tensor_split / n_gpu_layers
    //      / n_cpu_moe / model.path) and falls through to load_model()
    //      for the actual unload+reload cycle. COMBINED-mode bindings
    //      are torn down before the reload and re-attached after.
    //   4. Clear the staged state (T3 statics + pending_config JSON).
    //
    // On any failure: rollback to the pre-apply params_base and rebuild
    // from there. The exception path (GGML_ABORT) is reserved for the
    // catastrophic case where the rollback itself fails — the engine
    // would be unable to serve in any state and must exit.
    bool apply_pending_hydra_config() {
        const bool is_first_load = !ctx_tgt;
        if (is_first_load) {
            if (!first_load_pending) {
                return false;
            }
            // Don't check hydra_has_pending_config — ctx_tgt doesn't exist yet.
            // The T3 statics were staged by hydra_apply_t3_mutators() in the
            // CONFIGURE handler. Set a default tier for the rebuild path.
        } else if (!ctx_tgt->hydra_has_pending_config()) {
            return false;
        }

        // 1. Drain timeout — skipped for first load (no ctx_tgt timestamp).
        std::string tier;
        std::string pending_json;

        if (is_first_load) {
            tier = "T3";
            // pending_json stays empty — T3 statics are staged in global
            // overrides, not in pending_config (ctx_tgt doesn't exist yet).
        } else {
            constexpr time_t k_drain_timeout_default = 300;
            time_t now = std::time(nullptr);
            time_t elapsed = now - ctx_tgt->hydra_get_pending_config_set_at();
            int env_timeout = 0;
            if (const char * e = getenv("HYDRA_COORD_PROFILE_SWITCH_DRAIN_TIMEOUT")) {
                env_timeout = atoi(e);
            }
            time_t drain_timeout = env_timeout > 0 ? env_timeout : k_drain_timeout_default;
            if (elapsed > drain_timeout) {
                SRV_WRN("hydra: pending config drain timeout (elapsed=%lld, limit=%lld) — discarding, "
                        "tier='%s' payload_size=%zu\n",
                        (long long) elapsed, (long long) drain_timeout,
                        ctx_tgt->hydra_get_pending_config_tier().c_str(),
                        ctx_tgt->hydra_get_pending_config().size());
                ctx_tgt->hydra_clear_pending_config();
                llama_hydra_clear_pending_t3();
                // hydra#470: the staged generic (T4) subset must not
                // survive the discard — a stale config would be applied
                // on the next unrelated reload.
                g_pending_reload_config.clear();
                return false;
            }

            tier = ctx_tgt->hydra_get_pending_config_tier();
            pending_json = ctx_tgt->hydra_get_pending_config();
            SRV_INF("hydra: applying pending config (tier='%s', age=%llds, payload_size=%zu)\n",
                    tier.c_str(), (long long) elapsed, pending_json.size());
        }

        bool ok = true;

        // 2. T2 work: free + rebuild context with the new cparams.
        //    Skipped when tier is T3 (T3's load_model() handles both).
        if (tier == "T2") {
            if (!apply_t2_rebuild(pending_json)) {
                SRV_ERR("%s", "hydra: T2 rebuild failed; engine continues with old context\n");
                ok = false;
            }
        }

        // 3. T3 work: full model reload with the staged T3 statics.
        //    load_model() handles the unload+reload cycle. COMBINED-mode
        //    expert bindings are torn down before the reload and re-
        //    attached after, in the same pattern as SET_EXPERT_MODE.
        //    T4 (generic-arg) configs route here too: a model reload is
        //    the only apply that makes EVERY generic key take effect
        //    (speculative types need load_model's MTP/draft setup).
        if (tier == "T3" || tier == "T4") {
            // #470: force rebuild if a peer reconnection was detected
            const bool reconn_force = (ctx_tgt && ctx_tgt->peer_reconnection_pending);
            if (reconn_force) {
                ctx_tgt->peer_reconnection_pending = false;
                SRV_WRN("%s", "hydra: apply_pending: peer reconnection pending — forcing T3 rebuild\n");
            }
            if (!apply_t3_rebuild(reconn_force)) {
                SRV_ERR("%s", "hydra: T3 rebuild failed; engine continues with old model\n");
                ok = false;
            } else {
                // P1-6: T3 model changed — the cached server_context_meta
                // (model_path, split_mode, tensor_split, chat_params, …)
                // is now stale.  Refresh it on the task-queue thread
                // (safe — runs during the drain window when no slots are
                // processing and no new requests are being dispatched).
                if (routes_ptr) {
                    routes_ptr->refresh_meta();
                }
                // P0-1 (#49): after deferred first-load, apply staged capabilities
                // so ENGINE_INFO(0x41) and COMBINED-mode logic work correctly.
                if (is_first_load) {
                    hydra_rpc_backend_active = bootstrap_rpc_active;
                    hydra_peer               = bootstrap_peer;
                    hydra_peer_reachable     = bootstrap_peer_reachable;
                    hydra_combined_pattern   = bootstrap_pattern;
                    hydra_split_mode         = bootstrap_split_mode;
                    if (bootstrap_combined_static) {
                        hydra_combined_static = true;
                        SRV_INF("%s", "P0-1: deferred first-load — combined_static mode activated\n");
                    }
                    // Register local tensors and enable shared-backend compute
                    // lock so the model can serve inbound RPC requests.
                    if (model_tgt && ctx_tgt) {
                        llama_hydra_register_local_tensors_for_rpc(ctx_tgt);
                        llama_hydra_enable_shared_backend_compute_lock();
                    }
                    // Update the RPC server's compute backends now that the
                    // model is loaded. The RPC server was started with empty
                    // backends (head-bootstrap mode); now populate it.
                    if (ctx_tgt) {
                        std::vector<ggml_backend_t> backends(8);
                        size_t n = llama_hydra_get_compute_backends(ctx_tgt, backends.data(), backends.size());
                        if (n > backends.size()) {
                            backends.resize(n);
                            n = llama_hydra_get_compute_backends(ctx_tgt, backends.data(), backends.size());
                        }
                        backends.resize(n);
                        hydra_rpc::update_backends(backends);
                        SRV_INF("P0-1: updated RPC backends to %zu compute device(s)\n", backends.size());
                    }
                    SRV_INF("%s", "hydra-engine ready — model loaded via CONFIGURE T3\n");
                }
            }
        }

        // 4. Clear the staged state regardless of success. On failure
        //    the rollback in apply_t{2,3}_rebuild has restored the
        //    previous state; clearing the staged state prevents the
        //    next slot-free moment from re-attempting the same rebuild.
        if (is_first_load) {
            first_load_pending = false;
        } else {
            ctx_tgt->hydra_clear_pending_config();
        }
        llama_hydra_clear_pending_t3();
        return ok;
    }

    // T2 rebuild: free the live llama_context, rebuild llama_context_params
    // from the updated params_base (n_ctx / cache_type_k / cache_type_v /
    // RoPE / YaRN), recreate the context, re-init per-slot samplers.
    // On failure: rebuild with the old params_base (rollback).
    //
    // Helper: parse a wire-shape cache_type string ("f16" / "q8_0" / ...)
    // to a ggml_type. The wire spec uses llama.cpp's ggml type names.
    // There is no public ggml_parse_type() in upstream llama.cpp, so
    // we iterate ggml_type_traits via ggml_get_type_traits() and
    // match on ggml_type_name().
    static ggml_type hydra_parse_cache_type(const std::string & s) {
        if (s.empty()) return GGML_TYPE_COUNT;
        for (int i = 0; i < GGML_TYPE_COUNT; i++) {
            ggml_type t = (ggml_type) i;
            if (strcmp(ggml_type_name(t), s.c_str()) == 0) return t;
        }
        return GGML_TYPE_COUNT;
    }

    // hydra#470: apply the staged context-level (T2) + generic (T4) keys
    // from `cfg` to `params`. Shared by apply_t2_rebuild() (params_base,
    // before the context rebuild) and apply_t3_rebuild() (swapped_params,
    // before load_model()) so a mixed T2+T4 payload is handled identically
    // on both reload paths — the model-reload path consumes the same staged
    // config as the context-reload path and no T2 key is stranded.
    //
    // The explicit T2 keys keep their custom semantics (n_ctx clamp to
    // model_n_ctx_train, cache_type validation); every other key goes
    // through the llama.cpp arg table (hydra_apply_generic_key). A
    // present-but-unusable value is logged with SRV_WRN — never silent.
    void hydra_apply_staged_context_keys(common_params & params, const json & cfg) {
        // ── explicit T2 keys (context-level); each is optional, absence
        //    means "leave unchanged" ──
        if (cfg.contains("n_ctx") && cfg["n_ctx"].is_number_integer()) {
            const int32_t n_ctx = cfg["n_ctx"].get<int32_t>();
            if (model_tgt) {
                // Clamp to the model's training ctx. The wire spec does
                // not require a reject-on-too-large; we clamp and report.
                // At first load (model_tgt null) the value is used as-is
                // and llama.cpp validates it during context creation.
                const int32_t max_ctx = (int32_t) llama_model_n_ctx_train(model_tgt);
                if (n_ctx > max_ctx) {
                    SRV_WRN("hydra: n_ctx=%d exceeds model_n_ctx_train=%d; clamping\n",
                            n_ctx, max_ctx);
                    params.n_ctx = max_ctx;
                } else {
                    params.n_ctx = n_ctx;
                }
            } else {
                params.n_ctx = n_ctx;
            }
        }
        if (cfg.contains("cache_type_k") && cfg["cache_type_k"].is_string()) {
            const std::string & s = cfg["cache_type_k"].get_ref<const std::string &>();
            ggml_type t = hydra_parse_cache_type(s);
            if (t == GGML_TYPE_COUNT) {
                SRV_WRN("hydra: cache_type_k='%s' unparseable; ignoring\n", s.c_str());
            } else {
                params.cache_type_k = t;
            }
        }
        if (cfg.contains("cache_type_v") && cfg["cache_type_v"].is_string()) {
            const std::string & s = cfg["cache_type_v"].get_ref<const std::string &>();
            ggml_type t = hydra_parse_cache_type(s);
            if (t == GGML_TYPE_COUNT) {
                SRV_WRN("hydra: cache_type_v='%s' unparseable; ignoring\n", s.c_str());
            } else {
                params.cache_type_v = t;
            }
        }
        if (cfg.contains("rope_freq_base") && cfg["rope_freq_base"].is_number()) {
            params.rope_freq_base = cfg["rope_freq_base"].get<float>();
        }
        if (cfg.contains("rope_freq_scale") && cfg["rope_freq_scale"].is_number()) {
            params.rope_freq_scale = cfg["rope_freq_scale"].get<float>();
        }
        if (cfg.contains("yarn_ext_factor") && cfg["yarn_ext_factor"].is_number()) {
            params.yarn_ext_factor = cfg["yarn_ext_factor"].get<float>();
        }
        if (cfg.contains("yarn_attn_factor") && cfg["yarn_attn_factor"].is_number()) {
            params.yarn_attn_factor = cfg["yarn_attn_factor"].get<float>();
        }
        if (cfg.contains("yarn_beta_fast") && cfg["yarn_beta_fast"].is_number()) {
            params.yarn_beta_fast = cfg["yarn_beta_fast"].get<float>();
        }
        if (cfg.contains("yarn_beta_slow") && cfg["yarn_beta_slow"].is_number()) {
            params.yarn_beta_slow = cfg["yarn_beta_slow"].get<float>();
        }
        if (cfg.contains("yarn_orig_ctx") && cfg["yarn_orig_ctx"].is_number_integer()) {
            params.yarn_orig_ctx = cfg["yarn_orig_ctx"].get<int32_t>();
        }

        // ── generic (T4) fallback: every key the explicit T2 code above
        //    did NOT handle is applied via the llama.cpp arg table. This
        //    also covers special-classified keys with no explicit handling
        //    (rope_scale / rope_scaling classify as T2 via the rope_*
        //    prefix but have no dedicated code above). ──
        static const std::set<std::string> t2_handled = {
            "n_ctx", "cache_type_k", "cache_type_v",
            "rope_freq_base", "rope_freq_scale",
            "yarn_ext_factor", "yarn_attn_factor",
            "yarn_beta_fast", "yarn_beta_slow", "yarn_orig_ctx",
        };
        for (auto it = cfg.begin(); it != cfg.end(); ++it) {
            const std::string & key = it.key();
            if (t2_handled.count(key) > 0) {
                continue;  // explicit T2 code above
            }
            if (!hydra_apply_generic_key(params, key, it.value())) {
                SRV_WRN("hydra: staged context key '%s' not applied (see log)\n", key.c_str());
            }
        }

        // ── present-but-unusable T2 values must be loud, not silent ──
        for (const auto & key : t2_handled) {
            if (!cfg.contains(key)) {
                continue;
            }
            const json & v = cfg[key];
            const bool usable =
                (key == "n_ctx"      || key == "yarn_orig_ctx") ? v.is_number_integer() :
                (key == "cache_type_k" || key == "cache_type_v") ? v.is_string() :
                v.is_number();
            if (!usable) {
                SRV_WRN("hydra: T2 key '%s' has unusable value (JSON type %s) — value ignored\n",
                        key.c_str(), v.type_name());
            }
        }
    }

    bool apply_t2_rebuild(const std::string & pending_json) {
        if (!ctx_tgt || !model_tgt) return false;

        json cfg;
        try {
            cfg = json::parse(pending_json);
        } catch (const std::exception & e) {
            SRV_WRN("hydra: T2 apply: invalid JSON in pending_config: %s\n", e.what());
            return false;
        }

        // Snapshot the old params for rollback. params_base is the
        // canonical "what's in effect" state; restoring it plus a
        // recreate-cycle is the rollback path.
        common_params old_params = params_base;

        // hydra#470: apply the staged context-level + generic keys to
        // params_base (explicit T2 keys + arg-table fallback). Shared with
        // apply_t3_rebuild() so mixed T2+T4 payloads are handled identically
        // on both reload paths.
        hydra_apply_staged_context_keys(params_base, cfg);
        // The staged reload config was consumed above (its keys ride the
        // pending_config JSON); clear the static so a later T3 reload does
        // not re-apply stale values. The "last applied" marker is updated
        // only on the success path below.
        const std::string consumed_reload_config = g_pending_reload_config;
        g_pending_reload_config.clear();

        // Free the live context. KV cache is destroyed; this is the
        // T2 cost. The model is kept (T2 is context-only).
        llama_free(ctx_tgt);
        if (ctx_dft) {
            llama_free(ctx_dft.get());
            ctx_dft.reset();
        }

        // Build new cparams from the updated params_base. This is
        // the same call site load_model() uses internally.
        auto cparams = common_context_params_to_llama(params_base);

        // Recreate the context with the new cparams.
        ctx_tgt = llama_new_context_with_model(model_tgt, cparams);
        if (!ctx_tgt) {
            // Rollback: rebuild with the old params_base. The old
            // params must work (we just freed and recreated the
            // context with them). If they don't, the engine is in
            // a bad state — abort.
            SRV_WRN("hydra: T2 rebuild failed with n_ctx=%d cache_type=%d/%d; "
                    "rolling back to old params\n",
                    params_base.n_ctx, (int) params_base.cache_type_k,
                    (int) params_base.cache_type_v);
            params_base = old_params;
            auto cparams_old = common_context_params_to_llama(params_base);
            ctx_tgt = llama_new_context_with_model(model_tgt, cparams_old);
            if (!ctx_tgt) {
                GGML_ABORT("hydra: T2 rollback failed (cannot rebuild context with old params). "
                           "Engine exiting to prevent serving with corrupted state.");
            }
            return false;
        }

        // Re-init per-slot samplers. The old samplers were bound to
        // the now-freed context; common_sampler_init() on the new
        // model picks up the (possibly changed) sampling config.
        for (auto & slot : slots) {
            slot.smpl.reset(common_sampler_init(model_tgt, params_base.sampling));
        }

        n_ctx = llama_n_ctx(ctx_tgt);
        // hydra#470: record the staged reload config as applied. A later
        // T3-tier config carrying the same staged keys can then skip the
        // model reload via the early-exit (the values are already in
        // params_base, which feeds swapped_params).
        g_last_reload_config_applied = consumed_reload_config;
        SRV_INF("hydra: T2 rebuild applied (n_ctx=%d, cache=%d/%d, slots=%zu)\n",
                n_ctx, (int) params_base.cache_type_k,
                (int) params_base.cache_type_v, slots.size());
        return true;
    }

    // Register new RPC peer devices into the global ggml backend registry.
    // Called from apply_t3_rebuild() before load_model() so the new peer's
    // device exists when common_init_from_params() tries to place tensors
    // per tensor_split/split_mode.
    //
    // Repeated registration is NOT safe/idempotent in the underlying API,
    // so we track already-registered endpoints in a static set and only
    // register genuinely new ones.
    static void hydra_register_rpc_servers(const json & servers_arr) {
        static std::set<std::string> registered;

        if (!servers_arr.is_array() || servers_arr.empty()) {
            return;
        }

        ggml_backend_load_all();
        ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
        if (!rpc_reg) {
            SRV_WRN("%s", "hydra: rpc_servers: RPC backend not available\n");
            return;
        }

        typedef ggml_backend_reg_t (*ggml_backend_rpc_add_server_t)(const char * endpoint);
        auto add_server_fn = (ggml_backend_rpc_add_server_t)
            ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
        if (!add_server_fn) {
            SRV_WRN("%s", "hydra: rpc_servers: ggml_backend_rpc_add_server not found\n");
            return;
        }

        for (const auto & v : servers_arr) {
            if (!v.is_string()) continue;
            const std::string endpoint = v.get<std::string>();
            if (endpoint.empty()) continue;
            if (registered.count(endpoint)) {
                SRV_DBG("hydra: rpc_servers: endpoint '%s' already registered, skipping\n",
                        endpoint.c_str());
                continue;
            }
            ggml_backend_reg_t reg = add_server_fn(endpoint.c_str());
            if (reg) {
                ggml_backend_register(reg);
                registered.insert(endpoint);
                SRV_INF("hydra: rpc_servers: registered endpoint '%s'\n", endpoint.c_str());
            } else {
                SRV_WRN("hydra: rpc_servers: failed to register endpoint '%s'\n",
                        endpoint.c_str());
            }
        }
    }

    // #470: refresh the T3-current alias → file map after a successful
    // model load. The map's keys are the engine's own identity aliases
    // (model_name + model_aliases, as recomputed by load_model()); the
    // value is the resident file. Called from load_model() right after the
    // identity members are re-derived, so the map always tracks the CURRENT
    // resident — covering boot, apply_t3_rebuild(), the bare-alias swap
    // paths and T3 rollback alike.
    //
    // Why this map exists: preset_alias_to_path is the static INI mapping,
    // but the coordinator's T3 config (hydra_config.model_path) can load a
    // file that the INI does NOT associate with the engine's current alias
    // (e.g. the dense-27b-combined session T3-loads the 27B-Coder file
    // while the engine's identity still says qwen3.6-35B-balanced from an
    // earlier SOLO session). DECODE_APPLY must know that the requested
    // alias already refers to the resident before it decides to swap.
    void hydra_t3_record_current_alias_to_path() {
        t3_current_alias_to_path.clear();
        const std::string & resident = params_base.model.path;
        t3_current_alias_to_path[model_name] = resident;
        for (const auto & alias : model_aliases) {
            t3_current_alias_to_path[alias] = resident;
        }
        SRV_DBG("hydra: T3-current alias→file map recorded %zu alias(es) → '%s'\n",
                t3_current_alias_to_path.size(), resident.c_str());
    }

    // Tear down COMBINED-mode RPC peer bindings before a model reload.
    // Shared by apply_t3_rebuild() and the bare-alias swap paths (PREFILL,
    // DECODE_APPLY, server-context.cpp ~3800 / ~4310). Must run BEFORE
    // load_model() — otherwise the new ctx_tgt (post-reload) inherits a
    // stale binding to the old peer's device. The bare-alias paths used to
    // skip this entirely: the engine loaded the correct model file but kept
    // routing tokens through the old COMBINED config, which is #514
    // (throughput collapses to ~2-4 tok/s after a dynamic model swap).
    void hydra_teardown_combined_before_reload() {
        SRV_INF("hydra: tearing down COMBINED before model reload (was head_attached=%d, static=%d)\n",
                (int) hydra_combined_head_attached, (int) hydra_combined_static);
        llama_hydra_set_expert_mode(ctx_tgt, 0);
        if (!hydra_current_peer.empty()) {
            ctx_tgt->hydra_remove_combined_rpc_backend(hydra_current_peer.c_str());
        }
        llama_hydra_clear_combined_bindings(ctx_tgt, hydra_peer.c_str());
        hydra_combined_head_attached = false;
    }

    // epic #610 WS2: HYDRA task dispatch — declared here, defined in
    // hydra-server-context.cpp (included at the bottom of this TU). Keeps a
    // switch(task.type) wrapper so break/continue semantics are unchanged.
    void hydra_process_task(server_task & task);

    // Re-attach COMBINED-mode bindings after a model reload, mirroring
    // hydra_teardown_combined_before_reload() above. Layer-split (static)
    // just re-enables the mode flag — load_model() already preloaded the
    // peer device with the new tensor_split. Expert-split re-resolves the
    // peer's RPC device and rebinds the expert tensors; same fail-open
    // pattern as SET_EXPERT_MODE — if the peer is unreachable, the engine
    // stays solo and the coordinator's solo-fallback path handles it.
    void hydra_reattach_combined_after_reload() {
        SRV_INF("%s", "hydra: re-attaching COMBINED on new model\n");
        if (hydra_combined_static) {
            llama_hydra_set_expert_mode(ctx_tgt, 1);
        } else if (!hydra_peer.empty() && !hydra_combined_pattern.empty()) {
            if (llama_hydra_peer_reachable(hydra_peer.c_str())) {
                ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
                if (rpc_reg) {
                    using add_server_fn_t = ggml_backend_reg_t (*)(const char *);
                    auto add_server_fn = (add_server_fn_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
                    ggml_backend_reg_t peer_reg = add_server_fn ? add_server_fn(hydra_peer.c_str()) : nullptr;
                    ggml_backend_dev_t  peer_dev = (peer_reg && ggml_backend_reg_dev_count(peer_reg) > 0) ? ggml_backend_reg_dev_get(peer_reg, 0) : nullptr;
                    if (peer_dev) {
                        int32_t n_bound = llama_hydra_rebind_combined_experts(
                                ctx_tgt, hydra_peer.c_str(), peer_dev, hydra_combined_pattern.c_str());
                        if (n_bound > 0) {
                            hydra_combined_head_attached = true;
                            llama_hydra_set_expert_mode(ctx_tgt, 1);
                            SRV_INF("hydra: COMBINED re-attached on peer %s (%d layers bound)\n",
                                    hydra_peer.c_str(), n_bound);
                        } else {
                            SRV_WRN("hydra: rebind returned %d; staying solo\n", n_bound);
                        }
                    } else {
                        SRV_WRN("hydra: peer %s has no device; staying solo\n", hydra_peer.c_str());
                    }
                } else {
                    SRV_WRN("%s\n", "hydra: RPC backend not available; staying solo");
                }
            } else {
                SRV_WRN("hydra: peer %s unreachable; staying solo\n", hydra_peer.c_str());
            }
        }
    }

    // Re-pad tensor_buft_overrides to the nullptr-terminated capacity
    // llama_max_tensor_buft_overrides() after a preset's apply_to_params()
    // has push_back()'d entries onto a freshly-cleared vector (see the
    // bare-alias swap paths, server-context.cpp ~3800 / ~4310). Caps at
    // the limit with the same guard the override_tensor T3 path already
    // has (below) — apply_to_params() push_backs unconditionally, so an
    // unusually large preset could otherwise overflow the same 4096-entry
    // limit this whole clear/re-pad dance exists to respect.
    void hydra_repad_tensor_buft_overrides(common_params & p, const char * ctx_label) {
        const size_t ntbo = llama_max_tensor_buft_overrides();
        if (p.tensor_buft_overrides.size() + 1 > ntbo) {
            SRV_WRN("hydra: %s: %zu tensor_buft_overrides exceed the %zu-entry limit; keeping the first %zu\n",
                    ctx_label, p.tensor_buft_overrides.size(), ntbo, ntbo - 1);
            p.tensor_buft_overrides.resize(ntbo - 1);
        }
        p.tensor_buft_overrides.resize(ntbo, llama_model_tensor_buft_override{ nullptr, nullptr });
    }

    // T3 rebuild: full model reload. Uses the staged T3 statics
    // (override_tensor, split_mode, tensor_split, n_gpu_layers,
    // n_cpu_moe, model.path) populated by hydra_apply_t3_mutators().
    // Falls through to load_model() for the actual unload+reload
    // cycle (which handles mmproj, MTP/draft, slot rebuild, etc.).
    // On failure: rollback by reloading the old params_base.
    bool apply_t3_rebuild(bool force = false) {
        bool is_first_load = !ctx_tgt;

        // Track the last override_tensor string that was actually
        // applied so we can detect "nothing changed" on subsequent
        // calls and skip the expensive unload+reload cycle.
        static std::string old_override_applied;

        common_params old_params = params_base;
        common_params swapped_params = params_base;

        // Read the staged T3 statics and apply them to swapped_params.
        if (llama_hydra_get_pending_n_gpu_layers() >= 0) {
            swapped_params.n_gpu_layers = llama_hydra_get_pending_n_gpu_layers();
        }
        // n_cpu_moe is informational only — the actual MoE expert
        // offload is done via override_tensor (parsed below into
        // tensor_buft_overrides). The standard common_params struct
        // has no n_cpu_moe field; we just log the staged value for
        // operator visibility.
        if (llama_hydra_get_pending_n_cpu_moe() >= 0) {
            SRV_INF("hydra: T3 rebuild: staged n_cpu_moe=%d (informational; expert routing via override_tensor)\n",
                    llama_hydra_get_pending_n_cpu_moe());
        }
        const char * path = llama_hydra_get_pending_model_path();
        if (path && *path) {
            swapped_params.model.path = path;
        }
        const char * mode = llama_hydra_get_pending_split_mode();
        if (mode && *mode) {
            std::string m(mode);
            if (m == "none")      swapped_params.split_mode = LLAMA_SPLIT_MODE_NONE;
            else if (m == "layer") swapped_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
            else if (m == "row")   swapped_params.split_mode = LLAMA_SPLIT_MODE_ROW;
            else SRV_WRN("hydra: T3 split_mode='%s' unknown; keeping current\n", m.c_str());
        }
        const size_t n_split = llama_hydra_get_pending_tensor_split_count();
        if (n_split > 0) {
            const float * split = llama_hydra_get_pending_tensor_split();
            // common_params::tensor_split is a fixed-size array.
            const size_t cap = sizeof(swapped_params.tensor_split) /
                                sizeof(swapped_params.tensor_split[0]);
            const size_t n = n_split < cap ? n_split : cap;
            for (size_t i = 0; i < n; i++) {
                swapped_params.tensor_split[i] = split[i];
            }
            // Zero the rest so the engine doesn't see stale values.
            for (size_t i = n; i < cap; i++) {
                swapped_params.tensor_split[i] = 0.0f;
            }
        }
        const char * override = llama_hydra_get_pending_override_tensor();
        if (override && *override) {
            // Wire-shape: comma-separated "pattern=buft" pairs (e.g.
            // "blk.*.ffn_*_exps.weight=CPU"). The C++ side stores
            // these as a vector<llama_model_tensor_buft_override>.
            // Buft names are looked up
            // via ggml_backend_dev_buffer_type() + ggml_backend_buft_name()
            // (mirrors common/arg.cpp:parse_tensor_buffer_overrides).
            ggml_backend_load_all();
            std::map<std::string, ggml_backend_buffer_type_t> buft_list;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto * dev = ggml_backend_dev_get(i);
                auto * buft = ggml_backend_dev_buffer_type(dev);
                if (buft) {
                    buft_list[std::string(ggml_backend_buft_name(buft))] = buft;
                }
            }
            // CPU is the common case (MoE expert routing) — also lookup
            // explicitly since some backends may not register the CPU buft.
            buft_list["CPU"] = ggml_backend_cpu_buffer_type();

            // Keep pattern strings alive for the lifetime of the
            // process — entry.pattern is a const char* that must not
            // dangle.  Matches the safe pattern in common/arg.cpp.
            static std::list<std::string> buft_override_patterns;

            std::vector<llama_model_tensor_buft_override> staged;

            const std::string ovr(override);
            size_t start = 0;
            while (start < ovr.size()) {
                size_t comma = ovr.find(',', start);
                std::string part = ovr.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                size_t eq = part.find('=');
                if (eq != std::string::npos) {
                    std::string pattern = part.substr(0, eq);
                    std::string buft_name = part.substr(eq + 1);
                    auto it = buft_list.find(buft_name);
                    if (it != buft_list.end()) {
                        buft_override_patterns.push_back(pattern);
                        llama_model_tensor_buft_override entry;
                        entry.pattern = buft_override_patterns.back().c_str();
                        entry.buft = it->second;
                        staged.push_back(entry);
                    } else {
                        SRV_WRN("%s", "hydra: T3 rebuild: override_tensor buft name not in registered list; skipping pattern\n");
                    }
                }
                if (comma == std::string::npos) break;
                start = comma + 1;
            }

            // Install the staged patterns *in place of* the base ones instead of
            // appending to them.
            //
            // common_params_parse_ex() (common/arg.cpp) unconditionally pads this
            // vector out to llama_max_tensor_buft_overrides() entries of
            // {nullptr, nullptr}, so by the time we get here the real CLI overrides
            // sit at the head and the rest is terminator padding. push_back() would
            // land *behind* that padding, which breaks twice over:
            //   1. common_model_params_to_llama() asserts that back().pattern is
            //      nullptr, so the engine aborts before the model loads;
            //   2. even without that assert, llama_model_loader stops scanning at
            //      the first nullptr pattern, so appended entries are never read —
            //      the override would be silently dropped and the MoE experts would
            //      land on the GPU.
            // Replacing also matches the sibling fields handled above: model.path,
            // split_mode, n_gpu_layers and tensor_split are all overwritten by the
            // staged T3 config rather than merged into it.
            const size_t ntbo = llama_max_tensor_buft_overrides();
            if (staged.empty()) {
                // Nothing resolved (every buft name was unknown). Wiping the base
                // overrides here would silently change how the model is placed, so
                // keep them and make the no-op explicit.
                SRV_WRN("%s", "hydra: T3 rebuild: staged override_tensor resolved to no usable patterns; keeping base overrides\n");
            } else {
                if (staged.size() + 1 > ntbo) {
                    SRV_WRN("hydra: T3 rebuild: %zu override_tensor patterns exceed the %zu-entry limit; keeping the first %zu\n",
                            staged.size(), ntbo, ntbo - 1);
                    staged.resize(ntbo - 1);
                }
                // assign() re-establishes the full terminator padding, so everything
                // from staged.size() onward is {nullptr, nullptr}.
                swapped_params.tensor_buft_overrides.assign(ntbo, llama_model_tensor_buft_override{ nullptr, nullptr });
                for (size_t i = 0; i < staged.size(); ++i) {
                    swapped_params.tensor_buft_overrides[i] = staged[i];
                }
            }
        }

        // hydra#470: apply the staged reload config (context-level T2 keys
        // + generic T4 keys) to swapped_params BEFORE any model (re)load —
        // they must land in common_params before load_model() consumes them
        // (n_ctx, cache types, speculative types, ...). The staged JSON is
        // consumed here (cleared), so a later T2/T3 apply does not re-apply
        // stale values. Mixed T2+T4 payloads are applied by the same helper
        // the context-reload path uses — nothing is stranded.
        std::string staged_reload = g_pending_reload_config;
        g_pending_reload_config.clear();
        if (!staged_reload.empty()) {
            json reload_cfg;
            try {
                reload_cfg = json::parse(staged_reload);
            } catch (const std::exception & e) {
                SRV_WRN("hydra: T3 rebuild: staged reload config failed to parse: %s\n", e.what());
                staged_reload.clear();
            }
            if (!reload_cfg.is_null()) {
                hydra_apply_staged_context_keys(swapped_params, reload_cfg);
            }
        }

        // Early-exit: if the model, all T3-relevant params AND the staged
        // reload config (T2 + T4 keys) are identical to what is already
        // loaded, skip the expensive unload+reload cycle.  Without this,
        // every COMPLETION request that carries hydra_config triggers a
        // full model swap even when nothing changed (the coordinator sends
        // the same config on every decode request). The comparison is
        // against the last staged dump that was actually loaded — identical
        // config → identical dump → skip; a changed T2 or T4 key → forced
        // reload (no masked changes).
        const bool reload_unchanged = (staged_reload == g_last_reload_config_applied);
        if (!is_first_load) {
            const char * cur_override = llama_hydra_get_pending_override_tensor();
            bool params_unchanged =
                swapped_params.model.path == old_params.model.path &&
                swapped_params.n_gpu_layers == old_params.n_gpu_layers &&
                swapped_params.split_mode == old_params.split_mode &&
                ((cur_override == nullptr && old_override_applied.empty()) ||
                 (cur_override && old_override_applied == cur_override));
            if (params_unchanged && reload_unchanged && !force) {
                // T3 overrides (override_tensor, split_mode) were staged by
                // the COMPLETION hydra_config path. But the model reload is
                // being skipped. Clear the staged override so the next decode
                // uses the current tensor placement (not the staged override).
                llama_hydra_set_override_tensor(ctx_tgt, nullptr);
                SRV_INF("%s", "hydra: T3 rebuild: model and params unchanged — skipping reload, cleared staged overrides\n");
                return true;
            }
        }

        // COMBINED-mode teardown BEFORE the model reload — see
        // hydra_teardown_combined_before_reload() above.
        const bool was_combined = hydra_combined_head_attached || hydra_combined_static;
        if (!is_first_load && was_combined) {
            hydra_teardown_combined_before_reload();
        }

        // Register any new RPC peer devices before load_model() so the
        // peer's device exists in the global ggml backend registry when
        // common_init_from_params() tries to place tensors per
        // tensor_split/split_mode. Only genuinely new endpoints are
        // registered (hydra_register_rpc_servers tracks already-registered
        // endpoints to avoid unsafe repeated registration).
        if (!g_pending_rpc_servers.empty()) {
            json rpc_arr = json::array();
            for (const auto & s : g_pending_rpc_servers) {
                rpc_arr.push_back(s);
            }
            hydra_register_rpc_servers(rpc_arr);
            g_pending_rpc_servers.clear();
        }

        // Full model reload. load_model() handles the unload of the
        // current model, the load of the new model, the new context
        // creation, the MTP/draft paths, and the slot rebuild.
        // NOTE: load_model() does `params_base = params` internally
        // (line 844), so after a successful load params_base reflects
        // swapped_params — no explicit reassignment needed by us.
        //
        // #507: Skip the fit_params probe during T3 rebuild. The probe
        // does a full model-structure load with no_alloc=true to measure
        // GPU memory — expensive (~45-90s) and unnecessary here because:
        // (a) we just freed VRAM by destroying the old model, (b) the new
        // model's requirements are known (same or smaller), (c) a controlled
        // inference server has predictable VRAM. Disabling saves ~1 min.
        swapped_params.fit_params = false;
        if (!load_model(swapped_params)) {
            if (is_first_load) {
                SRV_WRN("%s", "hydra: T3 first load failed — engine stays empty\n");
                return false;
            }
            SRV_ERR("hydra: T3 reload to '%s' failed (load_model returned false); "
                    "rolling back to old model\n",
                    swapped_params.model.path.c_str());
            if (!load_model(old_params)) {
                SRV_ERR("%s", "hydra: T3 rollback also failed — engine in unrecoverable state\n");
                GGML_ABORT("hydra: T3 rollback failed (cannot reload old model). "
                           "Engine exiting to prevent serving with corrupted state.");
            }
            SRV_INF("hydra: T3 rollback succeeded — restored old model '%s'\n",
                    old_params.model.path.c_str());
            return false;
        }

        // Record the override_tensor that was just applied so the
        // next call can skip the reload if nothing changed.
        {
            const char * cur = llama_hydra_get_pending_override_tensor();
            old_override_applied = cur ? cur : "";
            // hydra#470: also record the staged reload config (T2+T4 keys)
            // that was just loaded, so a repeated identical CONFIGURE/decode
            // payload skips the reload (early-exit above).
            g_last_reload_config_applied = staged_reload;
        }

        // T3 reload confirmed. Log model identity for traceability.
        SRV_INF("hydra: T3 reload confirmed model_alias='%s' tokenizer='%s' model_name='%s' quant='%s' caps=0x%x model_path='%s'\n",
                swapped_params.model_alias.empty() ? "?" : swapped_params.model_alias.begin()->c_str(),
                model_tgt ? llama_model_get_tokenizer_model(model_tgt) : "",
                model_tgt ? llama_model_get_display_name(model_tgt) : "",
                model_tgt ? llama_model_get_quant_label(model_tgt) : "",
                model_tgt ? llama_model_get_capabilities_bitfield(model_tgt) : 0,
                swapped_params.model.path.c_str());

        // COMBINED-mode reattach AFTER the model reload — see
        // hydra_reattach_combined_after_reload() above.
        if (was_combined) {
            hydra_reattach_combined_after_reload();
        }

        SRV_INF("hydra: T3 rebuild applied (model='%s', split_mode=%d, n_gpu_layers=%d, slots=%zu)\n",
                params_base.model.path.c_str(), (int) params_base.split_mode,
                params_base.n_gpu_layers, slots.size());
        return true;
    }

    void update_slots() {
        // epic #610 WS1: in seam mode the extension may pre-empt the decode
        // loop (T3/CONFIGURE/reattach cluster). WS1 impl is a no-op — this is
        // a pure A/B switch, both modes run the inline body below.
        if (hydra_ext_active && hydra_ext && hydra_ext->pre_loop(*this)) {
            return;
        }
        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing() || slot.hydra_transferring->load()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_INF("%s", "all slots are idle\n");

                // Hydra #406: slot-free moment — if a tiered CONFIGURE
                // staged a T2/T3 rebuild, run the apply step now. The
                // apply step (in apply_pending_hydra_config below) does
                // the actual T2 context rebuild and/or T3 model reload,
                // then clears the staged state. The low-level helper
                // llama_hydra_apply_pending_config() is a no-op once the
                // staged state has been cleared.
                if (ctx_tgt && ctx_tgt->hydra_has_pending_config()) {
                    SRV_INF("hydra: slot-free moment — applying pending CONFIGURE (tier=%s)\n",
                            ctx_tgt->hydra_get_pending_config_tier().c_str());
                    apply_pending_hydra_config();
                } else if (!ctx_tgt && first_load_pending) {
                    SRV_INF("%s", "hydra: slot-free moment — first load (no context yet)\n");
                    apply_pending_hydra_config();
                }

                // #470 Option B: check if a peer reconnection was detected
                // during graph_compute. If so, trigger a T3 rebuild to
                // re-provision model layers on the fresh peer.
                if (ctx_tgt && ctx_tgt->peer_reconnection_pending) {
                    ctx_tgt->peer_reconnection_pending = false;
                    SRV_WRN("%s", "hydra: peer reconnection detected — triggering T3 rebuild\n");
                    // Force a T3 rebuild even if model config hasn't changed.
                    // The peer's buffers are gone, so we need to re-push.
                    if (!apply_t3_rebuild(true)) {
                        SRV_ERR("%s", "hydra: T3 rebuild after peer reconnection failed\n");
                    }
                }

                return;
            }
        }

        {
            SRV_DBG("%s", "posting NEXT_RESPONSE\n");

            server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            task.id = queue_tasks.get_new_id();
            queue_tasks.post(std::move(task));
        }

        // apply context-shift if needed
        // TODO: simplify and improve
        for (server_slot & slot : slots) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                const int n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                common_context_seq_rm (ctx_tgt, slot.id, n_keep            , n_keep + n_discard);
                common_context_seq_add(ctx_tgt, slot.id, n_keep + n_discard, slot.prompt.n_tokens(), -n_discard);

                // D4: seq_rm invalidates restored logits
                slot.logits_valid = false;
                slot.restored_logits.clear();

                if (ctx_dft) {
                    common_context_seq_rm (ctx_dft.get(), slot.id, n_keep            , n_keep + n_discard);
                    common_context_seq_add(ctx_dft.get(), slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);
                }

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.tokens.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        }

        // start populating the batch for this iteration
        common_batch_clear(batch);

        // track if given slot can be batched with slots already in the batch
        server_slot * slot_batched = nullptr;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING) {
                continue;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                continue;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .n_past   = */ slot.prompt.n_tokens(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        }

        // generate the actual drafts (if any)
        {
            common_speculative_draft(spec.get());
        }

        // make checkpoints if needed
        for (auto * slot_ptr : drafting) {
            auto & slot = *slot_ptr;

            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.n_draft_total += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                }

                common_context_seq_rm(ctx_dft.get(), slot.id, ckpt.pos_max + 1, -1);
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft.get()));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                }
            }
        }

        // update the batch with the sampled/drafted tokens
        for (auto * slot_ptr : generating) {
            auto & slot = *slot_ptr;

            slot.update_batch(batch);
        }

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        float  alora_scale       = -1.0f;
        size_t alora_disabled_id = 0;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.n_tokens == 0) {
            for (auto & slot : slots) {
                if (!slot.is_processing()) {
                    continue;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    continue;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    continue;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // #469 trace: log input tokens for cross-flow comparison
                    {
                        std::string tok_ids;
                        for (size_t i = 0; i < std::min<size_t>(16, input_tokens.size()); ++i) {
                            if (i > 0) tok_ids += ",";
                            tok_ids += std::to_string(input_tokens[i]);
                        }
                        SLT_DBG(slot, "#PD-TRACE HTTP_COMPLETION slot=%d input_tokens_first16=[%s] input_total=%zu cached=%d just_restored=%d\n",
                                slot.id, tok_ids.c_str(), input_tokens.size(),
                                slot.n_prompt_tokens_cache, slot.just_restored);
                    }

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.n_tokens;

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            continue;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                continue;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // #469 trace: log common prefix for cross-flow comparison
                                SLT_DBG(slot, "#PD-TRACE COMMON_PREFIX slot=%d n_past=%d cached=%d input_total=%zu just_restored=%d\n",
                                        slot.id, n_past, slot.n_prompt_tokens_cache, input_tokens.size(), slot.just_restored);
                                {
                                    // Log cached tokens if any
                                    if (slot.n_prompt_tokens_cache > 0) {
                                        std::string cached_ids;
                                        for (int i = 0; i < std::min<int>(16, slot.n_prompt_tokens_cache); ++i) {
                                            if (i > 0) cached_ids += ",";
                                            cached_ids += std::to_string(slot.prompt.tokens[i]);
                                        }
                                        SLT_DBG(slot, "#PD-TRACE CACHED_TOKENS slot=%d first16=[%s] total=%d\n",
                                                slot.id, cached_ids.c_str(), slot.n_prompt_tokens_cache);
                                    }
                                    // Log first mismatch point when common prefix < input size
                                    if (n_past < (int)input_tokens.size()) {
                                        if (n_past < (int)slot.prompt.tokens.size()) {
                                            SLT_WRN(slot, "#PD-TRACE MISMATCH slot=%d n_past=%d cached=%d input_total=%zu stored_tok[%d]=%d input_tok[%d]=%d\n",
                                                    slot.id, n_past, slot.n_prompt_tokens_cache, input_tokens.size(),
                                                    n_past, slot.prompt.tokens[n_past],
                                                    n_past, input_tokens[n_past]);
                                        } else {
                                            SLT_WRN(slot, "#PD-TRACE MISMATCH slot=%d n_past=%d cached=%d input_total=%zu stored_size=%zu input exceeds stored\n",
                                                    slot.id, n_past, slot.n_prompt_tokens_cache, input_tokens.size(),
                                                    slot.prompt.tokens.size());
                                        }
                                        // Log first 16 of both token lists for comparison
                                        {
                                            std::string stored_ids, input_ids;
                                            for (int i = 0; i < std::min<int>(16, (int)slot.prompt.tokens.size()); ++i) {
                                                if (i > 0) stored_ids += ",";
                                                stored_ids += std::to_string(slot.prompt.tokens[i]);
                                            }
                                            for (size_t i = 0; i < std::min<size_t>(16, input_tokens.size()); ++i) {
                                                if (i > 0) input_ids += ",";
                                                input_ids += std::to_string(input_tokens[i]);
                                            }
                                            SLT_WRN(slot, "#PD-TRACE MISMATCH_STORED slot=%d first16=[%s] total=%zu\n",
                                                    slot.id, stored_ids.c_str(), slot.prompt.tokens.size());
                                            SLT_WRN(slot, "#PD-TRACE MISMATCH_INPUT slot=%d first16=[%s] total=%zu\n",
                                                    slot.id, input_ids.c_str(), input_tokens.size());
                                        }
                                    }
                                }

                                // ── Hydra n_common decision rule ──────────────
                                // n_slot = tokens resident in KV (blob, prior turn, or prefill).
                                // n_new  = tokens in the incoming prompt.
                                // n_common = length of common prefix (== n_past before alora).
                                //
                                // SAFETY: Every slot must produce at least one batch row to
                                // reach common_sampler_sample() (server-context.cpp:6591).
                                // The batch-assembly loop [6170] adds tokens in
                                // [slot.prompt.n_tokens(), n_new).  When n_past == n_new
                                // (zero-prompt decode) that range is empty, leaving the slot
                                // with no batch row, no slot.i_batch, and a failed
                                // GGML_ASSERT(batch.n_tokens > 0) at [6240].  All branches
                                // therefore set n_past < n_new to ensure at least one token
                                // enters the batch; for the logits_reused path the restored
                                // logits are injected before sampling (line 6578), so the
                                // model-computed logits for that row are safely overwritten.
                                {
                                    const int n_slot = (int) slot.prompt.tokens.size();
                                    const int n_new  = (int) input_tokens.size();
                                    const int n_common_val = n_past; // before alora adjustment
                                    const bool logits_valid = slot.logits_valid;

                                    slot.n_common           = n_common_val;
                                    slot.logits_reused      = false;
                                    slot.n_prompt_processed = 0;

                                    SLT_DBG(slot, "#PD-TRACE N_COMMON slot=%d n_slot=%d n_new=%d n_common=%d logits_valid=%d just_restored=%d\n",
                                            slot.id, n_slot, n_new, n_common_val, (int) logits_valid, (int) slot.just_restored);

                                    if (n_common_val == n_new && logits_valid) {
                                        // Full prompt matches resident KV and restored logits are
                                        // valid.  Re-decode the final token so the slot gets a
                                        // batch row (required to reach common_sampler_sample);
                                        // the restored logits are injected before sampling,
                                        // overwriting the model-computed logits for that row.
                                        // NOTE: n_past cannot be set to n_new here because the
                                        // batch-assembly loop at [6170] only adds tokens in
                                        // [slot.prompt.n_tokens(), n_new) — with n_past == n_new
                                        // zero tokens would be added, leaving the slot with no
                                        // batch row, no slot.i_batch, and a failed assertion at
                                        // GGML_ASSERT(batch.n_tokens > 0) [6240].
                                        n_past = n_new - 1;
                                        slot.logits_reused = true;
                                        slot.n_prompt_processed = 1;
                                        SLT_INF(slot, "#PD-TRACE N_COMMON zero-prompt decode n_common=%d logits_reused=true (1-token batch row)\n", n_common_val);
                                    } else if (n_common_val == n_new && !logits_valid) {
                                        // Full prompt matches but restored logits are stale or
                                        // absent.  Re-decode the final token to regenerate logits
                                        // (the "1-token trick").  DEFAULT for warm/COMBINED.
                                        n_past = n_common_val - 1;
                                        slot.n_prompt_processed = 1;
                                        SLT_INF(slot, "#PD-TRACE N_COMMON 1-token re-decode n_common=%d logits_reused=false\n", n_common_val);
                                    } else if (n_common_val < n_slot) {
                                        // Partial match — trim KV from divergence point.
                                        // Discard restored logits FIRST (any seq_rm invalidates them).
                                        slot.logits_valid = false;
                                        slot.logits_reused = false;
                                        n_past = n_common_val;
                                        SLT_INF(slot, "#PD-TRACE N_COMMON trim n_common=%d < n_slot=%d, logits discarded\n", n_common_val, n_slot);
                                    } else {
                                        // Normal: n_common < n_new.  Process [n_common, n_new).
                                        n_past = n_common_val;
                                        slot.logits_reused = false;
                                    }
                                }

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            common_context_seq_rm (ctx_tgt, slot.id, head_p, head_c);
                                            common_context_seq_add(ctx_tgt, slot.id, head_c, head_c + n_match, kv_shift);

                                            if (ctx_dft) {
                                                common_context_seq_rm (ctx_dft.get(), slot.id, head_p, head_c);
                                                common_context_seq_add(ctx_dft.get(), slot.id, head_c, head_c + n_match, kv_shift);
                                            }

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - 1);

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // Hydra #641: post-decode KV restore (STATE_PUT / merged DECODE) freezes the
                                // slot's checkpoint at PREFILL end — checkpoints are only created during prompt
                                // processing, and the prompt loop breaks 4+n_ubatch/4 tokens early — so on the
                                // NEXT continuation that stale early checkpoint matches (is_rec: pos_max <= pos_next)
                                // and load_tgt() overwrites the whole sequence state (attention + SSM) with the
                                // old snapshot, re-prefilling ~800-1400 already-cached tokens (5.5s on RTX, 51s on
                                // P100 for the warm-affinity turn 2). A pure extension — the whole cache is a
                                // strict prefix of the new prompt and memory really ends at pos_next-1 — must not
                                // enter the checkpoint search. Logic is pinned by
                                // tests/test-hydra-checkpoint-policy.cpp (see server_should_rewind_to_checkpoint).
                                const auto pos_max_mem = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
                                const bool no_rewind_needed = !server_should_rewind_to_checkpoint(
                                        n_past,
                                        (llama_pos) slot.prompt.n_tokens(),
                                        (llama_pos) slot.task->n_tokens(),
                                        pos_next,
                                        pos_max_mem);

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - 4, 0);
                                    const int np1 = std::min<int>(n_past + 6, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold && !no_rewind_needed) {
                                    // For recurrent/hybrid models (e.g. Qwen3.x MTP) a checkpoint's
                                    // pos_min equals the full sequence length, so the usual
                                    // `pos_min < pos_min_thold` test is perpetually false → every turn
                                    // force-re-prefills. Match on pos_max <= pos_next instead so cached
                                    // KV is reused. Ref: ik_llama.cpp#1762 (port).
                                    const bool is_rec = llama_model_is_recurrent(model_tgt) ||
                                                        llama_model_is_hybrid(model_tgt);
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&, func_name = __func__](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            LOG_INF("slot %12.*s: id %2d | task %d | Checking checkpoint with [%d, %d] against %d...\n", 12,
                                                func_name, (slot).id, ((slot).task ? (slot).task->id : -1), cur.pos_min, cur.pos_max, pos_min_thold);
                                            if (is_rec) {
                                                return cur.pos_max <= pos_next;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    // #469 trace: log checkpoint search result and just_restored decision
                                    SLT_DBG(slot, "#PD-TRACE CHECKPOINT_SEARCH slot=%d do_reset=%d just_restored=%d n_past=%d checkpoints=%zu pos_next=%d\n",
                                            slot.id, do_reset, slot.just_restored, n_past, slot.prompt.checkpoints.size(), pos_next);

                                    // For slots restored via STATE_PUT (full context state),
                                    // skip the checkpoint search entirely. The restored state
                                    // already has the correct KV cache + logits. The checkpoint
                                    // check is needed for in-server reuse across turns, not for
                                    // cross-node migration where the full state is restored.
                                    if (do_reset && slot.just_restored && n_past > 0) {
                                        SLT_WRN(slot, "STATE_PUT restored slot — using cached n_past=%d, skipping checkpoint check\n", n_past);
                                        do_reset = false;
                                        pos_next = n_past;
                                        slot.just_restored = false;
                                    }

                                    if (!do_reset) {
                                        if (it != slot.prompt.checkpoints.rend()) {
                                            // restore the context checkpoint
                                            if (it->is_recr_only) {
                                                // Hydra M2 wire checkpoint (v3): the buffer holds a
                                                // recurrent-only (PARTIAL_ONLY) capture. The recurrent
                                                // (SSM) state is restored with matched PARTIAL_ONLY flags
                                                // (flag symmetry — a PARTIAL_ONLY buffer physically lacks
                                                // the mem_attn bytes). The attention cache must be trimmed
                                                // at pos_max first: after the full live-state restore it
                                                // still holds cells past the checkpoint position, and a
                                                // full restore has no attention bytes to overwrite them.
                                                // seq_rm is called on the ATTENTION cache directly — the
                                                // blanket llama_memory_hybrid::seq_rm would wipe the
                                                // recurrent cell first (n_rs_seq == 0 rollback path).
                                                if (llama_model_is_hybrid(model_tgt)) {
                                                    ((llama_memory_hybrid *) llama_get_memory(ctx_tgt))->get_mem_attn()->seq_rm(slot.id, it->pos_max, -1);
                                                }
                                                it->load_tgt_recr(ctx_tgt, slot.id);

                                                // Mirror for the draft (MTP) context when enabled.
                                                if (ctx_dft && llama_model_is_hybrid(model_tgt)) {
                                                    ((llama_memory_hybrid *) llama_get_memory(ctx_dft.get()))->get_mem_attn()->seq_rm(slot.id, it->pos_max, -1);
                                                }
                                                it->load_dft_recr(ctx_dft.get(), slot.id);
                                            } else {
                                                it->load_tgt(ctx_tgt,       slot.id, 0);
                                                it->load_dft(ctx_dft.get(), slot.id, 0);
                                            }

                                            pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                            n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                            SLT_WRN(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024);
                                            // One-shot: STATE_PUT flag consumed on first successful match
                                            slot.just_restored = false;
                                        }
                                        // else: just_restored override — KV already in place via STATE_PUT,
                                        // pos_next/n_past set above; no checkpoint to load from iterator.
                                    }

                                    if (do_reset) {
                                        SLT_WRN(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                } else if (pos_min >= pos_min_thold) {
                                    // #641: pure extension — the whole cached sequence is a strict prefix of
                                    // the new prompt and memory ends exactly at pos_next - 1, so no rewind is
                                    // needed and the stale PREFILL-end checkpoint must not be loaded on top of
                                    // the restored state (which would re-prefill already-cached tokens).
                                    SLT_INF(slot, "no rewind needed — memory at pos_next-1 (n_past = %d, prompt = %d, task = %d, pos_next = %d, pos_max_mem = %d); skipping checkpoint search\n",
                                            n_past, (int) slot.prompt.n_tokens(), (int) slot.task->n_tokens(), (int) pos_next, (int) pos_max_mem);
                                    // consume the one-shot STATE_PUT flag so it can't leak into a later turn
                                    slot.just_restored = false;
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_WRN(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        // When logits_reused is true (zero-prompt decode from restored
                        // logits), skip this guard — the entire prompt is cached.
                        if (!slot.logits_reused && n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.n_prompt_tokens_cache = n_past;
                        slot.n_prompt_tokens_processed = 0;

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    }

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.n_tokens + slot.task->n_tokens() > n_batch) {
                            continue;
                        }
                    }

                    const int64_t t_current = ggml_time_us();
                    slot.t_prompt_processing = (t_current - slot.t_start_process_prompt) / 1e3;
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    common_context_seq_rm(ctx_tgt, slot.id, p0, -1);
                    if (ctx_dft) {
                        common_context_seq_rm(ctx_dft.get(), slot.id, p0, -1);
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if (see server_should_create_checkpoint):
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    // - the model is recurrent/hybrid (see below)
                    // Hydra: when the binary RPC port is enabled this server participates in
                    // cross-node KV migration. The restore target may not support rollback
                    // (e.g. it reports SEQ_RM_TYPE_FULL for the same model), so create native
                    // checkpoints regardless of the local seq_rm verdict — STATE_GET ships
                    // the latest checkpoint in the v2 blob, and without one the receiver
                    // fabricates a checkpoint at the final position, which corrupts
                    // hybrid/recurrent decode (recurrent state ends up past the resume point).
                    // Hydra (#316): the generic seq_rm probe in common_context_can_seq_rm()
                    // reports PART (not RS) for this hybrid arch's mixed attention/recurrent
                    // memory, since llama_n_rs_seq() is 0 and the smoke-test removal succeeds.
                    // That left checkpoint creation gated on rpc_port (only true for in-cluster
                    // nodes), so a standalone server with no RPC peer never created checkpoints
                    // and every cache-search below (which already special-cases is_rec, see
                    // ik_llama.cpp#1762) found nothing to restore — forcing a full re-prefill
                    // on every request. Recurrent/hybrid models need checkpoints on their own
                    // merits, independent of rpc_port.
                    // Hydra (#8): the gate is extracted to server_should_create_checkpoint()
                    // and pinned by tests/test-hydra-checkpoint-policy.cpp so the is_rec term
                    // can't be silently dropped — doing so breaks hybrid KV-cache restore.
                    const bool is_rec = llama_model_is_recurrent(model_tgt) ||
                                        llama_model_is_hybrid(model_tgt);
                    bool do_checkpoint = server_should_create_checkpoint(
                            params_base.n_ctx_checkpoints,
                            slot.task->type == SERVER_TASK_TYPE_COMPLETION,
                            ctx_tgt_seq_rm_type,
                            n_swa,
                            is_rec,
                            params_base.rpc_port);

                    bool has_mtmd = false;

                    // check if we should process the image
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && input_tokens[slot.prompt.n_tokens()] == LLAMA_TOKEN_NULL) {
                        // process the image
                        size_t n_tokens_out = 0;
                        int32_t res = input_tokens.process_chunk(ctx_tgt, mctx, slot.prompt.n_tokens(), slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
                        if (res != 0) {
                            SLT_ERR(slot, "failed to process image, res = %d\n", res);
                            send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        if (ctx_dft) {
                            // TODO: in the future, figure out how to infuse target embeddings to the images
                            //       for now, we skip this for simplicity
                            //       maybe we simply need to call `common_speculative_process()` on the mtmd batches in the `process_chunk` above?
                            res = input_tokens.process_chunk(ctx_dft.get(), mctx, slot.prompt.n_tokens(), slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
                            if (res != 0) {
                                GGML_ABORT("failed to process multi-modal data on draft context\n");
                            }
                        }

                        slot.n_prompt_tokens_processed += n_tokens_out;

                        // add the image chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(slot.prompt.n_tokens());
                            slot.prompt.tokens.push_back(chunk.get()); // copy
                        }

                        has_mtmd = true;
                    }

                    const int32_t n_before_user = slot.task->params.n_before_user;
                    const bool n_before_user_known = n_before_user > 0;

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.n_tokens < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        common_batch_add(batch,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            { slot.id },
                            slot.need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;

                        // stop the prompt batch exactly before the latest user input, so a checkpoint
                        // can be created after the previous messages
                        if (n_before_user_known &&
                            slot.prompt.n_tokens() == n_before_user) {
                            break;
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.n_tokens - n_tokens_prev;

                    // Track total prompt tokens processed for n_common observability
                    if (!slot.logits_reused) {
                        slot.n_prompt_processed += n_tokens_cur;
                    }

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.n_tokens > 0);

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints
                        if (!n_before_user_known && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // checkpoints are created before the current batch is decoded, so
                    // their token position is the batch start rather than the prompt end
                    const int32_t n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    {
                        const bool is_on_user =
                            n_before_user_known &&
                            n_tokens_start == n_before_user;

                        const bool is_after_user =
                            n_before_user_known &&
                            n_tokens_start > n_before_user;

                        const bool is_allowed =
                            !n_before_user_known ||
                            is_on_user ||
                            (is_after_user && near_prompt_end);

                        if (do_checkpoint && !is_allowed) {
                            do_checkpoint = false;
                        }
                    }

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together.
                    // For recurrent/hybrid models, use a much smaller minimum spacing so short
                    // follow-up turns still get a checkpoint to resume from. Ref: ik_llama.cpp#1762.
                    const int eff_checkpoint_min_step =
                        (llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt))
                            ? std::min(params_base.checkpoint_min_step, 4)
                            : params_base.checkpoint_min_step;
                    do_checkpoint = do_checkpoint && (slot.prompt.checkpoints.empty() || n_tokens_start > slot.prompt.checkpoints.back().n_tokens + eff_checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }

                if (batch.n_tokens >= n_batch) {
                    break;
                }
            }
        }

        SRV_DBG("decoding batch, n_tokens = %d\n", batch.n_tokens);

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        if (slot_batched) {
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        if (batch.n_tokens == 0) {
            // epic #610 WS1: in seam mode the extension may handle the
            // empty-batch case (STATE_GET transfer suppression). WS1 impl is a
            // no-op — this is a pure A/B switch, both modes run the inline
            // suppression below.
            if (hydra_ext_active && hydra_ext && hydra_ext->on_empty_batch(*this)) {
                // extension handled the empty batch — skip the inline logic
            } else if (++n_empty_consecutive > 3) {
                // Hydra: a STATE_GET background stream holds the slot (hydra_transferring)
                // without contributing batch tokens — that is expected, not a stall.
                // Suppress the abort while a transfer is in flight, and for a short
                // grace window after it ends (the flag clears a few loop iterations
                // before the queue delivers the releasing task — without the grace
                // window those tail iterations trip the abort).
                static int64_t hydra_last_transfer_ms = 0;
                static int64_t hydra_suppress_count   = 0;
                bool any_transferring = false;
                for (const auto & s : slots) {
                    if (s.hydra_transferring && s.hydra_transferring->load()) {
                        any_transferring = true;
                        break;
                    }
                }
                const int64_t now_ms = ggml_time_us() / 1000;
                if (any_transferring) {
                    hydra_last_transfer_ms = now_ms;
                }
                if (any_transferring || now_ms - hydra_last_transfer_ms < 2000) {
                    // rate-limit: this branch runs in a hot loop — log once per 256 suppressions
                    if (hydra_suppress_count++ % 256 == 0) {
                        SRV_WRN("empty batch threshold exceeded (n_empty=%d, suppressed=%" PRId64 ") — hydra transfer %s, suppressing abort\n",
                                n_empty_consecutive, hydra_suppress_count,
                                any_transferring ? "in flight" : "just ended");
                    }
                    n_empty_consecutive = 0;
                    // avoid hot-spinning while the transfer holds the slot
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                } else {
                    SRV_WRN("%s", "no tokens to decode\n");
                    GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
                }
            }
        } else {
            n_empty_consecutive = 0;
        }

        int32_t i_next = 0;

        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i = i_next) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            const int ret = llama_decode(ctx_tgt, batch_view);

            metrics.on_decoded(slots);

            if (ret != 0) {
                {
                    std::string err;

                    if (n_batch == 1 && ret == 1) {
                        // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                        //       need to remove the tokens from the current batch too
                        err = "Context size has been exceeded.";
                    }

                    if (ret == -1) {
                        err = "Invalid input batch.";
                    }

                    if (ret < -1) {
                        // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                        err = "Compute error.";
                    }

                    // TODO: handle ret == 2 (abort) when we start aborting

                    if (!err.empty()) {
                        SRV_ERR("%s i = %d, n_batch = %d, ret = %d\n", err.c_str(), i, n_batch, ret);

                        for (auto & slot : slots) {
                            if (slot.is_processing() || slot.hydra_transferring->load()) {
                                send_error(slot, err);
                                slot.release();

                                // note: it's complicated to keep track of how much of the current batch has been
                                //       processed before the error occurred, so we simply clear the entire context
                                slot.prompt_clear(false);
                            }
                        }

                        break;
                    }
                }

                // retry with half the batch size to try to find a free slot in the KV cache
                if (!try_clear_idle_slots()) {
                    n_batch /= 2;
                }

                SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

                continue; // continue loop of n_batch
            }

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            //       for now, always re-evaluate for simplicity
            //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
            //
            // | spec type   | need re-eval |
            // | ---         | ---          |
            // | draft model | no           | because the draft model does not use embeddings from the target
            // | MTP (std)   | yes          |
            // | MTP Gemma4  | no           | because the KV cache is shared
            // | Eagle3      | yes          |
            // | DFlash      | yes          | https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4405406982
            //
            // note: this logic is now moved in `common_speculative_process()`
            //       keeping the sketch here until for a bit, until the logic is finalized
            //
            //if (ctx_dft) {
            //    // TODO: update as needed for MTP, Eagle3, etc.
            //    const bool need_tgt_embd = false;

            //    if (need_tgt_embd) {
            //        llama_synchronize(ctx_tgt);
            //    }

            //    // the logic here varies depending on the speculative decoding method
            //    //  - some draft contexts require embeddings from the target context, others don't
            //    //  - some draft contexts involve an encoder step to transform the target embeddings to draft embeddings
            //    // TODO: extract this in a function ?
            //    {
            //        // TODO: hook the embeddings from the last target batch here
            //        if (llama_model_has_encoder(model_dft.get())) {
            //            //llama_encode(ctx_dft, ...);

            //            GGML_ABORT("not implemented yet\n");
            //        }

            //        const int ret = llama_decode(ctx_dft.get(), batch_view);

            //        if (ret != 0) {
            //            SRV_ERR("failed to decode draft batch, ret = %d\n", ret);

            //            // TODO: handle error
            //            break;
            //        }
            //    }
            //}
            if (!common_speculative_process(spec.get(), batch_view)) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                break;
            }

            // move the head of the batch forward with the number of tokens we just processed
            i_next = i + n_tokens;

            // on successful decode, restore the original batch size
            n_batch = llama_n_batch(ctx_tgt);

            // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
            for (auto & slot : slots) {
                if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                    std::vector<server_slot *> children;
                    for (auto & other : slots) {
                        if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                            children.push_back(&other);
                        }
                    }

                    // all children slots should already launched by launch_slots_with_parent_task()
                    // copy state to the child slots
                    for (auto & child : children) {
                        SLT_INF(slot, " - copying state to child %d\n", child->id);

                        GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                        slot.copy_state_to(*child);
                        child->state = SLOT_STATE_DONE_PROMPT;
                    }
                }
            }

            for (auto & slot : slots) {
                // optionally send prompt processing progress
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->params.stream && slot.task->params.return_progress) {
                        send_partial_response(slot, {}, true);
                    }
                }

                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
                    continue; // continue loop of slots
                }

                if (slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                        // prompt evaluated for embedding
                        send_embedding(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                        send_rerank(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    GGML_ASSERT(slot.task->need_sampling());

                    // prompt evaluated for next-token prediction
                    slot.state = SLOT_STATE_GENERATING;

                    if (slot.can_speculate()) {
                        common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                    }
                } else if (slot.state != SLOT_STATE_GENERATING) {
                    continue; // continue loop of slots
                }

                if (slot.can_speculate() && !slot.spec_draft.empty()) {
                    continue; // sample using speculative decoding
                }

                const int tok_idx = slot.i_batch - i;

                // D4: Inject per-slot restored logits into the correct batch row before
                // first sample. The sampler reads via llama_get_logits_ith(ctx, tok_idx),
                // so we must write to that exact row — not row 0.
                if (slot.logits_valid && !slot.restored_logits.empty() && slot.n_decoded == 0) {
                    float * row_logits = llama_get_logits_ith(slot.ctx_tgt, tok_idx);
                    if (row_logits) {
                        const size_t n_floats = slot.restored_logits.size();
                        memcpy(row_logits, slot.restored_logits.data(), n_floats * sizeof(float));
                        SLT_INF(slot, "consumed %zu restored logits into row %d (tok_idx)\n", n_floats, tok_idx);
                    } else {
                        SLT_WRN(slot, "restored logits skipped: llama_get_logits_ith returned null for row %d\n", tok_idx);
                    }
                    slot.restored_logits.clear();
                    slot.logits_valid = false;
                }

                llama_token id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);

                slot.i_batch = -1;

                common_sampler_accept(slot.smpl.get(), id, true);

                // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
                const int64_t t_current = ggml_time_us();

                slot.n_decoded += 1;

                if (slot.n_decoded == 1) {
                    slot.t_start_generation = t_current;
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                completion_token_output result;
                result.tok          = id;
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

                if (slot.task->params.sampling.n_probs > 0) {
                    populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
                }

                if (!process_token(result, slot)) {
                    // release slot because of stop condition
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    continue;
                }

                slot.print_timings_tg();
            }

            // speculative decoding - main model sample and accept
            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() || slot.spec_draft.empty()) {
                    continue;
                }

                // save the original draft size
                const size_t n_draft = slot.spec_draft.size();

                GGML_ASSERT(n_draft > 0);

                // verify and try to accept the draft
                {
                    // save the sampler sampler state in case we need to restore it
                    common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                    GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                    auto accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                    slot.spec_i_batch.clear();

                    GGML_ASSERT(accepted.size() >= 1);

                    const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                    const bool use_ckpt_tgt =
                        ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                       (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));

                    // check for partial draft acceptance
                    if (n_rollback > 0) {
                        if (use_ckpt_tgt) {
                            if (trace > 0) {
                                SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                            }

                            // partial acceptance is not supported by the context -> truncate the draft and restore the state
                            slot.spec_draft = std::move(accepted);

                            const auto & ckpt = slot.spec_ckpt;

                            SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                            {
                                ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);

                                common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);
                            }

                            if (slot.ctx_dft) {
                                ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);

                                common_context_seq_rm(slot.ctx_dft, slot.id, ckpt.pos_max + 1, -1);
                            }

                            slot.prompt.tokens.keep_first(ckpt.n_tokens);
                            slot.smpl = std::move(smpl_save);

                            continue;
                        }
                    }

                    if (trace > 0) {
                        SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                    }

                    common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                    slot.spec_draft = std::move(accepted);
                }

                const int64_t t_current = ggml_time_us();

                const auto ids = std::move(slot.spec_draft);

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                // update how many tokens out of those tested were accepted
                slot.n_draft_accepted += ids.size() - 1;

                // add accepted tokens to the prompt
                slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
                slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

                slot.sampled = ids.back(); // last accepted token
                SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

                common_context_seq_rm(slot.ctx_tgt, slot.id, slot.prompt.tokens.pos_next(), -1);
                if (slot.ctx_dft) {
                    common_context_seq_rm(slot.ctx_dft, slot.id, slot.prompt.tokens.pos_next(), -1);
                }

                for (size_t i = 0; i < ids.size(); ++i) {
                    completion_token_output result;

                    result.tok          = ids[i];
                    result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                    result.prob         = 1.0f; // set later

                    // TODO: set result.probs

                    slot.n_decoded += 1;

                    if (!process_token(result, slot)) {
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();

                        break;
                    }
                }

                slot.print_timings_tg();

                SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) ids.size() - 1, (int) n_draft, slot.prompt.n_tokens());
            }
        }

        SRV_DBG("%s", "run slots completed\n");
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

void server_context::set_hydra_capabilities(bool rpc_backend_active, const std::string & peer,
        bool peer_reachable, const std::string & combined_pattern, const std::string & split_mode) {
    impl->hydra_rpc_backend_active = rpc_backend_active;
    impl->hydra_peer               = peer;
    impl->hydra_peer_reachable     = peer_reachable;
    impl->hydra_combined_pattern   = combined_pattern;
    impl->hydra_split_mode         = split_mode;
}

void server_context::set_hydra_combined_head_attached(bool attached) {
    impl->hydra_combined_head_attached = attached;
}

void server_context::set_hydra_combined_static(bool is_static) {
    impl->hydra_combined_static = is_static;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* json_webui_settings    */ impl->json_webui_settings,  // Deprecated
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
        /* split_mode             */ impl->params_base.split_mode,
        /* tensor_split           */ std::vector<float>(impl->params_base.tensor_split, impl->params_base.tensor_split + 128),
    };
}



// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_http_res {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::on_sleeping_changed(std::function<void(bool)> callback) {
    impl->queue_tasks.on_sleeping_state(std::move(callback));
}

void server_context::set_routes_ptr(server_routes * routes) {
    impl->routes_ptr = routes;
}

bool server_context::bootstrap_init() {
    return impl->bootstrap_init();
}

void server_context::set_bootstrap_capabilities(bool rpc_active, const std::string & peer,
        bool peer_reachable, const std::string & pattern,
        const std::string & split_mode, bool combined_static) {
    impl->bootstrap_rpc_active = rpc_active;
    impl->bootstrap_peer = peer;
    impl->bootstrap_peer_reachable = peer_reachable;
    impl->bootstrap_pattern = pattern;
    impl->bootstrap_split_mode = split_mode;
    impl->bootstrap_combined_static = combined_static;
}

// compute the number of tokens before the last user message in the prompt
static int32_t prompt_get_n_before_user(
        const json & message_spans,
        const std::string & prompt,
        const std::vector<raw_buffer> & files,
        const llama_vocab * vocab,
        mtmd_context * mctx) {
    int32_t result = -1;
    int32_t byte_pos = -1;

    for (const auto & span : message_spans) {
        const std::string role = json_value(span, "role", std::string());

        if (role == "user") {
            byte_pos = json_value(span, "pos", -1);
        }
    }

    if (byte_pos >= 0) {
        GGML_ASSERT((size_t) byte_pos <= prompt.size());

        const std::string prefix = prompt.substr(0, (size_t) byte_pos);

        const std::string marker = get_media_marker();
        size_t n_prefix_media = 0;
        for (size_t pos = 0; (pos = prefix.find(marker, pos)) != std::string::npos; pos += marker.size()) {
            n_prefix_media++;
        }

        GGML_ASSERT(n_prefix_media <= files.size());

        if (mctx != nullptr && n_prefix_media > 0) {
            // TODO: this makes a copy - avoid it
            std::vector<raw_buffer> prefix_files(files.begin(), files.begin() + n_prefix_media);

            result = (int32_t) process_mtmd_prompt(mctx, prefix, prefix_files).size();
        } else {
            result = (int32_t) tokenize_input_prompts(vocab, nullptr, prefix, true, true)[0].size();
        }

        SRV_TRC("message_spans: last user message: byte_pos=%d, media=%zu, n_before_user=%d\n",
                byte_pos, n_prefix_media, result);
    }

    return result;
}


//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type,
            const std::string & hydra_config_json) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    std::shared_lock meta_lock(meta_mutex);

    // P0-1 (#49): null-meta guard — meta is null until update_meta() is
    // called after model load. Return 503 instead of crashing.
    if (!meta) {
        auto res = create_response();
        res->error(format_error_response("model not loaded — waiting for CONFIGURE",
                                         ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(
                    ctx_server.vocab,
                    params,
                    meta->slot_n_ctx,
                    meta->logit_bias_eog,
                    data);

            const auto message_spans = json_value(data, "message_spans", json::array());
            if (prompt.is_string() && message_spans.is_array()) {
                task.params.n_before_user =
                    prompt_get_n_before_user(
                        message_spans,
                        prompt.get<std::string>(),
                        files,
                        ctx_server.vocab,
                        ctx_server.mctx);
            }

            task.id_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // Attach hydra_config (if any) so the task-queue thread can
            // apply it synchronously before processing the completion.
            if (!hydra_config_json.empty()) {
                task.hydra_config_json = hydra_config_json;
            }

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                auto * cmpl = dynamic_cast<server_task_result_cmpl_final*>(res.get());
                GGML_ASSERT(cmpl != nullptr);
                // P1-5/P1-6: build metrics inline from the (always-current)
                // meta rather than reading a class-level field that could
                // leak between concurrent requests.
                {
                    auto split_mode_str = [](enum llama_split_mode m) -> const char * {
                        switch (m) {
                            case LLAMA_SPLIT_MODE_NONE:  return "none";
                            case LLAMA_SPLIT_MODE_LAYER: return "layer";
                            case LLAMA_SPLIT_MODE_ROW:   return "row";
                            default: return "none";
                        }
                    };
                    json metrics = json::object();
                    metrics["model_path"]   = meta->model_path;
                    metrics["split_mode"]   = split_mode_str(meta->split_mode);
                    metrics["t3_reloaded"]  = false;
                    metrics["t3_reload_ms"] = 0.0;
                    json ts_arr = json::array();
                    for (const auto & v : meta->tensor_split) {
                        if (v != 0.0f) {
                            ts_arr.push_back(v);
                        } else {
                            break;
                        }
                    }
                    metrics["tensor_split"] = ts_arr;
                    cmpl->hydra_metrics = metrics;
                }
                arr.push_back(cmpl->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->next = [res_this = res.get(), res_type, &req, &params](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            try {
                if (req.should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &req, &start_time, &params]() {
                    if (req.should_stop()) {
                        return true; // should_stop condition met
                    } else if (params.sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)params.sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(req.should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        };
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

void server_routes::evict_decode_results_locked() {
    // Evict by TTL
    const int64_t now = std::time(nullptr);
    for (auto it = decode_results.begin(); it != decode_results.end(); ) {
        if (now - it->second.created_at >= it->second.ttl_s) {
            it = decode_results.erase(it);
        } else {
            ++it;
        }
    }
    // Evict oldest by insertion order when over capacity
    while ((int)decode_results.size() > decode_result_max) {
        decode_results.erase(decode_results.begin());
    }
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server_outer(ctx_server),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    // Merged DECODE result buffer config from env vars
    if (const char * e = getenv("HYDRA_DECODE_RESULT_TTL_S")) {
        decode_result_ttl_s = std::max(1, atoi(e));
    } else {
        decode_result_ttl_s = HYDRA_DECODE_RESULT_TTL_S_DEFAULT;
    }
    if (const char * e = getenv("HYDRA_DECODE_RESULT_MAX")) {
        decode_result_max = std::max(1, atoi(e));
    } else {
        decode_result_max = HYDRA_DECODE_RESULT_MAX_DEFAULT;
    }
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_task->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_task->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_task->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_task->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_task->n_decode_total}
            }, {
                    {"name",  "n_tokens_max"},
                    {"help",  "Largest observed n_tokens."},
                    {"value",  res_task->n_tokens_max}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  (uint64_t) res_task->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  (uint64_t) res_task->n_tasks_deferred}
            },{
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_task->n_busy_slots_total / std::max((float) res_task->n_decode_total, 1.f)}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = prometheus.str();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    // ── Hydra state streaming (M0.0) ───────────────────────────────────────
    this->get_state = [this](const server_http_req & req) {
        auto res = create_response();
        int id_slot;
        try {
            id_slot = std::stoi(req.get_param("id_slot"));
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        server_task task(SERVER_TASK_TYPE_HYDRA_STATE_GET);
        task.id = res->rd.get_new_id();
        task.hydra_action.id_slot = id_slot;
        task.hydra_action.hydra_fd = -1;
        res->rd.post_task(std::move(task));
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        auto * hr = dynamic_cast<server_task_result_hydra_state*>(result.get());
        GGML_ASSERT(hr != nullptr);
        if (hr->rpc_status != HYDRA_STATUS_OK) {
            res->status = hr->rpc_status == HYDRA_STATUS_NOT_FOUND ? 404 : 503;
            res->data = hr->error.empty() ? "" : hr->error;
            return res;
        }
        res->content_type = "application/octet-stream";
        res->headers["X-Hydra-State-Size"] = std::to_string(hr->state_data.size());
        res->headers["X-Hydra-N-Past"] = std::to_string(hr->n_past);
        res->data.assign((const char*)hr->state_data.data(), hr->state_data.size());
        return res;
    };
    this->put_state = [this](const server_http_req & req) {
        auto res = create_response();
        int id_slot;
        try {
            id_slot = std::stoi(req.get_param("id_slot"));
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        server_task task(SERVER_TASK_TYPE_HYDRA_STATE_PUT);
        task.id = res->rd.get_new_id();
        task.hydra_action.id_slot = id_slot;
        task.hydra_action.erase_existing = req.get_param("erase_existing") == "true";
        task.hydra_action.state_data.assign(req.body.begin(), req.body.end());
        res->rd.post_task(std::move(task));
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        auto * hr = dynamic_cast<server_task_result_hydra_state*>(result.get());
        GGML_ASSERT(hr != nullptr);
        if (hr->rpc_status != HYDRA_STATUS_OK) {
            res->status = 503;
            res->data  = hr->error.empty() ? "restore failed" : hr->error;
            return res;
        }
        res->ok(json{
            {"restored", hr->restored},
            {"n_past",   hr->n_past},
            {"bytes",    hr->bytes},
        });
        return res;
    };
    this->get_state_meta = [this](const server_http_req & req) {
        auto res = create_response();
        int id_slot;
        try {
            id_slot = std::stoi(req.get_param("id_slot"));
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        server_task task(SERVER_TASK_TYPE_HYDRA_STATE_META);
        task.id = res->rd.get_new_id();
        task.hydra_action.id_slot = id_slot;
        res->rd.post_task(std::move(task));
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        auto * hr = dynamic_cast<server_task_result_hydra_state*>(result.get());
        GGML_ASSERT(hr != nullptr);
        if (hr->rpc_status != HYDRA_STATUS_OK) {
            res->status = hr->rpc_status == HYDRA_STATUS_NOT_FOUND ? 404 : 503;
            res->data = hr->error.empty() ? "" : hr->error;
            return res;
        }
        res->ok(json{
            {"slot_id",        hr->id_slot},
            {"n_past",         hr->n_past},
            {"state_size",     (uint64_t)hr->state_size},
            {"is_processing",  hr->is_processing},
            {"is_transferring", hr->is_transferring},
            {"operation",      hr->operation},
            {"progress",       hr->progress},
            {"tokens_processed", hr->tokens_processed},
            {"tokens_total",   hr->tokens_total},
            {"elapsed_ms",     hr->elapsed_ms},
            // #470/A7: model identity — the Coordinator's merged-decode Gate A
            // compares these against kv_metadata. Without them the engine's
            // model_metadata came back empty and every COMBINED merged decode
            // was rejected (tokenizer/name mismatch) before KV restore.
            {"model_alias",    hr->model_alias},
            {"model_path",     hr->model_path},
            {"tokenizer",      hr->tokenizer},
            {"model_name",     hr->model_name},
            {"model_quant",    hr->model_quant},
            {"model_capabilities", hr->model_capabilities},
        });
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        std::shared_lock meta_lock(meta_mutex);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        // P0-1 (#49): null-meta guard
        if (!meta) {
            res->error(format_error_response("model not loaded — waiting for CONFIGURE",
                                             ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        task_params tparams;
        tparams.sampling = params.sampling;
        json default_generation_settings_for_props = json {
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel },
            { "model_alias",                 meta->model_name },
            { "model_path",                  meta->model_path },
            { "modalities",                  json {
                {"vision", meta->has_inp_image},
                {"audio",  meta->has_inp_audio},
            } },
            { "media_marker",                get_media_marker() },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            // New keys
            { "ui",                           params.ui },
            { "ui_settings",                  meta->json_ui_settings },
            // Deprecated: use ui/ui_settings instead (kept for backward compat)
            { "webui",                        params.webui },
            { "webui_settings",               meta->json_webui_settings },
            { "chat_template",               tmpl_default },
            { "chat_template_caps",          meta->chat_template_caps },
            { "bos_token",                   meta->bos_token_str },
            { "eos_token",                   meta->eos_token_str },
            { "build_info",                  meta->build_info },
            { "is_sleeping",                 queue_tasks.is_sleeping() },
            { "cors_proxy_enabled",          params.ui_mcp_proxy || params.webui_mcp_proxy },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // P0-1 (#49): null-meta guard
        {
            std::shared_lock meta_lock(meta_mutex);
            if (!meta) {
                res->error(format_error_response("model not loaded — waiting for CONFIGURE",
                                                 ERROR_TYPE_NOT_SUPPORTED));
                return res;
            }
        }

        // Validate input and compute infill prompt — these read meta->slot_n_ctx.
        // Scope the shared_lock so it releases before handle_completions_impl
        // takes its own lock (recursive shared_lock is UB on std::shared_mutex).
        json data;
        std::vector<raw_buffer> files;
        {
            std::shared_lock meta_lock(meta_mutex);

            // check model compatibility
            std::string err;
            if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
                err += "prefix token is missing. ";
            }
            if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
                err += "suffix token is missing. ";
            }
            if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
                err += "middle token is missing. ";
            }
            if (!err.empty()) {
                res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
                return res;
            }

            // validate input
            data = json::parse(req.body);
            if (data.contains("prompt") && !data.at("prompt").is_string()) {
                // prompt is optional
                res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
            }

            if (!data.contains("input_prefix")) {
                res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
            }

            if (!data.contains("input_suffix")) {
                res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
            }

            if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
                // input_extra is optional
                res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }

            json input_extra = json_value(data, "input_extra", json::array());
            for (const auto & chunk : input_extra) {
                // { "text": string, "filename": string }
                if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                    res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                    return res;
                }
                // filename is optional
                if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                    res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                    return res;
                }
            }
            data["input_extra"] = input_extra; // default to empty array if it's not exist

            std::string prompt = json_value(data, "prompt", std::string());
            std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
            SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
            data["prompt"] = format_prompt_infill(
                ctx_server.vocab,
                data.at("input_prefix"),
                data.at("input_suffix"),
                data.at("input_extra"),
                params.n_batch,
                params.n_predict,
                meta->slot_n_ctx,
                params.spm_infill,
                tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
            );
        }
        // meta_lock released here

        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        // P0-1 (#49): null-meta guard — meta is null until update_meta() is
        // called after model load. Return 503 instead of crashing.
        {
            std::shared_lock meta_lock(meta_mutex);
            if (!meta) {
                res->error(format_error_response("model not loaded — waiting for CONFIGURE",
                                                 ERROR_TYPE_NOT_SUPPORTED));
                return res;
            }
        }
        // Extract optional hydra_config from the raw body before it's
        // consumed by oaicompat_chat_params_parse(). The config is applied
        // on the task-queue thread (not here on the httplib worker thread)
        // to avoid racing the main inference loop.
        std::string hydra_config_str;
        {
            json raw_body = json::parse(req.body);
            if (raw_body.is_object() && raw_body.contains("hydra_config")
                && raw_body["hydra_config"].is_object()) {
                hydra_config_str = raw_body["hydra_config"].dump();
            }
        }
        json body_parsed;
        std::vector<raw_buffer> files;
        {
            std::shared_lock meta_lock(meta_mutex);
            json body = json::parse(req.body);
            body_parsed = oaicompat_chat_params_parse(body, meta->chat_params, files);
        }
        // meta_lock released before handle_completions_impl (which takes its own)

        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT,
            hydra_config_str);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        json body_parsed;
        std::vector<raw_buffer> files;
        {
            std::shared_lock meta_lock(meta_mutex);
            // P0-1 (#49): null-meta guard
            if (!meta) {
                res->error(format_error_response("model not loaded — waiting for CONFIGURE",
                                                 ERROR_TYPE_NOT_SUPPORTED));
                return res;
            }
            json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
            SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
            SRV_DBG("converted request: %s\n", body.dump().c_str());
            body_parsed = oaicompat_chat_params_parse(body, meta->chat_params, files);
        }
        // meta_lock released before handle_completions_impl

        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        json body_parsed;
        std::vector<raw_buffer> files;
        {
            std::shared_lock meta_lock(meta_mutex);

            // P0-1 (#49): null-meta guard
            if (!meta) {
                res->error(format_error_response("model not loaded — waiting for CONFIGURE",
                                                 ERROR_TYPE_NOT_SUPPORTED));
                return res;
            }

            if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
                res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
                return res;
            }

            json body = convert_transcriptions_to_chatcmpl(
                json::parse(req.body),
                meta->chat_params.tmpls.get(),
                req.files,
                files);
            SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
            SRV_DBG("converted request: %s\n", body.dump().c_str());
            body_parsed = oaicompat_chat_params_parse(body, meta->chat_params, files);
        }
        // meta_lock released before handle_completions_impl

        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        json body_parsed;
        std::vector<raw_buffer> files;
        {
            std::shared_lock meta_lock(meta_mutex);
            // P0-1 (#49): null-meta guard
            if (!meta) {
                res->error(format_error_response("model not loaded — waiting for CONFIGURE",
                                                 ERROR_TYPE_NOT_SUPPORTED));
                return res;
            }
            json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
            SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
            SRV_DBG("converted request: %s\n", body.dump().c_str());
            body_parsed = oaicompat_chat_params_parse(body, meta->chat_params, files);
        }
        // meta_lock released before handle_completions_impl

        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        auto res = create_response();
        std::shared_lock meta_lock(meta_mutex);
        // P0-1 (#49): null-meta guard
        if (!meta) {
            res->error(format_error_response("model not loaded — waiting for CONFIGURE",
                                             ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);

        json prompt = body_parsed.at("prompt");
        llama_tokens tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true);
        res->ok({{"input_tokens", static_cast<int>(tokens.size())}});
        return res;
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::shared_lock meta_lock(meta_mutex);
        // P0-1 (#49): null-meta guard
        if (!meta) {
            res->error(format_error_response("model not loaded — waiting for CONFIGURE",
                                             ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        std::shared_lock meta_lock(meta_mutex);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        // P0-1 (#49): null-meta guard
        if (!meta) {
            res->error(format_error_response("model not loaded — waiting for CONFIGURE",
                                             ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        json models = {
            {"models", {
                {
                    {"name",  meta->model_name},
                    {"model", meta->model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                get_model_info(),
            }}
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        std::shared_lock meta_lock(meta_mutex);
        // P0-1 (#49): null-meta guard
        if (!meta) {
            res->error(format_error_response("model not loaded — waiting for CONFIGURE",
                                             ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    // ── Merged DECODE result retrieval ─────────────────────────────────────
    this->get_decode_result = [this](const server_http_req & req) {
        auto res = create_response(true);
        int32_t decode_request_id;
        try {
            decode_request_id = std::stoi(req.get_param("decode_request_id"));
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid decode_request_id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::unique_lock lock(decode_results_mutex);
        evict_decode_results_locked();
        auto it = decode_results.find(decode_request_id);
        if (it == decode_results.end()) {
            lock.unlock();
            res->error(format_error_response("decode_request_id not found or expired", ERROR_TYPE_NOT_FOUND));
            return res;
        }

        // Read fields we need before unlocking
        const auto entry_state   = it->second.state;
        const auto entry_error   = it->second.error;
        const int32_t id_slot    = it->second.id_slot;
        const std::string completion_id = it->second.completion_id;
        const std::string oaicompat_model = it->second.oaicompat_model;
        const std::string content = it->second.content;
        const std::string reasoning_content = it->second.reasoning_content;
        const json tool_calls = it->second.tool_calls;
        const int32_t n_decoded  = it->second.n_decoded;
        const int32_t n_prompt_tokens = it->second.n_prompt_tokens;
        const int32_t n_prompt_tokens_cache = it->second.n_prompt_tokens_cache;
        const result_timings timings = it->second.timings;
        const stop_type stop     = it->second.stop;
        const bool include_usage = it->second.include_usage;
        const json hydra_metrics = it->second.hydra_metrics;
        const json match_json    = it->second.match_json;
        const double model_load_ms = it->second.model_load_ms;
        const double restore_slot_ms = it->second.restore_slot_ms;
        const json model_identity = it->second.model_identity;
        lock.unlock();

        // ── Terminal error ─────────────────────────────────────────────────
        if (!entry_error.empty()) {
            {
                std::lock_guard<std::mutex> lk(decode_results_mutex);
                decode_results.erase(decode_request_id);
            }
            json err_j = {
                {"error", entry_error},
                {"error_code", "DECODE_FAILED"},
                {"match", match_json},
            };
            res->error(format_error_response(entry_error, ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        // httplib's own Headers map is case-insensitive
        // (detail::case_ignore::hash, httplib.h), but server-http.cpp
        // get_headers() copies it into a plain case-sensitive
        // std::map<string,string>, so match the Accept header
        // case-insensitively here — otherwise clients sending
        // "Accept: text/event-stream" never reach the SSE branches.
        const bool stream = [&]() {
            for (const auto & [hname, hval] : req.headers) {
                if (hval.find("text/event-stream") == std::string::npos) {
                    continue;
                }
                if (hname.size() != 6) {
                    continue;
                }
                bool is_accept = true;
                for (size_t i = 0; i < 6; i++) {
                    char c = hname[i];
                    if (c >= 'A' && c <= 'Z') {
                        c = (char)(c - 'A' + 'a');
                    }
                    if (c != "accept"[i]) {
                        is_accept = false;
                        break;
                    }
                }
                if (is_accept) {
                    return true;
                }
            }
            return false;
        }();

        // ── In-progress states → 202 ──────────────────────────────────────
        if (entry_state == server_routes::DECODE_STATE_LOADING ||
            entry_state == server_routes::DECODE_STATE_RESTORING) {
            json state_j = {
                {"state", entry_state == server_routes::DECODE_STATE_LOADING ? "loading" : "restoring"},
                {"decode_request_id", decode_request_id},
                {"id_slot", id_slot},
                {"model_load_ms", model_load_ms},
                {"restore_slot_ms", restore_slot_ms},
                {"match", match_json},
            };
            res->status = 202;
            res->data = safe_json_to_str(state_j);
            return res;
        }

        // ── GENERATING + SSE → stream partials from relay queue ─────────
        if (entry_state == server_routes::DECODE_STATE_GENERATING && stream) {
            // Set up SSE response; partials arrive via the streaming_queue
            // populated by the background consumer thread.
            res->status = 200;
            res->content_type = "text/event-stream";
            res->data = ""; // no initial chunk — send headers immediately

            res->next = [this, res_this = res.get(), decode_request_id, &req, sent_final = false](
                    std::string & output) mutable -> bool {
                try {
                    if (req.should_stop()) {
                        return false;
                    }

                    std::unique_lock lock(decode_results_mutex);
                    auto it = decode_results.find(decode_request_id);
                    if (it == decode_results.end()) {
                        output = "data: [DONE]\n\n";
                        return false;
                    }
                    auto & entry = it->second;

                    // Drain streaming queue
                    if (entry.stream) {
                        std::lock_guard<std::mutex> slk(entry.stream->streaming_mutex);
                        if (!entry.stream->streaming_queue.empty()) {
                            auto result = std::move(entry.stream->streaming_queue.front());
                            entry.stream->streaming_queue.pop_front();
                            lock.unlock();

                            if (result->is_error()) {
                                json err_j = format_error_response("generation error", ERROR_TYPE_SERVER);
                                output = format_oai_sse(json{{"error", err_j}});
                                return false;
                            }
                            json j = result->to_json();
                            if (j.is_null()) {
                                // is_begin partial — skip
                                return true;
                            }
                            output = format_oai_sse(j);
                            return true;
                        }
                    }

                    // Queue empty — stream finished: emit the final DONE delta
                    // exactly once, then terminate with [DONE]. Without it a
                    // client attached during GENERATING never sees the final
                    // finish_reason / usage / hydra_metrics. Content is
                    // deliberately NOT repeated: the relay already streamed
                    // content/reasoning_content/tool_calls incrementally via
                    // the partial deltas, so this is OpenAI's empty final
                    // chunk ({"delta": {...}, "finish_reason": ...}) — echoing
                    // full content/tool_calls again would make concat-based
                    // clients see output twice. (The DONE+SSE single-delta
                    // branch below keeps full content: that one fires for
                    // attach-after-DONE clients that saw no partials.)
                    if (entry.stream && entry.stream->stream_finished) {
                        if (!sent_final && entry.state == server_routes::DECODE_STATE_DONE) {
                            sent_final = true;
                            std::time_t t = std::time(0);
                            json delta {
                                {"choices", json::array({
                                    json {
                                        {"finish_reason", entry.stop == STOP_TYPE_WORD || entry.stop == STOP_TYPE_EOS
                                            ? (entry.tool_calls.empty() ? "stop" : "tool_calls")
                                            : "length"},
                                        {"index", 0},
                                        {"delta", json{{"role", "assistant"}, {"content", ""}}},
                                    },
                                })},
                                {"created", t},
                                {"id", entry.completion_id},
                                {"model", entry.oaicompat_model},
                                {"system_fingerprint", std::string(llama_build_info())},
                                {"object", "chat.completion.chunk"},
                            };
                            if (entry.include_usage) {
                                delta["usage"] = json {
                                    {"completion_tokens", entry.n_decoded},
                                    {"prompt_tokens",     entry.n_prompt_tokens},
                                    {"total_tokens",      entry.n_decoded + entry.n_prompt_tokens},
                                    {"prompt_tokens_details", json{{"cached_tokens", entry.n_prompt_tokens_cache}}},
                                };
                            }
                            if (!entry.hydra_metrics.is_null()) {
                                delta["hydra_metrics"] = entry.hydra_metrics;
                            }
                            output = format_oai_sse(delta);
                            return true;
                        }
                        output = "data: [DONE]\n\n";
                        return false;
                    }

                    // Wait for next partial with ping interval
                    if (entry.stream) {
                        entry.stream->streaming_cv.wait_for(lock, std::chrono::seconds(30));
                    }
                    return true; // loop again

                } catch (const std::exception & e) {
                    json err_j = format_error_response(e.what(), ERROR_TYPE_SERVER);
                    output = format_oai_sse(json{{"error", err_j}});
                    return false;
                }
            };
            return res;
        }

        // ── DONE → return full result ─────────────────────────────────────
        if (entry_state == server_routes::DECODE_STATE_DONE || !content.empty()) {
            if (stream) {
                // SSE streaming: send full result as a single delta, then finish
                std::time_t t = std::time(0);
                json delta {
                    {"choices", json::array({
                        json {
                            {"finish_reason", stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS
                                ? (tool_calls.empty() ? "stop" : "tool_calls")
                                : "length"},
                            {"index", 0},
                            {"delta", json{{"role", "assistant"}, {"content", content}}},
                        },
                    })},
                    {"created", t},
                    {"id", completion_id},
                    {"model", oaicompat_model},
                    {"system_fingerprint", std::string(llama_build_info())},
                    {"object", "chat.completion.chunk"},
                };

                if (!reasoning_content.empty()) {
                    delta["choices"][0]["delta"]["reasoning_content"] = reasoning_content;
                }

                if (!tool_calls.empty()) {
                    delta["choices"][0]["delta"]["tool_calls"] = tool_calls;
                }

                if (include_usage) {
                    delta["usage"] = json {
                        {"completion_tokens", n_decoded},
                        {"prompt_tokens",     n_prompt_tokens},
                        {"total_tokens",      n_decoded + n_prompt_tokens},
                        {"prompt_tokens_details", json{{"cached_tokens", n_prompt_tokens_cache}}},
                    };
                }
                if (!hydra_metrics.is_null()) {
                    delta["hydra_metrics"] = hydra_metrics;
                }

                res->status = 200;
                res->content_type = "text/event-stream";
                res->data = format_oai_sse(delta);
            } else {
                // Buffered: full OAI chat completion response
                json message;
                message["role"] = "assistant";
                message["content"] = content;
                if (!reasoning_content.empty()) {
                    message["reasoning_content"] = reasoning_content;
                }
                if (!tool_calls.empty()) {
                    message["tool_calls"] = tool_calls;
                }

                json choice {
                    {"finish_reason", stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS
                        ? (tool_calls.empty() ? "stop" : "tool_calls")
                        : "length"},
                    {"index", 0},
                    {"message", message},
                };

                json oai_response {
                    {"choices", json::array({choice})},
                    {"created", std::time(0)},
                    {"model", oaicompat_model},
                    {"system_fingerprint", std::string(llama_build_info())},
                    {"object", "chat.completion"},
                    {"usage", {
                        {"completion_tokens", n_decoded},
                        {"prompt_tokens",     n_prompt_tokens},
                        {"total_tokens",      n_decoded + n_prompt_tokens},
                        {"prompt_tokens_details", json{{"cached_tokens", n_prompt_tokens_cache}}},
                    }},
                    {"id", completion_id},
                    {"id_slot", id_slot},
                    {"timings", timings.to_json()},
                };
                if (!hydra_metrics.is_null()) {
                    oai_response["hydra_metrics"] = hydra_metrics;
                }

                res->ok(oai_response);
            }
            return res;
        }

        // Fallback: no content yet
        res->error(format_error_response("decode_request_id not ready", ERROR_TYPE_NOT_FOUND));
        return res;
    };

    this->delete_decode_result = [this](const server_http_req & req) {
        auto res = create_response(true);
        int32_t decode_request_id;
        try {
            decode_request_id = std::stoi(req.get_param("decode_request_id"));
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid decode_request_id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int32_t id_slot = -1;
        int32_t task_id_to_cancel = -1;
        {
            std::unique_lock lock(decode_results_mutex);
            auto it = decode_results.find(decode_request_id);
            if (it == decode_results.end()) {
                lock.unlock();
                res->error(format_error_response("decode_request_id not found or expired", ERROR_TYPE_NOT_FOUND));
                return res;
            }
            id_slot = it->second.id_slot;
            task_id_to_cancel = it->second.stream ? it->second.stream->completion_task_id.load() : -1;

            // Signal streaming queue to finish so any waiting GET handler unblocks
            if (it->second.stream) {
                std::lock_guard<std::mutex> slk(it->second.stream->streaming_mutex);
                it->second.stream->stream_finished = true;
                it->second.stream->streaming_cv.notify_all();
            }

            decode_results.erase(it);
        }

        // Deterministically cancel the running completion task via the task queue.
        // SERVER_TASK_TYPE_CANCEL causes the inference thread to release the slot.
        if (task_id_to_cancel > 0) {
            server_task cancel_task(SERVER_TASK_TYPE_CANCEL);
            cancel_task.id = queue_tasks.get_new_id();
            cancel_task.id_target = task_id_to_cancel;
            queue_tasks.post(std::move(cancel_task), true);
            SRV_INF("hydra: DECODE_CANCEL id=%d slot=%d completion_task=%d (cancel posted)\n",
                    decode_request_id, id_slot, task_id_to_cancel);
        } else {
            SRV_INF("hydra: DECODE_CANCEL id=%d slot=%d (no active completion)\n",
                    decode_request_id, id_slot);
        }

        res->ok(json{{"cancelled", true}, {"decode_request_id", decode_request_id}});
        return res;
    };
}

json server_routes::get_model_info() const {
    std::shared_lock meta_lock(meta_mutex);

    // P0-1 (#49): null-meta guard — called from get_models and other paths
    if (!meta) {
        return json{{"error", "model not loaded — waiting for CONFIGURE"}};
    }

    return json {
        {"id",       meta->model_name},
        {"aliases",  meta->model_aliases},
        {"tags",     meta->model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta->model_vocab_type},
            {"n_vocab",     meta->model_vocab_n_tokens},
            {"n_ctx",       meta->slot_n_ctx},
            {"n_ctx_train", meta->model_n_ctx_train},
            {"n_embd",      meta->model_n_embd_inp},
            {"n_params",    meta->model_n_params},
            {"size",        meta->model_size},
        }},
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    std::shared_lock meta_lock(meta_mutex);

    auto res = create_response();
    // P0-1 (#49): null-meta guard
    if (!meta) {
        res->error(format_error_response("model not loaded — waiting for CONFIGURE",
                                         ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}


// epic #610 WS1: Hydra extension implementation. Compiled INTO this TU so the
// concrete class can reach server_context_impl private members via the friend
// declaration above. Do NOT add hydra-server-context.cpp to CMakeLists.txt.
#include "hydra-server-context.cpp"

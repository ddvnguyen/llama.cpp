
#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-checkpoint-policy.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-rpc.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "../src/llama-context.h"
#include "llama-hydra.h"
#include "log.h"
#include "../src/llama-memory-hybrid.h"
#include "preset.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cstddef>
#include <cinttypes>
#include <exception>
#include <memory>
#include <filesystem>
#include <thread>
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

// Forward-declared so the background thread lambda in process_single_task can use it
// before the full RPC helper definitions appear later in this file.
#if !defined(_WIN32)
static bool hydra_send_all(int fd, const void * buf, size_t n);
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

struct server_context_impl {
    friend struct server_context;

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

    // Hydra #383 T1: true when this engine was started in COMBINED static
    // (layer-split) mode. The split is fixed at model load time; the mode
    // cannot be changed at runtime — SET_EXPERT_MODE("solo") is rejected.
    bool hydra_combined_static = false;

    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
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

    bool sleeping = false;

    void destroy() {
        spec.reset();
        ctx_dft.reset();
        model_dft.reset();

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;

        llama_batch_free(batch);
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

        params_base = params;
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
                SRV_ERR("%s", "failed to create MTP context\n");
                return false;
            }

            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());

            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft.get();
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

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
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
            if (!slot.is_processing() && !slot.hydra_transferring->load() && slot.id != exclude_id_slot) {
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

        cur.update_tgt(ctx_tgt,       slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cur.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

        SLT_INF(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    void process_single_task(server_task && task) {
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

            // ── Hydra RPC state-transfer tasks (M1) ──────────────────────────
            // All three cases run on the inference thread so llama API access is safe.
            // The calling RPC thread blocks on queue_results.recv_with_timeout().

            case SERVER_TASK_TYPE_HYDRA_STATE_GET:
                {
                    // M1: background serialization thread — inference loop continues during state transfer.
                    // llama_state_seq_get_data reads KV cells for an IDLE sequence; llama_decode
                    // writes cells for ACTIVE sequences only — no memory overlap for different seq IDs.
                    const int id_slot = task.hydra_action.id_slot;
                    auto res = std::make_unique<server_task_result_hydra_state>();
                    res->id      = task.id;
                    res->id_slot = id_slot;
                    res->op      = HYDRA_OP_STATE_GET;

                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        res->rpc_status = HYDRA_STATUS_NOT_FOUND;
                        res->error      = "invalid slot ID";
                        queue_results.send(std::move(res));
                        break;
                    }
                    if (slot->is_processing() || slot->hydra_transferring->load()) {
                        res->rpc_status = HYDRA_STATUS_BUSY;
                        queue_results.send(std::move(res));
                        break;
                    }

                    // Snapshot on inference thread (cheap — dry-run serialization, no GPU copies).
                    const size_t state_size = llama_state_seq_get_size(ctx_tgt, slot->id);
                    int actual_n_past = slot->n_prompt_tokens_cache + slot->n_decoded;
                    // Cold prefill: n_prompt_tokens_cache is still 0 so n_decoded (1) dominates.
                    // Use prompt token count instead — matches STATE_META fallback.
                    if (slot->n_prompt_tokens_cache == 0 && slot->prompt.tokens.size() > 0) {
                        actual_n_past = (int)slot->prompt.tokens.size();
                    }
                    res->n_past     = actual_n_past;
                    res->rpc_status = HYDRA_STATUS_OK;
                    // M-Perf.9 #289: surface model identity alongside the state
                    // bytes so the Coordinator can record the model that built
                    // the KV (for cross-model safety on restore). The background
                    // thread that streams the bytes to the socket can mutate
                    // res->state_data freely; the model fields are immutable for
                    // the duration of the response.
                    res->model_alias = model_name;
                    res->model_path  = params_base.model.path;
                    if (model_tgt) {
                        const char * hash = llama_model_hash(model_tgt);
                        if (hash && hash[0]) res->model_hash = hash;
                    }
                    SRV_INF("hydra: STATE_GET slot=%d n_past=%d state=%.1f MiB — async\n",
                            id_slot, res->n_past, state_size / (1024.0 * 1024.0));

                    slot->hydra_transferring->store(true);

                    // M2: background thread streams directly to socket (zero-copy).
                    // Falls back to buffer path if fd < 0 (no fd in task).
                    const int      snap_seq_id = slot->id;
                    llama_context * snap_ctx   = ctx_tgt;
                    // shared_ptr keeps the atomic alive even if the slot is reallocated
                    std::shared_ptr<std::atomic<bool>> flag_ptr = slot->hydra_transferring;
                    const int      hydra_fd    = task.hydra_action.hydra_fd;

                    // Capture prompt tokens for M1 path header (slot is valid on inference thread)
                    const llama_tokens prompt_tokens_get = slot->prompt.tokens.get_text_tokens();
                    const int32_t     n_past_val         = res->n_past;

                    // Snapshot the most recent native checkpoint so STATE_PUT can
                    // register it instead of fabricating one at the final position.
                    // Fabricating at pos_max=n-1 corrupts hybrid/recurrent model
                    // decode because the recurrent state is one token ahead of the
                    // decode resume point — it has already processed the final token.
                    std::vector<uint8_t> snapshot_ckpt;
                    uint8_t hdr_flags = 0x00;
                    int32_t ckpt_pos_min = 0, ckpt_pos_max = 0;
                    int64_t ckpt_n_tokens = 0;
                    if (!slot->prompt.checkpoints.empty()) {
                        hdr_flags |= 0x01;
                        const auto & ckpt = slot->prompt.checkpoints.back();
                        ckpt_pos_min = ckpt.pos_min;
                        ckpt_pos_max = ckpt.pos_max;
                        ckpt_n_tokens = ckpt.n_tokens;

                        const uint64_t tgt_sz = ckpt.data_tgt.size();
                        const uint64_t dft_sz = ckpt.data_dft.size();
                        const size_t ckpt_hdr_sz = 4 + 4 + 8 + 8 + (size_t)tgt_sz + 8 + (size_t)dft_sz;
                        snapshot_ckpt.resize(ckpt_hdr_sz);
                        size_t off = 0;
                        memcpy(snapshot_ckpt.data() + off, &ckpt_pos_min, 4); off += 4;
                        memcpy(snapshot_ckpt.data() + off, &ckpt_pos_max, 4); off += 4;
                        memcpy(snapshot_ckpt.data() + off, &ckpt_n_tokens, 8); off += 8;
                        memcpy(snapshot_ckpt.data() + off, &tgt_sz, 8); off += 8;
                        if (tgt_sz > 0) { memcpy(snapshot_ckpt.data() + off, ckpt.data_tgt.data(), (size_t)tgt_sz); off += (size_t)tgt_sz; }
                        memcpy(snapshot_ckpt.data() + off, &dft_sz, 8); off += 8;
                        if (dft_sz > 0) memcpy(snapshot_ckpt.data() + off, ckpt.data_dft.data(), (size_t)dft_sz);
                    }

                    std::thread([snap_ctx, snap_seq_id, state_size, hydra_fd,
                                 res = std::move(res), flag_ptr,
                                 prompt_tokens_get, n_past_val,
                                 hdr_flags, snapshot_ckpt = std::move(snapshot_ckpt),
                                 &results = queue_results]() mutable {
                        SRV_INF("hydra: STATE_GET background thread starting (fd=%d state=%.1f MiB)\n",
                                hydra_fd, state_size / (1024.0 * 1024.0));
                        if (hydra_fd >= 0) {
                            // M2 path: stream v2 blob (header + checkpoint + GPU state) to fd.
                            // Response header + meta JSON sent first, then v2 header bytes,
                            // then llama_state_seq_get_data_to_fd writes GPU state directly.
                            const size_t n_tok = prompt_tokens_get.size();
                            const uint32_t hdr_n_tok = (uint32_t)n_tok;
                            const uint32_t hdr_n_past = (uint32_t)n_past_val;
                            const uint8_t version_byte = 0x02;
                            const size_t base_hdr_size = 1 + 4 + 4 + n_tok * sizeof(llama_token) + 1;
                            const size_t hdr_size = base_hdr_size + snapshot_ckpt.size();
                            const size_t total_payload = hdr_size + state_size;

                            // Build v2 header buffer
                            std::vector<uint8_t> v2_hdr(hdr_size);
                            {
                                size_t off = 0;
                                memcpy(v2_hdr.data() + off, &version_byte, 1); off += 1;
                                memcpy(v2_hdr.data() + off, &hdr_n_past, 4);    off += 4;
                                memcpy(v2_hdr.data() + off, &hdr_n_tok, 4);     off += 4;
                                memcpy(v2_hdr.data() + off, prompt_tokens_get.data(), n_tok * sizeof(llama_token)); off += n_tok * sizeof(llama_token);
                                memcpy(v2_hdr.data() + off, &hdr_flags, 1);     off += 1;
                                if (!snapshot_ckpt.empty()) {
                                    memcpy(v2_hdr.data() + off, snapshot_ckpt.data(), snapshot_ckpt.size());
                                    off += snapshot_ckpt.size();
                                }
                            }

                            {
                                json meta_j;
                                meta_j["n_past"]     = res->n_past;
                                meta_j["state_size"] = (uint64_t)state_size;
                                const std::string meta_str = meta_j.dump();

                                const uint32_t meta_len  = (uint32_t)meta_str.size();
                                const uint64_t payload_l = (uint64_t)total_payload;
                                uint8_t hdr[HYDRA_RES_HEADER_SIZE] = {};
                                hdr[0] = HYDRA_STATUS_OK;
                                hdr[1] = (meta_len)       & 0xFF;
                                hdr[2] = (meta_len >>  8) & 0xFF;
                                hdr[3] = (meta_len >> 16) & 0xFF;
                                memcpy(hdr + 4, &payload_l, 8);
                                hydra_send_all(hydra_fd, hdr,           HYDRA_RES_HEADER_SIZE);
                                hydra_send_all(hydra_fd, meta_str.data(), meta_str.size());

                                // Write v2 blob header before GPU state — STATE_PUT needs tokens + checkpoint
                                hydra_send_all(hydra_fd, v2_hdr.data(), v2_hdr.size());

                                res->header_sent = true; // META + header + v2-hdr before payload
                            }
                             // Stream GPU state to fd (zero-copy from GPU memory)
                             const size_t streamed = llama_state_seq_get_data_to_fd(snap_ctx, snap_seq_id, hydra_fd);
                             if (streamed != state_size) {
                                 // TOCTOU: state size changed between get_size (header already
                                 // promised state_size bytes) and the stream, or the stream
                                 // failed mid-way. The wire framing is now broken — the only
                                 // safe recovery is to kill the connection. Use shutdown(),
                                 // not close(): the RPC connection loop owns the fd and will
                                 // close it when its next read fails; closing here would race
                                 // (double-close / fd-reuse against unrelated threads).
                                 res->rpc_status = HYDRA_STATUS_ERROR;
                                 res->error      = "llama_state_seq_get_data_to_fd streamed " +
                                                   std::to_string(streamed) + " B, expected " +
                                                   std::to_string(state_size) + " B";
                                 ::shutdown(hydra_fd, SHUT_RDWR);
                             } else {
                                 res->streamed_bytes = total_payload;
                             }
                         } else {
                             // M1 path: buffer in memory, RPC thread sends afterwards.
                             // v2 blob format (0x02): [1B version][4B n_past][4B n_tok][n_tok*4B tokens]
                             //   [1B flags (bit 0 = has_checkpoint)]
                             //   [if flags & 0x01: 4B pos_min | 4B pos_max | 8B n_tokens | 8B tgt_sz | tgt_data | 8B dft_sz | dft_data]
                             //   [raw KV state from llama_state_seq_get_data]
                             const size_t n_tok = prompt_tokens_get.size();
                             const uint32_t hdr_n_tok = (uint32_t)n_tok;
                             const uint32_t hdr_n_past = (uint32_t)n_past_val;
                             const uint8_t version_byte = 0x02;
                             const size_t base_hdr_size = 1 + 4 + 4 + n_tok * sizeof(llama_token) + 1; // version + n_past + n_tok + tokens + flags
                             const size_t hdr_size = base_hdr_size + snapshot_ckpt.size();

                              // TOCTOU retry: if another slot grew the state between
                              // get_size (inference thread) and get_data (background thread),
                              // the copy returns 0. Retry up to 3 times with fresh sizing.
                              size_t buf_size = hdr_size + state_size;
                              res->state_data.resize(buf_size);
                              {
                                  size_t off = 0;
                                  memcpy(res->state_data.data() + off, &version_byte, 1); off += 1;
                                  memcpy(res->state_data.data() + off, &hdr_n_past, 4);    off += 4;
                                  memcpy(res->state_data.data() + off, &hdr_n_tok, 4);     off += 4;
                                  memcpy(res->state_data.data() + off, prompt_tokens_get.data(), n_tok * sizeof(llama_token)); off += n_tok * sizeof(llama_token);
                                  memcpy(res->state_data.data() + off, &hdr_flags, 1);     off += 1;
                                  if (!snapshot_ckpt.empty()) {
                                      memcpy(res->state_data.data() + off, snapshot_ckpt.data(), snapshot_ckpt.size());
                                      off += snapshot_ckpt.size();
                                  }
                              }

                              size_t cur_state_size = state_size;
                              size_t copied = 0;
                              int retries = 3;
                              while (retries-- > 0) {
                                  copied = llama_state_seq_get_data(
                                          snap_ctx, res->state_data.data() + hdr_size, cur_state_size, snap_seq_id);
                                  if (copied > 0) break;
                                  // State grew — re-measure and retry
                                  cur_state_size = llama_state_seq_get_size(snap_ctx, snap_seq_id);
                                  res->state_data.resize(hdr_size + cur_state_size);
                              }
                              if (copied == 0) {
                                  res->rpc_status = HYDRA_STATUS_ERROR;
                                  res->error      = "llama_state_get_data failed after 3 retries";
                                  res->state_data.clear();
                              }
                         }
                        flag_ptr->store(false);
                        // M2 streams to fd (streamed_bytes); M1 buffers into state_data.
                        const uint64_t out_bytes = (hydra_fd >= 0)
                                ? res->streamed_bytes
                                : (uint64_t) res->state_data.size();
                        SRV_INF("hydra: STATE_GET background done slot=%d rpc_status=%d path=%s bytes=%" PRIu64 "\n",
                                snap_seq_id, res->rpc_status,
                                hydra_fd >= 0 ? "M2-stream" : "M1-buffer", out_bytes);
                        results.send(std::move(res));
                    }).detach();

                    // Inference thread returns immediately — no stall on decode for other slots.
                } break;

            case SERVER_TASK_TYPE_HYDRA_STATE_PUT:
                {
                    const int id_slot = task.hydra_action.id_slot;
                    auto res = std::make_unique<server_task_result_hydra_state>();
                    res->id      = task.id;
                    res->id_slot = id_slot;
                    res->op      = HYDRA_OP_STATE_PUT;

                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        res->rpc_status = HYDRA_STATUS_NOT_FOUND;
                        res->error      = "invalid slot ID";
                        queue_results.send(std::move(res));
                        break;
                    }
                    if (slot->is_processing() || slot->hydra_transferring->load()) {
                        res->rpc_status = HYDRA_STATUS_BUSY;
                        queue_results.send(std::move(res));
                        break;
                    }

                    // Erase existing checkpoints to avoid collision with restored session state
                    if (task.hydra_action.erase_existing && !slot->prompt.checkpoints.empty()) {
                        SLT_INF(*slot, "erasing %zu existing checkpoints before STATE_PUT restore\n",
                                slot->prompt.checkpoints.size());
                        slot->prompt.checkpoints.clear();
                    }

                    const auto & buf = task.hydra_action.state_data;

                    // Detect v2 blob (0x02 at offset 0) vs legacy format (no version byte).
                    // v2: [1B version=0x02][4B n_past][4B n_tok][n_tok*4B tokens][1B flags][?ckpt?][KV state]
                    // legacy: [4B n_past][4B n_tok][n_tok*4B tokens][KV state]
                    const bool is_v2 = buf.size() >= 1 && buf[0] == 0x02;

                    size_t hdr_offset = 0;
                    int32_t hdr_n_tok = 0;
                    int32_t hdr_n_past = 0;
                    bool has_chkpt = false;
                    int32_t ckpt_pos_min_in = 0, ckpt_pos_max_in = 0;
                    int64_t ckpt_n_tokens_in = 0;
                    std::vector<uint8_t> ckpt_tgt_data, ckpt_dft_data;

                    if (is_v2) {
                        // v2: version at [0], n_past at [1..4], n_tok at [5..8]
                        if (buf.size() >= 9) {
                            memcpy(&hdr_n_past, buf.data() + 1, 4);
                            memcpy(&hdr_n_tok,  buf.data() + 5, 4);
                        }
                        const size_t token_start = 9;
                        const size_t token_end = token_start + (size_t)hdr_n_tok * sizeof(llama_token);
                        hdr_offset = token_end;
                        if (hdr_offset < buf.size()) {
                            const uint8_t flags = buf[hdr_offset];
                            hdr_offset += 1; // past flags byte
                            if (flags & 0x01) {
                                // Parse checkpoint: 4B pos_min | 4B pos_max | 8B n_tokens | 8B tgt_sz | tgt_data | 8B dft_sz | dft_data
                                if (hdr_offset + 4 + 4 + 8 + 8 <= buf.size()) {
                                    memcpy(&ckpt_pos_min_in, buf.data() + hdr_offset, 4); hdr_offset += 4;
                                    memcpy(&ckpt_pos_max_in, buf.data() + hdr_offset, 4); hdr_offset += 4;
                                    memcpy(&ckpt_n_tokens_in, buf.data() + hdr_offset, 8); hdr_offset += 8;
                                    uint64_t tgt_sz_in;
                                    memcpy(&tgt_sz_in, buf.data() + hdr_offset, 8); hdr_offset += 8;
                                    if (tgt_sz_in > 0 && hdr_offset + tgt_sz_in <= buf.size()) {
                                        ckpt_tgt_data.assign(buf.data() + hdr_offset, buf.data() + hdr_offset + (size_t)tgt_sz_in);
                                        hdr_offset += (size_t)tgt_sz_in;
                                    }
                                    if (hdr_offset + 8 <= buf.size()) {
                                        uint64_t dft_sz_in;
                                        memcpy(&dft_sz_in, buf.data() + hdr_offset, 8); hdr_offset += 8;
                                        if (dft_sz_in > 0 && hdr_offset + dft_sz_in <= buf.size()) {
                                            ckpt_dft_data.assign(buf.data() + hdr_offset, buf.data() + hdr_offset + (size_t)dft_sz_in);
                                            hdr_offset += (size_t)dft_sz_in;
                                        }
                                    }
                                    has_chkpt = true;
                                }
                            }
                        }
                        // Restore tokens from token_start
                        if (hdr_n_tok > 0 && token_start + (size_t)hdr_n_tok * sizeof(llama_token) <= buf.size()) {
                            slot->prompt.tokens.clear();
                            const llama_token * tok_ptr = (const llama_token *)(buf.data() + token_start);
                            llama_tokens restored_tokens(tok_ptr, tok_ptr + (size_t)hdr_n_tok);
                            slot->prompt.tokens.insert(restored_tokens);
                        }
                    } else {
                        // Legacy v1 format
                        if (buf.size() >= 8) {
                            memcpy(&hdr_n_past, buf.data(), 4);
                            memcpy(&hdr_n_tok, buf.data() + 4, 4);
                            hdr_offset = 8 + (size_t)hdr_n_tok * sizeof(llama_token);
                        }
                        if (hdr_offset > 0 && hdr_offset <= buf.size()) {
                            const size_t n_tokens = (size_t)hdr_n_tok;
                            slot->prompt.tokens.clear();
                            if (n_tokens > 0) {
                                const llama_token * tok_ptr = (const llama_token *)(buf.data() + 8);
                                llama_tokens restored_tokens(tok_ptr, tok_ptr + n_tokens);
                                slot->prompt.tokens.insert(restored_tokens);
                            }
                        }
                    }
                    const bool has_hdr = hdr_offset > 0 && hdr_offset <= buf.size();
                    const uint8_t * state_ptr = has_hdr ? buf.data() + hdr_offset : buf.data();
                    const size_t    state_len = has_hdr ? buf.size() - hdr_offset : buf.size();
                    const size_t n_read = llama_state_seq_set_data(ctx_tgt, state_ptr, state_len, slot->id);
                    if (n_read == 0) {
                        res->rpc_status = HYDRA_STATUS_ERROR;
                        res->error      = "llama_state_set_data returned 0";
                        // Tokens were registered before set_data — clear them so the slot
                        // is not left poisoned (n_past > 0 with no KV cells → pos_min == -1
                        // abort on the next decode that touches this slot).
                        slot->prompt.tokens.clear();
                        slot->prompt.checkpoints.clear();
                        slot->n_prompt_tokens_cache = 0;
                        llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot->id, -1, -1);
                    } else {
                        // Inject trailing logits if present — activation handoff from PREFILL.
                        // PREFILL appends n_vocab floats after the KV state so the decode GPU
                        // can call common_sampler_sample immediately without a re-prefill pass.
                        const size_t remaining = state_len - n_read;
                        const size_t expected_logits = (size_t)llama_vocab_n_tokens(vocab) * sizeof(float);
                        if (remaining == expected_logits) {
                            float * ctx_logits = llama_get_logits(ctx_tgt);
                            if (ctx_logits) {
                                memcpy(ctx_logits, state_ptr + n_read, expected_logits);
                                SRV_INF("hydra: STATE_PUT slot=%d injected %zu B logits\n",
                                        id_slot, expected_logits);
                            }
                        }

                        res->rpc_status = HYDRA_STATUS_OK;
                        res->restored   = true;
                        res->bytes      = (uint64_t)n_read;
                        if (hdr_n_tok > 0) {
                            slot->n_prompt_tokens_cache = hdr_n_tok;
                            slot->n_decoded = 0;
                            res->n_past = hdr_n_tok;

                            // Register native checkpoint from the blob (v2) or fabricate one (legacy).
                            // The native checkpoint has pos_max at n-4 (created before the last
                            // few prompt tokens were decoded), so loading it rewinds the recurrent
                            // state to a clean position. The old fabricated checkpoint at (0, n-1)
                            // puts the recurrent state at the final position — one token ahead of
                            // where decode must resume — corrupting hybrid/recurrent model output.
                            slot->prompt.checkpoints.clear();
                            if (has_chkpt) {
                                auto & ckpt = slot->prompt.checkpoints.emplace_back();
                                ckpt.n_tokens = ckpt_n_tokens_in;
                                ckpt.pos_min  = ckpt_pos_min_in;
                                ckpt.pos_max  = ckpt_pos_max_in;
                                ckpt.data_tgt = std::move(ckpt_tgt_data);
                                ckpt.data_dft = std::move(ckpt_dft_data);
                                SLT_INF(*slot, "STATE_PUT registered native checkpoint (pos_min=%d pos_max=%d n_tokens=%" PRId64 " tgt_sz=%zu)\n",
                                        ckpt.pos_min, ckpt.pos_max, ckpt.n_tokens, ckpt.data_tgt.size());
                            } else {
                                create_checkpoint(*slot, 0, 0, (llama_pos)(hdr_n_tok - 1));
                            }
                            slot->just_restored = true;
                        }
                        SRV_INF("hydra: STATE_PUT slot=%d restored=%zu B n_past=%d n_prompt_tok=%d\n",
                                id_slot, n_read, res->n_past, hdr_n_tok);
                    }
                    queue_results.send(std::move(res));
                } break;

            case SERVER_TASK_TYPE_HYDRA_STATE_META:
                {
                    const int id_slot = task.hydra_action.id_slot;
                    auto res = std::make_unique<server_task_result_hydra_state>();
                    res->id      = task.id;
                    res->id_slot = id_slot;
                    res->op      = HYDRA_OP_STATE_META;

                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        res->rpc_status = HYDRA_STATUS_NOT_FOUND;
                        res->error      = "invalid slot ID";
                        queue_results.send(std::move(res));
                        break;
                    }
                    // META is safe to serve even while processing or transferring (read-only metadata)
                    int actual_n_past = slot->n_prompt_tokens_cache + slot->n_decoded;
                    // For cold prefills n_prompt_tokens_cache is 0 — use prompt token count
                    if (slot->n_prompt_tokens_cache == 0 && slot->prompt.tokens.size() > 0) {
                        actual_n_past = (int)slot->prompt.tokens.size();
                    }
                    res->n_past = actual_n_past;
                    res->is_processing   = slot->is_processing();
                    res->is_transferring = slot->hydra_transferring->load();
                    res->state_size    = (uint64_t)llama_state_seq_get_size(ctx_tgt, slot->id);
                    // M-Perf.9 #289: surface model identity. The Coordinator uses
                    // these to detect cross-model restores — a slot holding a Mini
                    // KV cache must never have it decoded by a Balanced-loaded model.
                    res->model_alias = model_name;
                    res->model_path  = params_base.model.path;
                    if (model_tgt) {
                        const char * hash = llama_model_hash(model_tgt);
                        if (hash && hash[0]) res->model_hash = hash;
                    }
                    res->rpc_status    = HYDRA_STATUS_OK;
                    queue_results.send(std::move(res));
                } break;

            case SERVER_TASK_TYPE_HYDRA_ENGINE_CONFIGURE:
                {
                    auto res = std::make_unique<server_task_result_hydra_engine>();
                    res->id = task.id;
                    res->op = HYDRA_OP_CONFIGURE;
                    res->rpc_status = HYDRA_STATUS_OK;
                    res->success = true;
                    // hydra#334: "state_chunk_size" (bytes) tunes the STATE_GET
                    // socket-stream chunk size (llama_io_write_socket) without a
                    // rebuild. Unknown/absent keys are ignored — CONFIGURE is meant
                    // to accept a superset of engine params over time.
                    if (!task.hydra_action.config_json.empty()) {
                        try {
                            const json cfg = json::parse(task.hydra_action.config_json);
                            if (ctx_tgt && cfg.contains("state_chunk_size")) {
                                const size_t bytes = cfg.at("state_chunk_size").get<size_t>();
                                llama_hydra_set_state_chunk_size(ctx_tgt, bytes);
                                // Echo the post-clamp value back so the Coordinator can
                                // tell a silent clamp from "exactly what I asked for"
                                // instead of trusting an unconditional success response.
                                res->state_chunk_size_applied = (uint64_t)llama_hydra_get_state_chunk_size(ctx_tgt);
                                SRV_INF("hydra: CONFIGURE state_chunk_size requested=%zu applied=%" PRIu64 " (slot %d)\n",
                                        bytes, res->state_chunk_size_applied, task.hydra_action.id_slot);
                            }
                        } catch (const std::exception & e) {
                            res->success = false;
                            res->rpc_status = HYDRA_STATUS_ERROR;
                            res->error = std::string("CONFIGURE: invalid config_json: ") + e.what();
                            SRV_WRN("hydra: CONFIGURE failed to parse config_json (slot %d): %s\n",
                                    task.hydra_action.id_slot, e.what());
                        }
                    }
                    SRV_INF("hydra: CONFIGURE received (slot %d)\n", task.hydra_action.id_slot);
                    queue_results.send(std::move(res));
                } break;

            case SERVER_TASK_TYPE_HYDRA_ENGINE_INFO:
                {
                    auto res = std::make_unique<server_task_result_hydra_engine>();
                    res->id = task.id;
                    res->op = HYDRA_OP_INFO;
                    res->rpc_status = HYDRA_STATUS_OK;
                    // M-Perf.9 #289: advertise the model identity features so
                    // the Coordinator knows it can send `model` in PREFILL and
                    // expect model_alias/model_hash/model_path in META responses.
                    // `preset_aliases` lists every alias loaded from
                    // --models-preset (empty when no preset is configured).
                    json preset_aliases_j = json::array();
                    for (const auto & [alias, _path] : preset_alias_to_path) {
                        preset_aliases_j.push_back(alias);
                    }
                    // Hydra #287/#260/#348: two-engine "work together" status
                    // — see specs/rpc-protocol.md's ENGINE_INFO (0x41)
                    // contract. pipeline_capable stays false until #287's
                    // PIPELINE half lands; mode only ever reports
                    // solo/combined until then. solo_active/rpc_backend_active/
                    // peer_reachable/combined_head_attached are independent
                    // booleans (#348) — replaces the old single "role" string
                    // and the peer_connected/combined_capable field-aliasing.
                    const int32_t expert_mode = ctx_tgt ? llama_hydra_get_expert_mode(ctx_tgt) : 0;
                    // Hydra #383 T1 / #375: advertise "combined" capability when this
                    // engine is ready to serve in COMBINED mode — either via expert-split
                    // (hydra_combined_head_attached) or via layer-split (hydra_combined_static).
                    json capabilities_j = {"prefill", "decode", "state_transfer",
                                           "expert_mode", "quant_swap",
                                           "preset", "model_hash"};
                    if (hydra_combined_head_attached || hydra_combined_static) {
                        capabilities_j.push_back("combined");
                    }
                    // In layer-split static mode the engine is always in combined mode;
                    // in expert-split mode it follows the per-request SET_EXPERT_MODE state.
                    const std::string mode_str = hydra_combined_static ? "combined"
                                               : (expert_mode == 1 ? "combined" : "solo");
                    json info_j = {
                        {"engine", "llama-server-hydra"},
                        {"version", "E1"},
                        {"capabilities", capabilities_j},
                        {"preset_aliases", preset_aliases_j},
                        {"solo_active",            hydra_solo_active},
                        {"rpc_backend_active",     hydra_rpc_backend_active},
                        {"mode",                   mode_str},
                        {"split_mode",             hydra_split_mode},
                        {"peer_addr",              hydra_peer},
                        {"peer_reachable",         hydra_peer_reachable},
                        {"layer_split",            hydra_combined_pattern},
                        {"combined_head_attached", hydra_combined_head_attached || hydra_combined_static},
                        {"pipeline_capable",       false}
                    };
                    res->info_json = info_j.dump();
                    queue_results.send(std::move(res));
                } break;

            case SERVER_TASK_TYPE_HYDRA_ENGINE_PREFILL:
                {
                    const int id_slot = task.hydra_action.id_slot;
                    auto res = std::make_unique<server_task_result_hydra_engine>();
                    res->id = task.id;
                    res->op = HYDRA_OP_PREFILL;

                    // Set by the model-resolution block below when a real
                    // `load_model` swap happens. Used at the response site to
                    // decide whether the post-prefill model identity is the
                    // freshly loaded model (swap) or the original (no-swap /
                    // fallback).
                    bool model_was_swapped = false;

                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        res->rpc_status = HYDRA_STATUS_NOT_FOUND;
                        res->error = "invalid slot ID";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (slot->is_processing()) {
                        res->rpc_status = HYDRA_STATUS_BUSY;
                        res->error = "slot is busy";
                        queue_results.send(std::move(res));
                        break;
                    }

                    // M-Perf.9 #289: parse the optional `model` key from the
                    // request body and swap the resident model when the preset
                    // registry knows the alias. The parse is reused for the
                    // tokenization step below. Falls back to the resident model
                    // (with `model_fallback:true` in the response) when the
                    // alias is unknown or no preset is configured.
                    json parsed_body;
                    std::string requested_model;
                    if (!task.hydra_action.request_json.empty()) {
                        try {
                            parsed_body = json::parse(task.hydra_action.request_json);
                            if (parsed_body.is_object() && parsed_body.contains("model")
                                && parsed_body["model"].is_string()) {
                                requested_model = parsed_body["model"].get<std::string>();
                            }
                        } catch (const std::exception & e) {
                            res->rpc_status = HYDRA_STATUS_BAD_REQUEST;
                            res->error = std::string("invalid JSON: ") + e.what();
                            queue_results.send(std::move(res));
                            break;
                        }
                    }

                    if (!requested_model.empty()) {
                        auto it = preset_alias_to_path.find(requested_model);
                        if (it == preset_alias_to_path.end()) {
                            SRV_WRN("hydra: PREFILL model='%s' unknown (preset has %zu alias(es)) — falling back to resident '%s'\n",
                                    requested_model.c_str(), preset_alias_to_path.size(),
                                    model_name.c_str());
                            res->model_fallback = true;
                        } else if (it->second != params_base.model.path) {
                            SRV_INF("hydra: PREFILL model='%s' swapping %s -> %s\n",
                                    requested_model.c_str(), params_base.model.path.c_str(),
                                    it->second.c_str());
                            common_params swapped_params = params_base;
                            swapped_params.model.path   = it->second;
                            // Update the alias so model_name is re-derived
                            // correctly in load_model() (model_name is set from
                            // model_alias.first when non-empty).
                            swapped_params.model_alias  = { requested_model };
                            if (!load_model(swapped_params)) {
                                res->rpc_status = HYDRA_STATUS_ERROR;
                                res->error = "model swap to '" + requested_model + "' failed";
                                queue_results.send(std::move(res));
                                break;
                            }
                            // After load_model, `this` state is reset (new
                            // slots, new context). Re-look up the slot by id.
                            slot = get_slot_by_id(id_slot);
                            if (slot == nullptr) {
                                res->rpc_status = HYDRA_STATUS_NOT_FOUND;
                                res->error = "slot disappeared after model swap";
                                queue_results.send(std::move(res));
                                break;
                            }
                        } else {
                            SRV_DBG("hydra: PREFILL model='%s' already resident, no swap\n",
                                    requested_model.c_str());
                        }
                    }

                    // Tokenize from JSON messages if request_json is provided;
                    // otherwise fall back to pre-tokenized prompt_tokens for back-compat.
                    std::vector<llama_token> prompt_tokens = std::move(task.hydra_action.prompt_tokens);
                    if (!parsed_body.is_null()) {
                        try {
                            std::vector<raw_buffer> dummy_files;
                            json parsed = oaicompat_chat_params_parse(parsed_body, chat_params, dummy_files);
                            if (!parsed.contains("prompt")) {
                                res->rpc_status = HYDRA_STATUS_ERROR;
                                res->error = "chat template produced no prompt";
                                queue_results.send(std::move(res));
                                break;
                            }
                            auto tokenized = tokenize_input_prompts(vocab, mctx, parsed["prompt"], true, true);
                            if (tokenized.empty()) {
                                res->rpc_status = HYDRA_STATUS_ERROR;
                                res->error = "tokenization produced no tokens";
                                queue_results.send(std::move(res));
                                break;
                            }
                            prompt_tokens = tokenized[0].get_tokens();
                        } catch (const std::exception & e) {
                            res->rpc_status = HYDRA_STATUS_ERROR;
                            res->error = std::string("JSON/tokenization error: ") + e.what();
                            queue_results.send(std::move(res));
                            break;
                        }
                    }

                    SRV_INF("hydra: PREFILL slot=%d tokens=%zu\n", id_slot, prompt_tokens.size());

                    // Clear existing slot state
                    slot->prompt_clear(false);
                    slot->n_prompt_tokens_cache = 0;
                    slot->n_prompt_tokens_processed = 0;
                    slot->n_decoded = 0;

                    // Insert prompt tokens
                    if (prompt_tokens.empty()) {
                        res->rpc_status = HYDRA_STATUS_OK;
                        res->n_past = 0;
                        res->state_size = 0;
                        queue_results.send(std::move(res));
                        break;
                    }

                    slot->prompt.tokens.insert(prompt_tokens);
                    const auto & tokens = slot->prompt.tokens.get_tokens();
                    const int n_tokens = (int)tokens.size();

                    // Add BOS if needed (check if slot uses BOS)
                    int token_offset = 0;
                    llama_token bos = llama_vocab_bos(vocab);
                    if (add_bos_token && bos != LLAMA_TOKEN_NULL && (tokens.empty() || tokens[0] != bos)) {
                        token_offset = 1;
                    }

                    // Decode prompt in batches
                    const int n_ubatch = llama_n_ubatch(ctx_tgt);
                    bool decode_ok = true;
                    for (int i = 0; i < n_tokens + token_offset; i += n_ubatch) {
                        const int n_tokens_batch = std::min(n_ubatch, n_tokens + token_offset - i);
                        common_batch_clear(batch);
                        for (int j = 0; j < n_tokens_batch; j++) {
                            const int tok_idx = i + j;
                            llama_token id;
                            if (token_offset > 0 && tok_idx == 0) {
                                id = bos;
                            } else {
                                id = tokens[tok_idx - token_offset];
                            }
                            const bool need_logits = (tok_idx == n_tokens + token_offset - 1);
                            common_batch_add(batch, id, tok_idx, {slot->id}, need_logits);
                        }
                        if (llama_decode(ctx_tgt, batch) != 0) {
                            SRV_ERR("hydra: PREFILL slot=%d llama_decode failed at batch %d\n", id_slot, i);
                            decode_ok = false;
                            break;
                        }
                    }

                    if (!decode_ok) {
                        res->rpc_status = HYDRA_STATUS_ERROR;
                        res->error = "llama_decode failed during prefill";
                        queue_results.send(std::move(res));
                        break;
                    }

                    // Update slot tracking
                    slot->n_prompt_tokens_processed = n_tokens;
                    slot->n_prompt_tokens_cache = n_tokens;

                    // Register checkpoint BEFORE getting state so v2 header includes it
                    if (n_tokens > 0) {
                        create_checkpoint(*slot, 0, 0, (llama_pos)(n_tokens - 1));
                    }

                    // Build v2 blob: [1B version=0x02][4B n_past][4B n_tok][n_tok*4B tokens][1B flags][?ckpt?][raw KV state]
                    const uint32_t hdr_n_past = (uint32_t)n_tokens;
                    const uint32_t hdr_n_tok  = (uint32_t)(tokens.size());
                    uint8_t hdr_flags = 0x00;
                    std::vector<uint8_t> ckpt_buf;
                    int32_t ckpt_pos_min = 0, ckpt_pos_max = 0;
                    int64_t ckpt_n_tokens = 0;
                    if (!slot->prompt.checkpoints.empty()) {
                        hdr_flags |= 0x01;
                        const auto & ckpt = slot->prompt.checkpoints.back();
                        ckpt_pos_min = ckpt.pos_min;
                        ckpt_pos_max = ckpt.pos_max;
                        ckpt_n_tokens = ckpt.n_tokens;
                        const uint64_t tgt_sz = ckpt.data_tgt.size();
                        const uint64_t dft_sz = ckpt.data_dft.size();
                        ckpt_buf.resize(4 + 4 + 8 + 8 + (size_t)tgt_sz + 8 + (size_t)dft_sz);
                        size_t off = 0;
                        memcpy(ckpt_buf.data() + off, &ckpt_pos_min, 4); off += 4;
                        memcpy(ckpt_buf.data() + off, &ckpt_pos_max, 4); off += 4;
                        memcpy(ckpt_buf.data() + off, &ckpt_n_tokens, 8); off += 8;
                        memcpy(ckpt_buf.data() + off, &tgt_sz, 8); off += 8;
                        if (tgt_sz > 0) { memcpy(ckpt_buf.data() + off, ckpt.data_tgt.data(), (size_t)tgt_sz); off += (size_t)tgt_sz; }
                        memcpy(ckpt_buf.data() + off, &dft_sz, 8); off += 8;
                        if (dft_sz > 0) memcpy(ckpt_buf.data() + off, ckpt.data_dft.data(), (size_t)dft_sz);
                    }
                    const size_t base_hdr_size = 1 + 4 + 4 + hdr_n_tok * sizeof(llama_token) + 1;
                    const size_t v2_size = base_hdr_size + ckpt_buf.size();

                    // Get raw KV state
                    const size_t state_size = llama_state_seq_get_size(ctx_tgt, slot->id);
                    std::vector<uint8_t> v2_blob(v2_size + state_size);
                    {
                        size_t off = 0;
                        const uint8_t version_byte = 0x02;
                        memcpy(v2_blob.data() + off, &version_byte, 1); off += 1;
                        memcpy(v2_blob.data() + off, &hdr_n_past, 4);   off += 4;
                        memcpy(v2_blob.data() + off, &hdr_n_tok, 4);    off += 4;
                        if (hdr_n_tok > 0) {
                            const auto & toks = slot->prompt.tokens.get_text_tokens();
                            memcpy(v2_blob.data() + off, toks.data(), toks.size() * sizeof(llama_token));
                            off += toks.size() * sizeof(llama_token);
                        }
                        memcpy(v2_blob.data() + off, &hdr_flags, 1);    off += 1;
                        if (!ckpt_buf.empty()) {
                            memcpy(v2_blob.data() + off, ckpt_buf.data(), ckpt_buf.size());
                            off += ckpt_buf.size();
                        }
                        if (state_size > 0) {
                            llama_state_seq_get_data(ctx_tgt, v2_blob.data() + off, state_size, slot->id);
                        }
                    }

                    // Append logits for activation handoff — eliminates the 1-token trick on the
                    // decode GPU. llama_state_seq_get_data saves KV (k/v tensors) but not the
                    // logits buffer; without these, common_sampler_sample reads garbage after
                    // StatePut. Appending n_vocab floats here lets STATE_PUT inject them directly
                    // into ctx->logits so DECODE can sample immediately.
                    uint64_t logits_size = 0;
                    const int n_vocab = llama_vocab_n_tokens(vocab);
                    const float * logits_ptr = llama_get_logits(ctx_tgt);
                    if (logits_ptr && n_vocab > 0) {
                        logits_size = (uint64_t)n_vocab * sizeof(float);
                        const size_t old_sz = v2_blob.size();
                        v2_blob.resize(old_sz + (size_t)logits_size);
                        memcpy(v2_blob.data() + old_sz, logits_ptr, (size_t)logits_size);
                    }

                    SRV_INF("hydra: PREFILL slot=%d done n_past=%d kv=%zu logits=%" PRIu64 "B total=%zu\n",
                            id_slot, n_tokens, state_size, logits_size, v2_blob.size());

                    // M-Perf.9 #289: model identity for the slot the prefill
                    // was just built on. Coordinator uses this to populate
                    // item.KvModelAlias/Hash and to gate RestoreKvAsync. When
                    // a `model` swap happened earlier in this handler, the
                    // post-swap `model_name` / `params_base.model.path` /
                    // `model` are used. `res->model_fallback` was set by the
                    // model-resolution block above; we preserve it here.
                    res->model_alias    = model_name;
                    res->model_path     = params_base.model.path;
                    // res->model_fallback may already be true (alias unknown
                    // or no preset); only set false when no swap was needed.
                    if (!model_was_swapped && !res->model_fallback) {
                        // nothing to do — leave as-is
                    }
                    if (model_tgt) {
                        const char * hash = llama_model_hash(model_tgt);
                        if (hash && hash[0]) res->model_hash = hash;
                    }

                    res->rpc_status  = HYDRA_STATUS_OK;
                    res->n_past      = n_tokens;
                    res->state_data  = std::move(v2_blob);
                    res->state_size  = state_size;
                    res->logits_size = logits_size;
                    queue_results.send(std::move(res));
                } break;

            case SERVER_TASK_TYPE_HYDRA_ENGINE_DECODE:
                {
                    const int id_slot = task.hydra_action.id_slot;
                    auto res = std::make_unique<server_task_result_hydra_engine>();
                    res->id = task.id;
                    res->op = HYDRA_OP_DECODE;

                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        res->rpc_status = HYDRA_STATUS_NOT_FOUND;
                        res->error = "invalid slot ID";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (slot->is_processing()) {
                        res->rpc_status = HYDRA_STATUS_BUSY;
                        res->error = "slot is busy";
                        queue_results.send(std::move(res));
                        break;
                    }

                    const int n_predict = task.hydra_action.n_predict;
                    const int stream_fd = task.hydra_action.stream_fd;

                    // Determine if atomic (has messages) or cross-GPU (KV already on slot).
                    // Always called via RPC with a JSON payload; cross-GPU sends
                    // {"n_predict":N,"messages":null} while atomic sends messages as an array.
                    bool atomically_prefill = false;
                    std::vector<llama_token> prompt_tokens;
                    if (!task.hydra_action.request_json.empty()) {
                        try {
                            json body = json::parse(task.hydra_action.request_json);
                            // Only do atomic prefill when messages is present AND not null
                            if (body.contains("messages") && !body["messages"].is_null()) {
                                atomically_prefill = true;
                                std::vector<raw_buffer> dummy_files;
                                json parsed = oaicompat_chat_params_parse(body, chat_params, dummy_files);
                                if (!parsed.contains("prompt")) {
                                    res->rpc_status = HYDRA_STATUS_ERROR;
                                    res->error = "chat template produced no prompt";
                                    queue_results.send(std::move(res));
                                    break;
                                }
                                auto tokenized = tokenize_input_prompts(vocab, mctx, parsed["prompt"], true, true);
                                if (tokenized.empty()) {
                                    res->rpc_status = HYDRA_STATUS_ERROR;
                                    res->error = "tokenization produced no tokens";
                                    queue_results.send(std::move(res));
                                    break;
                                }
                                prompt_tokens = tokenized[0].get_tokens();
                            }
                        } catch (const std::exception & e) {
                            res->rpc_status = HYDRA_STATUS_ERROR;
                            res->error = std::string("JSON/tokenization error: ") + e.what();
                            queue_results.send(std::move(res));
                            break;
                        }
                    }

                    if (atomically_prefill) {
                        SRV_INF("hydra: DECODE slot=%d n_predict=%d atomic mode (%zu tokens)\n",
                                id_slot, n_predict, prompt_tokens.size());

                        // Clear existing slot state
                        slot->prompt_clear(false);
                        slot->n_prompt_tokens_cache = 0;
                        slot->n_prompt_tokens_processed = 0;
                        slot->n_decoded = 0;

                        // Insert prompt tokens
                        slot->prompt.tokens.insert(prompt_tokens);
                        const auto & tokens = slot->prompt.tokens.get_tokens();
                        const int n_tokens = (int)tokens.size();

                        // Add BOS if needed
                        int token_offset = 0;
                        llama_token bos = llama_vocab_bos(vocab);
                        if (add_bos_token && bos != LLAMA_TOKEN_NULL && (tokens.empty() || tokens[0] != bos)) {
                            token_offset = 1;
                        }

                        // Prefill in batches
                        const int n_ubatch = llama_n_ubatch(ctx_tgt);
                        for (int i = 0; i < n_tokens + token_offset; i += n_ubatch) {
                            const int n_tokens_batch = std::min(n_ubatch, n_tokens + token_offset - i);
                            common_batch_clear(batch);
                            for (int j = 0; j < n_tokens_batch; j++) {
                                const int tok_idx = i + j;
                                llama_token id;
                                if (token_offset > 0 && tok_idx == 0) {
                                    id = bos;
                                } else {
                                    id = tokens[tok_idx - token_offset];
                                }
                                const bool need_logits = (tok_idx == n_tokens + token_offset - 1);
                                common_batch_add(batch, id, tok_idx, {slot->id}, need_logits);
                            }
                            if (llama_decode(ctx_tgt, batch) != 0) {
                                SRV_ERR("hydra: DECODE slot=%d atomic prefill failed at batch %d\n", id_slot, i);
                                slot->n_prompt_tokens_processed = i;
                                break;
                            }
                        }
                        slot->n_prompt_tokens_processed = n_tokens;
                        slot->n_prompt_tokens_cache = n_tokens;
                        slot->n_decoded = 0;

                        // Register checkpoint
                        if (n_tokens > 0) {
                            create_checkpoint(*slot, 0, 0, (llama_pos)(n_tokens - 1));
                        }

                        SRV_INF("hydra: DECODE slot=%d atomic prefill done tokens=%d\n",
                                id_slot, n_tokens);
                    } else {
                        SRV_INF("hydra: DECODE slot=%d n_predict=%d cross-GPU / KV mode\n",
                                id_slot, n_predict);

                        // Skip checkpoint loading for cross-GPU decode.
                        // The KV cache restore via STATE_PUT already restores the attention states.
                        // For hybrid models, the recurrent/SSM state is NOT restored by KV cache restore,
                        // but loading the checkpoint on P100 takes 50+ seconds (65 MB at 1.3 MB/s).
                        // This overhead makes P/D split slower than baseline.
                        // TODO: investigate if checkpoint is actually needed or if KV restore is sufficient.
                        if (!slot->prompt.checkpoints.empty()) {
                            SLT_INF(*slot, "skipping checkpoint load for cross-GPU decode (checkpoint size=%zu B)\n",
                                    slot->prompt.checkpoints.back().data_tgt.size());
                            slot->prompt.checkpoints.clear();
                        }

                        // Append any additional prompt tokens (cross-GPU: none expected)
                        if (!task.hydra_action.prompt_tokens.empty()) {
                            slot->prompt.tokens.insert(task.hydra_action.prompt_tokens);
                            const auto & all_tokens = slot->prompt.tokens.get_tokens();
                            const int n_total = (int)all_tokens.size();
                            const int n_cached = slot->n_prompt_tokens_cache + slot->n_decoded;

                            if (n_total > n_cached) {
                                const int n_ubatch = llama_n_ubatch(ctx_tgt);
                                for (int i = n_cached; i < n_total; i += n_ubatch) {
                                    const int n_tokens_batch = std::min(n_ubatch, n_total - i);
                                    common_batch_clear(batch);
                                    for (int j = 0; j < n_tokens_batch; j++) {
                                        const bool need_logits = (i + j == n_total - 1);
                                        common_batch_add(batch, all_tokens[i + j], i + j, {slot->id}, need_logits);
                                    }
                                    if (llama_decode(ctx_tgt, batch) != 0) {
                                        SRV_ERR("hydra: DECODE slot=%d prefill failed at batch %d\n", id_slot, i);
                                        break;
                                    }
                                }
                                slot->n_prompt_tokens_processed = n_total;
                                slot->n_prompt_tokens_cache = n_total;
                                slot->n_decoded = 0;
                            }
                        }
                    }

                    // Set up greedy sampler
                    common_params_sampling sparams;
                    sparams.top_k = 1;
                    sparams.temp = 0.0f;
                    slot->smpl.reset(common_sampler_init(model_tgt, sparams));

                    // Generation loop
                    int n_decoded = 0;
                    int n_tokens_cached = slot->n_prompt_tokens_cache;
                    std::string accumulated;

                    while (n_decoded < n_predict) {
                        // Sample from the last logits
                        llama_token id = common_sampler_sample(slot->smpl.get(), ctx_tgt, -1);

                        if (llama_vocab_is_eog(vocab, id)) {
                            SRV_INF("hydra: DECODE slot=%d stopped at EOS after %d tokens\n", id_slot, n_decoded);
                            break;
                        }

                        // Convert to token string and accumulate
                        std::string token_str = common_token_to_piece(ctx_tgt, id);
                        res->tokens.push_back(id);
                        accumulated += token_str;

                        // If streaming enabled, write token bytes to socket
                        if (stream_fd >= 0) {
                            uint32_t len = (uint32_t)token_str.size();
                            hydra_send_all(stream_fd, &len, sizeof(len));
                            if (len > 0) {
                                hydra_send_all(stream_fd, token_str.data(), len);
                            }
                        }

                        // Prepare next decode
                        const int next_pos = n_tokens_cached + n_decoded;
                        common_batch_clear(batch);
                        common_batch_add(batch, id, next_pos, {slot->id}, true);

                        if (llama_decode(ctx_tgt, batch) != 0) {
                            SRV_ERR("hydra: DECODE slot=%d llama_decode failed at step %d\n", id_slot, n_decoded);
                            break;
                        }

                        common_sampler_accept(slot->smpl.get(), id, true);
                        n_decoded++;
                    }

                    SRV_INF("hydra: DECODE slot=%d completed %d tokens\n", id_slot, n_decoded);

                    // Store generated text for RPC handler to send as payload
                    res->generated_text = std::move(accumulated);

                    // Update slot tracking
                    slot->n_decoded = n_decoded;
                    res->n_past = slot->n_prompt_tokens_cache + slot->n_decoded;

                    res->rpc_status = HYDRA_STATUS_OK;
                    queue_results.send(std::move(res));
                } break;

            case SERVER_TASK_TYPE_HYDRA_ENGINE_SET_EXPERT_MODE:
                {
                    auto res = std::make_unique<server_task_result_hydra_engine>();
                    res->id = task.id;
                    res->op = HYDRA_OP_SET_EXPERT_MODE;

                    // Parse the payload. For backward compatibility, a raw string
                    // ("solo" or "combined") is accepted. Phase D (C# side) sends
                    // a JSON payload: {"mode":"combined","peer":"host:port",...}.
                    std::string requested;
                    std::string peer_override;
                    const std::string & raw = task.hydra_action.expert_mode;
                    if (!raw.empty() && raw[0] == '{') {
                        try {
                            json j = json::parse(raw);
                            requested    = j.value("mode", "solo");
                            peer_override = j.value("peer", "");
                        } catch (...) {
                            requested = "solo";
                        }
                    } else {
                        requested = raw;
                    }

                    if (requested != "solo" && requested != "combined") {
                        res->rpc_status = HYDRA_STATUS_ERROR;
                        res->success = false;
                        res->error = "expert_mode must be 'solo' or 'combined'";
                        queue_results.send(std::move(res));
                        break;
                    }

                    // #29 Phase B: per-request peer switching. If the request
                    // specifies a different peer, clean up the old one first.
                    static std::string g_current_peer;
                    if (!peer_override.empty() && peer_override != g_current_peer) {
                        if (!g_current_peer.empty()) {
                            SRV_INF("hydra: switching from peer %s to %s — cleaning up old binding\n",
                                    g_current_peer.c_str(), peer_override.c_str());
                            ctx_tgt->hydra_remove_combined_rpc_backend(g_current_peer.c_str());
                        }
                        g_current_peer = peer_override;
                        // Override the configured peer for the rest of this handler
                        const_cast<server_context_impl *>(this)->hydra_peer = peer_override;
                    }

                    // Hydra #383 T1: layer-split (static combined) engines cannot
                    // switch modes at runtime — the split is baked in at model load.
                    // "combined" is a no-op (already combined); "solo" is rejected.
                    if (hydra_combined_static) {
                        if (requested == "solo") {
                            res->rpc_status = HYDRA_STATUS_ERROR;
                            res->success = false;
                            res->error = "combined_static: this engine loaded in layer-split COMBINED mode; cannot switch to solo at runtime";
                            LOG_WRN("srv  %12.*s: hydra: SET_EXPERT_MODE solo rejected — engine is combined_static (layer-split)\n", 12, __func__);
                            queue_results.send(std::move(res));
                            break;
                        }
                        // requested == "combined": success no-op
                        res->expert_mode_applied = "combined";
                        res->rpc_status = HYDRA_STATUS_OK;
                        res->success = true;
                        LOG_INF("srv  %12.*s: hydra: SET_EXPERT_MODE combined no-op — engine is combined_static (layer-split)\n", 12, __func__);
                        queue_results.send(std::move(res));
                        break;
                    }

                    // #368 fix: gate on "configured as combined head" (non-empty
                    // peer addr + OT pattern), NOT on whether the startup
                    // dual-load succeeded. The rebind path below is fail-open —
                    // if the peer is still unreachable it stays solo — so
                    // hydra_combined_head_attached (set only when startup
                    // succeeded) must NOT block the attempt. Hydra #287/#260/#348
                    // intent is preserved: an unconfigured engine (no peer/
                    // pattern) still falls back to solo immediately.
                    const bool want_combined = requested == "combined" &&
                        !hydra_peer.empty() && !hydra_combined_pattern.empty();

                    // #368 (#357 fix): bind-on-activation. Re-bind the peer's
                    // expert tensors on each SET_EXPERT_MODE("combined") request
                    // so a peer that was down at boot is picked up on the first
                    // COMBINED request after it comes up. Fail-open: if the
                    // rebind fails we stay solo and the Coordinator's
                    // ReportsSolo path handles it.
                    bool actually_combined = want_combined;
                    if (want_combined) {
                        if (hydra_peer.empty() || hydra_combined_pattern.empty()) {
                            SRV_WRN("%s\n", "hydra: SET_EXPERT_MODE(combined) but no peer/pattern configured; staying solo");
                            actually_combined = false;
                        } else {
                            // ggml_backend_rpc_add_server is idempotent — returns
                            // the existing reg if the peer was registered before.
                            ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
                            if (!rpc_reg) {
                                SRV_WRN("%s\n", "hydra: SET_EXPERT_MODE(combined) but RPC backend not available; staying solo");
                                actually_combined = false;
                            } else {
                                using add_server_fn_t = ggml_backend_reg_t (*)(const char *);
                                auto add_server_fn = (add_server_fn_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
                                ggml_backend_reg_t peer_reg = add_server_fn ? add_server_fn(hydra_peer.c_str()) : nullptr;
                                ggml_backend_dev_t  peer_dev = (peer_reg && ggml_backend_reg_dev_count(peer_reg) > 0) ? ggml_backend_reg_dev_get(peer_reg, 0) : nullptr;
                                if (!peer_dev) {
                                    SRV_WRN("hydra: SET_EXPERT_MODE(combined) but peer %s has no registered device; staying solo\n",
                                            hydra_peer.c_str());
                                    actually_combined = false;
                                } else {
                                    int32_t n_bound = llama_hydra_rebind_combined_experts(
                                            ctx_tgt, hydra_peer.c_str(), peer_dev, hydra_combined_pattern.c_str());
                                    if (n_bound <= 0) {
                                        SRV_WRN("hydra: SET_EXPERT_MODE(combined) rebind on peer %s returned %d; staying solo\n",
                                                hydra_peer.c_str(), n_bound);
                                        actually_combined = false;
                                    } else {
                                        // Peer is up — latch so INFO RPC advertises combined.
                                        hydra_combined_head_attached = true;
                                    }
                                }
                            }
                        }
                    }

                    llama_hydra_set_expert_mode(ctx_tgt, actually_combined ? 1 : 0);
                    res->expert_mode_applied = actually_combined ? "combined" : "solo";

                    res->rpc_status = HYDRA_STATUS_OK;
                    res->success = true;
                    SRV_INF("hydra: SET_EXPERT_MODE requested='%s' applied='%s' (slot %d)\n",
                            requested.c_str(), res->expert_mode_applied.c_str(), task.hydra_action.id_slot);
                    queue_results.send(std::move(res));
                } break;

            case SERVER_TASK_TYPE_HYDRA_ENGINE_SWAP_QUANT:
                {
                    auto res = std::make_unique<server_task_result_hydra_engine>();
                    res->id = task.id;
                    res->op = HYDRA_OP_SWAP_QUANT;
                    res->rpc_status = HYDRA_STATUS_OK;
                    res->success = true;
                    SRV_INF("hydra: SWAP_QUANT quant='%s' pattern='%s' (slot %d)\n",
                            task.hydra_action.quant_key.c_str(),
                            task.hydra_action.tensor_pattern.c_str(),
                            task.hydra_action.id_slot);
                    queue_results.send(std::move(res));
                } break;

            // M-Perf.9 (#289) / issue #287: PIPELINE_ATTACH is part of the
            // two-engine "work together" routing tracked in #287. The
            // coordinator wires the request; the engine-side scaffolding
            // (--override-tensor local-load, activation passing, COMBINED
            // expert mode) is the next deliverable. For now this opcode
            // returns NOT_IMPLEMENTED so the wire stays in sync — the
            // coordinator will treat that as a fallback to solo mode.
            case SERVER_TASK_TYPE_HYDRA_ENGINE_PIPELINE_ATTACH:
                {
                    auto res = std::make_unique<server_task_result_hydra_engine>();
                    res->id = task.id;
                    res->op = HYDRA_OP_PIPELINE_ATTACH;
                    res->rpc_status = HYDRA_STATUS_NOT_IMPLEMENTED;
                    res->success = false;
                    res->error = "HYDRA_OP_PIPELINE_ATTACH not yet implemented in this build (see issue #287)";
                    SRV_WRN("hydra: PIPELINE_ATTACH received (slot %d) — stubbed, issue #287\n",
                            task.hydra_action.id_slot);
                    queue_results.send(std::move(res));
                } break;
        }
    }

    void update_slots() {
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

                                if (pos_min >= pos_min_thold) {
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
                                            it->load_tgt(ctx_tgt,       slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                            it->load_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

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
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
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
            if (++n_empty_consecutive > 3) {
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
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

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
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
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

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
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
        });
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

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
        json data = json::parse(req.body);
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

        std::vector<raw_buffer> files; // dummy
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
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
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
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        auto res = create_response();
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

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

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
}

json server_routes::get_model_info() const {
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
    auto res = create_response();
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

// ═══════════════════════════════════════════════════════════════════════════════
// Hydra RPC server — KV state transfer (M1: task-queue based)
// Wire format: specs/rpc-protocol.md  |  constants: server-rpc.h
// Ops implemented: STATE_GET (0x30), STATE_PUT (0x31), STATE_META (0x32)
// M1: All llama API calls routed through task queue (inference thread safe)
// ═══════════════════════════════════════════════════════════════════════════════

#if !defined(_WIN32)

// ── Context for RPC thread — pass to handlers ─────────────────────────────────

struct hydra_rpc_ctx {
    server_queue * queue_tasks = nullptr;
    server_response * queue_results = nullptr;
};

// ── Low-level I/O helpers ─────────────────────────────────────────────────────

static bool hydra_recv_all(int fd, void * buf, size_t n) {
    char * p = reinterpret_cast<char *>(buf);
    while (n > 0) {
        ssize_t r = ::recv(fd, p, n, 0);
        if (r <= 0) return false;
        p += r; n -= r;
    }
    return true;
}

static bool hydra_send_all(int fd, const void * buf, size_t n) {
    const char * p = reinterpret_cast<const char *>(buf);
    while (n > 0) {
        ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
        if (w <= 0) return false;
        p += w; n -= w;
    }
    return true;
}

// Response header: status(1) | meta_len(3 LE uint24) | payload_len(8 LE) — 12 bytes
static void hydra_write_res(int fd, uint8_t status, uint32_t meta_len, uint64_t payload_len) {
    uint8_t buf[HYDRA_RES_HEADER_SIZE] = {};
    buf[0] = status;
    buf[1] = (meta_len)       & 0xFF;
    buf[2] = (meta_len >>  8) & 0xFF;
    buf[3] = (meta_len >> 16) & 0xFF;
    memcpy(buf + 4, &payload_len, 8); // little-endian (x86/arm64)
    hydra_send_all(fd, buf, HYDRA_RES_HEADER_SIZE);
}

// ── Op handlers (M1: dispatch via task queue) ─────────────────────────────────

// STATE_GET (0x30): Post task, wait for result.
//
// M1 path (hydra_fd < 0): inference thread serializes 800 MB into result buffer;
//   RPC thread sends response header + meta JSON + buffer here.
//
// M2 path (hydra_fd = fd): background thread streams GPU→socket directly using
//   llama_state_seq_get_data_to_fd; result carries only n_past + streamed_bytes.
//   Response header + meta are sent BEFORE the task (we know size from STATE_META),
//   so the payload is already on the wire before we even get the result back.
//   Actually: we must send header AFTER knowing state_size. So:
//   - If M2: we get state_size first from a quick STATE_META query (n_past already known),
//     OR we embed state_size in the result from get_size() on the inference thread.
//   The inference thread always calls llama_state_seq_get_size (cheap) and stores it
//   in res->state_size for M2 so we can send the header before the stream completes.
//
// Timeout: 30s — streaming 800 MB over localhost may take a few seconds.
static void hydra_handle_state_get(int fd, int slot_id, const hydra_rpc_ctx & ctx) {
    // Build task — pass fd for M2 zero-copy streaming
    server_task task(SERVER_TASK_TYPE_HYDRA_STATE_GET);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot  = slot_id;
    task.hydra_action.hydra_fd = fd;    // M2: background thread streams here
    const int task_id = task.id;
    // Register BEFORE posting — server_response::send() silently drops results
    // for ids not in waiting_task_ids.
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->post(std::move(task));

    // Wait for result (n_past + state_size always set; state_data only on M1)
    std::unordered_set<int> task_ids = {task_id};
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 30); // seconds
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        SRV_WRN("hydra rpc: STATE_GET timeout for slot %d\n", slot_id);
        // M2 caveat: the background thread may own the fd (header possibly sent);
        // writing an error header here could interleave with the stream. Shut the
        // socket down instead so the client unblocks with a clean EOF.
        ::shutdown(fd, SHUT_RDWR);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_state*>(res_ptr.get());
    if (!res) {
        SRV_WRN("hydra rpc: STATE_GET result type mismatch for slot %d\n", slot_id);
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    if (res->rpc_status != HYDRA_STATUS_OK) {
        if (res->header_sent) {
            // M2 failure: header already sent but stream failed; background thread
            // shut the socket down — connection loop will close the fd on next read.
            // Log and return without sending a second response header.
            SRV_WRN("hydra rpc: STATE_GET slot=%d M2 stream failed: %s\n",
                    slot_id, res->error.c_str());
            return;
        }
        hydra_write_res(fd, res->rpc_status, 0, 0);
        if (!res->error.empty()) {
            hydra_send_all(fd, res->error.data(), res->error.size());
        }
        return;
    }

    if (res->streamed_bytes > 0) {
        // M2 path: data already on the wire — response header + meta were sent by background thread.
        // Nothing left for RPC thread to do. The protocol framing (header + meta + payload)
        // was completed inside llama_io_write_socket / the background thread.
        // Note: header was sent AFTER state_size was known (inference thread called get_size).
        SRV_INF("hydra rpc: STATE_GET slot=%d M2 streamed %.1f MiB directly\n",
                slot_id, res->streamed_bytes / (1024.0 * 1024.0));
    } else {
        // M1 path: inference thread buffered 800 MB; send it now.
        const uint64_t payload = (uint64_t)res->state_data.size();
        json meta_j;
        meta_j["n_past"]     = res->n_past;
        meta_j["state_size"] = payload;
        const std::string meta_str = meta_j.dump();
        hydra_write_res(fd, HYDRA_STATUS_OK, (uint32_t)meta_str.size(), payload);
        hydra_send_all(fd, meta_str.data(), meta_str.size());
        hydra_send_all(fd, res->state_data.data(), (size_t)payload);
        SRV_INF("hydra rpc: STATE_GET slot=%d M1 sent %.1f MiB from buffer\n",
                slot_id, payload / (1024.0 * 1024.0));
    }
}

// STATE_PUT (0x31): Receive payload, post task, wait for result, send ack.
static void hydra_handle_state_put(int fd, int slot_id, uint64_t payload_len, const hydra_rpc_ctx & ctx) {
    if (payload_len > HYDRA_MAX_STATE_BYTES) {
        SRV_WRN("hydra rpc: STATE_PUT payload %" PRIu64 " B exceeds cap %" PRIu64 " B\n",
                payload_len, HYDRA_MAX_STATE_BYTES);
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        // Drain to keep persistent connection alive
        std::vector<uint8_t> drain(65536);
        for (uint64_t rem = payload_len; rem > 0; ) {
            size_t chunk = (size_t)std::min(rem, (uint64_t)drain.size());
            if (!hydra_recv_all(fd, drain.data(), chunk)) break;
            rem -= chunk;
        }
        return;
    }

    // Read payload from socket
    std::vector<uint8_t> buf((size_t)payload_len);
    if (!hydra_recv_all(fd, buf.data(), (size_t)payload_len)) {
        SRV_WRN("%s", "hydra rpc: STATE_PUT failed to read payload\n");
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    // Post task to inference thread
    server_task task(SERVER_TASK_TYPE_HYDRA_STATE_PUT);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot = slot_id;
    task.hydra_action.erase_existing = true; // RPC restore always replaces slot state
    task.hydra_action.state_data = std::move(buf);
    const int task_id = task.id;
    // Register BEFORE posting — results for unregistered ids are dropped.
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->post(std::move(task));

    // Wait for result from inference thread (30s timeout for large restore)
    std::unordered_set<int> task_ids = {task_id};
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 30); // seconds
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        SRV_WRN("hydra rpc: STATE_PUT timeout for slot %d\n", slot_id);
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_state*>(res_ptr.get());
    if (!res) {
        SRV_WRN("hydra rpc: STATE_PUT result type mismatch for slot %d\n", slot_id);
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    // Send result back to client
    uint8_t rpc_status = res->rpc_status;
    if (rpc_status == HYDRA_STATUS_OK) {
        json meta_j;
        meta_j["restored"] = true;
        meta_j["bytes"]    = res->bytes;
        const std::string meta_str = meta_j.dump();
        hydra_write_res(fd, HYDRA_STATUS_OK, (uint32_t)meta_str.size(), 0);
        hydra_send_all(fd, meta_str.data(), meta_str.size());
    } else {
        json err_j;
        err_j["error"] = res->error;
        const std::string err_str = err_j.dump();
        hydra_write_res(fd, rpc_status, (uint32_t)err_str.size(), 0);
        hydra_send_all(fd, err_str.data(), err_str.size());
    }
}

// STATE_META (0x32): Post task, wait for result, send JSON metadata.
static void hydra_handle_state_meta(int fd, int slot_id, const hydra_rpc_ctx & ctx) {
    server_task task(SERVER_TASK_TYPE_HYDRA_STATE_META);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot = slot_id;
    const int task_id = task.id;
    // Register BEFORE posting — results for unregistered ids are dropped.
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->post(std::move(task));

    // Wait for result from inference thread (5s timeout — allows for queue congestion)
    std::unordered_set<int> task_ids = {task_id};
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 5); // seconds
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        SRV_WRN("hydra rpc: STATE_META timeout for slot %d\n", slot_id);
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_state*>(res_ptr.get());
    if (!res) {
        SRV_WRN("hydra rpc: STATE_META result type mismatch for slot %d\n", slot_id);
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    // Send result back to client
    uint8_t rpc_status = res->rpc_status;
    if (rpc_status == HYDRA_STATUS_OK) {
        json meta_j;
        meta_j["slot_id"]         = res->id_slot;
        meta_j["n_past"]          = res->n_past;
        meta_j["state_size"]      = res->state_size;
        meta_j["is_processing"]   = res->is_processing;
        meta_j["is_transferring"] = res->is_transferring;
        const std::string meta_str = meta_j.dump();
        hydra_write_res(fd, HYDRA_STATUS_OK, (uint32_t)meta_str.size(), 0);
        hydra_send_all(fd, meta_str.data(), meta_str.size());
    } else {
        hydra_write_res(fd, rpc_status, 0, 0);
    }
}

// ── E1 Engine control handlers ────────────────────────────────────────────────

// CONFIGURE (0x33): Read JSON config payload, post task, return success.
static void hydra_handle_configure(int fd, int slot_id, uint64_t payload_len, const hydra_rpc_ctx & ctx) {
    std::string config_json(payload_len, '\0');
    if (payload_len > 0 && !hydra_recv_all(fd, config_json.data(), payload_len)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    server_task task(SERVER_TASK_TYPE_HYDRA_ENGINE_CONFIGURE);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot = slot_id;
    task.hydra_action.config_json = std::move(config_json);
    const int task_id = task.id;
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->post(std::move(task));

    std::unordered_set<int> task_ids = {task_id};
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 5);
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_engine*>(res_ptr.get());
    if (!res || !res->success) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    json meta_j = {{"success", true}};
    // hydra#334: echo the post-clamp value so the Coordinator can tell a
    // silent clamp from "exactly what I asked for" instead of trusting an
    // unconditional success response.
    if (res->state_chunk_size_applied > 0) {
        meta_j["state_chunk_size_applied"] = res->state_chunk_size_applied;
    }
    const std::string meta_str = meta_j.dump();
    hydra_write_res(fd, HYDRA_STATUS_OK, (uint32_t)meta_str.size(), 0);
    hydra_send_all(fd, meta_str.data(), meta_str.size());
}

// INFO (0x34): Return engine capabilities as JSON.
static void hydra_handle_info(int fd, int slot_id, const hydra_rpc_ctx & ctx) {
    server_task task(SERVER_TASK_TYPE_HYDRA_ENGINE_INFO);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot = slot_id;
    const int task_id = task.id;
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->post(std::move(task));

    std::unordered_set<int> task_ids = {task_id};
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 5);
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_engine*>(res_ptr.get());
    if (!res) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    const std::string & info_str = res->info_json;
    hydra_write_res(fd, HYDRA_STATUS_OK, (uint32_t)info_str.size(), 0);
    hydra_send_all(fd, info_str.data(), info_str.size());
}

// PREFILL (0x35): Read JSON payload with {"messages": [...]},
// tokenize internally, run prefill, return n_past + KV state blob.
static void hydra_handle_prefill(int fd, int slot_id, uint64_t payload_len, const hydra_rpc_ctx & ctx) {
    if (payload_len == 0) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    std::string json_str((size_t)payload_len, '\0');
    if (!hydra_recv_all(fd, json_str.data(), (size_t)payload_len)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    server_task task(SERVER_TASK_TYPE_HYDRA_ENGINE_PREFILL);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot = slot_id;
    task.hydra_action.request_json = std::move(json_str);
    const int task_id = task.id;
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->post(std::move(task));

    std::unordered_set<int> task_ids = {task_id};
    // Bumped from 60s to 180s. Prefill for 32k+ token prompts exceeds 120s
    // (we measured 32s for 22k tokens; 48k ≈ 70s, 100k ≈ 150s+). Long autoregressive
    // decode on P100 (28 tok/s) for 4k+ token outputs also exceeds 120s. The C++
    // side was timing out and returning HYDRA_STATUS_ERROR before the C# client
    // gave up, surfacing as a 503 from the coordinator even though the model was
    // still working.
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 180);
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_engine*>(res_ptr.get());
    if (!res || res->rpc_status != HYDRA_STATUS_OK) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    // Return n_past + sizes in meta; full blob (v2 header + KV + logits) as payload.
    // logits_size > 0 signals the decode GPU to inject them into ctx->logits via STATE_PUT.
    json meta_j = {
        {"n_past",      res->n_past},
        {"state_size",  res->state_size},
        {"logits_size", res->logits_size}
    };
    const std::string meta_str = meta_j.dump();
    const uint64_t total_payload = (uint64_t)res->state_data.size();
    hydra_write_res(fd, HYDRA_STATUS_OK, (uint32_t)meta_str.size(), total_payload);
    hydra_send_all(fd, meta_str.data(), meta_str.size());
    if (total_payload > 0) {
        hydra_send_all(fd, res->state_data.data(), (size_t)total_payload);
    }
    SRV_INF("hydra: PREFILL slot=%d sent n_past=%d kv=%" PRIu64 "B logits=%" PRIu64 "B total=%" PRIu64 "B\n",
            slot_id, res->n_past, res->state_size, res->logits_size, total_payload);
}

// DECODE (0x36): Read JSON payload with {"n_predict": N, "messages": [...]}.
// If "messages" is present, tokenize + prefill first (atomic / single-node path).
// If "messages" is absent, decode from existing slot KV state (cross-GPU path).
// Returns generated text as payload.
static void hydra_handle_decode(int fd, int slot_id, uint64_t payload_len, const hydra_rpc_ctx & ctx) {
    if (payload_len == 0) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    std::string json_str((size_t)payload_len, '\0');
    if (!hydra_recv_all(fd, json_str.data(), (size_t)payload_len)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    int32_t n_predict = -1;

    try {
        json req = json::parse(json_str);
        if (req.contains("n_predict")) {
            n_predict = req["n_predict"].get<int32_t>();
        }
    } catch (...) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    if (n_predict < 0) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    server_task task(SERVER_TASK_TYPE_HYDRA_ENGINE_DECODE);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot = slot_id;
    task.hydra_action.n_predict = n_predict;
    task.hydra_action.request_json = std::move(json_str);
    // Enable streaming: pass the RPC socket fd so tokens are streamed as generated
    task.hydra_action.stream_fd = fd;
    const int task_id = task.id;
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->post(std::move(task));

    // Send response header (status OK, no meta) before streaming tokens
    // The DECODE handler will write tokens directly to fd via stream_fd
    hydra_write_res(fd, HYDRA_STATUS_OK, 0, 0);

    // Wait for task completion (tokens are streamed during execution)
    std::unordered_set<int> task_ids = {task_id};
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 180);
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        // Timeout — connection already has header sent, just close
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_engine*>(res_ptr.get());
    if (!res || res->rpc_status != HYDRA_STATUS_OK) {
        // Error during decode — header already sent, can't send error header
        // Just close the connection
        return;
    }

    SRV_INF("hydra: DECODE slot=%d generated %d tokens (streamed)\n",
            slot_id, (int)res->tokens.size());
}

// SET_EXPERT_MODE (0x37): Read mode string, post task, return success.
static void hydra_handle_set_expert_mode(int fd, int slot_id, uint64_t payload_len, const hydra_rpc_ctx & ctx) {
    std::string mode(payload_len, '\0');
    if (payload_len > 0 && !hydra_recv_all(fd, mode.data(), payload_len)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    server_task task(SERVER_TASK_TYPE_HYDRA_ENGINE_SET_EXPERT_MODE);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot = slot_id;
    task.hydra_action.expert_mode = std::move(mode);
    const int task_id = task.id;
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->post(std::move(task));

    std::unordered_set<int> task_ids = {task_id};
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 5);
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_engine*>(res_ptr.get());
    if (!res || !res->success) {
        const std::string err = (res && !res->error.empty()) ? res->error : std::string();
        json err_j = {{"success", false}};
        if (!err.empty()) err_j["error"] = err;
        const std::string err_str = err_j.dump();
        hydra_write_res(fd, HYDRA_STATUS_ERROR, (uint32_t)err_str.size(), 0);
        hydra_send_all(fd, err_str.data(), err_str.size());
        return;
    }

    // Report the ACTUAL mode applied (may be "solo" even though "combined" was
    // requested, if this engine never dual-loaded combined experts) — the
    // Coordinator's ReportsSolo() reads this key to detect the fallback.
    json meta_j = {{"success", true}, {"mode", res->expert_mode_applied}};
    const std::string meta_str = meta_j.dump();
    hydra_write_res(fd, HYDRA_STATUS_OK, (uint32_t)meta_str.size(), 0);
    hydra_send_all(fd, meta_str.data(), meta_str.size());
}

// SWAP_QUANT (0x38): Read quant_key + tensor_pattern, post task, return success.
static void hydra_handle_swap_quant(int fd, int slot_id, uint64_t payload_len, const hydra_rpc_ctx & ctx) {
    if (payload_len < sizeof(uint16_t)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    uint16_t quant_key_len = 0;
    if (!hydra_recv_all(fd, &quant_key_len, sizeof(quant_key_len))) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    std::string quant_key(quant_key_len, '\0');
    if (quant_key_len > 0 && !hydra_recv_all(fd, quant_key.data(), quant_key_len)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    const uint64_t pattern_len = payload_len - sizeof(uint16_t) - quant_key_len;
    std::string tensor_pattern(pattern_len, '\0');
    if (pattern_len > 0 && !hydra_recv_all(fd, tensor_pattern.data(), pattern_len)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    server_task task(SERVER_TASK_TYPE_HYDRA_ENGINE_SWAP_QUANT);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot = slot_id;
    task.hydra_action.quant_key = std::move(quant_key);
    task.hydra_action.tensor_pattern = std::move(tensor_pattern);
    const int task_id = task.id;
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->post(std::move(task));

    std::unordered_set<int> task_ids = {task_id};
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 30);
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_engine*>(res_ptr.get());
    if (!res || !res->success) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    json meta_j = {{"success", true}};
    const std::string meta_str = meta_j.dump();
    hydra_write_res(fd, HYDRA_STATUS_OK, (uint32_t)meta_str.size(), 0);
    hydra_send_all(fd, meta_str.data(), meta_str.size());
}

// PIPELINE_ATTACH (0x46): M-Perf.9 (#289) / issue #287 — two-engine "work
// together" routing scaffolding. The C# Coordinator sends the peer address
// and the --override-tensor regex; the engine should load the assigned
// tensor slice from its OWN local model (no weight transfer). This opcode
// is stubbed for now (returns NOT_IMPLEMENTED) — full implementation is
// tracked under issue #287.
static void hydra_handle_pipeline_attach(int fd, int slot_id, uint64_t payload_len, const hydra_rpc_ctx & ctx) {
    std::string json_body(payload_len, '\0');
    if (payload_len > 0 && !hydra_recv_all(fd, json_body.data(), payload_len)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    server_task task(SERVER_TASK_TYPE_HYDRA_ENGINE_PIPELINE_ATTACH);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot = slot_id;
    task.hydra_action.request_json = std::move(json_body);
    const int task_id = task.id;
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->post(std::move(task));

    std::unordered_set<int> task_ids = {task_id};
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 5);
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_engine*>(res_ptr.get());
    // Stubbed: server returns NOT_IMPLEMENTED until issue #287 lands.
    // Propagate that status to the client so the Coordinator can
    // distinguish "not yet built" from a real error and fall back to solo.
    const uint8_t status = (res && res->rpc_status == HYDRA_STATUS_NOT_IMPLEMENTED)
        ? HYDRA_STATUS_NOT_IMPLEMENTED : HYDRA_STATUS_ERROR;
    json meta_j;
    if (res && !res->error.empty()) meta_j["error"] = res->error;
    meta_j["success"] = res && res->success;
    const std::string meta_str = meta_j.dump();
    hydra_write_res(fd, status, (uint32_t)meta_str.size(), 0);
    hydra_send_all(fd, meta_str.data(), meta_str.size());
}

// ── Per-connection loop ───────────────────────────────────────────────────────
// Persistent: one TCP connection handles many sequential requests.

static void hydra_handle_connection(int fd, const hydra_rpc_ctx & ctx) {
    // Set receive timeout to prevent hung connections on stalled clients
    struct timeval tv;
    tv.tv_sec  = 120; // 2 min inactivity timeout
    tv.tv_usec = 0;
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    while (true) {
        uint8_t hdr[HYDRA_REQ_HEADER_SIZE];
        if (!hydra_recv_all(fd, hdr, HYDRA_REQ_HEADER_SIZE)) break;

        uint16_t magic = 0;
        memcpy(&magic, hdr + 0, 2);
        if (magic != HYDRA_MAGIC) {
            SRV_WRN("hydra rpc: bad magic 0x%04x — closing connection\n", (unsigned)magic);
            break;
        }

        const uint8_t op = hdr[2];
        // hdr[3] = flags (reserved, unused in M1)
        uint16_t key_len = 0, trace_len = 0;
        uint64_t payload_len = 0;
        memcpy(&key_len,     hdr + 4,  2);
        memcpy(&payload_len, hdr + 6,  8);
        memcpy(&trace_len,   hdr + 14, 2);

        std::string key(key_len, '\0');
        std::string trace_id(trace_len, '\0');
        if (!hydra_recv_all(fd, key.data(),      key_len))   break;
        if (!hydra_recv_all(fd, trace_id.data(), trace_len)) break;

        int slot_id = -1;
        try { slot_id = std::stoi(key); }
        catch (...) {
            SRV_WRN("hydra rpc: invalid slot key '%s'\n", key.c_str());
            hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
            continue;
        }

        // Dispatch to handler via task queue (no direct slot access)
        switch (op) {
            case HYDRA_OP_STATE_GET:
                SRV_DBG("hydra rpc: STATE_GET  slot=%d trace=%s\n", slot_id, trace_id.c_str());
                hydra_handle_state_get(fd, slot_id, ctx);
                break;
            case HYDRA_OP_STATE_PUT:
                SRV_DBG("hydra rpc: STATE_PUT  slot=%d payload=%" PRIu64 " trace=%s\n",
                        slot_id, payload_len, trace_id.c_str());
                hydra_handle_state_put(fd, slot_id, payload_len, ctx);
                break;
            case HYDRA_OP_STATE_META:
                SRV_DBG("hydra rpc: STATE_META slot=%d trace=%s\n", slot_id, trace_id.c_str());
                hydra_handle_state_meta(fd, slot_id, ctx);
                break;
            case HYDRA_OP_CONFIGURE:
                SRV_DBG("hydra rpc: CONFIGURE slot=%d payload=%" PRIu64 " trace=%s\n",
                        slot_id, payload_len, trace_id.c_str());
                hydra_handle_configure(fd, slot_id, payload_len, ctx);
                break;
            case HYDRA_OP_INFO:
                SRV_DBG("hydra rpc: INFO slot=%d trace=%s\n", slot_id, trace_id.c_str());
                hydra_handle_info(fd, slot_id, ctx);
                break;
            case HYDRA_OP_PREFILL:
                SRV_DBG("hydra rpc: PREFILL slot=%d payload=%" PRIu64 " trace=%s\n",
                        slot_id, payload_len, trace_id.c_str());
                hydra_handle_prefill(fd, slot_id, payload_len, ctx);
                break;
            case HYDRA_OP_DECODE:
                SRV_DBG("hydra rpc: DECODE slot=%d payload=%" PRIu64 " trace=%s\n",
                        slot_id, payload_len, trace_id.c_str());
                hydra_handle_decode(fd, slot_id, payload_len, ctx);
                break;
            case HYDRA_OP_SET_EXPERT_MODE:
                SRV_DBG("hydra rpc: SET_EXPERT_MODE slot=%d payload=%" PRIu64 " trace=%s\n",
                        slot_id, payload_len, trace_id.c_str());
                hydra_handle_set_expert_mode(fd, slot_id, payload_len, ctx);
                break;
            case HYDRA_OP_SWAP_QUANT:
                SRV_DBG("hydra rpc: SWAP_QUANT slot=%d payload=%" PRIu64 " trace=%s\n",
                        slot_id, payload_len, trace_id.c_str());
                hydra_handle_swap_quant(fd, slot_id, payload_len, ctx);
                break;
            // M-Perf.9 (#289) / issue #287: PIPELINE_ATTACH (0x46) is the
            // two-engine "work together" attach. Stubbed: full impl in #287.
            case HYDRA_OP_PIPELINE_ATTACH:
                SRV_DBG("hydra rpc: PIPELINE_ATTACH slot=%d payload=%" PRIu64 " trace=%s\n",
                        slot_id, payload_len, trace_id.c_str());
                hydra_handle_pipeline_attach(fd, slot_id, payload_len, ctx);
                break;
            default:
                SRV_WRN("hydra rpc: unknown op 0x%02x — ignoring\n", (unsigned)op);
                hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        }
    }
    ::close(fd);
}

// ── Unified RPC server ──────────────────────────────────────────────────────
// Single TCP listener on `port` that serves BOTH:
//   - ggml-RPC protocol (first byte = RPC_CMD_HELLO = 0x0E) via
//     ggml_backend_rpc_handle_client — for GPU compute (COMBINE peer role)
//   - Hydra protocol (first byte in 0x30-0x46 range) via
//     hydra_handle_connection — for KV state transfer / coordination
//
// The ggml-RPC handler uses the given backend list (shared with local inference
// in the model-loaded path, or globally enumerated devices in the no-model path).
// When `this` is null (no server_context), Hydra connections are rejected.

// Forward declaration for ggml backend library symbol
extern void ggml_backend_rpc_handle_client(int fd, const char * cache_dir,
                                            size_t n_backends, ggml_backend_t * backends);

void server_context::start_rpc_server(int port,
                                       const std::vector<ggml_backend *> & rpc_backends) {
    if (port <= 0) return;

    // Extract Hydra queue pointers (only needed if server_context exists)
    hydra_rpc_ctx ctx{};
    if (this && impl) {
        ctx.queue_tasks   = &impl->queue_tasks;
        ctx.queue_results = &impl->queue_results;
    }

    // Copy backends for the accept thread — the vector is captured by value.
    std::vector<ggml_backend *> backends = rpc_backends;

    std::thread([port, backends, ctx, has_hydra = (this != nullptr && impl != nullptr)]() {
        const int srv_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (srv_fd < 0) {
            SRV_ERR("hydra rpc: socket() failed: %s\n", strerror(errno));
            return;
        }
        const int opt = 1;
        ::setsockopt(srv_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port        = htons((uint16_t)port);

        if (::bind(srv_fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
            SRV_ERR("hydra rpc: bind() on port %d failed: %s\n", port, strerror(errno));
            ::close(srv_fd);
            return;
        }
        ::listen(srv_fd, 16);

        if (has_hydra) {
            SRV_INF("hydra rpc: unified server on 0.0.0.0:%d (ggml-RPC + Hydra protocol)\n", port);
        } else {
            SRV_INF("hydra rpc: unified server on 0.0.0.0:%d (ggml-RPC only, no model loaded)\n", port);
        }

        while (true) {
            const int conn_fd = ::accept(srv_fd, nullptr, nullptr);
            if (conn_fd < 0) continue;

            std::thread([conn_fd, backends, ctx, has_hydra]() {
                // Peek first byte to determine protocol — MSG_PEEK does not consume it.
                uint8_t first_byte;
                if (::recv(conn_fd, &first_byte, 1, MSG_PEEK) != 1) {
                    ::close(conn_fd);
                    return;
                }

                // RPC_CMD_HELLO = 14 = 0x0E. Hydra opcodes are 0x30-0x46.
                if (first_byte == 0x0E) {
                    // ggml-RPC client — dispatch to ggml backend library handler
                    ggml_backend_rpc_handle_client(conn_fd, nullptr,
                        backends.size(), const_cast<ggml_backend_t *>(backends.data()));
                } else if (has_hydra) {
                    // Hydra protocol client
                    hydra_handle_connection(conn_fd, ctx);
                } else {
                    // No model loaded — Hydra protocol not available
                    ::close(conn_fd);
                }
            }).detach();
        }
    }).detach();
}

#else
// Windows: RPC server not implemented — target hardware is Linux-only for M0.
void server_context::start_rpc_server(int port, const std::vector<ggml_backend *> &) {
    if (port > 0) {
        SRV_WRN("hydra rpc: not supported on Windows (port %d ignored)\n", port);
    }
    GGML_UNUSED(port);
}
#endif // !_WIN32

#pragma once

#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"

#include <nlohmann/json_fwd.hpp>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <set>
#include <shared_mutex>
#include <vector>

struct ggml_backend;

struct server_context_impl; // private implementation

// ── Hydra #470: generic CONFIGURE key pass-through (T4) ────────────────────
//
// hydra_classify_config_key() is the tiered CONFIGURE classifier:
//   1 = T1  apply immediately (no rebuild) — sampling.*, n_predict, ...
//   2 = T2  defer to slot-free moment, context reload — n_ctx, cache_type_*,
//           rope_*, ...
//   3 = T3  defer to slot-free moment, model reload — n_gpu_layers,
//           override_tensor, split_mode, model_path, ...
//   4 = T4  generic-arg pass-through: any key not in the special lists is
//           mapped to llama.cpp's own CLI arg table (snake_case → kebab-case)
//           and applied through the arg's canonical handler, so a valid
//           llama.cpp arg ALWAYS lands in common_params. No silent drops.
//
// Exported (non-static) so the header-only configure-tier test can pin the
// contract: every coordinator-emitted key must classify as T1/T2/T3 (special)
// or as a T4 key that is APPLIABLE / explicitly DENIED — never silently
// ignored.
int hydra_classify_config_key(const std::string & key);

// For a T4 (generic) key: its reload disposition.
//   NONE         — not a T4 key (classified T1/T2/T3 by the classifier)
//   APPLIABLE    — has a reload-safe entry in the llama.cpp arg table; the
//                  value will be applied via the arg's canonical handler
//   DENIED       — startup-only / flag / two-value arg; reported as a
//                  rejected_key with a visible SRV_WRN ("cannot change at
//                  reload") instead of being silently dropped
//   UNRECOGNIZED — no entry in the llama.cpp arg table; reported as an
//                  unrecognized_key with a visible SRV_WRN
enum class hydra_generic_key_status {
    NONE,
    APPLIABLE,
    DENIED,
    UNRECOGNIZED,
};

hydra_generic_key_status hydra_classify_generic_key(const std::string & key);

// Apply one generic key (snake_case name, JSON value from the CONFIGURE
// payload) to `params` via the arg-table handler. Returns true when the
// value was applied; false when the key is not appliable or the value's
// JSON type cannot be converted (the failure is logged with SRV_WRN).
bool hydra_apply_generic_key(common_params & params, const std::string & key, const json & value);

// hydra#648 (review): clear the draft/MTP context bindings held in `params`.
// On a T3 reload-after-prior-MTP-success, params_base carries the OLD
// ctx_tgt/ctx_dft pointers into load_model(); if the new MTP draft context
// then fails to build, those pointers are dangling-non-null and the
// common_speculative_init() null-gate would let the MTP impl dereference
// freed memory (use-after-free, speculative.cpp:447). Nulling them makes the
// existing ctx_dft == nullptr degradation gate work. Called from the MTP
// build-failure branch in load_model() and from destroy(). Exported so the
// configure-tier test can pin the reload-after-success-then-fail bookkeeping.
void hydra_clear_stale_draft_bindings(common_params & params);

// hydra#470: a synchronous hydra_config apply whose highest tier is >= 3
// may have rebuilt the slots (T3 statics or a T4-only generic config both
// route through apply_t3_rebuild() → load_model() → slots.clear()). Callers
// that hold a server_slot* across hydra_apply_config(sync=true) MUST use
// this predicate to decide whether to re-look-up the slot — otherwise the
// held pointer dangles (use-after-free on the first T4-triggered reload).
// Exported so the configure-tier test can pin the condition.
static inline bool hydra_config_requires_slot_relookup(int highest_tier) {
    return highest_tier >= 3;
}

struct server_context_meta {
    std::string build_info;
    std::string model_name;
    std::set<std::string> model_aliases;
    std::set<std::string> model_tags;
    std::string model_path;
    bool has_mtmd;
    bool has_inp_image;
    bool has_inp_audio;
    json json_ui_settings;            // Primary: new name
    json json_webui_settings;            // Deprecated: use json_ui_settings instead (kept for backward compat)
    int slot_n_ctx;
    enum llama_pooling_type pooling_type;

    // chat params
    server_chat_params & chat_params;
    std::map<std::string, bool> chat_template_caps;

    // tokens
    std::string bos_token_str;
    std::string eos_token_str;
    llama_token fim_pre_token;
    llama_token fim_sub_token;
    llama_token fim_mid_token;
    llama_token fim_pad_token;
    llama_token fim_rep_token;
    llama_token fim_sep_token;

    // sampling
    std::vector<llama_logit_bias> logit_bias_eog;

    // model meta
    enum llama_vocab_type model_vocab_type;
    int32_t model_vocab_n_tokens;
    int32_t model_n_ctx_train;
    int32_t model_n_embd_inp;
    uint64_t model_n_params;
    uint64_t model_size;
    enum llama_split_mode split_mode;
    std::vector<float> tensor_split;
};

struct server_context {
    std::unique_ptr<server_context_impl> impl;

    server_context();
    ~server_context();

    // load the model and initialize llama_context
    // returns true on success
    bool load_model(common_params & params);

    // this function will block main thread until termination
    void start_loop();

    // terminate main loop (will unblock start_loop)
    void terminate();

    // get the underlaying llama_context, can return nullptr if sleeping
    // not thread-safe, should only be used from the main thread
    llama_context * get_llama_context() const;

    // get a new response reader, used by CLI application
    server_response_reader get_response_reader();

    // get server metadata (read-only), can only be called after load_model()
    // not thread-safe, should only be used from the main thread
    server_context_meta get_meta() const;

    // register a callback to be called when sleeping state changes
    // must be set before load_model() is called
    void on_sleeping_changed(std::function<void(bool)> callback);

    // P0-1 (#49): bootstrap_init wires up queue callbacks and metrics without
    // requiring a model to be loaded. Used by llama-engine's head-bootstrap
    // mode (no model at startup, model loaded later via CONFIGURE T3).
    // Must be called before start_loop().
    bool bootstrap_init();

    // P0-1 (#49): set the back-pointer to server_routes so
    // apply_pending_hydra_config() can call routes_ptr->refresh_meta()
    // after the first load. Must be called before start_loop().
    void set_routes_ptr(struct server_routes * routes);

    // P0-1 (#49): stage capability flags for deferred first-load.
    // apply_pending_hydra_config() will call set_hydra_capabilities() and
    // set_hydra_combined_static() after the first load succeeds.
    void set_bootstrap_capabilities(bool rpc_active, const std::string & peer,
            bool peer_reachable, const std::string & pattern,
            const std::string & split_mode, bool combined_static);

    // Hydra RPC: start a binary TCP listener for KV state transfer.
    // Ops: STATE_GET (0x30), STATE_PUT (0x31), STATE_META (0x32).
    // port = 0 is a no-op (disabled). Must be called after load_model().
    // Thread safety: slots must be idle (is_processing() == false) when ops run.
    // TODO(M1): route through task queue for full thread safety under load.
    void start_rpc_server(int port, std::vector<ggml_backend *> rpc_backends = {});

    // Hydra #348: record this engine's independent capability flags (replaces
    // the old single hydra_role string) + (for a COMBINED head) the
    // configured peer/tensor-pattern, so ENGINE_INFO (0x41) can report them.
    // `rpc_backend_active`: true once the shared-backend RPC thread is
    // running (--ggml-rpc-port was set). `peer_reachable`: the startup
    // TCP-probe result for `peer`, distinct from whether the dual-load
    // (set_hydra_combined_head_attached) actually succeeded.
    // `split_mode`: "expert" (default, --combined-ot-pattern path) or
    // "layer" (#383 T1, --combined-tensor-split pre-load path).
    void set_hydra_capabilities(bool rpc_backend_active, const std::string & peer,
            bool peer_reachable, const std::string & combined_pattern,
            const std::string & split_mode = "expert");

    // Hydra #348 (renamed from set_hydra_combined_capable): record whether
    // COMBINED expert dual-loading succeeded at startup. Must be called (if
    // at all) after load_model() and before serving requests.
    // SET_EXPERT_MODE("combined") falls back to solo when this is false
    // (peer was unreachable, had no matching expert tensors, or didn't have
    // enough free VRAM for the dual-load).
    void set_hydra_combined_head_attached(bool attached);

    // Hydra #383 T1: record that this engine started in COMBINED static
    // (layer-split) mode. Unlike expert-split, the split is fixed at model
    // load time — SET_EXPERT_MODE("solo") is an error, "combined" is a no-op.
    void set_hydra_combined_static(bool is_static);
};


// forward declarations
struct server_res_generator;

struct server_routes {
    server_routes(const common_params & params, server_context & ctx_server);

    void init_routes();

    // Guards meta against concurrent read/write: update_meta takes exclusive,
    // HTTP handlers take shared.  refresh_meta() calls update_meta() so it
    // already holds the exclusive lock — do NOT add a shared lock there.
    mutable std::shared_mutex meta_mutex;

    // note: this is not thread-safe and can only when ctx_http.is_ready is false
    void update_meta(const server_context & ctx_server) {
        std::unique_lock lock(meta_mutex);
        this->meta = std::make_unique<server_context_meta>(ctx_server.get_meta());
    }

    // Hydra P1-6: refresh meta from the stored ctx_server_outer reference.
    // Called from the task-queue thread (apply_pending_hydra_config) during
    // the drain window — no concurrent readers at this point.
    void refresh_meta() {
        this->update_meta(ctx_server_outer);
    }

    // handlers using lambda function, so that they can capture `this` without `std::bind`
    // they won't be called until ctx_http.is_ready is set to true
    server_http_context::handler_t get_health;
    server_http_context::handler_t get_metrics;
    server_http_context::handler_t get_slots;
    server_http_context::handler_t post_slots;
    // Hydra state streaming (M0.0)
    server_http_context::handler_t get_state;       // GET  /slots/:id_slot/state
    server_http_context::handler_t put_state;       // PUT  /slots/:id_slot/state
    server_http_context::handler_t get_state_meta;  // GET  /slots/:id_slot/state/meta
    server_http_context::handler_t get_props;
    server_http_context::handler_t post_props;
    server_http_context::handler_t post_infill;
    server_http_context::handler_t post_completions;
    server_http_context::handler_t post_completions_oai;
    server_http_context::handler_t post_chat_completions;
    server_http_context::handler_t post_control;
    server_http_context::handler_t post_responses_oai;
    server_http_context::handler_t post_transcriptions_oai;
    server_http_context::handler_t post_anthropic_messages;
    server_http_context::handler_t post_anthropic_count_tokens;
    server_http_context::handler_t post_apply_template;
    server_http_context::handler_t get_models;
    server_http_context::handler_t post_tokenize;
    server_http_context::handler_t post_detokenize;
    server_http_context::handler_t post_embeddings;
    server_http_context::handler_t post_embeddings_oai;
    server_http_context::handler_t post_rerank;
    server_http_context::handler_t get_lora_adapters;
    server_http_context::handler_t post_lora_adapters;

    // Merged DECODE result retrieval
    server_http_context::handler_t get_decode_result;   // GET  /v1/decode/:decode_request_id
    server_http_context::handler_t delete_decode_result; // DELETE /v1/decode/:decode_request_id

    // Merged DECODE result buffer: keyed by decode_request_id
    enum decode_state {
        DECODE_STATE_LOADING,    // sync DECODE passed, awaiting DECODE_APPLY
        DECODE_STATE_RESTORING,  // DECODE_APPLY running (model swap / KV restore)
        DECODE_STATE_GENERATING, // COMPLETION task posted, generation in progress
        DECODE_STATE_DONE,       // generation complete, final result buffered
        DECODE_STATE_ERROR,      // terminal error
    };
    struct decode_result_entry {
        int32_t         id_slot = -1;
        decode_state    state = DECODE_STATE_LOADING;
        std::string     completion_id;
        std::string     oaicompat_model;
        json            generation_params;
        std::string     content;           // full generated text
        std::string     reasoning_content; // model thinking text (reasoning_content from OAI chat message)
        json            tool_calls;        // parsed OAI tool calls (array of {type,function{name,arguments},id}); empty when none
        int32_t         n_decoded = 0;
        int32_t         n_prompt_tokens = 0;
        int32_t         n_prompt_tokens_cache = 0;
        result_timings  timings;
        stop_type       stop = STOP_TYPE_NONE;
        bool            include_usage = false;
        json            hydra_metrics;
        json            match_json;        // match result from inference thread
        int64_t         created_at = 0;    // std::time(nullptr) at creation
        int             ttl_s = 300;       // TTL in seconds
        std::string     error;             // non-empty => request rejected/failed;
                                            // GET /v1/decode/:id returns this instead
                                            // of a completion body (content is unset)
        double          model_load_ms = 0.0;
        double          restore_slot_ms = 0.0;
        double          decode_init_ms = 0.0;
        int32_t         n_past = 0;
        json            model_identity;
        json            model_metadata;

        // Streaming relay: background consumer posts partials here;
        // GET handler drains via streaming_cv.
        // Wrapped in unique_ptr because std::mutex/std::condition_variable
        // are non-movable, and decode_result_entry is move-assigned.
        struct streaming_state {
            std::mutex              streaming_mutex;
            std::condition_variable streaming_cv;
            std::deque<server_task_result_ptr> streaming_queue;
            bool                    stream_finished = false;
            std::atomic<int32_t>    completion_task_id{-1};
        };
        std::unique_ptr<streaming_state> stream = std::make_unique<streaming_state>();

        // Hydra n_common observability (set at GENERATING, read by background consumer)
        int32_t         n_common           = 0;
        int32_t         n_prompt_processed = 0;
        bool            logits_reused      = false;
    };
    mutable std::mutex decode_results_mutex;
    std::map<int32_t, decode_result_entry> decode_results;
    int decode_result_max = 1024;
    int decode_result_ttl_s = 300;

    // Evict expired entries from decode_results (call with decode_results_mutex held)
    void evict_decode_results_locked();

    // to be used in router mode
    json get_model_info() const;

private:
    std::unique_ptr<server_res_generator> handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type,
            const std::string & hydra_config_json = "");
    std::unique_ptr<server_res_generator> handle_slots_save(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_restore(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_erase(const server_http_req &, int id_slot);
    std::unique_ptr<server_res_generator> handle_embeddings_impl(const server_http_req & req, task_response_type res_type);

    // using unique_ptr to allow late initialization of const
    std::unique_ptr<const server_context_meta> meta;

    const common_params & params;
    const server_context & ctx_server_outer; // P1-6: outer wrapper, needed by update_meta()
    const server_context_impl & ctx_server;

    server_queue & queue_tasks;
    server_response & queue_results;
    std::unique_ptr<server_res_generator> create_response(bool bypass_sleep = false);
};

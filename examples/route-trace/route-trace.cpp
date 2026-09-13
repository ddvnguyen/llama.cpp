#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "ggml-backend.h"

#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Dumps the MoE expert selection of every routed layer in the ROUTE_TRACE
// format consumed by colibri's offline routing tools:
//
//   <call> <pos> <layer> <expert>:<gate> <expert>:<gate> ...
//
// one line per (position, routed layer). The first field is an opaque call
// counter; the offline consumers group lines by (forward, position) and only
// use <pos>, <layer> and the expert ids, so the gate placeholder is ignored.
//
// The graph names the selected-expert tensor "ffn_moe_topk-<il>" (see
// llama_context::graph_get_cb), so we can capture it with the existing
// ggml_backend_sched eval callback and no core changes are needed.
//
// Enable with:  LLAMA_ROUTE_TRACE=/path/to/trace.txt llama-route-trace -m ...
//
// Analysis (from a colibri checkout):
//   python3 c/tools/route_coupling_report.py trace.txt
//   python3 c/tools/route_pairs.py out.coli_pairs trace.txt

static constexpr const char * ROUTE_TRACE_PREFIX = "ffn_moe_topk-";

struct route_trace_state {
    FILE *    fp   = nullptr;
    llama_pos pos  = 0;
    uint64_t  call = 0;
};

static bool route_trace_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<route_trace_state *>(user_data);
    if (st == nullptr || st->fp == nullptr) {
        return false;
    }

    const bool is_topk = strncmp(t->name, ROUTE_TRACE_PREFIX, strlen(ROUTE_TRACE_PREFIX)) == 0;
    if (ask) {
        return is_topk;
    }
    if (!is_topk || t->type != GGML_TYPE_I32) {
        return true;
    }

    const int     layer         = atoi(t->name + strlen(ROUTE_TRACE_PREFIX));
    const int64_t n_expert_used = t->ne[0];
    const int64_t n_tokens      = t->ne[1];

    std::vector<int32_t> ids((size_t) n_expert_used * n_tokens);
    ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));

    for (int64_t j = 0; j < n_tokens; ++j) {
        fprintf(st->fp, "%llu %d %d", (unsigned long long) st->call, (int) (st->pos + j), layer);
        for (int64_t k = 0; k < n_expert_used; ++k) {
            fprintf(st->fp, " %d:1", ids[(size_t) j * n_expert_used + k]);
        }
        fputc('\n', st->fp);
        st->call++;
    }
    return true;
}

static bool decode_one(llama_context * ctx, llama_batch & batch, route_trace_state & st, llama_token token) {
    common_batch_clear(batch);
    common_batch_add(batch, token, st.pos, { 0 }, true);
    if (llama_decode(ctx, batch)) {
        LOG_ERR("%s: llama_decode failed at pos %d\n", __func__, (int) st.pos);
        return false;
    }
    st.pos++;
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * trace_path = getenv("LLAMA_ROUTE_TRACE");
    if (trace_path == nullptr || trace_path[0] == '\0') {
        LOG_ERR("%s: LLAMA_ROUTE_TRACE=<path> is required\n", __func__);
        return 1;
    }

    route_trace_state st;

    // register the readback callback before the context is created; the file
    // stays null until we start decoding so warmup/reserve graphs are skipped
    params.cb_eval           = route_trace_cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab  = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, true);
    if (tokens.empty()) {
        LOG_ERR("%s: empty prompt\n", __func__);
        return 1;
    }

    st.fp = fopen(trace_path, "w");
    if (st.fp == nullptr) {
        LOG_ERR("%s: failed to open %s\n", __func__, trace_path);
        return 1;
    }

    llama_batch batch = llama_batch_init(1, 0, 1);

    // prompt is evaluated one token at a time so every row maps to a single position
    for (llama_token token : tokens) {
        if (!decode_one(ctx, batch, st, token)) {
            fclose(st.fp);
            return 1;
        }
    }

    // optional greedy continuation, so decode-time routing is captured too
    const int n_gen = params.n_predict > 0 ? params.n_predict : 0;
    for (int i = 0; i < n_gen; ++i) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            break;
        }
        llama_token best = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (logits[v] > logits[best]) {
                best = v;
            }
        }
        if (!decode_one(ctx, batch, st, best)) {
            break;
        }
    }

    fclose(st.fp);
    st.fp = nullptr;

    LOG_INF("%s: wrote %llu routing rows (%d positions) to %s\n",
            __func__, (unsigned long long) st.call, (int) st.pos, trace_path);

    llama_batch_free(batch);
    llama_backend_free();

    return 0;
}

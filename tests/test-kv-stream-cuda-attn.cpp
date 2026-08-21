#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "testing.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace {

constexpr int64_t HEAD_DIM = 256;
constexpr int64_t N_KV     = 512;
constexpr int64_t N_KV_HEAD = 4;
constexpr int64_t N_Q_HEAD  = 24;
constexpr int64_t N_BATCH   = 2;

size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1)/alignment*alignment;
}

std::vector<uint8_t> pack_token_block(
        const std::vector<uint8_t> & source,
        ggml_type type,
        int64_t token_begin,
        int64_t token_count) {
    const size_t row_bytes = ggml_row_size(type, HEAD_DIM);
    const size_t head_bytes = row_bytes*N_KV;
    std::vector<uint8_t> result(row_bytes*token_count*N_KV_HEAD);
    for (int64_t head = 0; head < N_KV_HEAD; ++head) {
        const uint8_t * src = source.data() + head*head_bytes + token_begin*row_bytes;
        uint8_t * dst = result.data() + head*token_count*row_bytes;
        std::copy_n(src, token_count*row_bytes, dst);
    }
    return result;
}

struct attention_inputs {
    std::vector<float> q;
    std::vector<uint8_t> k;
    std::vector<uint8_t> v;
    std::vector<uint16_t> mask;
};

attention_inputs make_inputs() {
    attention_inputs result;

    result.q.resize(HEAD_DIM*N_BATCH*N_Q_HEAD);
    for (size_t i = 0; i < result.q.size(); ++i) {
        result.q[i] = 0.15f*std::sin(float(i)*0.03125f) + 0.05f*std::cos(float(i)*0.0078125f);
    }

    const int64_t nrows = N_KV*N_KV_HEAD;
    std::vector<float> source(HEAD_DIM*nrows);
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = 0.4f*std::sin(float(i)*0.001953125f) + 0.2f*std::cos(float(i)*0.00048828125f);
    }

    result.k.resize(ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*nrows);
    const size_t k_written = ggml_quantize_chunk(
        GGML_TYPE_Q8_0, source.data(), result.k.data(), 0, nrows, HEAD_DIM, nullptr);
    GGML_ASSERT(k_written == result.k.size());

    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = 0.35f*std::cos(float(i)*0.00146484375f) - 0.1f*std::sin(float(i)*0.00390625f);
    }
    result.v.resize(ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*nrows);
    const size_t v_written = ggml_quantize_chunk(
        GGML_TYPE_Q4_0, source.data(), result.v.data(), 0, nrows, HEAD_DIM, nullptr);
    GGML_ASSERT(v_written == result.v.size());

    result.mask.resize(N_KV*N_BATCH);
    for (int64_t batch = 0; batch < N_BATCH; ++batch) {
        for (int64_t token = 0; token < N_KV; ++token) {
            // Vary both token blocks and both query rows. This catches a
            // streamed implementation that offsets the first mask row but
            // accidentally uses the compact block width as the next-row pitch.
            const float bias = -0.015625f*float((token + 73*batch) % 127);
            result.mask[batch*N_KV + token] = ggml_fp32_to_fp16(bias);
        }
    }
    return result;
}

std::vector<float> run_attention(
        ggml_backend_t backend,
        const attention_inputs & inputs,
        ggml_backend_buffer_type_t kv_buft) {
    constexpr size_t N_TENSORS = 16;
    const size_t context_bytes = ggml_tensor_overhead()*N_TENSORS + ggml_graph_overhead_custom(N_TENSORS, false);

    ggml_init_params params{
        /* .mem_size   = */ context_bytes,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr compute_ctx(ggml_init(params));
    ggml_context_ptr kv_ctx(ggml_init(params));
    GGML_ASSERT(compute_ctx && kv_ctx);

    ggml_tensor * q = ggml_new_tensor_4d(
        compute_ctx.get(), GGML_TYPE_F32, HEAD_DIM, N_BATCH, N_Q_HEAD, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(
        compute_ctx.get(), GGML_TYPE_F16, N_KV, N_BATCH, 1, 1);
    ggml_tensor * k = ggml_new_tensor_4d(
        kv_ctx.get(), GGML_TYPE_Q8_0, HEAD_DIM, N_KV, N_KV_HEAD, 1);
    ggml_tensor * v = ggml_new_tensor_4d(
        kv_ctx.get(), GGML_TYPE_Q4_0, HEAD_DIM, N_KV, N_KV_HEAD, 1);

    ggml_tensor * out = ggml_flash_attn_ext(
        compute_ctx.get(), q, k, v, mask, 1.0f/std::sqrt(float(HEAD_DIM)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_set_name(out, "streamed-attention-output");

    ggml_backend_buffer_ptr kv_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(kv_ctx.get(), kv_buft));
    ggml_backend_buffer_ptr compute_buffer(
        ggml_backend_alloc_ctx_tensors(compute_ctx.get(), backend));
    GGML_ASSERT(kv_buffer && compute_buffer);

    ggml_backend_tensor_set(q, inputs.q.data(), 0, inputs.q.size()*sizeof(float));
    ggml_backend_tensor_set(k, inputs.k.data(), 0, inputs.k.size());
    ggml_backend_tensor_set(v, inputs.v.data(), 0, inputs.v.size());
    ggml_backend_tensor_set(mask, inputs.mask.data(), 0, inputs.mask.size()*sizeof(uint16_t));

    ggml_cgraph * graph = ggml_new_graph_custom(compute_ctx.get(), N_TENSORS, false);
    ggml_build_forward_expand(graph, out);
    GGML_ASSERT(ggml_backend_supports_op(backend, out));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size()*sizeof(float));
    return result;
}

} // namespace

int main() {
    testing t;

    t.test("two streamed Q8/Q4 blocks and query rows match ordinary CUDA attention", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs();
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()));

        ggml_backend_cuda_kv_stream_params params{};
        params.device      = 0;
        params.stage_bytes = 512*1024;
        params.stage_slots = 1;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
            return;
        }

        std::vector<uint8_t> cleared(params.stage_bytes, 0);
        GGML_ASSERT(ggml_backend_cuda_kv_stream_stage_upload(
            runtime, 0, 0, cleared.data(), cleared.size()));

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime));

        const std::vector<uint8_t> expected_k = pack_token_block(
            inputs.k, GGML_TYPE_Q8_0, 256, 256);
        const std::vector<uint8_t> expected_v = pack_token_block(
            inputs.v, GGML_TYPE_Q4_0, 256, 256);
        const size_t v_offset = align_up(expected_k.size(), 128);
        std::vector<uint8_t> staged(v_offset + expected_v.size());
        GGML_ASSERT(ggml_backend_cuda_kv_stream_stage_download(
            runtime, 0, 0, staged.data(), staged.size()));
        t.assert_true(
            "final K token block was packed into the managed stage",
            std::equal(expected_k.begin(), expected_k.end(), staged.begin()));
        t.assert_true(
            "final V token block was packed into the managed stage",
            std::equal(expected_v.begin(), expected_v.end(), staged.begin() + v_offset));
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }

        float max_abs = 0.0f;
        float max_rel = 0.0f;
        for (size_t i = 0; i < expected.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(expected[i] - actual[i]));
            max_rel = std::max(max_rel, std::abs(expected[i] - actual[i])/(std::abs(expected[i]) + 1e-6f));
        }
        std::fprintf(stderr, "streamed attention max_abs=%g max_rel=%g\n", max_abs, max_rel);
        t.assert_true("outputs remain finite", std::isfinite(max_abs) && std::isfinite(max_rel));
        t.assert_true("streamed output is numerically equivalent", max_abs <= 1e-5f);
    });

    ggml_quantize_free();
    return t.summary();
}

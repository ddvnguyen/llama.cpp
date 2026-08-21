#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"

#include <algorithm>
#include <unordered_map>
#include <vector>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, const int ne30, const int n_kv_max,
        const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int values_per_lane = 8;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;
    const int sequence = blockIdx.y;
    const int query    = blockIdx.x;

    const half * mask = mask_ptr + sequence*s33 + query*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + query)*n_kv_max;

    __shared__ int warp_offsets[256/WARP_SIZE];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        uint32_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const bool selected = i < ne30 && isfinite(__half2float(mask[i]));
            selected_warp[item] = __ballot_sync(0xFFFFFFFF, selected);
            warp_count += __popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/WARP_SIZE; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const uint32_t lane_mask = lane == 0 ? 0 : (1u << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + __popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & (uint32_t(1) << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += __popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = row_count;
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t n_kv_max, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(mask, indices, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
#else
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const dim3 blocks_num(mask->ne[1], mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    ggml_cuda_kernel_launch(flash_attn_mask_to_sparse_indices, launch_params,
        (const half *) mask->data, indices, int(mask->ne[0]), n_kv_max, s31, s33);
    CUDA_CHECK(cudaGetLastError());
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(ctx, dst);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, 2LL*n_kv_max);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}
struct ggml_cuda_kv_stream_transfer_ring {
    char * pool_data = nullptr;
    size_t page_bytes = 0;
    uint32_t stage_slots = 0;
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t producer_ready = nullptr;
    std::vector<cudaEvent_t> ready;
    std::vector<cudaEvent_t> consumed;
    std::vector<uint8_t> slot_used;
    uint64_t asynchronous_page_uploads = 0;
    uint64_t compute_stream_waits = 0;
    uint64_t stage_slot_reuses = 0;
};

ggml_cuda_kv_stream_transfer_ring * ggml_cuda_kv_stream_transfer_ring_new(
        void * pool_data, size_t page_bytes, uint32_t stage_slots) {
    if (pool_data == nullptr || page_bytes == 0 || stage_slots == 0) {
        return nullptr;
    }

    auto * ring = new ggml_cuda_kv_stream_transfer_ring;
    ring->pool_data = static_cast<char *>(pool_data);
    ring->page_bytes = page_bytes;
    ring->stage_slots = stage_slots;
    ring->ready.resize(stage_slots, nullptr);
    ring->consumed.resize(stage_slots, nullptr);
    ring->slot_used.resize(stage_slots, 0);

    auto cleanup = [&]() {
        for (cudaEvent_t event : ring->ready) {
            if (event != nullptr) {
                (void) cudaEventDestroy(event);
            }
        }
        for (cudaEvent_t event : ring->consumed) {
            if (event != nullptr) {
                (void) cudaEventDestroy(event);
            }
        }
        if (ring->producer_ready != nullptr) {
            (void) cudaEventDestroy(ring->producer_ready);
        }
        if (ring->copy_stream != nullptr) {
            (void) cudaStreamDestroy(ring->copy_stream);
        }
        delete ring;
    };

    if (cudaStreamCreateWithFlags(&ring->copy_stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&ring->producer_ready, cudaEventDisableTiming) != cudaSuccess) {
        (void) cudaGetLastError();
        cleanup();
        return nullptr;
    }
    for (uint32_t slot = 0; slot < stage_slots; ++slot) {
        if (cudaEventCreateWithFlags(&ring->ready[slot], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ring->consumed[slot], cudaEventDisableTiming) != cudaSuccess) {
            (void) cudaGetLastError();
            cleanup();
            return nullptr;
        }
    }
    return ring;
}

void ggml_cuda_kv_stream_transfer_ring_free(ggml_cuda_kv_stream_transfer_ring * ring) {
    if (ring == nullptr) {
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(ring->copy_stream));
    for (cudaEvent_t event : ring->ready) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
    for (cudaEvent_t event : ring->consumed) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
    CUDA_CHECK(cudaEventDestroy(ring->producer_ready));
    CUDA_CHECK(cudaStreamDestroy(ring->copy_stream));
    delete ring;
}

ggml_cuda_kv_stream_transfer_stats ggml_cuda_kv_stream_transfer_ring_get_stats(
        const ggml_cuda_kv_stream_transfer_ring * ring) {
    if (ring == nullptr) {
        return {};
    }
    return {
        ring->asynchronous_page_uploads,
        ring->compute_stream_waits,
        ring->stage_slot_reuses,
    };
}

struct ggml_cuda_kv_stream_resident_cache {
    char * pool_data = nullptr;
    size_t pool_bytes = 0;
    size_t scratch_bytes = 0;
    size_t page_bytes = 0;
    uint32_t layer_count = 0;
    uint32_t page_tokens = 0;
    uint32_t resident_pages_per_layer = 0;
    uint32_t next_layer = 0;

    std::unordered_map<const void *, uint32_t> layer_by_k;
    std::vector<uint8_t> loaded;
    ggml_cuda_kv_stream_resident_stats stats;
};

ggml_cuda_kv_stream_resident_cache * ggml_cuda_kv_stream_resident_cache_new(
        void * pool_data, size_t pool_bytes, size_t scratch_bytes, size_t page_bytes,
        uint32_t layer_count, uint32_t page_tokens) {
    if (pool_data == nullptr || scratch_bytes == 0 || scratch_bytes >= pool_bytes ||
            page_bytes == 0 || scratch_bytes%page_bytes != 0 ||
            layer_count == 0 || page_tokens != 256) {
        return nullptr;
    }

    const size_t resident_pages = (pool_bytes - scratch_bytes)/(page_bytes*layer_count);
    if (resident_pages == 0 || resident_pages > UINT32_MAX) {
        return nullptr;
    }

    auto * cache = new ggml_cuda_kv_stream_resident_cache;
    cache->pool_data = static_cast<char *>(pool_data);
    cache->pool_bytes = pool_bytes;
    cache->scratch_bytes = scratch_bytes;
    cache->page_bytes = page_bytes;
    cache->layer_count = layer_count;
    cache->page_tokens = page_tokens;
    cache->resident_pages_per_layer = resident_pages;
    cache->loaded.resize(size_t(layer_count)*resident_pages, 0);
    return cache;
}

void ggml_cuda_kv_stream_resident_cache_free(ggml_cuda_kv_stream_resident_cache * cache) {
    delete cache;
}

void ggml_cuda_kv_stream_resident_cache_reset(ggml_cuda_kv_stream_resident_cache * cache) {
    if (cache == nullptr) {
        return;
    }
    std::fill(cache->loaded.begin(), cache->loaded.end(), 0);
    cache->layer_by_k.clear();
    cache->next_layer = 0;
    cache->stats = {};
}

ggml_cuda_kv_stream_resident_stats ggml_cuda_kv_stream_resident_cache_get_stats(
        const ggml_cuda_kv_stream_resident_cache * cache) {
    return cache == nullptr ? ggml_cuda_kv_stream_resident_stats{} : cache->stats;
}

namespace {

constexpr int KV_STREAM_HEAD_DIM = 256;
constexpr int KV_STREAM_PARTS_PER_CHUNK = 2;

static int64_t kv_stream_block_tokens(const ggml_tensor * dst, size_t stage_bytes) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const size_t k_row_bytes = ggml_row_size(K->type, K->ne[0]);
    const size_t v_row_bytes = ggml_row_size(V->type, V->ne[0]);
    const size_t bytes_per_token = k_row_bytes*K->ne[2] + v_row_bytes*V->ne[2];
    if (bytes_per_token == 0) {
        return 0;
    }

    int64_t tokens = std::min<int64_t>(K->ne[1], stage_bytes/bytes_per_token);
    tokens = tokens/FATTN_KQ_STRIDE*FATTN_KQ_STRIDE;
    while (tokens > 0) {
        const size_t k_bytes = k_row_bytes*tokens*K->ne[2];
        const size_t v_offset = GGML_PAD(k_bytes, 128);
        const size_t v_bytes = v_row_bytes*tokens*V->ne[2];
        if (v_offset <= stage_bytes && v_bytes <= stage_bytes - v_offset) {
            return tokens;
        }
        tokens -= FATTN_KQ_STRIDE;
    }
    return 0;
}

template<int D>
static __global__ void kv_stream_combine_chunk_results(
        const float * parts,
        const float2 * meta,
        float * dst,
        int nrows,
        int nchunks) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= nrows || tid >= D) {
        return;
    }

    float maximum = -FLT_MAX;
    for (int chunk = 0; chunk < nchunks; ++chunk) {
        const int base = (chunk*nrows + row)*KV_STREAM_PARTS_PER_CHUNK;
        for (int part = 0; part < KV_STREAM_PARTS_PER_CHUNK; ++part) {
            maximum = fmaxf(maximum, meta[base + part].x);
        }
    }

    float numerator = 0.0f;
    float denominator = 0.0f;
    for (int chunk = 0; chunk < nchunks; ++chunk) {
        const int base = (chunk*nrows + row)*KV_STREAM_PARTS_PER_CHUNK;
        for (int part = 0; part < KV_STREAM_PARTS_PER_CHUNK; ++part) {
            const float weight = expf(meta[base + part].x - maximum);
            numerator += weight*parts[(base + part)*D + tid];
            denominator += weight*meta[base + part].y;
        }
    }
    dst[row*D + tid] = numerator/denominator;
}

template<bool use_logit_softcap>
static void kv_stream_launch_q8_q4_partial(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        float * parts,
        float2 * meta) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    constexpr int ncols = 1;
    constexpr int nthreads = 128;
    constexpr int nwarps = nthreads/WARP_SIZE;
    fattn_kernel_t kernel = flash_attn_ext_vec<
        KV_STREAM_HEAD_DIM, ncols, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, use_logit_softcap>;

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float m0 = powf(2.0f, -(max_bias       )/n_head_log2);
    const float m1 = powf(2.0f, -(max_bias/2.0f)/n_head_log2);
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    const dim3 blocks(Q->ne[1], KV_STREAM_PARTS_PER_CHUNK, Q->ne[2]*Q->ne[3]);
    const dim3 threads(WARP_SIZE, nwarps, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
    ggml_cuda_kernel_launch(kernel, launch_params,
        (const char *) Q->data,
        (const char *) K->data,
        (const char *) V->data,
        mask ? (const char *) mask->data : nullptr,
        nullptr,
        nullptr,
        parts,
        meta,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3], K->nb[1], K->nb[2], K->nb[3],
        V->nb[1], V->nb[2], V->nb[3],
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool ggml_cuda_flash_attn_ext_streamed_supported(const ggml_tensor * dst, size_t stage_bytes) {
    if (dst == nullptr || dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    return Q != nullptr && K != nullptr && V != nullptr &&
        Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0 &&
        Q->ne[0] == KV_STREAM_HEAD_DIM && V->ne[0] == KV_STREAM_HEAD_DIM &&
        Q->ne[1] >= 1 && Q->ne[1] <= 256 && Q->ne[3] == 1 && K->ne[3] == 1 && V->ne[3] == 1 &&
        K->ne[1] == V->ne[1] && K->ne[2] == V->ne[2] &&
        K->ne[1] % FATTN_KQ_STRIDE == 0 &&
        K->nb[0] == ggml_element_size(K) && V->nb[0] == ggml_element_size(V) &&
        K->nb[1] >= ggml_row_size(K->type, K->ne[0]) &&
        V->nb[1] >= ggml_row_size(V->type, V->ne[0]) &&
        (mask == nullptr || (mask->type == GGML_TYPE_F16 && ggml_is_contiguous(mask))) &&
        sinks == nullptr && kv_stream_block_tokens(dst, stage_bytes) > 0;
}

void ggml_cuda_flash_attn_ext_streamed(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        ggml_cuda_kv_stream_transfer_ring * transfer_ring,
        ggml_cuda_kv_stream_resident_cache * resident_cache) {
    GGML_ASSERT(transfer_ring != nullptr);
    void * stage_data = transfer_ring->pool_data;
    const size_t stage_bytes = transfer_ring->page_bytes;
    GGML_ASSERT(ggml_cuda_flash_attn_ext_streamed_supported(dst, stage_bytes));

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int64_t block_tokens = resident_cache != nullptr ?
        resident_cache->page_tokens : kv_stream_block_tokens(dst, stage_bytes);
    const int nchunks = (K->ne[1] + block_tokens - 1)/block_tokens;
    const int nrows = ggml_nrows(dst);

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float> parts(pool, size_t(nchunks)*KV_STREAM_PARTS_PER_CHUNK*ggml_nelements(dst));
    ggml_cuda_pool_alloc<float2> meta(pool, size_t(nchunks)*KV_STREAM_PARTS_PER_CHUNK*nrows);

    struct chunk_descriptor {
        int64_t token_begin = 0;
        int64_t token_count = 0;
        size_t k_row_bytes = 0;
        size_t v_row_bytes = 0;
        size_t k_head_bytes = 0;
        size_t k_bytes = 0;
        size_t v_offset = 0;
        size_t v_head_bytes = 0;
        size_t v_bytes = 0;
        char * stage = nullptr;
        bool upload = true;
        bool streamed = false;
        uint32_t slot = 0;
    };

    uint32_t resident_layer = 0;
    if (resident_cache != nullptr) {
        auto [it, inserted] = resident_cache->layer_by_k.emplace(K->data, resident_cache->next_layer);
        if (inserted) {
            GGML_ASSERT(resident_cache->next_layer < resident_cache->layer_count);
            ++resident_cache->next_layer;
        }
        resident_layer = it->second;
    }

    std::vector<chunk_descriptor> chunks(nchunks);
    std::vector<size_t> streamed_chunks;
    streamed_chunks.reserve(nchunks);

    for (int chunk = 0; chunk < nchunks; ++chunk) {
        auto & desc = chunks[chunk];
        const int64_t token_begin = chunk*block_tokens;
        const int64_t token_count = std::min<int64_t>(block_tokens, K->ne[1] - token_begin);
        const size_t k_row_bytes = ggml_row_size(K->type, K->ne[0]);
        const size_t v_row_bytes = ggml_row_size(V->type, V->ne[0]);
        const size_t k_head_bytes = k_row_bytes*token_count;
        const size_t k_bytes = k_head_bytes*K->ne[2];
        const size_t v_offset = GGML_PAD(k_bytes, 128);
        const size_t v_head_bytes = v_row_bytes*token_count;
        const size_t v_bytes = v_head_bytes*V->ne[2];
        GGML_ASSERT(v_offset <= stage_bytes && v_bytes <= stage_bytes - v_offset);

        desc.token_begin = token_begin;
        desc.token_count = token_count;
        desc.k_row_bytes = k_row_bytes;
        desc.v_row_bytes = v_row_bytes;
        desc.k_head_bytes = k_head_bytes;
        desc.k_bytes = k_bytes;
        desc.v_offset = v_offset;
        desc.v_head_bytes = v_head_bytes;
        desc.v_bytes = v_bytes;
        desc.stage = static_cast<char *>(stage_data);

        if (resident_cache != nullptr) {
            GGML_ASSERT(token_count == resident_cache->page_tokens);
            GGML_ASSERT(v_offset + v_bytes == resident_cache->page_bytes);

            const uint32_t page = token_begin/resident_cache->page_tokens;
            if (page < resident_cache->resident_pages_per_layer) {
                const size_t resident_index = size_t(resident_layer)*resident_cache->resident_pages_per_layer + page;
                desc.stage = resident_cache->pool_data + resident_cache->scratch_bytes +
                    resident_index*resident_cache->page_bytes;
                if (resident_cache->loaded[resident_index]) {
                    ++resident_cache->stats.resident_hits;
                    // The final page contains the rows most recently changed by SET_ROWS.
                    desc.upload = chunk == nchunks - 1;
                } else {
                    ++resident_cache->stats.resident_misses;
                    resident_cache->loaded[resident_index] = 1;
                }
            } else {
                ++resident_cache->stats.streamed_pages;
                desc.streamed = true;
            }
        } else {
            desc.streamed = true;
        }

        if (desc.streamed) {
            const size_t stream_index = streamed_chunks.size();
            desc.slot = uint32_t(stream_index%transfer_ring->stage_slots);
            desc.stage = transfer_ring->pool_data + size_t(desc.slot)*transfer_ring->page_bytes;
            streamed_chunks.push_back(chunk);
        }
    }

    auto upload = [&](const chunk_descriptor & desc, cudaStream_t stream) {
        if (!desc.upload) {
            return;
        }
        for (int64_t head = 0; head < K->ne[2]; ++head) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                desc.stage + head*desc.k_head_bytes, desc.k_row_bytes,
                static_cast<const char *>(K->data) + head*K->nb[2] + desc.token_begin*K->nb[1], K->nb[1],
                desc.k_row_bytes, desc.token_count, cudaMemcpyHostToDevice, stream));
        }
        for (int64_t head = 0; head < V->ne[2]; ++head) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                desc.stage + desc.v_offset + head*desc.v_head_bytes, desc.v_row_bytes,
                static_cast<const char *>(V->data) + head*V->nb[2] + desc.token_begin*V->nb[1], V->nb[1],
                desc.v_row_bytes, desc.token_count, cudaMemcpyHostToDevice, stream));
        }
        if (resident_cache != nullptr) {
            resident_cache->stats.host_to_device_bytes += desc.k_bytes + desc.v_bytes;
        }
    };

    auto schedule_streamed = [&](size_t stream_index) {
        auto & desc = chunks[streamed_chunks[stream_index]];
        const uint32_t slot = desc.slot;
        if (transfer_ring->slot_used[slot]) {
            CUDA_CHECK(cudaStreamWaitEvent(
                transfer_ring->copy_stream, transfer_ring->consumed[slot], 0));
            ++transfer_ring->stage_slot_reuses;
        }
        upload(desc, transfer_ring->copy_stream);
        CUDA_CHECK(cudaEventRecord(transfer_ring->ready[slot], transfer_ring->copy_stream));
        transfer_ring->slot_used[slot] = 1;
        ++transfer_ring->asynchronous_page_uploads;
    };

    if (!streamed_chunks.empty()) {
        // SET_ROWS and all other producers for this layer are ordered before
        // this marker on the compute stream. The copy stream may then run
        // independently while attention consumes previously prepared pages.
        CUDA_CHECK(cudaEventRecord(transfer_ring->producer_ready, ctx.stream()));
        CUDA_CHECK(cudaStreamWaitEvent(
            transfer_ring->copy_stream, transfer_ring->producer_ready, 0));
        const size_t initial = std::min<size_t>(
            transfer_ring->stage_slots, streamed_chunks.size());
        for (size_t i = 0; i < initial; ++i) {
            schedule_streamed(i);
        }
    }

    size_t stream_index = 0;
    for (int chunk = 0; chunk < nchunks; ++chunk) {
        auto & desc = chunks[chunk];
        if (desc.streamed) {
            CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), transfer_ring->ready[desc.slot], 0));
            ++transfer_ring->compute_stream_waits;
        } else if (desc.upload) {
            upload(desc, ctx.stream());
        }

        ggml_tensor staged_k = *K;
        ggml_tensor staged_v = *V;
        staged_k.data = desc.stage;
        staged_k.ne[1] = desc.token_count;
        staged_k.nb[1] = desc.k_row_bytes;
        staged_k.nb[2] = desc.k_head_bytes;
        staged_k.nb[3] = desc.k_bytes;
        staged_v.data = desc.stage + desc.v_offset;
        staged_v.ne[1] = desc.token_count;
        staged_v.nb[1] = desc.v_row_bytes;
        staged_v.nb[2] = desc.v_head_bytes;
        staged_v.nb[3] = desc.v_bytes;

        ggml_tensor staged_mask{};
        ggml_tensor * staged_mask_ptr = nullptr;
        if (mask != nullptr) {
            staged_mask = *mask;
            staged_mask.data = static_cast<char *>(mask->data) + desc.token_begin*mask->nb[0];
            staged_mask.ne[0] = desc.token_count;
            staged_mask_ptr = &staged_mask;
        }

        ggml_tensor staged_dst = *dst;
        staged_dst.src[1] = &staged_k;
        staged_dst.src[2] = &staged_v;
        staged_dst.src[3] = staged_mask_ptr;

        // Preserve the normal CUDA flash-attention path when the active cache
        // fits in a single page. Besides avoiding an unnecessary partial
        // reduction, this keeps short-context logits numerically identical to
        // a non-streamed cache.
        if (nchunks == 1) {
            ggml_cuda_flash_attn_ext(ctx, &staged_dst);
            if (desc.streamed) {
                CUDA_CHECK(cudaEventRecord(transfer_ring->consumed[desc.slot], ctx.stream()));
            }
            return;
        }

        float * chunk_parts = parts.ptr + size_t(chunk)*KV_STREAM_PARTS_PER_CHUNK*ggml_nelements(dst);
        float2 * chunk_meta = meta.ptr + size_t(chunk)*KV_STREAM_PARTS_PER_CHUNK*nrows;
        float logit_softcap = 0.0f;
        memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
        if (logit_softcap == 0.0f) {
            kv_stream_launch_q8_q4_partial<false>(ctx, &staged_dst, chunk_parts, chunk_meta);
        } else {
            kv_stream_launch_q8_q4_partial<true>(ctx, &staged_dst, chunk_parts, chunk_meta);
        }

        if (desc.streamed) {
            CUDA_CHECK(cudaEventRecord(transfer_ring->consumed[desc.slot], ctx.stream()));
            const size_t next = stream_index + transfer_ring->stage_slots;
            if (next < streamed_chunks.size()) {
                schedule_streamed(next);
            }
            ++stream_index;
        }
    }

    const dim3 blocks(nrows, 1, 1);
    const dim3 threads(KV_STREAM_HEAD_DIM, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
    ggml_cuda_kernel_launch(kv_stream_combine_chunk_results<KV_STREAM_HEAD_DIM>, launch_params,
        parts.ptr, meta.ptr, static_cast<float *>(dst->data), nrows, nchunks);
    CUDA_CHECK(cudaGetLastError());
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        if (ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(ctx, dst)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

static bool ggml_cuda_fattn_kv_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            return true;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return false;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC:
            need_f16_K = K->type == GGML_TYPE_F32;
            need_f16_V = V->type == GGML_TYPE_F32;
            break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}

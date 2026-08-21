#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_streamed_supported(const ggml_tensor * dst, size_t stage_bytes);

void ggml_cuda_flash_attn_ext_streamed(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    void * stage_data,
    size_t stage_bytes);

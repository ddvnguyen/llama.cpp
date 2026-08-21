#include "ggml-cuda.h"
#include "testing.h"

#include <cstdint>

int main() {
    testing t;

    t.test("runtime exposes exact staging geometry", [](testing & t) {
        ggml_backend_cuda_kv_stream_params params{};
        params.device      = 0;
        params.stage_bytes = 1024*1024;
        params.stage_slots = 2;

        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("runtime allocation succeeds", runtime != nullptr)) {
            return;
        }

        t.assert_equal(params.stage_bytes, ggml_backend_cuda_kv_stream_stage_bytes(runtime));
        t.assert_equal(params.stage_slots, ggml_backend_cuda_kv_stream_stage_slots(runtime));
        t.assert_true("streamed buffer type exists", ggml_backend_cuda_kv_stream_buffer_type(runtime) != nullptr);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);
    });

    t.test("authoritative buffer is pinned-host accessible", [](testing & t) {
        ggml_backend_cuda_kv_stream_params params{};
        params.device      = 0;
        params.stage_bytes = 1024*1024;
        params.stage_slots = 1;

        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("runtime allocation succeeds", runtime != nullptr)) {
            return;
        }

        auto buft = ggml_backend_cuda_kv_stream_buffer_type(runtime);
        auto buffer = ggml_backend_buft_alloc_buffer(buft, 256*1024);
        if (!t.assert_true("host buffer allocation succeeds", buffer != nullptr)) {
            ggml_backend_cuda_kv_stream_runtime_free(runtime);
            return;
        }

        t.assert_true("buffer reports host accessibility", ggml_backend_buffer_is_host(buffer));
        t.assert_equal(size_t(256*1024), ggml_backend_buffer_get_size(buffer));
        t.assert_true("buffer base exists", ggml_backend_buffer_get_base(buffer) != nullptr);

        ggml_backend_cuda_kv_stream_runtime_free(runtime);
        ggml_backend_buffer_free(buffer);
    });

    t.test("invalid runtime geometry is rejected", [](testing & t) {
        ggml_backend_cuda_kv_stream_params params{};
        params.device      = -1;
        params.stage_bytes = 1024*1024;
        params.stage_slots = 1;
        t.assert_true("negative device is rejected", ggml_backend_cuda_kv_stream_runtime_new(params) == nullptr);

        params.device      = 0;
        params.stage_bytes = 0;
        t.assert_true("zero stage size is rejected", ggml_backend_cuda_kv_stream_runtime_new(params) == nullptr);

        params.stage_bytes = 1024*1024;
        params.stage_slots = 0;
        t.assert_true("zero stage slots are rejected", ggml_backend_cuda_kv_stream_runtime_new(params) == nullptr);
    });

    return t.summary();
}

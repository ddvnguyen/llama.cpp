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
        auto buft = ggml_backend_cuda_kv_stream_buffer_type(runtime);
        if (!t.assert_true("streamed buffer type exists", buft != nullptr)) {
            ggml_backend_cuda_kv_stream_runtime_free(runtime);
            return;
        }
        t.assert_true(
            "owning CUDA device accepts streamed buffers",
            ggml_backend_dev_supports_buft(ggml_backend_buft_get_device(buft), buft));
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

    t.test("unimplemented operations cannot consume streamed storage", [](testing & t) {
        ggml_backend_cuda_kv_stream_params params{};
        params.device      = 0;
        params.stage_bytes = 1024*1024;
        params.stage_slots = 1;

        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("runtime allocation succeeds", runtime != nullptr)) {
            return;
        }

        auto buft = ggml_backend_cuda_kv_stream_buffer_type(runtime);
        auto buffer = ggml_backend_buft_alloc_buffer(buft, 4096);
        if (!t.assert_true("host buffer allocation succeeds", buffer != nullptr)) {
            ggml_backend_cuda_kv_stream_runtime_free(runtime);
            return;
        }

        ggml_tensor source{};
        source.type   = GGML_TYPE_F32;
        source.buffer = buffer;
        source.data   = ggml_backend_buffer_get_base(buffer);
        source.ne[0]  = 1;
        source.ne[1]  = 1;
        source.ne[2]  = 1;
        source.ne[3]  = 1;
        source.nb[0]  = sizeof(float);
        source.nb[1]  = sizeof(float);
        source.nb[2]  = sizeof(float);
        source.nb[3]  = sizeof(float);

        ggml_tensor fill{};
        fill.op   = GGML_OP_FILL;
        fill.type = GGML_TYPE_F32;
        fill.ne[0] = fill.ne[1] = fill.ne[2] = fill.ne[3] = 1;
        fill.nb[0] = fill.nb[1] = fill.nb[2] = fill.nb[3] = sizeof(float);

        auto device = ggml_backend_buft_get_device(buft);
        t.assert_true("ordinary fill is supported", ggml_backend_dev_supports_op(device, &fill));
        fill.src[0] = &source;
        t.assert_true("fill from streamed storage is rejected", !ggml_backend_dev_supports_op(device, &fill));

        ggml_backend_buffer_free(buffer);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);
    });

    return t.summary();
}

#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    4
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    0

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 96, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

// Hydra #348: like ggml_backend_rpc_start_server, but serves pre-built backend
// instances (e.g. the ones a local llama_context's ggml_backend_sched already
// created) instead of creating independent ones for the same devices. The
// caller retains ownership of `backends` - this never frees them. Pair with
// ggml_backend_rpc_server_compute_lock/unlock to serialize GPU compute against
// a caller that also dispatches local work to the same backend instances.
GGML_BACKEND_API void ggml_backend_rpc_start_server_with_backends(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_backends, ggml_backend_t * backends);

// Hydra #348: acquire/release the per-device mutex this RPC server holds
// while computing a graph (rpc_server::graph_compute/graph_recompute). A
// caller sharing the same backend instance for local inference (see
// ggml_backend_rpc_start_server_with_backends) must hold the same lock around
// its own dispatch so the two never execute on the device concurrently.
GGML_BACKEND_API void ggml_backend_rpc_server_compute_lock(uint32_t device);
GGML_BACKEND_API void ggml_backend_rpc_server_compute_unlock(uint32_t device);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

// Hydra: zero-copy COMBINED expert tensors (llama.cpp#20). A worker registers
// its own already-loaded tensors by name (once, at model-load time); a peer
// resolves + binds directly to that memory instead of allocating a buffer
// and copying bytes over the wire. See ggml-rpc.cpp for the wire protocol
// (RPC_CMD_RESOLVE_TENSOR).
GGML_BACKEND_API void ggml_backend_rpc_register_local_tensor(const char * name, struct ggml_tensor * tensor);
GGML_BACKEND_API struct ggml_tensor * ggml_backend_rpc_bind_remote_tensor(const char * endpoint, uint32_t device,
                                                          struct ggml_context * ctx, const char * name);

#ifdef  __cplusplus
}
#endif

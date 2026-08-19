#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "transport.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <algorithm>

static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)


namespace fs = std::filesystem;

// Macro for nicer error messages on server crash. NOTE (issue #634, smoke
// #8, 2026-08-12): this macro GGML_ABORTs the whole process, so it must ONLY
// be used for genuinely invariant-breaking conditions (malformed response,
// protocol corruption, server crashed mid-operation). A send/recv failure on
// a stale/broken PEER CONNECTION is a recoverable runtime condition — handle
// it by logging + returning the caller's error/fallback path, never here.
#define RPC_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// All RPC structures must be packed
// #470: mirrors MATRIX_ROW_PADDING in ggml-cuda/ggml-cuda.cu — the CUDA
// backend allocates every quantized row on a 512-element boundary (MMQ
// requirement), so a plain ggml_nbytes() fallback under-estimates the peer's
// allocation and the peer's wide batched prefill reads out of bounds.
static constexpr size_t GGML_RPC_QUANT_ROW_PADDING = 512;

// #634/#470: SAFE over-estimate of what the peer will allocate for `tensor`,
// used when RPC_CMD_GET_ALLOC_SIZE fails (peer died in the window between the
// liveness probe and the send). Must NEVER return less than the peer's
// ggml_backend_buft_get_alloc_size:
//  - quantized types: peer pads rows to a 512-element boundary (MMQ), so
//    ggml_nbytes alone under-allocates peer weight buffers → OOB read on
//    wide batched prefill.
//  - GGML_OP_FLASH_ATTN_EXT: peer reserves f16 K/V conversion scratch after
//    the dst (ggml_cuda_flash_attn_ext_get_alloc_size); bound it by the
//    source tensors' own bytes.
size_t ggml_backend_rpc_get_alloc_size_fallback(const ggml_tensor * tensor) {
    size_t estimate = ggml_nbytes(tensor);
    if (ggml_is_quantized(tensor->type)) {
        const int64_t ne0 = tensor->ne[0];
        if (ne0 % GGML_RPC_QUANT_ROW_PADDING != 0) {
            estimate += ggml_row_size(tensor->type, GGML_RPC_QUANT_ROW_PADDING - ne0 % GGML_RPC_QUANT_ROW_PADDING);
        }
    }
    if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
        for (int i = 1; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] != nullptr) {
                estimate += ggml_nbytes(tensor->src[i]);
            }
        }
    }
    return estimate;
}

#pragma pack(push, 1)
// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];

    char padding[4];
};

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    // Hydra: zero-copy COMBINED expert tensors (llama.cpp#20). Resolves a
    // tensor the server already has resident (loaded from its own local
    // model, not via RPC_CMD_ALLOC_BUFFER) by name, so a peer can bind
    // directly to it instead of allocating + copying bytes over the wire.
    RPC_CMD_RESOLVE_TENSOR,
    RPC_CMD_COUNT,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
const size_t HASH_THRESHOLD = 10 * 1024 * 1024;

struct rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_device_count_rsp {
    uint32_t device_count;
};

struct rpc_msg_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_req {
    uint32_t device;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_req {
    uint32_t device;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

struct rpc_msg_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

struct rpc_msg_set_tensor_hash_rsp {
    uint8_t result;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_get_device_memory_req {
    uint32_t device;
};

struct rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rpc_msg_graph_recompute_req {
    uint32_t device;
};

// Hydra: zero-copy COMBINED expert tensors (llama.cpp#20).
struct rpc_msg_resolve_tensor_req {
    char name[GGML_MAX_NAME];
};

struct rpc_msg_resolve_tensor_rsp {
    uint8_t  found;
    uint32_t registry_epoch; // #368: monotonic version of the local-tensor registry.
                             // Bumped on every clear (or on the first register after a clear
                             // is a no-op). The head records this on every bound tensor and
                             // rejects a later use of the binding when the peer has since
                             // swapped (no UAF on freed resident memory).
    uint32_t type;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint64_t buffer;       // server-side ggml_backend_buffer_t handle (opaque, same convention as ALLOC_BUFFER's remote_ptr)
    uint64_t buffer_size;  // size of the *owning* buffer (for get_base/bounds-check parity with allocated buffers)
    uint64_t data;         // absolute tensor->data pointer in the server's address space
};

#pragma pack(pop)

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
    // #470: atomic — written under get_rpc_mutex() by
    // ggml_backend_rpc_invalidate_recompute (any thread freeing a buffer) and
    // by the compute path, read lock-free by ggml_backend_rpc_graph_compute.
    std::atomic<uint64_t> last_graph_uid;
    // #470: the socket object this device context last computed a graph over.
    // A different object means a reconnect — the peer's rpc_server is a NEW
    // instance with an EMPTY stored-graph cache, so the recompute fast-path
    // must be invalidated (next compute = full GRAPH_COMPUTE). Compared by
    // identity at compute time (lock-free; get_socket never touches the
    // registry mutex, so no lock-order inversion with ggml_backend_rpc_add_server).
    std::weak_ptr<socket_t> last_sock;
    // #470 Option B: set when a peer reconnection is detected (last_sock changed).
    // The engine checks this after graph_compute and triggers a T3 rebuild to
    // re-provision model layers on the fresh peer. Cleared by check function.
    std::atomic<bool> peer_reconnected{false};
};

// Forward declaration — defined after ggml_backend_rpc_reg_context (needs the
// device list). Resets last_graph_uid on every device context bound to
// `endpoint` when a buffer is freed (#470): the peer frees the memory its
// stored graph points into, so the next compute with a recycled uid must be a
// full GRAPH_COMPUTE (re-serialized with current data pointers).
static void ggml_backend_rpc_invalidate_recompute(const std::string & endpoint);

// #470 Option B: helper to set peer_reconnected flag on the device that owns
// a buffer, given the buffer's buffer_type. Used by fail-soft paths in buffer
// functions when the peer restarts (RPC fails, stale pointers, etc.).
static void rpc_set_reconnect_flag(ggml_backend_buffer_t buffer, const char * func) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buffer->buft);
    if (dev) {
        ggml_backend_rpc_device_context * dev_ctx =
            (ggml_backend_rpc_device_context *)dev->context;
        if (!dev_ctx->peer_reconnected.exchange(true, std::memory_order_acq_rel)) {
            GGML_LOG_WARN("[%s] peer reconnection detected — "
                          "engine will trigger T3 rebuild\n", func);
        }
    }
}

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

struct ggml_backend_rpc_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
};

struct ggml_backend_rpc_buffer_context {
    // #470 (COMBINED crash): `sock` is the socket captured ONCE at alloc
    // time. The data-path ops below re-resolve it on EVERY call via
    // get_socket(ctx->endpoint) instead of sending over this stored one —
    // during a COMBINED teardown/re-attach the head-side connection drops
    // while the peer stays up, leaving pre-existing buffers holding a dead
    // fd whose first boundary send used to GGML_ABORT the engine. The
    // re-resolve reuses get_socket's liveness probe + reconnect (smoke #8).
    // `sock` is kept only for the same-server identity check in cpy_tensor.
    std::shared_ptr<socket_t> sock;
    std::string endpoint;
    void * base_ptr;
    uint64_t remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static bool send_msg(socket_ptr sock, const void * msg, size_t msg_size) {
    if (!sock->send_data(&msg_size, sizeof(msg_size))) {
        return false;
    }
    return sock->send_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, void * msg, size_t msg_size) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return sock->recv_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    } catch (const std::bad_alloc & e) {
        GGML_LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    return sock->recv_data(input.data(), size);
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    try {
        port = std::stoi(endpoint.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    uint8_t cmd_byte = cmd;
    if (!sock->send_data(&cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    if (!sock->send_data(&input_size, sizeof(input_size))) {
        return false;
    }
    if (!sock->send_data(input, input_size)) {
        return false;
    }
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    if (!send_rpc_cmd(sock, cmd, input, input_size)) {
        return false;
    }
    uint64_t out_size;
    if (!sock->recv_data(&out_size, sizeof(out_size))) {
        return false;
    }
    if (out_size != output_size) {
        return false;
    }
    if (!sock->recv_data(output, output_size)) {
        return false;
    }
    return true;
}

// RPC client-side implementation

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
static bool negotiate_hello(const std::shared_ptr<socket_t> & sock) {
    rpc_msg_hello_req request = {};
    rpc_msg_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, sizeof(request), &response, sizeof(response));
    if (!status) {
        GGML_LOG_ERROR("RPC handshake (HELLO) failed — RPC server not ready or incompatible\n");
        return false;
    }

    // #368: rpc_msg_resolve_tensor_rsp grew a field (registry_epoch); an old
    // server (minor < 1) would size-mismatch on RESOLVE_TENSOR and silently
    // return indeterminate bytes. Reject old AND new servers at hello so the
    // mismatch is diagnosed here, not buried in a silent fail-open solo.
    if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor != RPC_PROTO_MINOR_VERSION) {
        GGML_LOG_ERROR("RPC server version mismatch: %d.%d.%d (expected %d.%d.x)\n",
                       response.major, response.minor, response.patch,
                       RPC_PROTO_MAJOR_VERSION, RPC_PROTO_MINOR_VERSION);
        return false;
    }

    sock->update_caps(response.conn_caps);
    return true;
}

static std::shared_ptr<socket_t> get_socket(const std::string & endpoint) {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::weak_ptr<socket_t>> sockets;

    // smoke #8 (2026-08-12): the rtx engine (sm_120) crashed mid P/D-prefill
    // with 'send failed (bytes_sent=0, size_to_send=1)' at ggml-rpc.cpp:657 ->
    // RPC_STATUS_ASSERT -> GGML_ABORT. The mini P/D-prefill model spreads
    // tensors onto the 3060 peer (RPC0, localhost:9504); ggml-rpc caches ONE
    // socket per endpoint in this map and used to hand it out with NO liveness
    // check. After 9 hydra teardown/reattach cycles the 10th swap left a dead
    // cached fd ('hydra rpc: send failed on fd=39 ... Bad file descriptor'),
    // so the first 1-byte send on the stale socket failed and the assert
    // aborted the entire engine. Fix: probe the cached socket before reuse
    // (same poll mechanism as tcp_peer_closed); if dead, evict it from the
    // cache, reconnect to the endpoint and re-negotiate the HELLO handshake
    // below. A dead peer is a recoverable condition — never GGML_ABORT here.
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = sockets.find(endpoint);
        if (it != sockets.end()) {
            if (auto sock = it->second.lock()) {
                if (sock->is_peer_alive()) {
                    return sock;
                }
                LOG_DBG("[%s] cached socket for %s is dead, reconnecting\n", __func__, endpoint.c_str());
                it = sockets.erase(it);
            }
        }
    }
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        GGML_LOG_ERROR("Failed to parse endpoint: %s\n", endpoint.c_str());
        return nullptr;
    }

    if (!rpc_transport_init()) {
        return nullptr;
    }
    // #470 (PR#98 deadlock): the connect + HELLO handshake below are blocking
    // network operations with NO socket timeout (neither SO_RCVTIMEO nor a
    // bounded connect), and this function runs on the decode hot path
    // (ggml_backend_rpc_graph_compute) and under ggml_backend_rpc_add_server.
    // They must NOT run while holding the cache mutex: a peer that stalls its
    // HELLO response would hold the mutex forever and convoy every RPC user
    // in the process into a single-mutex futex deadlock (the observed 42-
    // thread wedge). The cache mutex is now scoped to the probe/evict/insert
    // critical sections only; a stale duplicate connection is harmless (the
    // peer sees a HELLO'd connection that never sends and closes it on idle).
    auto sock = socket_t::connect(host.c_str(), port);
    if (sock == nullptr) {
        return nullptr;
    }
    if (!negotiate_hello(sock)) {
        return nullptr;
    }
    // Double-checked insert: another thread may have connected + cached a
    // socket while we were blocked on the network. Prefer the cached one and
    // let ours drop (its fd closes when the last refcount expires).
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = sockets.find(endpoint);
        if (it != sockets.end()) {
            if (auto existing = it->second.lock()) {
                LOG_DBG("[%s] socket for %s appeared while connecting, using cached\n", __func__, endpoint.c_str());
                return existing;
            }
        }
        LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
        sockets[endpoint] = sock;
    }
    return sock;
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // #470: re-resolve — the free must ride the CURRENT connection, not the
    // socket captured at alloc time (dead after a head teardown/re-attach).
    auto sock = get_socket(ctx->endpoint);
    if (sock == nullptr) {
        // #470 Option B: fail-soft
        rpc_set_reconnect_flag(buffer, __func__);
        return;
    }
    rpc_msg_free_buffer_req request = {ctx->remote_ptr};
    bool status = send_rpc_cmd(sock, RPC_CMD_FREE_BUFFER, &request, sizeof(request), nullptr, 0);
    if (!status) {
        rpc_set_reconnect_flag(buffer, __func__);
    }
    // #470: freeing the buffer frees the peer memory the server's stored graph
    // points into. Reset the recompute fast-path so the next compute with a
    // recycled graph uid is a full GRAPH_COMPUTE (fresh data pointers), never
    // a GRAPH_RECOMPUTE executed against freed buffers.
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buffer->buft->context;
    ggml_backend_rpc_invalidate_recompute(buft_ctx->endpoint);
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    // #470: re-resolve (see free_buffer) — a cached-dead sock would make the
    // GET_BASE round trip abort; reconnecting here is safe and cheap on the
    // happy path (mutex + map lookup, no network).
    auto sock = get_socket(ctx->endpoint);
    if (sock == nullptr) {
        // #470 Option B: fail-soft
        rpc_set_reconnect_flag(buffer, __func__);
        return nullptr;
    }
    rpc_msg_buffer_get_base_req request = {ctx->remote_ptr};
    rpc_msg_buffer_get_base_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_BUFFER_GET_BASE, &request, sizeof(request), &response, sizeof(response));
    if (!status) {
        rpc_set_reconnect_flag(buffer, __func__);
        return nullptr;
    }
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor) {
    rpc_tensor result;
    if (!tensor) {
        memset(&result, 0, sizeof(result));
        return result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        result.buffer = ctx != nullptr ? ctx->remote_ptr : 0;
        result.data = reinterpret_cast<uint64_t>(tensor->data);
    } else {
        result.buffer = 0;
        result.data   = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, sizeof(result.name));
    memset(result.padding, 0, sizeof(result.padding));

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

static enum ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // #470: re-resolve — the stored sock may be dead after a head
    // teardown/re-attach while this buffer predates it.
    auto sock = get_socket(ctx->endpoint);

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        rpc_msg_init_tensor_req request;

        request.tensor = serialize_tensor(tensor);

        if (sock == nullptr) {
            // #470 Option B: fail-soft
            rpc_set_reconnect_flag(buffer, __func__);
            return GGML_STATUS_SUCCESS;
        }
        bool status = send_rpc_cmd(sock, RPC_CMD_INIT_TENSOR, &request, sizeof(request), nullptr, 0);
        if (!status) {
            rpc_set_reconnect_flag(buffer, __func__);
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // #470: re-resolve — NEVER silently skip a set on a dead peer: an
    // unwritten weight/KV while the engine reports healthy is silent data
    // corruption. Only a genuinely unreachable peer aborts here; get_socket's
    // reconnect fixes the dead fd in the COMBINED teardown/re-attach case.
    auto sock = get_socket(ctx->endpoint);
    if (sock == nullptr) {
        // #470 Option B: fail-soft
        rpc_set_reconnect_flag(buffer, __func__);
        return;
    }
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    if (size > HASH_THRESHOLD) {
        rpc_msg_set_tensor_hash_req request;
        request.tensor = rpc_tensor;
        request.offset = offset;
        request.hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        bool status = send_rpc_cmd(sock, RPC_CMD_SET_TENSOR_HASH, &request, sizeof(request), &response, sizeof(response));
        if (!status) {
            rpc_set_reconnect_flag(buffer, __func__);
            return;
        }
        if (response.result) {
            // the server has the same data, no need to send it
            return;
        }
    }
    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes)
    size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);
    bool status = send_rpc_cmd(sock, RPC_CMD_SET_TENSOR, input.data(), input.size());
    if (!status) {
        rpc_set_reconnect_flag(buffer, __func__);
    }
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // #470: re-resolve (see set_tensor) — a dead cached sock must not abort
    // the read; the reconnect rides the current connection instead.
    auto sock = get_socket(ctx->endpoint);
    if (sock == nullptr) {
        // #470 Option B: fail-soft
        rpc_set_reconnect_flag(buffer, __func__);
        return;
    }
    rpc_msg_get_tensor_req request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size = size;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_TENSOR, &request, sizeof(request), data, size);
    if (!status) {
        // #470 Option B: fail-soft instead of RPC_STATUS_ASSERT crash
        rpc_set_reconnect_flag(buffer, __func__);
    }
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        // #470: re-resolve — the stored sock may be dead after a head
        // teardown/re-attach; ride the CURRENT connection instead.
        auto sock = get_socket(ctx->endpoint);
        if (sock == nullptr) {
            // Fail-open: ggml_backend_tensor_copy treats a false return as a
            // normal miss and falls back to a host round-trip — never abort
            // a copy on a merely-unreachable peer.
            GGML_LOG_ERROR("[%s] cannot reach peer %s (reconnect failed) for copy — fail-open, returning false\n",
                    __func__, ctx->endpoint.c_str());
            return false;
        }
        if (src_ctx->sock != dst_ctx->sock) {
            return false;
        }
        rpc_msg_copy_tensor_req request;
        request.src = serialize_tensor(src);
        request.dst = serialize_tensor(dst);
        rpc_msg_copy_tensor_rsp response;
        bool status = send_rpc_cmd(sock, RPC_CMD_COPY_TENSOR, &request, sizeof(request), &response, sizeof(response));
        if (!status) {
            // Fail-open (mirrors the bind path at ggml_backend_rpc_bind_remote_tensor):
            // a copy failure must degrade to the host round-trip, not GGML_ABORT.
            GGML_LOG_ERROR("[%s] RPC_CMD_COPY_TENSOR RPC failed for '%s' -> '%s' on %s — fail-open, returning false\n",
                    __func__, src->name, dst->name, ctx->endpoint.c_str());
            return false;
        }
        return response.result;
    }
    return false;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // #470: re-resolve (see set_tensor) — a clear on the CURRENT connection,
    // never on a dead cached sock.
    auto sock = get_socket(ctx->endpoint);
    if (sock == nullptr) {
        // #470 Option B: fail-soft
        rpc_set_reconnect_flag(buffer, __func__);
        return;
    }
    rpc_msg_buffer_clear_req request = {ctx->remote_ptr, value};
    bool status = send_rpc_cmd(sock, RPC_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr, 0);
    if (!status) {
        rpc_set_reconnect_flag(buffer, __func__);
    }
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    rpc_msg_alloc_buffer_req request = {buft_ctx->device, size};
    rpc_msg_alloc_buffer_rsp response;
    auto sock = get_socket(buft_ctx->endpoint);
    bool status = send_rpc_cmd(sock, RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    if (response.remote_ptr != 0) {
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{sock, buft_ctx->endpoint, nullptr, response.remote_ptr},
            response.remote_size);
        return buffer;
    } else {
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_alignment_req request = {device};
    rpc_msg_get_alignment_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALIGNMENT, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_max_size_req request = {device};
    rpc_msg_get_max_size_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_MAX_SIZE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // ops that require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= tensor->op == GGML_OP_FLASH_ATTN_EXT;
    rpc_get |= tensor->op == GGML_OP_MUL_MAT_ID;

    if (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
        auto sock = get_socket(buft_ctx->endpoint);

        rpc_msg_get_alloc_size_req request = {
            /*.device =*/ buft_ctx->device,
            /*.tensor =*/ serialize_tensor(tensor),
            /*.srcs   =*/ {},
        };

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request.srcs[i] = serialize_tensor(tensor->src[i]);
        }

        // TODO: cache the alloc responses to avoid extra RPC calls?
        rpc_msg_get_alloc_size_rsp response;
        bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALLOC_SIZE, &request, sizeof(request), &response, sizeof(response));
        // smoke #8 (issue #634): a stale/broken peer connection used to hit
        // RPC_STATUS_ASSERT here and GGML_ABORT the entire engine mid-decode
        // ('send failed (bytes_sent=0, size_to_send=1)' -> abort). get_socket()
        // now reconnects dead cached sockets, but the peer can still die in
        // the window between the liveness probe and this send. A peer failure
        // is recoverable: log and degrade to the local byte estimate — the
        // same fallback this function already returns when rpc_get is false —
        // and let the graph allocator's existing path handle a mismatch.
        if (!status) {
            // #634/#470: never fall back to the naive local size — the peer's
            // allocator pads quantized rows to 512 elements and reserves
            // flash-attn scratch, so ggml_nbytes under-estimates and the peer
            // weight buffer comes out too small → OOB read on wide prefill.
            const size_t estimate = ggml_backend_rpc_get_alloc_size_fallback(tensor);
            GGML_LOG_ERROR("[%s] RPC_CMD_GET_ALLOC_SIZE failed for %s on %s, using safe over-estimate (%zu bytes)\n",
                           __func__, tensor->name, buft_ctx->endpoint.c_str(), estimate);
            return estimate;
        }

        return response.alloc_size;
    }

    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    // this is no-op because we don't have any async operations
}

static void add_tensor(ggml_tensor * tensor, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], tensors, visited);
    }
    add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(serialize_tensor(tensor));
}

static void serialize_graph(uint32_t device, const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    int output_size = 2*sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(output_size, 0);
    uint8_t * dest = output.data();
    memcpy(dest, &device, sizeof(device));
    dest += sizeof(device);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes * sizeof(uint64_t);
    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
}

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ggml_backend_dev_t rpc_dev = ggml_backend_get_device(backend);
    ggml_backend_rpc_device_context * rpc_dev_ctx = (ggml_backend_rpc_device_context *)rpc_dev->context;

    GGML_ASSERT(cgraph->n_nodes > 0);
    auto sock = get_socket(rpc_ctx->endpoint);
    // #470: a reconnect (new socket object) means the peer's rpc_server is a
    // fresh instance with an empty stored-graph cache — a GRAPH_RECOMPUTE
    // would be refused and the connection would churn. Detect it here (lock
    // free, by socket identity) and fall back to a full GRAPH_COMPUTE.
    if (rpc_dev_ctx->last_sock.lock() != sock) {
        rpc_dev_ctx->last_graph_uid = 0;
        rpc_dev_ctx->last_sock      = sock;
        // #470 Option B: signal that the peer reconnected — the engine
        // must re-provision model layers (T3 rebuild) before compute can
        // succeed on this peer again.
        rpc_dev_ctx->peer_reconnected.store(true, std::memory_order_release);
        GGML_LOG_WARN("[%s] peer %s reconnected — stale buffers, "
                      "engine should trigger re-provision\n",
                      __func__, rpc_ctx->endpoint.c_str());
        // Return FAILED so the engine knows the peer needs re-provision.
        // The caller (engine) will detect this and trigger T3 rebuild.
        return GGML_STATUS_FAILED;
    }
    bool reuse = cgraph->uid != 0 && rpc_dev_ctx->last_graph_uid == cgraph->uid;
    if (reuse) {
        rpc_msg_graph_recompute_req request;
        request.device = rpc_ctx->device;
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request));
        RPC_STATUS_ASSERT(status);
    } else {
        rpc_dev_ctx->last_graph_uid = cgraph->uid;
        std::vector<uint8_t> input;
        serialize_graph(rpc_ctx->device, cgraph, input);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size());
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return nullptr;
    }
    size_t alignment = get_alignment(sock, device);
    size_t max_size = get_max_size(sock, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .endpoint       = */ endpoint,
        /* .device         = */ device,
        /* .name           = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    return backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

static void get_device_memory(const std::shared_ptr<socket_t> & sock, uint32_t device, size_t * free, size_t * total) {
    rpc_msg_get_device_memory_req request;
    request.device = device;
    rpc_msg_get_device_memory_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_DEVICE_MEMORY, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    *free = response.free_mem;
    *total = response.total_mem;
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        *free = 0;
        *total = 0;
        return;
    }
    get_device_memory(sock, device, free, total);
}

// Hydra: zero-copy COMBINED expert tensors (llama.cpp#20).
//
// A worker process that already has a model resident (its own SOLO load,
// possibly a different quant than any future peer's) registers its own
// tensors here by name, once, at model-load time. Any RPC connection can
// then RESOLVE_TENSOR by that name and bind directly to the already-loaded
// memory — no RPC_CMD_ALLOC_BUFFER, no bytes copied. The registry is
// process-wide (not per-connection, unlike rpc_server::buffers) because
// registration happens once at startup, before any peer has connected.
//
// #368 (epoch + re-callable registration): the registry carries a monotonic
// version (g_hydra_registry_epoch). Bumped on every clear, and observed by
// a binding head so it can refuse to use a stale binding after the peer's
// model was swapped (the resident buffer has been freed/replaced; the raw
// tensor->data pointer is no longer safe to read). Foreign buffers cleared
// on every clear too — a binding made at epoch N is invalid for any later
// epoch.
namespace {
struct hydra_local_tensor_info {
    ggml_backend_buffer_t buffer; // the tensor's own (already-allocated) owning buffer
    uint64_t              data;   // absolute tensor->data pointer
    uint32_t              type;
    uint32_t               ne[GGML_MAX_DIMS];
    uint32_t               nb[GGML_MAX_DIMS];
};
std::mutex g_hydra_local_tensors_mutex;
std::unordered_map<std::string, hydra_local_tensor_info> g_hydra_local_tensors;
std::atomic<uint32_t> g_hydra_registry_epoch{0};
} // namespace

uint32_t ggml_backend_rpc_get_registry_epoch(void) {
    return g_hydra_registry_epoch.load(std::memory_order_acquire);
}

// #470 Option B: check if the peer for a given device reconnected since the
// last call. Returns true once per reconnection event (flag is cleared on read).
// The engine calls this after graph_compute fails to decide whether to trigger
// a T3 rebuild (re-provision) or treat it as a transient error.
bool ggml_backend_rpc_check_peer_reconnection(uint32_t device_idx) {
    // Get the RPC backend registry (index 0 is the first registered backend)
    ggml_backend_reg_t reg = ggml_backend_reg_get(0);
    if (!reg) {
        return false;
    }
    // Check if the device index is valid
    size_t dev_count = ggml_backend_reg_dev_count(reg);
    if (device_idx >= dev_count) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, device_idx);
    if (!dev) {
        return false;
    }
    // Check if this is an RPC device by name
    const char * name = ggml_backend_dev_name(dev);
    if (!name || strncmp(name, "RPC", 3) != 0) {
        return false;
    }
    // The device context is stored in dev->context — but we need to cast it
    // to our specific context type. Since we know this is an RPC device
    // (checked by name), we can safely cast.
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    return ctx->peer_reconnected.exchange(false, std::memory_order_acquire);
}

void ggml_backend_rpc_register_local_tensor(const char * name, struct ggml_tensor * tensor) {
    if (!tensor || !tensor->buffer || !tensor->data) {
        return;
    }
    hydra_local_tensor_info info;
    info.buffer = tensor->buffer;
    info.data   = reinterpret_cast<uint64_t>(tensor->data);
    info.type   = tensor->type;
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        info.ne[i] = tensor->ne[i];
        info.nb[i] = tensor->nb[i];
    }
    std::lock_guard<std::mutex> lock(g_hydra_local_tensors_mutex);
    g_hydra_local_tensors[name] = info;
}

// #368: clear the entire local-tensor registry and bump the epoch so any
// outstanding binding (which the head recorded at the previous epoch)
// becomes observably stale on its next use. Called before a fresh batch
// of register_local_tensor — e.g. after a model swap (SWAP_QUANT) frees
// and reloads the resident tensors, or before llama_hydra_register_local_
// tensors_for_rpc is called a second time. Idempotent.
//
// TODO(#368-foreign): also clear any per-rpc_server::foreign_buffers entries
// that point into the freed registry's buffers. The head's epoch check is
// the primary safety; foreign_buffers clearing is defense-in-depth and can
// be a follow-up. The registry is process-global; the rpc_server instances
// are not, so the cleanest path is making foreign_buffers a global too.
void ggml_backend_rpc_clear_local_tensors(void) {
    std::lock_guard<std::mutex> lock(g_hydra_local_tensors_mutex);
    g_hydra_local_tensors.clear();
    g_hydra_registry_epoch.fetch_add(1, std::memory_order_acq_rel);
}

// Client side: resolve `name` on `endpoint` and bind a local ggml_tensor
// directly to the peer's already-resident memory (no allocation, no copy).
// Returns nullptr if the peer doesn't have a tensor registered under that
// name (caller falls back, same "log + stay solo" philosophy as the rest of
// the COMBINED path).
//
// #368 (fail-open + ne-guard + epoch):
// - Fail-open: if the RPC call itself fails (network, malformed response),
//   return nullptr — do NOT GGML_ABORT. The head is allowed to keep running
//   solo if the peer is unreachable or temporarily wedged. This is the
//   "fail-open" behavior called out in issue #368 §2 and the review-finding
//   #1 from PR #19: RPC_STATUS_ASSERT was wrong here, RPC failures must not
//   crash the engine.
// - ne-guard: the head's expectation of a tensor's shape (`expected_ne`)
//   is checked against the peer's response. Only `type`/`nb` (i.e. the
//   quant layout) may differ across bindings — `ne` (the count along each
//   axis) is structural. A mismatch means the peer's resident model is not
//   the one the head was built for (a different llama-arch build, a
//   different routing rule, etc.) and the binding must be refused.
// - epoch: written through to *out_epoch (or NULL) so the caller can stash
//   the binding's generation and refuse to use it after a later
//   ggml_backend_rpc_clear_local_tensors on the peer.
struct ggml_tensor * ggml_backend_rpc_bind_remote_tensor(const char * endpoint, uint32_t device,
                                                          struct ggml_context * ctx, const char * name,
                                                          const uint32_t * expected_ne,
                                                          uint32_t * out_epoch) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        return nullptr;
    }
    rpc_msg_resolve_tensor_req request = {};
    snprintf(request.name, GGML_MAX_NAME, "%s", name);
    rpc_msg_resolve_tensor_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_RESOLVE_TENSOR, &request, sizeof(request), &response, sizeof(response));
    if (!status) {
        GGML_LOG_ERROR("[%s] RPC_CMD_RESOLVE_TENSOR RPC failed for '%s' on %s — fail-open, returning null\n",
                __func__, name, endpoint);
        return nullptr;
    }
    if (!response.found) {
        return nullptr;
    }
    if (response.type >= GGML_TYPE_COUNT) {
        GGML_LOG_ERROR("[%s] peer returned invalid tensor type %u for '%s'\n", __func__, response.type, name);
        return nullptr;
    }
    // ne-guard: only `type`/`nb` (the quant layout) may differ between
    // bindings; the shape (ne[]) is structural. Reject mismatches so a
    // head bound to a different peer's resident model fails loud, not by
    // running on the wrong weights (which would corrupt the KV cache
    // silently — harder to diagnose than a refused bind).
    if (expected_ne) {
        for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
            if (response.ne[i] != expected_ne[i]) {
                GGML_LOG_ERROR("[%s] ne-guard rejected '%s' on %s: dim %u expected %u got %u (peer model is not the one this head was built for)\n",
                        __func__, name, endpoint, i, expected_ne[i], response.ne[i]);
                return nullptr;
            }
        }
    }

    ggml_backend_buffer_type_t buft = ggml_backend_rpc_buffer_type(endpoint, device);
    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
        ggml_backend_rpc_buffer_interface,
        // #470: carry the endpoint so the data-path ops on this bound-tensor
        // buffer can re-resolve via get_socket (COMBINED expert tensors are
        // exactly the pre-re-attach buffers that used to hold the dead sock).
        new ggml_backend_rpc_buffer_context{sock, endpoint, nullptr, response.buffer},
        response.buffer_size);
    // Every current caller binds a weight tensor (COMBINED expert tensors).
    // ggml_backend_buffer_init defaults usage to ANY; the scheduler's
    // weight-affinity heuristic (ggml-backend.cpp's
    // ggml_backend_sched_backend_id_from_cur, "operations with weights are
    // preferably run on the same backend as the weights") only fires for
    // GGML_BACKEND_BUFFER_USAGE_WEIGHTS buffers — llama-model.cpp sets this
    // for normal model loads (llama_model::load_tensors). Without it, the
    // scheduler doesn't recognize this buffer as weights, assigns the
    // consuming op to the wrong backend, and ends up fetching the whole
    // tensor back over RPC_CMD_GET_TENSOR instead of computing where the
    // weight already lives — confirmed on real hardware as a multi-hundred-
    // MB stall under concurrent load (llama.cpp#21).
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_tensor * tensor = ggml_new_tensor_4d(ctx, (ggml_type) response.type,
            response.ne[0], response.ne[1], response.ne[2], response.ne[3]);
    if (!tensor) {
        return nullptr;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = response.nb[i];
    }
    tensor->buffer = buffer;
    tensor->data   = reinterpret_cast<void *>(response.data);
    ggml_set_name(tensor, name);
    if (out_epoch) {
        *out_epoch = response.registry_epoch;
    }
    return tensor;
}

// #368: fetch the peer's current registry epoch over RPC. Implementation
// note: we reuse RPC_CMD_RESOLVE_TENSOR with a sentinel name — the server
// returns `found = 0` for an unknown name but still populates the epoch
// field of the response, so this single round trip is enough. Returns 0
// on any failure (peer unreachable, RPC error) — callers must treat
// "got 0" as "couldn't verify" (not "epoch is 0") and fall back to
// rebinding or solo.
uint32_t ggml_backend_rpc_get_remote_registry_epoch(const char * endpoint) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        return 0;
    }
    rpc_msg_resolve_tensor_req request = {};
    snprintf(request.name, GGML_MAX_NAME, "%s", "__hydra_epoch_probe__");
    rpc_msg_resolve_tensor_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_RESOLVE_TENSOR, &request, sizeof(request), &response, sizeof(response));
    if (!status) {
        return 0;
    }
    return response.registry_epoch;
}

// RPC server-side implementation

class rpc_server {
public:
    rpc_server(std::vector<ggml_backend_t> all_backends, const char * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
    }
    ~rpc_server();

    void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);
    bool resolve_tensor(const rpc_msg_resolve_tensor_req & request, rpc_msg_resolve_tensor_rsp & response);

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

private:
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);
    bool is_known_buffer(ggml_backend_buffer_t buffer) const;

    std::vector<ggml_backend_t> backends;
    const char * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // Hydra: buffers resolved via RPC_CMD_RESOLVE_TENSOR (llama.cpp#20) — the
    // server doesn't own these (they belong to its own resident model), so
    // unlike `buffers` they are never freed in the destructor. Valid for
    // deserialize_tensor's ownership check (read paths / graph_compute), but
    // deliberately NOT accepted by free_buffer/buffer_clear, which must stay
    // restricted to buffers this rpc_server actually allocated.
    std::unordered_set<ggml_backend_buffer_t> foreign_buffers;
    // store the last computed graph for each backend
    std::vector<stored_graph> stored_graphs;
};

bool rpc_server::is_known_buffer(ggml_backend_buffer_t buffer) const {
    return buffers.find(buffer) != buffers.end() || foreign_buffers.find(buffer) != foreign_buffers.end();
}

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG("[%s] device: %d, buffer: %p, data: %p\n", __func__, dev_id, (void*)tensor->buffer, tensor->data);
    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    return true;
}

bool rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 "\n",
            __func__, dev_id, request.size, response.remote_ptr, response.remote_size);
        buffers.insert(buffer);
    } else {
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> failed\n", __func__, dev_id, request.size);
    }
    return true;
}

bool rpc_server::get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG("[%s] device: %d, alignment: %lu\n", __func__, dev_id, alignment);
    response.alignment = alignment;
    return true;
}

bool rpc_server::get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG("[%s] device: %d, max_size: %lu\n", __func__, dev_id, max_size);
    response.max_size = max_size;
    return true;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (!is_known_buffer(buffer)) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    // #470: the freed buffer's memory may be referenced by a stored graph's
    // deserialized tensors (absolute data pointers). Invalidate ALL stored
    // graphs so the next compute re-serializes with current pointers instead
    // of GRAPH_RECOMPUTE'ing stale ones against freed memory (batched GEMM
    // dereferences an unmapped VA → Xid 13/31 on the peer).
    for (auto & sg : stored_graphs) {
        sg.graph = nullptr;
    }
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    // Validate tensor type before using it
    if (tensor->type >= GGML_TYPE_COUNT) {
        GGML_LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        return nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    if (ggml_blck_size((enum ggml_type)tensor->type) == 0) {
        GGML_LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        return nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    if (result == nullptr) {
        GGML_LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        return nullptr;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && !is_known_buffer(result->buffer)) {
        result->buffer = nullptr;
    }

    if (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        GGML_ASSERT(tensor->data + tensor_size >= tensor->data); // check for overflow
        GGML_ASSERT(tensor->data >= buffer_start && tensor->data + tensor_size <= buffer_start + buffer_size);
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}


bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            return false;
        }
    }

    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    if (cache_dir && size > HASH_THRESHOLD) {
        uint64_t hash = fnv_hash((const uint8_t*)data, size);
        char hash_str[17];
        snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.write((const char *)data, size);
        GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    return true;
}

bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    if (!cache_dir) {
        return false;
    }
    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    if (!fs::exists(cache_file, ec)) {
        return false;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.read((char *)data.data(), size);
    return true;
}

bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
{
    std::vector<uint8_t> cached_file;
    if (!get_cached_file(request.hash, cached_file)) {
        response.result = 0;
        return true;
    }
    size_t size = cached_file.size();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu, hash: %" PRIx64 "\n",
            __func__, (void*)tensor->buffer, tensor->data, request.offset, size, request.hash);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            return false;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p\n", __func__, (void*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        if (!buffer) {
            GGML_LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        GGML_LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }

    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                GGML_LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                return false;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        GGML_LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        return false;
    }

    LOG_DBG("[%s] src->buffer: %p, dst->buffer: %p\n",
            __func__, (void*) src->buffer, (void*) dst->buffer);

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    if (it_ptr == tensor_ptrs.end()) {
        return nullptr;
    }
    const rpc_tensor * tensor = it_ptr->second;

    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    if (result->buffer == nullptr && result->data != nullptr) {
        GGML_LOG_ERROR("[%s] invalid data ptr", __func__);
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        if (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } else {
            result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            if (result->src[i] == nullptr) {
                GGML_LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                return nullptr;
            }
        }
    }

    // Handle view_src similarly
    if (tensor->view_src == 0) {
        result->view_src = nullptr;
    } else {
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        if (result->view_src == nullptr) {
            GGML_LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            return nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    return result;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < 2*sizeof(uint32_t)) {
        GGML_LOG_ERROR("[%s] #376: truncated graph header (%zu bytes)\n", __func__, input.size());
        return false;
    }
    const uint8_t * src = input.data();
    uint32_t device;
    memcpy(&device, src, sizeof(device));
    src += sizeof(device);
    if (device >= backends.size()) {
        // #376: prime suspect for COMBINE "peer closes connection" — the head
        // serialized a device index this peer never enumerated. Log both sides.
        GGML_LOG_ERROR("[%s] #376: device index %u out of range (peer has %zu backend(s)) — "
                       "COMBINE layer-split device-index mismatch\n",
                       __func__, device, backends.size());
        return false;
    }
    uint32_t n_nodes;
    memcpy(&n_nodes, src, sizeof(n_nodes));
    src += sizeof(n_nodes);
    if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        GGML_LOG_ERROR("[%s] #376: truncated graph (nodes section, n_nodes=%u, %zu bytes)\n",
                       __func__, n_nodes, input.size());
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes*sizeof(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, sizeof(n_tensors));
    src += sizeof(n_tensors);
    if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
        GGML_LOG_ERROR("[%s] #376: truncated graph (tensors section, n_tensors=%u, %zu bytes)\n",
                       __func__, n_tensors, input.size());
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, n_nodes: %u, n_tensors: %u\n", __func__, device, n_nodes, n_tensors);

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    if (stored_graphs[device].buffer.size() < buf_size) {
        stored_graphs[device].buffer.resize(buf_size);
    }
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ stored_graphs[device].buffer.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);

        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        if (graph->nodes[i] == nullptr && id != 0) {
            GGML_LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            return false;
        }
    }
    // Hydra #348: serialize against a caller that shares this backend
    // instance for local inference (ggml_backend_rpc_start_server_with_backends).
    ggml_backend_rpc_server_compute_lock(device);
    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
    ggml_backend_rpc_server_compute_unlock(device);
    // Hydra #376: was GGML_ASSERT — a peer-side abort on a failed compute killed
    // the whole peer. Return false instead so the client sees a clean RPC error
    // (fail-stop at the client) and can degrade-to-solo. Peer stays alive.
    if (status != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("[%s] #376: graph compute returned status %d on device %u\n",
                       __func__, (int)status, device);
        return false;
    }
    stored_graphs[device].graph = graph;
    return true;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    if (stored_graphs[device].graph == nullptr) {
        return false;
    }
    ggml_cgraph * graph = stored_graphs[device].graph;
    LOG_DBG("[%s] device: %u\n", __func__, device);

    // #470: re-validate that every stored tensor's buffer is still owned by
    // this connection before re-executing the cached graph. A buffer freed
    // behind the client's back (rpc_server teardown, model swap, head-side
    // FREE_BUFFER on sched re-reserve) leaves the deserialized tensors
    // pointing at unmapped VA — batched GEMM then faults (Xid 13 / Xid 31).
    // Refuse here: the connection drops, the client reconnects (resetting its
    // recompute fast-path) and re-sends a full GRAPH_COMPUTE with current
    // data pointers.
    auto buffer_valid = [this](const ggml_tensor * t) -> bool {
        if (t == nullptr) {
            return true;
        }
        if (t->buffer != nullptr && !is_known_buffer(t->buffer)) {
            return false;
        }
        if (t->view_src != nullptr && t->view_src->buffer != nullptr && !is_known_buffer(t->view_src->buffer)) {
            return false;
        }
        return true;
    };
    for (int i = 0; i < graph->n_nodes; i++) {
        if (!buffer_valid(graph->nodes[i])) {
            GGML_LOG_ERROR("[%s] device %u: stored graph node %d references a freed buffer — invalidating stored graph, forcing full GRAPH_COMPUTE\n",
                           __func__, device, i);
            stored_graphs[device].graph = nullptr;
            return false;
        }
    }
    for (int i = 0; i < graph->n_leafs; i++) {
        if (!buffer_valid(graph->leafs[i])) {
            GGML_LOG_ERROR("[%s] device %u: stored graph leaf %d references a freed buffer — invalidating stored graph, forcing full GRAPH_COMPUTE\n",
                           __func__, device, i);
            stored_graphs[device].graph = nullptr;
            return false;
        }
    }

    ggml_backend_rpc_server_compute_lock(device);
    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
    ggml_backend_rpc_server_compute_unlock(device);
    // Hydra #376: was GGML_ASSERT — return false instead of aborting the peer.
    if (status != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("[%s] #376: graph recompute returned status %d on device %u\n",
                       __func__, (int)status, device);
        return false;
    }
    return true;
}

bool rpc_server::resolve_tensor(const rpc_msg_resolve_tensor_req & request, rpc_msg_resolve_tensor_rsp & response) {
    // request.name may not be NUL-terminated if the client sent a full
    // GGML_MAX_NAME buffer; force termination before using it as a C string.
    char name[GGML_MAX_NAME];
    memcpy(name, request.name, GGML_MAX_NAME);
    name[GGML_MAX_NAME - 1] = '\0';

    hydra_local_tensor_info info;
    {
        std::lock_guard<std::mutex> lock(g_hydra_local_tensors_mutex);
        auto it = g_hydra_local_tensors.find(name);
        if (it == g_hydra_local_tensors.end()) {
            response.found = 0;
            // Always populate registry_epoch even on not-found — the sentinel
            // probe in ggml_backend_rpc_get_remote_registry_epoch relies on
            // this. Without it, indeterminate stack bytes would be returned.
            response.registry_epoch = g_hydra_registry_epoch.load(std::memory_order_acquire);
            return true;
        }
        info = it->second;
    }

    // Bless this connection to read through the tensor's owning buffer —
    // it's not in `buffers` (this rpc_server never allocated it), so
    // deserialize_tensor would otherwise reject it. Never freed here (see
    // foreign_buffers' declaration comment).
    foreign_buffers.insert(info.buffer);

    response.found       = 1;
    response.registry_epoch = g_hydra_registry_epoch.load(std::memory_order_acquire);
    response.type        = info.type;
    response.buffer       = reinterpret_cast<uint64_t>(info.buffer);
    response.buffer_size  = ggml_backend_buffer_get_size(info.buffer);
    response.data         = info.data;
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        response.ne[i] = info.ne[i];
        response.nb[i] = info.nb[i];
    }
    LOG_DBG("[%s] name: %s, buffer: %p, data: 0x%" PRIx64 "\n", __func__, name, (void*)info.buffer, info.data);
    return true;
}

bool rpc_server::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    size_t free, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &free, &total);
    response.free_mem = free;
    response.total_mem = total;
    LOG_DBG("[%s] device: %u, free_mem: %" PRIu64 ", total_mem: %" PRIu64 "\n", __func__, dev_id, response.free_mem, response.total_mem);
    return true;
}

rpc_server::~rpc_server() {
    // #470: drop stored-graph references before freeing the buffers they point
    // into (defense in depth — graph_recompute also re-validates, but a stored
    // graph must never outlive the memory its tensors reference).
    for (auto & sg : stored_graphs) {
        sg.graph = nullptr;
    }
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const char * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    if (!sock->recv_data(&cmd, 1)) {
        return;
    }
    if (cmd != RPC_CMD_HELLO) {
        GGML_LOG_ERROR("Expected HELLO command, update client\n");
        return;
    }

    // Read input_size and validate protocol version
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return;
    }

    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        GGML_LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_input_size, sizeof(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        return;
    }

    rpc_msg_hello_req req = {};
    if (!sock->recv_data(&req, sizeof(req))) {
        return;
    }

    rpc_msg_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    if (!send_msg(sock, &rsp, sizeof(rsp))) {
        return;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    while (true) {
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            GGML_LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        switch (cmd) {
            case RPC_CMD_HELLO: {
                // HELLO command is handled above
                return;
            }
            case RPC_CMD_DEVICE_COUNT: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_device_count_rsp response;
                response.device_count = backends.size();
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_alloc_buffer_rsp response;
                if (!server.alloc_buffer(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alloc_size_rsp response;
                if (!server.get_alloc_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                rpc_msg_get_alignment_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alignment_rsp response;
                if (!server.get_alignment(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                rpc_msg_get_max_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_max_size_rsp response;
                if (!server.get_max_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_buffer_get_base_rsp response;
                if (!server.buffer_get_base(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.free_buffer(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.buffer_clear(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_HASH: {
                rpc_msg_set_tensor_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_hash(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sock, &request,sizeof(request))) {
                    return;
                }
                if (!server.init_tensor(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.graph_compute(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_msg_graph_recompute_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.graph_recompute(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_msg_get_device_memory_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_device_memory_rsp response;
                if (!server.get_device_memory(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_RESOLVE_TENSOR: {
                rpc_msg_resolve_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_resolve_tensor_rsp response;
                if (!server.resolve_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            default: {
                GGML_LOG_ERROR("Unknown command: %d\n", cmd);
                return;
            }
        }
    }
}

void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    if (n_devices == 0 || devices == nullptr) {
        fprintf(stderr, "Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    std::vector<ggml_backend_t> backends;
    printf("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    printf("  endpoint       : %s\n", endpoint);
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    printf("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "Failed to create backend for device %s\n", dev->iface.get_name(dev));
            return;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }

#ifdef GGML_RPC_RDMA
    printf("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    printf("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        fprintf(stderr, "Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        printf("Accepted client connection\n");
        fflush(stdout);
        rpc_serve_client(backends, cache_dir, client_socket);
        printf("Client connection closed\n");
        fflush(stdout);
    }
    rpc_transport_shutdown();
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

// Hydra #348: per-device mutex serializing GPU compute between this embedded
// RPC server and a caller that shares the same backend instance for local
// inference (ggml_backend_rpc_start_server_with_backends). A standalone
// rpc-server process with no such caller just pays the cost of an
// uncontended lock/unlock per graph_compute.
//
// HYDRA_RPC_MAX_LOCAL_DEVICES=8 is generous headroom (this fork only ever
// runs one GPU per node); device >= this bound asserts rather than silently
// skipping the lock, since a silent no-op here would mean "no serialization
// happened" with no signal that the safety property is no longer held.
static constexpr size_t HYDRA_RPC_MAX_LOCAL_DEVICES = 8;
static std::array<std::mutex, HYDRA_RPC_MAX_LOCAL_DEVICES> g_hydra_server_compute_mutexes;

void ggml_backend_rpc_server_compute_lock(uint32_t device) {
    GGML_ASSERT(device < g_hydra_server_compute_mutexes.size());
    g_hydra_server_compute_mutexes[device].lock();
}

void ggml_backend_rpc_server_compute_unlock(uint32_t device) {
    GGML_ASSERT(device < g_hydra_server_compute_mutexes.size());
    g_hydra_server_compute_mutexes[device].unlock();
}

// Hydra #348: like ggml_backend_rpc_start_server above, but serves the given
// pre-built backend instances instead of creating independent ones via
// ggml_backend_dev_init for the same devices. This is what lets a process
// expose its own already-loaded llama_context's backend(s) over the embedded
// RPC server - one backend instance per physical device for the whole
// process, shared between local inference and inbound RPC compute, rather
// than two independent CUDA contexts contending for the same device. The
// caller retains ownership of `backends_in` - this function never frees them
// (unlike ggml_backend_rpc_start_server's cleanup loop above, which created
// and therefore owns its own).
void ggml_backend_rpc_start_server_with_backends(const char * endpoint, const char * cache_dir,
                                                 size_t n_threads, size_t n_backends, ggml_backend_t * backends_in) {
    if (n_backends == 0 || backends_in == nullptr) {
        fprintf(stderr, "Invalid arguments to ggml_backend_rpc_start_server_with_backends\n");
        return;
    }
    std::vector<ggml_backend_t> backends(backends_in, backends_in + n_backends);
    printf("Starting RPC server v%d.%d.%d (shared local backend instances)\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    printf("  endpoint       : %s\n", endpoint);
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    printf("Devices (shared with local inference):\n");
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backends[i]);
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               total / 1024 / 1024, free / 1024 / 1024);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backends[i], n_threads);
            }
        }
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }

#ifdef GGML_RPC_RDMA
    printf("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    printf("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        fprintf(stderr, "Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        printf("Accepted client connection\n");
        fflush(stdout);
        rpc_serve_client(backends, cache_dir, client_socket);
        printf("Client connection closed\n");
        fflush(stdout);
    }
    rpc_transport_shutdown();
    // Hydra #348: deliberately do not free `backends` here - they are owned
    // by the caller (e.g. a llama_context's sched), not by this function.
}

// Unified RPC server: handle a pre-accepted client fd as a ggml-RPC connection.
// The caller must have accepted the connection and determined it is a ggml-RPC
// client (e.g. via protocol detection with MSG_PEEK).
void ggml_backend_rpc_handle_client(int fd, const char * cache_dir,
                                     size_t n_backends, ggml_backend_t * backends) {
    std::vector<ggml_backend_t> b(backends, backends + n_backends);
    rpc_serve_client(b, cache_dir, socket_t::from_fd(fd));
}

static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    // #376: Flash-attn ops crash the peer when no CUDA kernel is available for
    // the tensor config on the peer's GPU (e.g. sm_86 vs sm_120 differences).
    // Short-term: reject FLASH_ATTN_EXT so the scheduler puts it on a local
    // backend. TODO: forward the query to the remote backend and cache results.
    if (op->op == GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    return true;
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    return buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

static const struct ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

struct ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->name.c_str() : "RPC";
}

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->devices.size() : 0;
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    if (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } else {
        GGML_ASSERT(index < ctx->devices.size());
        return ctx->devices[index];
    }
}

// Forward declarations for functions defined after get_proc_address
void ggml_backend_rpc_handle_client(int fd, const char * cache_dir,
                                     size_t n_backends, ggml_backend_t * backends);
bool ggml_backend_rpc_remove_server(const char * endpoint);

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        return (void *)ggml_backend_rpc_add_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        return (void *)ggml_backend_rpc_start_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server_with_backends") == 0) {
        return (void *)ggml_backend_rpc_start_server_with_backends;
    }
    if (std::strcmp(name, "ggml_backend_rpc_server_compute_lock") == 0) {
        return (void *)ggml_backend_rpc_server_compute_lock;
    }
    if (std::strcmp(name, "ggml_backend_rpc_server_compute_unlock") == 0) {
        return (void *)ggml_backend_rpc_server_compute_unlock;
    }
    if (std::strcmp(name, "ggml_backend_rpc_get_device_memory") == 0) {
        return (void *)ggml_backend_rpc_get_device_memory;
    }
    if (std::strcmp(name, "ggml_backend_rpc_register_local_tensor") == 0) {
        return (void *)ggml_backend_rpc_register_local_tensor;
    }
    if (std::strcmp(name, "ggml_backend_rpc_bind_remote_tensor") == 0) {
        return (void *)ggml_backend_rpc_bind_remote_tensor;
    }
    if (std::strcmp(name, "ggml_backend_rpc_clear_local_tensors") == 0) {
        return (void *)ggml_backend_rpc_clear_local_tensors;
    }
    if (std::strcmp(name, "ggml_backend_rpc_get_registry_epoch") == 0) {
        return (void *)ggml_backend_rpc_get_registry_epoch;
    }
    if (std::strcmp(name, "ggml_backend_rpc_check_peer_reconnection") == 0) {
        return (void *)ggml_backend_rpc_check_peer_reconnection;
    }
    if (std::strcmp(name, "ggml_backend_rpc_get_remote_registry_epoch") == 0) {
        return (void *)ggml_backend_rpc_get_remote_registry_epoch;
    }
    if (std::strcmp(name, "ggml_backend_rpc_handle_client") == 0) {
        return (void *)ggml_backend_rpc_handle_client;
    }
    if (std::strcmp(name, "ggml_backend_rpc_remove_server") == 0) {
        return (void *)ggml_backend_rpc_remove_server;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static struct ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_rpc_reg;
}

static uint32_t ggml_backend_rpc_get_device_count(const char * endpoint) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return 0;
    }
    rpc_msg_device_count_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_DEVICE_COUNT, nullptr, 0, &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.device_count;
}

static const ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

namespace {
std::unordered_map<std::string, ggml_backend_reg_t> & get_rpc_reg_map() {
    static std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    return reg_map;
}
std::mutex & get_rpc_mutex() {
    static std::mutex mutex;
    return mutex;
}
uint32_t & get_rpc_dev_id() {
    static uint32_t dev_id = 0;
    return dev_id;
}
} // anonymous namespace

// #470: resets the GRAPH_RECOMPUTE fast-path (last_graph_uid) on every device
// bound to `endpoint`. Called when a buffer is freed — the peer frees the
// memory its stored graph points into, so the next compute with a recycled
// uid must be a full GRAPH_COMPUTE with fresh data pointers. (Reconnects are
// handled separately in ggml_backend_rpc_graph_compute via socket identity.)
//
// PR#98-deadlock note: this runs on the decode hot path (buffer free during
// compute). It takes get_rpc_mutex() — safe ONLY because that mutex is now
// never held across blocking network: ggml_backend_rpc_add_server performs its
// device-count RPC outside the mutex (double-checked insert), so every
// get_rpc_mutex() critical section is a bounded map access. If a future change
// adds network under get_rpc_mutex() again, this function becomes a convoy
// point for the whole compute path — do not do that.
static void ggml_backend_rpc_invalidate_recompute(const std::string & endpoint) {
    std::lock_guard<std::mutex> lock(get_rpc_mutex());
    for (auto & entry : get_rpc_reg_map()) {
        if (entry.first != endpoint) {
            continue;
        }
        ggml_backend_rpc_reg_context * reg_ctx = (ggml_backend_rpc_reg_context *)entry.second->context;
        for (auto dev : reg_ctx->devices) {
            ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
            dev_ctx->last_graph_uid = 0;
        }
    }
}

ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint) {
    auto & reg_map = get_rpc_reg_map();
    auto & mutex   = get_rpc_mutex();
    auto & dev_id  = get_rpc_dev_id();
    {
        // fast path: already registered
        std::lock_guard<std::mutex> lock(mutex);
        auto it = reg_map.find(endpoint);
        if (it != reg_map.end()) {
            return it->second;
        }
    }
    // #470 (PR#98 deadlock): the device-count RPC below is a blocking
    // round-trip (connect + HELLO handshake + request/response, no socket
    // timeouts). It must NOT run while holding get_rpc_mutex() — that same
    // process-global mutex is taken by ggml_backend_rpc_invalidate_recompute
    // on the decode hot path (every RPC buffer free). A peer that stalls its
    // HELLO/DEVICE_COUNT response (or is wedged by recompute-refusal churn)
    // used to hold the mutex forever, convoying every buffer free during
    // compute into a process-wide futex deadlock (the 42-thread wedge — the
    // durable build served only when PR#98's invalidate was absent). The
    // mutex now protects only the map's check-and-insert critical section.
    uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);
    if (dev_count == 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex);
    // Double-checked insert: another thread may have registered the endpoint
    // while we were blocked on the network — return the existing reg instead
    // of leaking a second registration for the same endpoint.
    auto it = reg_map.find(endpoint);
    if (it != reg_map.end()) {
        return it->second;
    }
    ggml_backend_rpc_reg_context * ctx = new ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
            /* .last_graph_uid = */ 0,
            /* .last_sock   = */    {},
        };

        ggml_backend_dev_t dev = new ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = new ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    return reg;
}

GGML_BACKEND_API bool ggml_backend_rpc_remove_server(const char * endpoint) {
    auto & reg_map = get_rpc_reg_map();
    auto & mutex   = get_rpc_mutex();
    std::lock_guard<std::mutex> lock(mutex);
    auto it = reg_map.find(endpoint);
    if (it == reg_map.end()) {
        return false;
    }
    ggml_backend_reg_t reg = it->second;
    // Unregister from global registry
    ggml_backend_unload(reg);
    // Free the context and its devices
    auto * ctx = static_cast<ggml_backend_rpc_reg_context *>(reg->context);
    for (auto dev : ctx->devices) {
        auto * dev_ctx = static_cast<ggml_backend_rpc_device_context *>(dev->context);
        delete dev_ctx;
        delete dev;
    }
    delete ctx;
    delete reg;
    reg_map.erase(it);
    return true;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)

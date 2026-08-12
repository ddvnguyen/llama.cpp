#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);

    // Non-blocking liveness probe for a client socket. Returns false when the
    // underlying fd is known-dead: peer closed/reset the connection, or the
    // fd was closed underneath us (e.g. EBADF on a stale cached socket after a
    // peer teardown). Used by get_socket() to evict stale cached sockets
    // instead of handing them out (issue #634, smoke #8).
    bool is_peer_alive() const;

    socket_ptr accept();

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);
    static socket_ptr from_fd(int fd);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();

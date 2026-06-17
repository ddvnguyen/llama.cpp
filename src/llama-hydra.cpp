#include "llama-hydra.h"
#include "llama-context.h"
#include "llama-impl.h"

#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <unistd.h>
#endif

size_t llama_state_seq_get_data_to_fd(struct llama_context * ctx, llama_seq_id seq_id, int fd) {
    return ctx->state_seq_get_data_to_fd(seq_id, fd);
}

bool llama_hydra_peer_reachable(const char * host_port) {
#if !defined(_WIN32)
    std::string hp(host_port);
    auto colon = hp.rfind(':');
    if (colon == std::string::npos) return false;
    std::string host = hp.substr(0, colon);
    int port = std::stoi(hp.substr(colon + 1));

    struct addrinfo hints = {}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0) return false;

    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); return false; }

    struct timeval tv = { .tv_sec = 3 };
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    bool ok = (connect(fd, res->ai_addr, res->ai_addrlen) == 0);
    close(fd);
    freeaddrinfo(res);
    return ok;
#else
    (void)host_port;
    return false;
#endif
}

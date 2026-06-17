#include "llama-engine.h"
#include "server-context.h"
#include "server-task.h"
#include "server-rpc.h"
#include "server-http.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <atomic>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <signal.h>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }
    shutdown_handler(signal);
}

static bool try_tcp_connect(const std::string & host, int port, int timeout_sec) {
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
        struct hostent * he = gethostbyname(host.c_str());
        if (!he) {
            return false;
        }
        memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);
    }

#if defined(_WIN32)
    SOCKET fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd == INVALID_SOCKET) {
        return false;
    }
    u_long mode = 1;
    ioctlsocket(fd, FIONBIO, &mode);
#else
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return false;
    }
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
#endif

    int ret = connect(fd, (struct sockaddr *) &addr, sizeof(addr));

#if defined(_WIN32)
    if (ret == SOCKET_ERROR && WSAGetLastError() != WSAEWOULDBLOCK) {
        closesocket(fd);
        return false;
    }
#else
    if (ret < 0 && errno != EINPROGRESS) {
        close(fd);
        return false;
    }
#endif

    fd_set wfds;
    FD_ZERO(&wfds);
    FD_SET(fd, &wfds);

    struct timeval tv;
    tv.tv_sec = timeout_sec;
    tv.tv_usec = 0;

    ret = select(fd + 1, NULL, &wfds, NULL, &tv);

#if defined(_WIN32)
    closesocket(fd);
#else
    close(fd);
#endif

    return ret > 0;
}

static std::string extract_rpc_engine_peer(int argc, char ** argv, std::vector<char *> & filtered_argv) {
    std::string peer;
    filtered_argv.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rpc-engine") == 0 && i + 1 < argc) {
            peer = argv[i + 1];
            i++;
            continue;
        }
        filtered_argv.push_back(argv[i]);
    }

    return peer;
}

int llama_engine(int argc, char ** argv);

int llama_engine(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    std::vector<char *> filtered_argv;
    std::string rpc_engine_peer = extract_rpc_engine_peer(argc, argv, filtered_argv);

    int filtered_argc = filtered_argv.size();
    char ** filtered_argv_ptr = filtered_argv.data();

    if (!common_params_parse(filtered_argc, filtered_argv_ptr, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    if (!rpc_engine_peer.empty()) {
        std::string host = rpc_engine_peer;
        int port = 8080;
        size_t colon = rpc_engine_peer.find(':');
        if (colon != std::string::npos) {
            host = rpc_engine_peer.substr(0, colon);
            port = std::stoi(rpc_engine_peer.substr(colon + 1));
        }

        if (!try_tcp_connect(host, port, 3)) {
            LOG_WRN("eng  %12.*s: rpc-engine peer %s unreachable, entering solo mode\n",
                    12, __func__, rpc_engine_peer.c_str());
        }
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    common_params_print_info(params, true);

    server_context ctx_server;

    if (!ctx_server.load_model(params)) {
        LOG_ERR("eng  %12.*s: failed to load model\n", 12, __func__);
        llama_backend_free();
        return 1;
    }

    if (params.rpc_port > 0) {
        ctx_server.start_rpc_server(params.rpc_port);
    }

    server_http_context ctx_http;
    if (params.port > 0) {
        if (!ctx_http.init(params)) {
            LOG_ERR("eng  %12.*s: failed to initialize HTTP server\n", 12, __func__);
            llama_backend_free();
            return 1;
        }

        ctx_http.get("/health", [&ctx_server](const server_http_req &) {
            auto res = std::make_unique<server_http_res>();
            res->status = 200;
            res->data = "{\"status\":\"ok\"}";
            return res;
        });

        ctx_http.get("/version", [](const server_http_req &) {
            auto res = std::make_unique<server_http_res>();
            res->status = 200;
            res->data = "{\"version\":\"E1\",\"engine\":\"llama-engine\"}";
            return res;
        });

        ctx_http.get("/slots", [&ctx_server](const server_http_req &) {
            auto res = std::make_unique<server_http_res>();
            res->status = 200;
            res->data = "[]";
            return res;
        });

        ctx_http.get("/slots/:id/state/meta", [&ctx_server](const server_http_req & req) {
            auto res = std::make_unique<server_http_res>();
            int slot_id = std::stoi(req.get_param("id"));
            res->status = 200;
            res->data = "{\"slot_id\":" + std::to_string(slot_id) + ",\"n_past\":0,\"state_size\":0}";
            return res;
        });

        if (!ctx_http.start()) {
            LOG_ERR("eng  %12.*s: failed to start HTTP server\n", 12, __func__);
            llama_backend_free();
            return 1;
        }
    }

    shutdown_handler = [&](int) {
        ctx_server.terminate();
    };

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined(_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    ctx_server.start_loop();

    ctx_server.terminate();
    llama_backend_free();

    return 0;
}

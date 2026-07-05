#include "llama-engine.h"
#include "server-common.h"
#include "server-context.h"
#include "server-task.h"
#include "server-rpc.h"
#include "server-http.h"
#include "server-queue.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "llama.h"
#include "llama-hydra.h"
#include "log.h"

#include "ggml-backend.h"
#include "ggml-rpc.h"

// ggml-backend RPC per-connection handler — exported from ggml-rpc library
extern void ggml_backend_rpc_handle_client(int fd, const char * cache_dir,
                                            size_t n_backends, ggml_backend_t * backends);

#include <atomic>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <signal.h>
#include <sstream>
#include <string>
#include <thread>
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

// Hydra #356: ex_wrapper is a verbatim copy of tools/server/server.cpp's
// static helper. The HTTP handlers registered below (post_chat_completions,
// post_completions_oai) return std::unique_ptr<server_res_generator>, but
// server_http_context::handler_t expects std::unique_ptr<server_http_res>.
// ex_wrapper bridges the two: it owns the streaming generator, drains it
// into a server_http_res, and converts any thrown exception into a JSON
// error response with the right HTTP status (400 invalid_argument, 500
// otherwise). The chat handler reads meta->chat_params; we must call
// routes.update_meta(ctx_server) before flipping ctx_http.is_ready.
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

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
    if (ret > 0) {
        // select() reports writable even on failed non-blocking connects
        // (e.g. ECONNREFUSED). Check SO_ERROR to distinguish success.
        int so_error = 0;
        socklen_t so_len = sizeof(so_error);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, (char *)&so_error, &so_len) < 0 || so_error != 0) {
            ret = 0;
        }
    }

#if defined(_WIN32)
    closesocket(fd);
#else
    close(fd);
#endif

    return ret > 0;
}

// Hydra: capability flags filtered out of argv before common_params_parse.
// Only --rpc-engine is stored; the rest are stripped to avoid "unknown arg"
// errors from the stock arg parser (see extract_hydra_capability_flags).
// In production all COMBINE peer info comes from the Hydra request body at
// runtime — CLI flags are only for testing.
struct hydra_capability_flags {
    std::string rpc_engine_peer;     // testing shortcut: reach out to this peer

    bool wants_combined_head() const { return !rpc_engine_peer.empty(); }
};

// Filters Hydra flags out of argv before common_params_parse sees them.
// Keeps only --rpc-engine; other flags are consumed but their values discarded
// (they are now request-driven, not startup-driven).
static hydra_capability_flags extract_hydra_capability_flags(int argc, char ** argv, std::vector<char *> & filtered_argv) {
    hydra_capability_flags flags;
    filtered_argv.clear();
    if (argc > 0) filtered_argv.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rpc-engine") == 0 && i + 1 < argc) {
            flags.rpc_engine_peer = argv[++i];
            continue;
        }
        // Strip remaining Hydra-only flags so they never reach common_params_parse
        if (strcmp(argv[i], "--ggml-rpc-port") == 0 && i + 1 < argc) { ++i; continue; }
        if (strcmp(argv[i], "--peer-only") == 0) { continue; }
        if (strcmp(argv[i], "--combined-ot-pattern") == 0 && i + 1 < argc) { ++i; continue; }
        if (strcmp(argv[i], "--combined-split-mode") == 0 && i + 1 < argc) { ++i; continue; }
        if (strcmp(argv[i], "--combined-tensor-split") == 0 && i + 1 < argc) { ++i; continue; }
        filtered_argv.push_back(argv[i]);
    }

    return flags;
}

// Extract model compute backends into a vector. Returns empty if no non-CPU
// backends are found.
static std::vector<ggml_backend_t> get_model_compute_backends(struct llama_context * ctx) {
    if (!ctx) return {};
    std::vector<ggml_backend_t> backends(8);
    size_t n = llama_hydra_get_compute_backends(ctx, backends.data(), backends.size());
    if (n > backends.size()) {
        backends.resize(n);
        n = llama_hydra_get_compute_backends(ctx, backends.data(), backends.size());
    }
    backends.resize(n);
    return backends;
}

// Enumerate globally registered non-CPU devices. Used in the no-model path.
static std::vector<ggml_backend_t> enumerate_non_cpu_devices() {
    std::vector<ggml_backend_t> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend) {
                devices.push_back(backend);
            }
        }
    }
    return devices;
}

int llama_engine(int argc, char ** argv);

int llama_engine(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    std::vector<char *> filtered_argv;
    hydra_capability_flags flags = extract_hydra_capability_flags(argc, argv, filtered_argv);

    int filtered_argc = filtered_argv.size();
    char ** filtered_argv_ptr = filtered_argv.data();

    if (!common_params_parse(filtered_argc, filtered_argv_ptr, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    const bool has_model = !params.model.path.empty();
    const bool has_peer  = flags.wants_combined_head();

    if (!has_model) {
        LOG_INF("eng  %12.*s: no model specified — starting in compute-only mode\n", 12, __func__);
    }

    // Determine RPC port: use --rpc-port from common_params, or derive from HTTP port.
    int rpc_port = params.rpc_port;
    if (rpc_port <= 0) {
        rpc_port = (params.port > 0) ? params.port + 1 : 9504;
    }

    // If --rpc-engine was given (testing shortcut), pre-connect the peer now.
    if (has_peer) {
        ggml_backend_load_all();
        if (ggml_backend_rpc_add_server(flags.rpc_engine_peer.c_str()) == nullptr) {
            LOG_WRN("eng  %12.*s: rpc-engine peer %s unreachable — running in solo mode\n",
                    12, __func__, flags.rpc_engine_peer.c_str());
        }
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // ── Model-loaded path ──
    if (has_model) {
        server_context ctx_server;

        common_params_print_info(params, true);

        if (!ctx_server.load_model(params)) {
            LOG_ERR("eng  %12.*s: failed to load model\n", 12, __func__);
            llama_backend_free();
            return 1;
        }

        ctx_server.set_hydra_capabilities(true, flags.rpc_engine_peer,
                false, "", "solo");

        // Register tensors for expert-split COMBINE (used at runtime when
        // a request specifies a peer).
        llama_hydra_register_local_tensors_for_rpc(ctx_server.get_llama_context());

        // Enable shared-backend compute lock for inbound RPC + local inference.
        llama_hydra_enable_shared_backend_compute_lock();

        // Start the unified RPC server on the auto-derived port, sharing the
        // model's compute backends.
        auto backends = get_model_compute_backends(ctx_server.get_llama_context());
        ctx_server.start_rpc_server(rpc_port, backends);

        // ── HTTP server with full inference routes ──
        server_routes routes(params, ctx_server);

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

            // /slots data
            ctx_http.get("/slots", [&ctx_server](const server_http_req & req) {
                auto res = std::make_unique<server_http_res>();
                auto rd = ctx_server.get_response_reader();
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = rd.get_new_id();
                rd.post_task(std::move(task), true);
                auto result = rd.next(req.should_stop);
                if (!result || result->is_error()) {
                    res->status = 200;
                    res->data = "[]";
                    return res;
                }
                auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
                if (!res_task) {
                    res->status = 200;
                    res->data = "[]";
                    return res;
                }
                res->data = res_task->slots_data.dump();
                return res;
            });

            // /metrics Prometheus endpoint
            ctx_http.get("/metrics", [&ctx_server](const server_http_req & req) {
                auto res = std::make_unique<server_http_res>();
                auto rd = ctx_server.get_response_reader();
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = rd.get_new_id();
                rd.post_task(std::move(task), true);
                auto result = rd.next(req.should_stop);
                if (!result || result->is_error()) {
                    res->status = 200;
                    res->content_type = "text/plain; version=0.0.4";
                    res->data = "";
                    return res;
                }
                auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
                if (!res_task) {
                    res->status = 200;
                    res->content_type = "text/plain; version=0.0.4";
                    res->data = "";
                    return res;
                }
                res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
                res->content_type = "text/plain; version=0.0.4";
                std::stringstream prom;
                auto emit = [&](const std::string & name, const std::string & help,
                                const std::string & type, double value) {
                    prom << "# HELP llamacpp:" << name << " " << help << "\n"
                         << "# TYPE llamacpp:" << name << " " << type << "\n"
                         << "llamacpp:"        << name << " " << value << "\n";
                };
                emit("prompt_tokens_total",       "Number of prompt tokens processed.",                               "counter", (uint64_t)res_task->n_prompt_tokens_processed_total);
                emit("prompt_seconds_total",      "Prompt process time",                                              "counter", (uint64_t)res_task->t_prompt_processing_total / 1.e3);
                emit("tokens_predicted_total",    "Number of generation tokens processed.",                            "counter", (uint64_t)res_task->n_tokens_predicted_total);
                emit("tokens_predicted_seconds_total", "Predict process time",                                        "counter", (uint64_t)res_task->t_tokens_generation_total / 1.e3);
                emit("n_decode_total",            "Total number of llama_decode() calls",                             "counter", res_task->n_decode_total);
                emit("n_tokens_max",              "Largest observed n_tokens.",                                       "counter", res_task->n_tokens_max);
                emit("prompt_tokens_seconds",     "Average prompt throughput in tokens/s.",                           "gauge",   res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.);
                emit("predicted_tokens_seconds",  "Average generation throughput in tokens/s.",                       "gauge",   res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.);
                emit("requests_processing",       "Number of requests processing.",                                   "gauge",   (uint64_t)res_task->n_processing_slots);
                emit("requests_deferred",         "Number of requests deferred.",                                     "gauge",   (uint64_t)res_task->n_tasks_deferred);
                emit("n_busy_slots_per_decode",   "Average number of busy slots per llama_decode() call",             "gauge",   (float)res_task->n_busy_slots_total / std::max((float)res_task->n_decode_total, 1.f));
                res->status = 200;
                res->data = prom.str();
                return res;
            });

            // Slot erase
            ctx_http.post("/slots/:id_slot", [&ctx_server](const server_http_req & req) {
                auto res = std::make_unique<server_http_res>();
                std::string action = req.get_param("action");
                if (action != "erase") {
                    res->status = 400;
                    res->data = "{\"error\":\"only erase action is supported\"}";
                    return res;
                }
                int id_slot;
                try {
                    id_slot = std::stoi(req.get_param("id_slot"));
                } catch (...) {
                    res->status = 400;
                    res->data = "{\"error\":\"invalid slot ID\"}";
                    return res;
                }
                auto rd = ctx_server.get_response_reader();
                server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
                task.id = rd.get_new_id();
                task.slot_action.id_slot = id_slot;
                rd.post_task(std::move(task));
                auto result = rd.next(req.should_stop);
                if (!result || result->is_error()) {
                    res->status = 500;
                    res->data = "{\"error\":\"slot erase failed\"}";
                    return res;
                }
                res->data = result->to_json().dump();
                return res;
            });

            // /v1/models stub
            ctx_http.get("/v1/models", [&ctx_server](const server_http_req &) {
                auto res = std::make_unique<server_http_res>();
                auto meta = ctx_server.get_meta();
                res->data = (json{
                    {"object", "list"},
                    {"data", {{
                        {"id", meta.model_name},
                        {"object", "model"},
                        {"created", 0},
                        {"owned_by", "llama-engine"}
                    }}}
                }).dump();
                return res;
            });
            ctx_http.get("/models", [&ctx_server](const server_http_req &) {
                auto res = std::make_unique<server_http_res>();
                auto meta = ctx_server.get_meta();
                res->data = (json{
                    {"object", "list"},
                    {"data", {{
                        {"id", meta.model_name},
                        {"object", "model"},
                        {"created", 0},
                        {"owned_by", "llama-engine"}
                    }}}
                }).dump();
                return res;
            });

            // State meta
            ctx_http.get("/slots/:id/state/meta", [&ctx_server](const server_http_req & req) {
                auto res = std::make_unique<server_http_res>();
                int slot_id;
                try {
                    slot_id = std::stoi(req.get_param("id"));
                } catch (...) {
                    res->status = 400;
                    res->data = "{\"error\":\"invalid slot ID\"}";
                    return res;
                }
                auto rd = ctx_server.get_response_reader();
                server_task task(SERVER_TASK_TYPE_HYDRA_STATE_META);
                task.id = rd.get_new_id();
                task.hydra_action.id_slot = slot_id;
                rd.post_task(std::move(task));
                auto result = rd.next(req.should_stop);
                if (!result || result->is_error()) {
                    res->status = 503;
                    res->data = "{\"error\":\"state meta unavailable\"}";
                    return res;
                }
                auto * hr = dynamic_cast<server_task_result_hydra_state*>(result.get());
                if (!hr) {
                    res->status = 503;
                    res->data = "{\"error\":\"state meta type mismatch\"}";
                    return res;
                }
                json j = {
                    {"slot_id", slot_id},
                    {"n_past", hr->n_past},
                    {"state_size", hr->state_size},
                    {"is_processing", hr->is_processing}
                };
                res->data = j.dump();
                return res;
            });

            // OpenAI-schema completions
            ctx_http.post("/v1/chat/completions", ex_wrapper(routes.post_chat_completions));
            ctx_http.post("/chat/completions",    ex_wrapper(routes.post_chat_completions));
            ctx_http.post("/v1/completions",      ex_wrapper(routes.post_completions_oai));

            if (!ctx_http.start()) {
                LOG_ERR("eng  %12.*s: failed to start HTTP server\n", 12, __func__);
                llama_backend_free();
                return 1;
            }

            routes.update_meta(ctx_server);
            ctx_http.is_ready.store(true);
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

        if (params.port > 0) {
            ctx_http.stop();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
        }

        ctx_server.terminate();
        llama_backend_free();

        return 0;
    }

    // ── No-model path: compute-only backend ──
    {
        // Start the unified RPC server, serving globally registered non-CPU
        // devices via ggml-RPC. No Hydra protocol — no server_context exists.
        auto backends = enumerate_non_cpu_devices();
        if (backends.empty()) {
            LOG_ERR("eng  %12.*s: no non-CPU backends found\n", 12, __func__);
            llama_backend_free();
            return 1;
        }

        // The unified RPC server is started as a standalone (no server_context),
        // so only ggml-RPC protocol is accepted — Hydra connections are rejected.
        // We start an inline TCP server accepting connections and forwarding them
        // to ggml_backend_rpc_handle_client.
        std::thread([rpc_port, backends]() {
            // We can't use server_context::start_rpc_server because there's no
            // server_context. Inline the accept loop here for the no-model case.
            const int srv_fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (srv_fd < 0) {
                SRV_ERR("eng  %12.*s: socket() failed: %s\n", 12, __func__);
                return;
            }
            const int opt = 1;
            ::setsockopt(srv_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
            struct sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = INADDR_ANY;
            addr.sin_port = htons((uint16_t)rpc_port);
            if (::bind(srv_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
                SRV_ERR("eng  %12.*s: bind() on port %d failed\n", 12, __func__, rpc_port);
                ::close(srv_fd);
                return;
            }
            ::listen(srv_fd, 16);
            LOG_INF("eng  %12.*s: no-model RPC server on 0.0.0.0:%d (%zu GPU backend(s))\n",
                    12, __func__, rpc_port, backends.size());

            while (true) {
                int conn_fd = ::accept(srv_fd, nullptr, nullptr);
                if (conn_fd < 0) continue;
                std::thread([conn_fd, backends]() {
                    ggml_backend_rpc_handle_client(conn_fd, nullptr,
                        backends.size(), const_cast<ggml_backend_t *>(backends.data()));
                }).detach();
            }
        }).detach();

        // ── Minimal HTTP server (health + version only) ──
        server_http_context ctx_http;
        if (params.port > 0) {
            if (!ctx_http.init(params)) {
                LOG_ERR("eng  %12.*s: failed to init HTTP server\n", 12, __func__);
                llama_backend_free();
                return 1;
            }

            ctx_http.get("/health", [](const server_http_req &) {
                auto res = std::make_unique<server_http_res>();
                res->status = 200;
                res->data = "{\"status\":\"ok\"}";
                return res;
            });

            ctx_http.get("/version", [](const server_http_req &) {
                auto res = std::make_unique<server_http_res>();
                res->status = 200;
                res->data = "{\"version\":\"E1\",\"engine\":\"llama-engine\",\"mode\":\"peer\"}";
                return res;
            });

            ctx_http.get("/slots", [](const server_http_req &) {
                auto res = std::make_unique<server_http_res>();
                res->status = 200;
                res->data = "[]";
                return res;
            });

            if (!ctx_http.start()) {
                LOG_ERR("eng  %12.*s: failed to start HTTP server\n", 12, __func__);
                llama_backend_free();
                return 1;
            }
            ctx_http.is_ready.store(true);
        }

        LOG_INF("eng  %12.*s: no-model engine ready — RPC on :%d, HTTP on :%d\n",
                12, __func__, rpc_port, params.port);

        // Block until signal.
        {
            struct sigaction sa;
            sa.sa_handler = +[](int) {};
            sigemptyset(&sa.sa_mask);
            sa.sa_flags = 0;
            sigaction(SIGINT, &sa, NULL);
            sigaction(SIGTERM, &sa, NULL);
        }
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        if (params.port > 0) {
            ctx_http.stop();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
        }
        llama_backend_free();
        return 0;
    }
}

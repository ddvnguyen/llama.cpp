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

#if defined(_WIN32)
    closesocket(fd);
#else
    close(fd);
#endif

    return ret > 0;
}

// Hydra #348: composable, independent capability flags — replaces the
// tri-state --role standalone|head|worker. Every engine always loads its
// model (SOLO). A node can independently, additionally, opt into exposing
// itself as a COMBINED-peer RPC backend and/or reaching out to a peer as a
// COMBINED head — any combination, no restart to change which. Filtered out
// of argv before common_params_parse (these aren't stock llama.cpp args).
//   --ggml-rpc-port <port>        opt-in: expose this engine's local GPU(s)
//                                 as an embedded ggml-RPC backend on <port>,
//                                 sharing the SAME backend instances this
//                                 engine already built for local inference
//                                 (not a duplicate context — see
//                                 start_shared_backend_rpc_server below).
//   --rpc-engine <host:port>      opt-in: this engine acts as a COMBINED
//                                 head reaching out to the named peer.
//   --combined-ot-pattern <regex> required alongside --rpc-engine: which
//                                 expert tensors (by name) get a dual-resident
//                                 copy on the peer.
struct hydra_capability_flags {
    int         ggml_rpc_port = 0;
    std::string rpc_engine_peer;
    std::string combined_ot_pattern;

    bool wants_rpc_backend()   const { return ggml_rpc_port > 0; }
    bool wants_combined_head() const { return !rpc_engine_peer.empty(); }
};

// Filters Hydra capability flags out of argv into filtered_argv, mirroring
// the previous extract_rpc_engine_peer()/extract_hydra_role_flags()'s
// filtering style (now merged into one pass).
static hydra_capability_flags extract_hydra_capability_flags(int argc, char ** argv, std::vector<char *> & filtered_argv) {
    hydra_capability_flags flags;
    filtered_argv.clear();
    if (argc > 0) filtered_argv.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rpc-engine") == 0 && i + 1 < argc) {
            flags.rpc_engine_peer = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--combined-ot-pattern") == 0 && i + 1 < argc) {
            flags.combined_ot_pattern = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--ggml-rpc-port") == 0 && i + 1 < argc) {
            flags.ggml_rpc_port = std::atoi(argv[++i]);
            continue;
        }
        filtered_argv.push_back(argv[i]);
    }

    return flags;
}

// Hydra #348: starts the embedded ggml-RPC backend on a background thread,
// serving the SAME backend instances `ctx`'s scheduler already built for
// local inference — not independent ones for the same devices (the previous
// design's run_combined_worker() created its own via ggml_backend_dev_init,
// which is why a --role worker process could never also serve SOLO). One
// backend instance per physical device for the whole process, shared
// between local inference and inbound RPC compute, serialized via the
// per-device lock enabled below (see llama_hydra_enable_shared_backend_compute_lock,
// llama-hydra.cpp, and ggml-rpc.cpp's ggml_backend_rpc_server_compute_lock).
// The thread is detached: ggml_backend_rpc_start_server_with_backends blocks
// in its own accept loop with no stop hook, the same "runs for process
// lifetime, no teardown path" choice already made for
// llama_hydra_load_combined_experts's static leak-on-purpose buffers.
static void start_shared_backend_rpc_server(struct llama_context * ctx, int ggml_rpc_port) {
    ggml_backend_load_all();
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        LOG_ERR("eng  %12.*s: RPC backend not available in this build — --ggml-rpc-port ignored\n", 12, __func__);
        return;
    }
    using start_server_fn_t = void (*)(const char *, const char *, size_t, size_t, ggml_backend_t *);
    auto start_server_fn = (start_server_fn_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_start_server_with_backends");
    if (!start_server_fn) {
        LOG_ERR("eng  %12.*s: failed to resolve ggml_backend_rpc_start_server_with_backends\n", 12, __func__);
        return;
    }

    std::vector<ggml_backend_t> backends(8);
    size_t n_backends = llama_hydra_get_compute_backends(ctx, backends.data(), backends.size());
    if (n_backends > backends.size()) {
        backends.resize(n_backends);
        n_backends = llama_hydra_get_compute_backends(ctx, backends.data(), backends.size());
    }
    backends.resize(n_backends);
    if (backends.empty()) {
        LOG_ERR("eng  %12.*s: no non-CPU backends found to expose for the embedded RPC server\n", 12, __func__);
        return;
    }

    // Local decode dispatch (llama-context.cpp's graph_compute) must now
    // serialize against this RPC server's compute on the same backend(s).
    llama_hydra_enable_shared_backend_compute_lock();

    const std::string endpoint = "0.0.0.0:" + std::to_string(ggml_rpc_port);
    const size_t n_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    LOG_INF("eng  %12.*s: exposing %zu shared backend(s) as a ggml-RPC server on %s\n",
            12, __func__, backends.size(), endpoint.c_str());

    std::thread([start_server_fn, endpoint, n_threads, backends]() mutable {
        start_server_fn(endpoint.c_str(), nullptr, n_threads, backends.size(), backends.data());
    }).detach();
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

    const bool wants_combined_head = flags.wants_combined_head() && !flags.combined_ot_pattern.empty();
    if (flags.wants_combined_head() && flags.combined_ot_pattern.empty()) {
        LOG_WRN("eng  %12.*s: --rpc-engine given without --combined-ot-pattern — "
                "COMBINED head capability disabled, running solo-only\n", 12, __func__);
    }

    bool peer_reachable = false;
    if (wants_combined_head) {
        std::string host = flags.rpc_engine_peer;
        int port = 8080;
        size_t colon = flags.rpc_engine_peer.find(':');
        if (colon != std::string::npos) {
            host = flags.rpc_engine_peer.substr(0, colon);
            port = std::stoi(flags.rpc_engine_peer.substr(colon + 1));
        }

        peer_reachable = try_tcp_connect(host, port, 3);
        if (!peer_reachable) {
            LOG_WRN("eng  %12.*s: rpc-engine peer %s unreachable, entering solo mode\n",
                    12, __func__, flags.rpc_engine_peer.c_str());
        }
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    common_params_print_info(params, true);

    server_context ctx_server;

    // Hydra #348: every engine always loads its model now — there is no more
    // --role worker path that skips this. SOLO duty is unconditional; the
    // capabilities below layer on top of it instead of replacing it.
    if (!ctx_server.load_model(params)) {
        LOG_ERR("eng  %12.*s: failed to load model\n", 12, __func__);
        llama_backend_free();
        return 1;
    }

    ctx_server.set_hydra_capabilities(flags.wants_rpc_backend(), flags.rpc_engine_peer,
            peer_reachable, flags.combined_ot_pattern);

    // Hydra #348: expose this engine's own backend(s) as an embedded
    // ggml-RPC server, shared with local inference rather than a duplicate
    // context — see start_shared_backend_rpc_server.
    if (flags.wants_rpc_backend()) {
        start_shared_backend_rpc_server(ctx_server.get_llama_context(), flags.ggml_rpc_port);
        // Hydra llama.cpp#20: make this engine's own resident tensors
        // resolvable by name, so a COMBINED peer can zero-copy bind to them
        // instead of dual-loading a copy (llama_hydra_load_combined_experts).
        llama_hydra_register_local_tensors_for_rpc(ctx_server.get_llama_context());
    }

    // Hydra #287/#260/#348: a COMBINED head dual-loads its configured expert
    // tensors onto the --rpc-engine peer once, here, before serving any
    // requests. An unreachable peer or insufficient peer VRAM (see
    // llama_hydra_load_combined_experts's headroom check) degrades to
    // solo-only (never aborts startup) — SET_EXPERT_MODE("combined") will
    // report "solo" until fixed.
    if (wants_combined_head) {
        int32_t n = llama_hydra_load_combined_experts(ctx_server.get_llama_context(),
                flags.rpc_engine_peer.c_str(), flags.combined_ot_pattern.c_str());
        ctx_server.set_hydra_combined_head_attached(n > 0);
        if (n > 0) {
            LOG_INF("eng  %12.*s: COMBINED ready — %d layer(s) dual-loaded onto %s\n",
                    12, __func__, n, flags.rpc_engine_peer.c_str());
        } else {
            LOG_WRN("eng  %12.*s: COMBINED dual-load failed — running solo-only\n", 12, __func__);
        }
    }

    if (params.rpc_port > 0) {
        ctx_server.start_rpc_server(params.rpc_port);
    }

    // Hydra #356: server_routes owns the schema-correct handlers
    // (post_chat_completions, post_completions_oai) and the chat-template
    // meta (read by oaicompat_chat_params_parse). Declared BEFORE
    // server_http_context ctx_http so the captured-lambda handlers stored
    // in ctx_http are destroyed before `routes` is destroyed — reverse
    // declaration order would leave dangling captures in ctx_http. Mirrors
    // the lifetime layout in tools/server/server.cpp (the routes struct is
    // constructed in the same scope as ctx_http and used until shutdown).
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

        // A1: Real /slots data
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

        // A2: /metrics Prometheus endpoint
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

        // A3: Slot erase
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

        // A4: /v1/models stub
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
            // Reuse same handler as /v1/models
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

        // State meta: real data from task queue
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

        // Hydra #356: route OpenAI-schema completions through the
        // schema-correct server_routes handlers so the existing
        // CompletionProxyService on Hydra.Core can target engine-mode
        // workers. Hand-rolled routes above are unchanged; the routes
        // struct just adds these three. Unused server_routes handlers
        // (embeddings, anthropic, lora, etc.) stay unregistered — full
        // llama-server parity is a larger, riskier refactor deferred as
        // a follow-up.
        ctx_http.post("/v1/chat/completions", ex_wrapper(routes.post_chat_completions));
        ctx_http.post("/chat/completions",    ex_wrapper(routes.post_chat_completions));
        ctx_http.post("/v1/completions",      ex_wrapper(routes.post_completions_oai));

        if (!ctx_http.start()) {
            LOG_ERR("eng  %12.*s: failed to start HTTP server\n", 12, __func__);
            llama_backend_free();
            return 1;
        }

        // Pre-existing gap, found while live-verifying #348: unlike
        // tools/server/server.cpp's main(), llama_engine() never flipped
        // ctx_http.is_ready — server-http.cpp's middleware_server_state
        // gates EVERY endpoint on this flag, so every HTTP route 503'd
        // "Loading model" forever, even after the model finished loading.
        // Hydra #356: server_routes::post_chat_completions reads
        // meta->chat_params, which is only populated by update_meta() —
        // must run before ctx_http.is_ready flips or the first chat
        // request will see a default-constructed (empty) chat template.
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

    ctx_server.terminate();
    llama_backend_free();

    return 0;
}

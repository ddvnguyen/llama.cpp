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

// Hydra #287/#260: one-binary role flag, filtered out before common_params_parse
// the same way extract_rpc_engine_peer() is (these aren't stock llama.cpp args).
//   --role <standalone|head|worker>   default: standalone (no behavior change)
//   --combined-ot-pattern <regex>     head only: which expert tensors (matched
//                                     by name) get a dual-resident copy on the
//                                     peer named by --rpc-engine (COMBINED mode)
//   --ggml-rpc-port <port>            worker only: port for the embedded
//                                     ggml-RPC backend server (NOT the Hydra
//                                     control-RPC port — different wire protocol)
struct hydra_role_flags {
    std::string role = "standalone";
    std::string combined_ot_pattern;
    int         ggml_rpc_port = 0;
};

// Filters role flags out of argv_inout in place (same vector serves as input
// and output), mirroring extract_rpc_engine_peer()'s filtering style.
static hydra_role_flags extract_hydra_role_flags(std::vector<char *> & argv_inout) {
    hydra_role_flags flags;
    std::vector<char *> in = std::move(argv_inout);
    argv_inout.clear();
    if (!in.empty()) argv_inout.push_back(in[0]);

    for (size_t i = 1; i < in.size(); i++) {
        if (strcmp(in[i], "--role") == 0 && i + 1 < in.size()) {
            flags.role = in[++i];
            continue;
        }
        if (strcmp(in[i], "--combined-ot-pattern") == 0 && i + 1 < in.size()) {
            flags.combined_ot_pattern = in[++i];
            continue;
        }
        if (strcmp(in[i], "--ggml-rpc-port") == 0 && i + 1 < in.size()) {
            flags.ggml_rpc_port = std::atoi(in[++i]);
            continue;
        }
        argv_inout.push_back(in[i]);
    }

    return flags;
}

// Hydra #287/#260 — COMBINED worker role: this engine process does not load a
// model or serve completions at all. It is purely an embedded ggml-RPC backend
// server exposing its local GPU(s) for a head engine's --rpc-engine link, the
// same protocol tools/rpc/rpc-server.cpp serves as a standalone process — here
// it's embedded directly in the engine binary so a single role flag switches
// between solo/head/worker without a separate process to manage. Blocks forever
// (the worker has nothing else to do; this sidesteps ever running local
// inference and the embedded RPC server concurrently against the same GPU
// context, which is not a supported pattern upstream).
static int run_combined_worker(int ggml_rpc_port) {
    if (ggml_rpc_port <= 0) {
        LOG_ERR("eng  %12.*s: --role worker requires --ggml-rpc-port\n", 12, __func__);
        return 1;
    }

    ggml_backend_load_all();
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        LOG_ERR("eng  %12.*s: RPC backend not available in this build\n", 12, __func__);
        return 1;
    }
    using start_server_fn_t = void (*)(const char *, const char *, size_t, size_t, ggml_backend_dev_t *);
    auto start_server_fn = (start_server_fn_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_start_server");
    if (!start_server_fn) {
        LOG_ERR("eng  %12.*s: failed to resolve ggml_backend_rpc_start_server\n", 12, __func__);
        return 1;
    }

    std::vector<ggml_backend_dev_t> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            devices.push_back(dev);
        }
    }
    if (devices.empty()) {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu) devices.push_back(cpu);
    }
    if (devices.empty()) {
        LOG_ERR("eng  %12.*s: no devices found for the embedded RPC server\n", 12, __func__);
        return 1;
    }

    const std::string endpoint = "0.0.0.0:" + std::to_string(ggml_rpc_port);
    LOG_INF("eng  %12.*s: role=worker — serving %zu device(s) as a ggml-RPC backend on %s\n",
            12, __func__, devices.size(), endpoint.c_str());

    start_server_fn(endpoint.c_str(), nullptr, std::max(1u, std::thread::hardware_concurrency() / 2),
                     devices.size(), devices.data());
    return 0;
}

int llama_engine(int argc, char ** argv);

int llama_engine(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    std::vector<char *> filtered_argv;
    std::string rpc_engine_peer = extract_rpc_engine_peer(argc, argv, filtered_argv);
    hydra_role_flags role_flags = extract_hydra_role_flags(filtered_argv);

    if (role_flags.role != "standalone" && role_flags.role != "head" && role_flags.role != "worker") {
        LOG_ERR("eng  %12.*s: --role must be 'standalone', 'head', or 'worker' (got '%s')\n",
                12, __func__, role_flags.role.c_str());
        return 1;
    }

    int filtered_argc = filtered_argv.size();
    char ** filtered_argv_ptr = filtered_argv.data();

    if (!common_params_parse(filtered_argc, filtered_argv_ptr, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    // Hydra #287/#260: a COMBINED worker is purely an embedded ggml-RPC
    // backend — no model load, no HTTP/control-RPC serving. It blocks here
    // for the lifetime of the process.
    if (role_flags.role == "worker") {
        llama_backend_init();
        int rc = run_combined_worker(role_flags.ggml_rpc_port);
        llama_backend_free();
        return rc;
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

    ctx_server.set_hydra_role(role_flags.role, rpc_engine_peer, role_flags.combined_ot_pattern);

    // Hydra #287/#260: a head dual-loads its configured expert tensors onto
    // the --rpc-engine peer once, here, before serving any requests. Missing
    // config or an unreachable peer degrades to solo-only (never aborts
    // startup) — SET_EXPERT_MODE("combined") will report "solo" until fixed.
    if (role_flags.role == "head") {
        if (rpc_engine_peer.empty() || role_flags.combined_ot_pattern.empty()) {
            LOG_WRN("eng  %12.*s: role=head needs --rpc-engine and --combined-ot-pattern for "
                    "COMBINED — running solo-only\n", 12, __func__);
        } else {
            int32_t n = llama_hydra_load_combined_experts(ctx_server.get_llama_context(),
                    rpc_engine_peer.c_str(), role_flags.combined_ot_pattern.c_str());
            ctx_server.set_hydra_combined_capable(n > 0);
            if (n > 0) {
                LOG_INF("eng  %12.*s: COMBINED ready — %d layer(s) dual-loaded onto %s\n",
                        12, __func__, n, rpc_engine_peer.c_str());
            } else {
                LOG_WRN("eng  %12.*s: COMBINED dual-load failed — running solo-only\n", 12, __func__);
            }
        }
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

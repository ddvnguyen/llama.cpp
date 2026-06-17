#pragma once

#include "server-context.h"
#include "server-task.h"

#include <atomic>
#include <string>
#include <thread>

struct engine_params {
    common_params base;
    std::string   rpc_engine_peer;
    int           rpc_port = 0;
    bool          solo_mode = false;
};

struct llama_engine_context {
    server_context ctx_server;
    engine_params  params;

    bool load_model();
    void start_rpc();
    void start_loop();
    void terminate();

    llama_context * get_llama_context() const;
};

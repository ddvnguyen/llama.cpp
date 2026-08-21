#pragma once

#include <cstdint>
#include <string>

struct llama_kv_stream_config {
    uint64_t stage_bytes         = 0;
    uint64_t minimum_stage_bytes = 0;

    bool arch_qwen35     = false;
    bool context_default = false;
    bool single_sequence = false;
    bool flash_attention = false;
    bool kv_offload      = false;
    bool cache_q8_q4     = false;
};

struct llama_kv_stream_config_result {
    bool valid   = false;
    bool enabled = false;
    std::string error;
};

llama_kv_stream_config_result llama_kv_stream_config_validate(const llama_kv_stream_config & config);

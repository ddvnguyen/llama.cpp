#include "llama-kv-stream-config.h"

llama_kv_stream_config_result llama_kv_stream_config_validate(const llama_kv_stream_config & config) {
    if (config.stage_bytes == 0) {
        return { true, false, {} };
    }

    auto invalid = [](const char * error) {
        return llama_kv_stream_config_result { false, false, error };
    };

    if (!config.arch_qwen35) {
        return invalid("block KV streaming currently supports only Qwen3.5");
    }
    if (!config.context_default) {
        return invalid("block KV streaming currently supports only the target context, not MTP/draft contexts");
    }
    if (!config.single_sequence) {
        return invalid("block KV streaming requires exactly one sequence (-np 1)");
    }
    if (!config.flash_attention) {
        return invalid("block KV streaming requires Flash Attention");
    }
    if (!config.kv_offload) {
        return invalid("block KV streaming requires GPU KV offload");
    }
    if (!config.cache_q8_q4) {
        return invalid("block KV streaming currently requires K Q8_0 and V Q4_0");
    }
    if (config.minimum_stage_bytes == 0 || config.stage_bytes < config.minimum_stage_bytes) {
        return invalid("block KV streaming stage is too small for one 256-token cache page");
    }

    return { true, true, {} };
}

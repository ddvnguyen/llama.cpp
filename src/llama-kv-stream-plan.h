#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class llama_kv_stream_region_role : uint8_t {
    target,
    mtp,
};

// Smallest independently resident piece of a logical KV cache. The CUDA copy
// layer may coalesce adjacent regions into a larger transfer slot.
struct llama_kv_stream_region {
    llama_kv_stream_region_role role = llama_kv_stream_region_role::target;

    int32_t  layer_id    = -1;
    uint32_t token_begin = 0;
    uint32_t token_count = 0;

    uint64_t bytes = 0;

    // Lower values have stronger residency preference. The ordering must be
    // stable across replans to avoid unnecessary promotion/demotion churn.
    uint64_t residency_priority = 0;

    // Pinned is a logical pool policy, not cudaMalloc pinning. A pinned region
    // must receive permanent pool space or planning fails.
    bool pinned = false;
};

struct llama_kv_stream_plan_params {
    uint64_t pool_bytes       = 0;
    uint64_t stage_slot_bytes = 0;
    uint32_t stage_slots      = 0;

    std::vector<llama_kv_stream_region> regions;
};

struct llama_kv_stream_plan {
    bool valid = false;
    std::string error;

    uint64_t pool_bytes           = 0;
    uint64_t reserved_stage_bytes = 0;
    uint64_t resident_bytes       = 0;
    uint64_t streamed_bytes       = 0;
    uint64_t unused_bytes         = 0;

    std::vector<size_t> resident_regions;
    std::vector<size_t> streamed_regions;
};

llama_kv_stream_plan llama_kv_stream_plan_make(const llama_kv_stream_plan_params & params);

struct llama_kv_stream_extent_params {
    uint32_t live_tokens              = 0;
    uint32_t reserve_tokens           = 0;
    uint32_t page_tokens              = 256;
    uint32_t previous_extent          = 0;
    uint32_t shrink_hysteresis_tokens = 0;
    uint32_t maximum_tokens           = 0;

    bool force_shrink = false;
};

struct llama_kv_stream_extent {
    bool valid = false;
    std::string error;

    uint32_t tokens = 0;
    bool grew   = false;
    bool shrunk = false;
};

llama_kv_stream_extent llama_kv_stream_extent_make(const llama_kv_stream_extent_params & params);

struct llama_kv_stream_layer_layout {
    llama_kv_stream_region_role role = llama_kv_stream_region_role::target;

    int32_t  layer_id        = -1;
    uint32_t n_tokens        = 0;
    uint64_t bytes_per_token = 0;

    // Lower values spread residency preference within the same token page.
    uint32_t layer_priority = 0;

    bool pin_all  = false;
    bool pin_tail = false;
};

struct llama_kv_stream_regions_params {
    uint32_t page_tokens = 256;
    std::vector<llama_kv_stream_layer_layout> layers;
};

struct llama_kv_stream_regions {
    bool valid = false;
    std::string error;

    uint64_t total_bytes = 0;
    std::vector<llama_kv_stream_region> regions;
};

llama_kv_stream_regions llama_kv_stream_regions_make(const llama_kv_stream_regions_params & params);

#include "llama-kv-stream-plan.h"

#include <algorithm>
#include <limits>
#include <set>
#include <tuple>

namespace {

struct logical_interval {
    llama_kv_stream_region_role role;
    int32_t  layer_id;
    uint64_t begin;
    uint64_t end;
};

bool checked_add(uint64_t a, uint64_t b, uint64_t & result) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }

    result = a + b;
    return true;
}

bool checked_mul(uint64_t a, uint64_t b, uint64_t & result) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max()/a) {
        return false;
    }

    result = a*b;
    return true;
}

llama_kv_stream_plan fail(llama_kv_stream_plan result, const char * message) {
    result.valid = false;
    result.error = message;
    return result;
}

} // namespace

llama_kv_stream_plan llama_kv_stream_plan_make(const llama_kv_stream_plan_params & params) {
    llama_kv_stream_plan result;
    result.pool_bytes = params.pool_bytes;

    if (params.pool_bytes == 0) {
        return fail(std::move(result), "KV stream pool must be non-zero");
    }

    if (params.stage_slots > 0 && params.stage_slot_bytes == 0) {
        return fail(std::move(result), "non-zero stage count requires a non-zero stage size");
    }

    if (!checked_mul(params.stage_slots, params.stage_slot_bytes, result.reserved_stage_bytes)) {
        return fail(std::move(result), "stage reservation size overflow");
    }

    if (result.reserved_stage_bytes > params.pool_bytes) {
        return fail(std::move(result), "stage reservations exceed the KV stream pool");
    }

    std::vector<size_t> optional_regions;
    std::vector<logical_interval> logical_intervals;
    optional_regions.reserve(params.regions.size());
    logical_intervals.reserve(params.regions.size());
    result.resident_regions.reserve(params.regions.size());
    result.streamed_regions.reserve(params.regions.size());

    uint64_t pinned_bytes = 0;

    for (size_t index = 0; index < params.regions.size(); ++index) {
        const auto & region = params.regions[index];

        if (region.layer_id < 0) {
            return fail(std::move(result), "KV stream region has an invalid layer id");
        }

        if (region.token_count == 0 || region.bytes == 0) {
            return fail(std::move(result), "KV stream region must contain tokens and bytes");
        }

        const uint64_t token_end = uint64_t(region.token_begin) + region.token_count;
        if (token_end > uint64_t(std::numeric_limits<uint32_t>::max()) + 1) {
            return fail(std::move(result), "KV stream region token range overflow");
        }

        logical_intervals.push_back({ region.role, region.layer_id, region.token_begin, token_end });

        if (region.pinned) {
            if (!checked_add(pinned_bytes, region.bytes, pinned_bytes)) {
                return fail(std::move(result), "pinned KV region size overflow");
            }
            result.resident_regions.push_back(index);
        } else {
            optional_regions.push_back(index);
        }
    }

    std::sort(logical_intervals.begin(), logical_intervals.end(), [](const auto & lhs, const auto & rhs) {
        return std::tie(lhs.role, lhs.layer_id, lhs.begin, lhs.end) <
               std::tie(rhs.role, rhs.layer_id, rhs.begin, rhs.end);
    });

    for (size_t i = 1; i < logical_intervals.size(); ++i) {
        const auto & previous = logical_intervals[i - 1];
        const auto & current  = logical_intervals[i];

        if (previous.role == current.role &&
            previous.layer_id == current.layer_id &&
            current.begin < previous.end) {
            return fail(std::move(result), "KV stream regions overlap in one logical layer");
        }
    }

    uint64_t required_bytes = 0;
    if (!checked_add(result.reserved_stage_bytes, pinned_bytes, required_bytes)) {
        return fail(std::move(result), "mandatory KV stream pool size overflow");
    }

    if (required_bytes > params.pool_bytes) {
        return fail(std::move(result), "pinned KV regions and stages exceed the pool");
    }

    result.resident_bytes = pinned_bytes;
    uint64_t remaining_bytes = params.pool_bytes - required_bytes;

    std::sort(optional_regions.begin(), optional_regions.end(), [&](size_t a, size_t b) {
        const auto & lhs = params.regions[a];
        const auto & rhs = params.regions[b];

        return std::tie(lhs.residency_priority, lhs.role, lhs.layer_id, lhs.token_begin, a) <
               std::tie(rhs.residency_priority, rhs.role, rhs.layer_id, rhs.token_begin, b);
    });

    for (size_t index : optional_regions) {
        const auto & region = params.regions[index];

        if (region.bytes <= remaining_bytes) {
            result.resident_regions.push_back(index);
            result.resident_bytes += region.bytes;
            remaining_bytes -= region.bytes;
            continue;
        }

        if (params.stage_slots == 0) {
            return fail(std::move(result), "streaming is required but no stage slots are configured");
        }

        if (region.bytes > params.stage_slot_bytes) {
            return fail(std::move(result), "a streamed KV region exceeds the stage slot size");
        }

        uint64_t streamed_bytes = 0;
        if (!checked_add(result.streamed_bytes, region.bytes, streamed_bytes)) {
            return fail(std::move(result), "streamed KV region size overflow");
        }

        result.streamed_bytes = streamed_bytes;
        result.streamed_regions.push_back(index);
    }

    result.unused_bytes = remaining_bytes;
    result.valid = true;
    return result;
}

llama_kv_stream_extent llama_kv_stream_extent_make(const llama_kv_stream_extent_params & params) {
    llama_kv_stream_extent result;

    auto fail_extent = [&](const char * message) {
        result.valid = false;
        result.error = message;
        return result;
    };

    if (params.page_tokens == 0) {
        return fail_extent("KV stream page size must be non-zero");
    }

    if (params.maximum_tokens == 0 || params.maximum_tokens%params.page_tokens != 0) {
        return fail_extent("KV stream maximum must be non-zero and page aligned");
    }

    if (params.previous_extent > params.maximum_tokens ||
        params.previous_extent%params.page_tokens != 0) {
        return fail_extent("previous KV stream extent is invalid");
    }

    const uint64_t requested_tokens = uint64_t(params.live_tokens) + params.reserve_tokens;
    if (requested_tokens > params.maximum_tokens) {
        return fail_extent("live and reserved KV tokens exceed the configured maximum");
    }

    uint64_t desired_tokens = 0;
    if (requested_tokens > 0) {
        desired_tokens = ((requested_tokens + params.page_tokens - 1)/params.page_tokens)*params.page_tokens;
    }

    if (desired_tokens > params.maximum_tokens) {
        return fail_extent("padded KV stream extent exceeds the configured maximum");
    }

    uint32_t selected_tokens = uint32_t(desired_tokens);
    if (!params.force_shrink && params.previous_extent > selected_tokens) {
        const uint32_t released_tokens = params.previous_extent - selected_tokens;
        if (released_tokens < params.shrink_hysteresis_tokens) {
            selected_tokens = params.previous_extent;
        }
    }

    result.valid  = true;
    result.tokens = selected_tokens;
    result.grew   = selected_tokens > params.previous_extent;
    result.shrunk = selected_tokens < params.previous_extent;
    return result;
}

llama_kv_stream_regions llama_kv_stream_regions_make(const llama_kv_stream_regions_params & params) {
    llama_kv_stream_regions result;

    auto fail_regions = [&](const char * message) {
        result.valid = false;
        result.error = message;
        result.regions.clear();
        result.total_bytes = 0;
        return result;
    };

    if (params.page_tokens == 0) {
        return fail_regions("KV stream region page size must be non-zero");
    }

    std::set<std::pair<llama_kv_stream_region_role, int32_t>> logical_layers;

    for (const auto & layer : params.layers) {
        if (layer.layer_id < 0) {
            return fail_regions("KV stream layer layout has an invalid layer id");
        }

        if (!logical_layers.emplace(layer.role, layer.layer_id).second) {
            return fail_regions("KV stream layer layout is duplicated");
        }

        if (layer.n_tokens > 0 && layer.bytes_per_token == 0) {
            return fail_regions("non-empty KV stream layer must have a non-zero token size");
        }

        for (uint64_t token_begin = 0; token_begin < layer.n_tokens; token_begin += params.page_tokens) {
            const uint32_t token_count = uint32_t(std::min<uint64_t>(
                params.page_tokens, uint64_t(layer.n_tokens) - token_begin));

            llama_kv_stream_region region;
            region.role        = layer.role;
            region.layer_id    = layer.layer_id;
            region.token_begin = uint32_t(token_begin);
            region.token_count = token_count;
            region.pinned      = layer.pin_all ||
                (layer.pin_tail && token_begin + token_count == layer.n_tokens);

            if (!checked_mul(token_count, layer.bytes_per_token, region.bytes)) {
                return fail_regions("KV stream region byte size overflow");
            }

            uint64_t total_bytes = 0;
            if (!checked_add(result.total_bytes, region.bytes, total_bytes)) {
                return fail_regions("KV stream layout total byte size overflow");
            }
            result.total_bytes = total_bytes;

            const uint64_t page_index = token_begin/params.page_tokens;
            region.residency_priority = (page_index << 32) | layer.layer_priority;
            result.regions.push_back(region);
        }
    }

    result.valid = true;
    return result;
}

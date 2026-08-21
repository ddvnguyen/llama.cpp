#include "llama-kv-stream-plan.h"

#include <algorithm>
#include <limits>
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

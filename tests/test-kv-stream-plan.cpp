#include "llama-kv-stream-plan.h"
#include "testing.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr uint64_t MIB = 1024ULL*1024ULL;

constexpr uint32_t N_TARGET_LAYERS = 16;
constexpr uint32_t N_MTP_LAYERS    = 1;
constexpr uint32_t PAGE_TOKENS     = 256;
constexpr uint32_t STAGE_TOKENS    = 16*1024;
constexpr uint64_t BYTES_PER_TOKEN = 1664;

std::vector<llama_kv_stream_region> make_qwen38_regions(uint32_t n_kv) {
    std::vector<llama_kv_stream_region> result;

    auto append_layer = [&](llama_kv_stream_region_role role, int32_t layer_id, bool pinned) {
        for (uint32_t token_begin = 0; token_begin < n_kv; token_begin += PAGE_TOKENS) {
            const uint32_t token_count = std::min(PAGE_TOKENS, n_kv - token_begin);

            llama_kv_stream_region region;
            region.role               = role;
            region.layer_id           = layer_id;
            region.token_begin        = token_begin;
            region.token_count        = token_count;
            region.bytes              = uint64_t(token_count)*BYTES_PER_TOKEN;
            region.residency_priority = uint64_t(layer_id)*1'000'000ULL + token_begin/PAGE_TOKENS;
            region.pinned             = pinned;
            result.push_back(region);
        }
    };

    for (uint32_t il = 0; il < N_TARGET_LAYERS; ++il) {
        append_layer(llama_kv_stream_region_role::target, 3 + 4*il, false);
    }

    append_layer(llama_kv_stream_region_role::mtp, 64, true);

    return result;
}

llama_kv_stream_plan_params make_params(uint32_t n_kv, uint64_t pool_mib = 2526) {
    llama_kv_stream_plan_params params;
    params.pool_bytes       = pool_mib*MIB;
    params.stage_slot_bytes = STAGE_TOKENS*BYTES_PER_TOKEN;
    params.stage_slots      = 2;
    params.regions          = make_qwen38_regions(n_kv);
    return params;
}

size_t count_role(
        const llama_kv_stream_plan_params & params,
        const std::vector<size_t> & region_indices,
        llama_kv_stream_region_role role) {
    return std::count_if(region_indices.begin(), region_indices.end(), [&](size_t index) {
        return params.regions.at(index).role == role;
    });
}

} // namespace

int main() {
    testing t;

    t.test("small context remains fully resident while transfer slots stay reserved", [](testing & t) {
        const auto params = make_params(1024);
        const auto plan = llama_kv_stream_plan_make(params);

        t.assert_true("plan is valid", plan.valid);
        t.assert_equal(2*STAGE_TOKENS*BYTES_PER_TOKEN, plan.reserved_stage_bytes);
        t.assert_equal(uint64_t(0), plan.streamed_bytes);
        t.assert_equal(size_t(0), plan.streamed_regions.size());
        t.assert_equal(params.regions.size(), plan.resident_regions.size());
        t.assert_equal(
            plan.pool_bytes,
            plan.reserved_stage_bytes + plan.resident_bytes + plan.unused_bytes);
    });

    t.test("pinned MTP pages remain resident at 196K", [](testing & t) {
        const auto params = make_params(196608);
        const auto plan = llama_kv_stream_plan_make(params);

        t.assert_true("plan is valid", plan.valid);
        t.assert_equal(size_t(0), count_role(params, plan.streamed_regions, llama_kv_stream_region_role::mtp));
        t.assert_true(
            "some target pages stream after the pool fills",
            count_role(params, plan.streamed_regions, llama_kv_stream_region_role::target) > 0);
        t.assert_true(
            "resident bytes never exceed the pool",
            plan.reserved_stage_bytes + plan.resident_bytes <= plan.pool_bytes);
    });

    t.test("one padded context step increases streaming smoothly", [](testing & t) {
        const auto params_before = make_params(196608);
        const auto params_after  = make_params(196608 + PAGE_TOKENS);

        const auto before = llama_kv_stream_plan_make(params_before);
        const auto after  = llama_kv_stream_plan_make(params_after);

        t.assert_true("before plan is valid", before.valid);
        t.assert_true("after plan is valid", after.valid);
        t.assert_equal(
            uint64_t((N_TARGET_LAYERS + N_MTP_LAYERS)*PAGE_TOKENS)*BYTES_PER_TOKEN,
            after.streamed_bytes - before.streamed_bytes);
    });

    t.test("planning fails when pinned pages and stages do not fit", [](testing & t) {
        auto params = make_params(196608);
        params.pool_bytes = 300*MIB;

        const auto plan = llama_kv_stream_plan_make(params);
        t.assert_true("plan is rejected", !plan.valid);
        t.assert_true("error is reported", !plan.error.empty());
    });

    t.test("a streamed region must fit one transfer slot", [](testing & t) {
        llama_kv_stream_plan_params params;
        params.pool_bytes       = 1*MIB;
        params.stage_slot_bytes = 1*MIB;
        params.stage_slots      = 1;

        llama_kv_stream_region region;
        region.layer_id    = 3;
        region.token_count = 1;
        region.bytes       = 2*MIB;
        params.regions.push_back(region);

        const auto plan = llama_kv_stream_plan_make(params);
        t.assert_true("plan is rejected", !plan.valid);
        t.assert_true("error is reported", !plan.error.empty());
    });

    t.test("every region is assigned exactly once and accounting closes", [](testing & t) {
        for (uint32_t n_kv : { 1024U, 92160U, 196608U, 262144U }) {
            const auto params = make_params(n_kv);
            const auto plan = llama_kv_stream_plan_make(params);

            t.assert_true("plan is valid", plan.valid);

            std::vector<uint8_t> seen(params.regions.size(), 0);
            uint64_t logical_bytes = 0;

            for (const auto & region : params.regions) {
                logical_bytes += region.bytes;
            }

            bool indices_in_range = true;
            for (size_t index : plan.resident_regions) {
                if (index < seen.size()) {
                    ++seen[index];
                } else {
                    indices_in_range = false;
                }
            }

            for (size_t index : plan.streamed_regions) {
                if (index < seen.size()) {
                    ++seen[index];
                } else {
                    indices_in_range = false;
                }
            }

            t.assert_true("all assignment indices are in range", indices_in_range);
            t.assert_true("every region is assigned once", std::all_of(seen.begin(), seen.end(), [](uint8_t count) {
                return count == 1;
            }));
            t.assert_equal(logical_bytes, plan.resident_bytes + plan.streamed_bytes);
            t.assert_equal(
                plan.pool_bytes,
                plan.reserved_stage_bytes + plan.resident_bytes + plan.unused_bytes);
        }
    });

    t.test("streamed bytes grow monotonically without a context cliff", [](testing & t) {
        std::vector<uint32_t> contexts = { 1024, 65536, 89088 };
        for (uint32_t n_kv = 89344; n_kv <= 94208; n_kv += PAGE_TOKENS) {
            contexts.push_back(n_kv);
        }
        contexts.insert(contexts.end(), { 131072, 196608, 196864, 262144 });

        uint64_t previous_streamed = 0;
        uint32_t previous_context = 0;

        for (uint32_t n_kv : contexts) {
            const auto plan = llama_kv_stream_plan_make(make_params(n_kv));

            if (!t.assert_true("plan is valid", plan.valid)) {
                break;
            }

            t.assert_true("streamed bytes are monotonic", plan.streamed_bytes >= previous_streamed);
            if (previous_context != 0) {
                const uint64_t maximum_step =
                    uint64_t(N_TARGET_LAYERS + N_MTP_LAYERS)*(n_kv - previous_context)*BYTES_PER_TOKEN;
                t.assert_true(
                    "context growth cannot stream more bytes than the added logical KV",
                    plan.streamed_bytes - previous_streamed <= maximum_step);
            }

            previous_streamed = plan.streamed_bytes;
            previous_context  = n_kv;
        }
    });

    t.test("duplicate logical regions are rejected", [](testing & t) {
        llama_kv_stream_plan_params params;
        params.pool_bytes       = 64*MIB;
        params.stage_slot_bytes = 1*MIB;
        params.stage_slots      = 1;

        llama_kv_stream_region region;
        region.layer_id    = 3;
        region.token_begin = 0;
        region.token_count = PAGE_TOKENS;
        region.bytes       = PAGE_TOKENS*BYTES_PER_TOKEN;

        params.regions.push_back(region);
        params.regions.push_back(region);

        const auto plan = llama_kv_stream_plan_make(params);
        t.assert_true("duplicate plan is rejected", !plan.valid);
    });

    t.test("overlapping regions in one logical layer are rejected", [](testing & t) {
        llama_kv_stream_plan_params params;
        params.pool_bytes       = 64*MIB;
        params.stage_slot_bytes = 1*MIB;
        params.stage_slots      = 1;

        llama_kv_stream_region first;
        first.layer_id    = 3;
        first.token_begin = 0;
        first.token_count = PAGE_TOKENS;
        first.bytes       = PAGE_TOKENS*BYTES_PER_TOKEN;

        auto second = first;
        second.token_begin = PAGE_TOKENS/2;

        params.regions.push_back(first);
        params.regions.push_back(second);

        const auto plan = llama_kv_stream_plan_make(params);
        t.assert_true("overlapping plan is rejected", !plan.valid);
    });

    t.test("overflowing token ranges are rejected", [](testing & t) {
        llama_kv_stream_plan_params params;
        params.pool_bytes       = 64*MIB;
        params.stage_slot_bytes = 1*MIB;
        params.stage_slots      = 1;

        llama_kv_stream_region region;
        region.layer_id    = 3;
        region.token_begin = std::numeric_limits<uint32_t>::max() - 127;
        region.token_count = PAGE_TOKENS;
        region.bytes       = PAGE_TOKENS*BYTES_PER_TOKEN;
        params.regions.push_back(region);

        const auto plan = llama_kv_stream_plan_make(params);
        t.assert_true("overflowing plan is rejected", !plan.valid);
    });

    t.test("extent includes pending speculative positions and page padding", [](testing & t) {
        llama_kv_stream_extent_params params;
        params.live_tokens    = 196607;
        params.reserve_tokens = 3;
        params.page_tokens    = PAGE_TOKENS;
        params.maximum_tokens = 262144;

        const auto extent = llama_kv_stream_extent_make(params);
        t.assert_true("extent is valid", extent.valid);
        t.assert_equal(uint32_t(196864), extent.tokens);
        t.assert_true("initial extent is growth", extent.grew);
        t.assert_true("initial extent is not shrink", !extent.shrunk);
    });

    t.test("growth is immediate", [](testing & t) {
        llama_kv_stream_extent_params params;
        params.live_tokens              = 131072;
        params.reserve_tokens           = 3;
        params.page_tokens              = PAGE_TOKENS;
        params.previous_extent          = 131072;
        params.shrink_hysteresis_tokens = 4096;
        params.maximum_tokens           = 262144;

        const auto extent = llama_kv_stream_extent_make(params);
        t.assert_true("extent is valid", extent.valid);
        t.assert_equal(uint32_t(131328), extent.tokens);
        t.assert_true("extent grew", extent.grew);
    });

    t.test("small rollback retains the previous extent", [](testing & t) {
        llama_kv_stream_extent_params params;
        params.live_tokens              = 196605;
        params.reserve_tokens           = 3;
        params.page_tokens              = PAGE_TOKENS;
        params.previous_extent          = 196864;
        params.shrink_hysteresis_tokens = 4096;
        params.maximum_tokens           = 262144;

        const auto extent = llama_kv_stream_extent_make(params);
        t.assert_true("extent is valid", extent.valid);
        t.assert_equal(uint32_t(196864), extent.tokens);
        t.assert_true("extent did not shrink", !extent.shrunk);
    });

    t.test("large shrink and forced reset release residency", [](testing & t) {
        llama_kv_stream_extent_params params;
        params.live_tokens              = 180000;
        params.page_tokens              = PAGE_TOKENS;
        params.previous_extent          = 196864;
        params.shrink_hysteresis_tokens = 4096;
        params.maximum_tokens           = 262144;

        auto extent = llama_kv_stream_extent_make(params);
        t.assert_true("large-shrink extent is valid", extent.valid);
        t.assert_equal(uint32_t(180224), extent.tokens);
        t.assert_true("large shrink is reported", extent.shrunk);

        params.live_tokens     = 0;
        params.force_shrink    = true;
        extent = llama_kv_stream_extent_make(params);
        t.assert_true("reset extent is valid", extent.valid);
        t.assert_equal(uint32_t(0), extent.tokens);
        t.assert_true("reset shrink is reported", extent.shrunk);
    });

    t.test("extent rejects invalid alignment and capacity overflow", [](testing & t) {
        llama_kv_stream_extent_params params;
        params.live_tokens    = 262143;
        params.reserve_tokens = 3;
        params.page_tokens    = PAGE_TOKENS;
        params.maximum_tokens = 262144;

        auto extent = llama_kv_stream_extent_make(params);
        t.assert_true("capacity overflow is rejected", !extent.valid);

        params.live_tokens    = 1024;
        params.reserve_tokens = 0;
        params.page_tokens    = 0;
        extent = llama_kv_stream_extent_make(params);
        t.assert_true("zero page size is rejected", !extent.valid);

        params.page_tokens      = PAGE_TOKENS;
        params.maximum_tokens   = 262143;
        extent = llama_kv_stream_extent_make(params);
        t.assert_true("unaligned maximum is rejected", !extent.valid);
    });

    return t.summary();
}

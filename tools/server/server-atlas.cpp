// hydra: implementation of the expert-atlas read surface (see server-atlas.h).
// Geometry source of truth = the model's own GGUF (arch KVs + per-layer expert
// tensor presence), so no architecture is hardcoded and the UI never guesses a
// layer mapping — the Colibri GLM `row+3`/MTP-78 hardcode dies here (design §D).

#include "server-atlas.h"

#include "ggml.h"
#include "gguf.h"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>

namespace hydra_atlas {

namespace {

using json = nlohmann::json;

uint64_t fnv1a64(const std::string & s) {
    uint64_t h = 1469598103934665603ULL;
    for (const unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

std::string hex64(const uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) v);
    return buf;
}

bool kv_u32(const gguf_context * gguf, const std::string & key, uint32_t & out) {
    const int idx = gguf_find_key(gguf, key.c_str());
    if (idx < 0) return false;
    if (gguf_get_kv_type(gguf, idx) != GGUF_TYPE_UINT32) return false;
    out = gguf_get_val_u32(gguf, idx);
    return true;
}

// Layer index from a gguf tensor name "blk.<il>.<rest>"; -1 if not a blk tensor.
int blk_layer(const char * name) {
    if (std::strncmp(name, "blk.", 4) != 0) return -1;
    if (!std::isdigit((unsigned char) name[4])) return -1;
    return std::atoi(name + 4);
}

bool load_impl(const std::string & path, geometry & g) {
    gguf_init_params p = { /* no_alloc */ true, /* ctx */ nullptr };
    gguf_context * gguf = gguf_init_from_file(path.c_str(), p);
    if (!gguf) return false;

    const int ai = gguf_find_key(gguf, "general.architecture");
    if (ai < 0 || gguf_get_kv_type(gguf, ai) != GGUF_TYPE_STRING) {
        gguf_free(gguf);
        return false;
    }
    g.arch = gguf_get_val_str(gguf, ai);

    uint32_t block_count = 0, expert_count = 0, expert_used = 0, nextn = 0;
    kv_u32(gguf, g.arch + ".block_count", block_count);
    kv_u32(gguf, g.arch + ".expert_count", expert_count);
    kv_u32(gguf, g.arch + ".expert_used_count", expert_used);
    // NextN/MTP layers are optional (kv stays 0 when absent).
    kv_u32(gguf, g.arch + ".nextn_predict_layers", nextn);

    if (expert_count == 0) { // dense model: no expert atlas surface
        gguf_free(gguf);
        return false;
    }

    // Routed-expert presence per layer, straight from tensor names — arch-agnostic.
    std::vector<bool> moe(block_count + nextn, false);
    const size_t n_tensors = gguf_get_n_tensors(gguf);
    for (size_t i = 0; i < n_tensors; i++) {
        const char * tn = gguf_get_tensor_name(gguf, (int) i);
        if (std::strstr(tn, "exps") == nullptr) continue;
        const int il = blk_layer(tn);
        if (il >= 0 && il < (int) moe.size()) moe[il] = true;
    }

    for (int il = 0; il < (int) block_count; il++) {
        if (moe[il]) g.moe_rows.push_back(il);
    }
    for (int il = (int) block_count; il < (int) (block_count + nextn); il++) {
        if (moe[il]) g.nextn_rows.push_back(il); // MTP rows reported separately (design §A)
    }

    g.cols           = (int) expert_count;
    g.n_expert_used  = (int) expert_used;
    g.dense_prefix   = g.moe_rows.empty() ? 0 : g.moe_rows[0];
    g.rows           = (int) (g.moe_rows.size() + g.nextn_rows.size());
    g.model_hash     = path.substr(path.find_last_of("/\\") + 1);

    std::error_code fs_ec;
    const auto size = std::filesystem::file_size(path, fs_ec);
    g.engine_id = hex64(fnv1a64(g.arch + ":" + g.model_hash + ":" +
                                std::to_string(fs_ec ? 0 : (uint64_t) size)));

    gguf_free(gguf);
    return g.rows > 0;
}

std::mutex g_mtx;
std::optional<geometry> g_geom;

} // namespace

bool init(const std::string & model_path) {
    if (model_path.empty()) return false;
    std::lock_guard<std::mutex> lock(g_mtx);
    if (g_geom) return true;
    geometry g;
    if (!load_impl(model_path, g)) return false;
    g_geom = std::move(g);
    return true;
}

std::optional<std::string> experts_json() {
    if (std::getenv("HYDRA_EXPERT_META") == nullptr) return std::nullopt;
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_geom) return std::nullopt;
    const geometry & g = *g_geom;

    const size_t cells = (size_t) g.rows * (size_t) g.cols;
    // Telemetry (Stage A) absent: honest zeros — tier 0 / heat 0 everywhere.
    std::string map_hex(cells * 2, '0');
    std::string hits_hex(((cells + 7) / 8) * 2, '0');

    const json j = {
        {"seq",               0},
        {"rows",              g.rows},
        {"cols",              g.cols},
        {"map",               map_hex},
        {"hits",              hits_hex},
        {"telemetry_enabled", false}, // Stage A flips this and fills map/hits
        {"geometry", {
            {"engine_id",      g.engine_id},
            {"model_hash",     g.model_hash},
            {"dense_prefix",   g.dense_prefix},
            {"moe_rows",       g.moe_rows},
            {"nextn_rows",     g.nextn_rows},
            {"n_expert_used",  g.n_expert_used},
        }},
    };
    return j.dump();
}

} // namespace hydra_atlas

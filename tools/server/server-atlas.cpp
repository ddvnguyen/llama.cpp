// hydra: implementation of the expert-atlas read surface (see server-atlas.h).
// Geometry source of truth = the model's own GGUF (arch KVs + per-layer expert
// tensor presence), so no architecture is hardcoded and the UI never guesses a
// layer mapping — the Colibri GLM `row+3`/MTP-78 hardcode dies here (design §D).

// hydra #785: allocation snapshot for honest tier/hwinfo reporting on engine
// /health. Uses the public llama_get_memory_breakdown() API (declared in
// ../src/llama-ext.h — a staging header, included here and NOT leaked into
// server-atlas.h, which only forward-declares llama_context). Device/host
// split: ggml_backend_buft_is_host() → ram; else vram (device-side model +
// KV + compute). disk = 0 always (nothing is disk-paged in this mode).
#include "server-atlas.h"

#include "../src/llama-ext.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

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

    // hydra #771: expert_count/expert_used_count live under the *model*
    // block namespace (e.g. qwen3next), not general.architecture — probe all
    // "*.<key>" KV names so a new MoE arch inherits the surface (design §B).
    auto kv_find_u32 = [&](const char * key, uint32_t & out) {
        for (int i = 0; i < gguf_get_n_kv(gguf); i++) {
            const char * kn = gguf_get_key(gguf, i);
            if (kn == nullptr || gguf_get_kv_type(gguf, i) != GGUF_TYPE_UINT32) continue;
            const char * dot = std::strrchr(kn, '.');
            if (dot != nullptr && std::strcmp(dot + 1, key) == 0) {
                out = gguf_get_val_u32(gguf, i);
                return true;
            }
        }
        return false;
    };
    uint32_t block_count = 0, expert_count = 0, expert_used = 0, nextn = 0;
    kv_find_u32("block_count", block_count);
    kv_find_u32("expert_count", expert_count);
    kv_find_u32("expert_used_count", expert_used);
    // NextN/MTP layers are optional (kv stays 0 when absent).
    kv_find_u32("nextn_predict_layers", nextn);

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
// Stage A counters. counts[] indexed gridRow*cols+expert; hits_step[] is the
// last-decode-step bitmap; seq++ once per counted decode step. Sized lazily
// on first accumulate (geometry known by then). Mutex = the file mutex.
std::vector<uint64_t> g_counts;
uint64_t g_seq = 0;
std::vector<uint8_t> g_hits_step;
bool g_stats_on = false;
bool g_stats_checked = false;
// hydra #785: load-time allocation snapshot for /health (sleep-safe copy).
alloc_info g_alloc;
bool g_alloc_ready = false;

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
bool enabled() {
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_stats_checked) {
        g_stats_on = std::getenv("HYDRA_EXPERT_STATS") != nullptr;
        g_stats_checked = true;
    }
    return g_stats_on;
}

void accumulate(int grid_row, const int32_t * ids, int n_ids, int cols) {
    if (ids == nullptr || n_ids <= 0 || cols <= 0 || grid_row < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_stats_on || !g_geom) {
        return;
    }
    const size_t cells = (size_t) g_geom->rows * (size_t) g_geom->cols;
    if (g_counts.size() != cells) {
        g_counts.assign(cells, 0);
        g_hits_step.assign((cells + 7) / 8, 0);
    }
    if (grid_row >= g_geom->rows) {
        return;
    }
    for (int i = 0; i < n_ids; ++i) {
        const int32_t id = ids[i];
        if (id < 0 || id >= cols || id >= g_geom->cols) {
            continue;
        }
        const size_t cell = (size_t) grid_row * (size_t) g_geom->cols + (size_t) id;
        g_counts[cell] += 1;
        g_hits_step[cell >> 3] |= (uint8_t) (1u << (cell & 7));
    }
}

void end_step(int rows, int cols) {
    (void) rows;
    (void) cols;
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_stats_on) {
        return;
    }
    g_seq += 1; // one counted decode step
}

std::vector<int> trunk_rows() {
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_geom) return {};
    return g_geom->moe_rows;
}

int grid_cols() {
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_geom) return 0;
    return g_geom->cols;
}

int expert_used() {
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_geom) return 0;
    return g_geom->n_expert_used;
}
void reset() {
    std::lock_guard<std::mutex> lock(g_mtx);
    g_counts.clear();
    g_hits_step.clear();
    g_seq = 0;
}

// hydra #785: one-shot load-time capture. Breakdown split mirrors
// common/memory_breakdown_print (fit.cpp:938-965): host buft → ram,
// device buft → vram. Device totals via ggml_backend_dev_memory; host
// totals via /proc/meminfo (Linux) with CPU-device fallback. Never throws:
// on any failure the snapshot stays partial and have_tiers=false so the
// /health handler omits tiers instead of fabricating them.
void snapshot_alloc(const struct llama_context * ctx) {
    alloc_info a;
    a.cores = (int) std::thread::hardware_concurrency();
    try {
        const size_t ndev = ggml_backend_dev_count();
        for (size_t i = 0; i < ndev; i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!dev) continue;
            const auto type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                size_t free = 0, total = 0;
                ggml_backend_dev_memory(dev, &free, &total);
                if (total > 0) {
                    a.gpus++;
                    a.vram_total += (uint64_t) total;
                    if (a.gpu.empty()) {
                        const char * d = ggml_backend_dev_description(dev);
                        if (d) a.gpu = d;
                    }
                }
            }
        }
    } catch (...) {}
#if !defined(_WIN32)
    try {
        std::ifstream mi("/proc/meminfo");
        std::string k;
        unsigned long long v = 0;
        std::string u;
        while (mi >> k >> v >> u) {
            if (k == "MemTotal:") a.ram_total = v * 1024ULL;
            else if (k == "MemAvailable:") a.ram_avail = v * 1024ULL;
            if (a.ram_total && a.ram_avail) break;
        }
    } catch (...) {}
    if (a.cpu.empty()) {
        try {
            std::ifstream ci("/proc/cpuinfo");
            std::string line;
            while (std::getline(ci, line)) {
                if (line.compare(0, 10, "model name") == 0) {
                    const auto p = line.find(':');
                    if (p != std::string::npos) {
                        a.cpu = line.substr(p + 1);
                        while (!a.cpu.empty() && (a.cpu.front() == ' ' || a.cpu.front() == '\t')) a.cpu.erase(a.cpu.begin());
                    }
                    break;
                }
            }
        } catch (...) {}
    }
#endif
    if (a.ram_total == 0) {
        try {
            ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (cpu) {
                size_t free = 0, total = 0;
                ggml_backend_dev_memory(cpu, &free, &total);
                a.ram_total = (uint64_t) total;
                a.ram_avail = (uint64_t) free;
            }
        } catch (...) {}
    }
    if (ctx != nullptr) {
        try {
            const llama_memory_breakdown bd = llama_get_memory_breakdown(ctx);
            uint64_t vram = 0, ram = 0;
            for (const auto & kv : bd) {
                const uint64_t self = (uint64_t) kv.second.model +
                                      (uint64_t) kv.second.context +
                                      (uint64_t) kv.second.compute;
                if (ggml_backend_buft_is_host(kv.first)) ram += self;
                else vram += self;
            }
            if (!bd.empty()) {
                a.vram_bytes = vram;
                a.ram_bytes  = ram;
                a.have_tiers = true;
            }
        } catch (...) {}
    }
    std::lock_guard<std::mutex> lock(g_mtx);
    g_alloc = a;
    g_alloc_ready = true;
}

bool health_snapshot(alloc_info & out) {
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_alloc_ready) return false;
    out = g_alloc;
    return true;
}



std::optional<std::string> experts_json() {
    if (std::getenv("HYDRA_EXPERT_META") == nullptr) return std::nullopt;
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_geom) return std::nullopt;
    const geometry & g = *g_geom;

    const size_t cells = (size_t) g.rows * (size_t) g.cols;
    const bool live = g_stats_on && g_counts.size() == cells;

    // Colibri EMAP byte encoding verbatim (tier=byte>>6, heat=byte&63).
    // Tier bits are PROVISIONAL (design §8 honesty rule): our fork has no
    // disk tier, so counted experts report tier 1 (RAM-resident, the truth
    // until hybrid residency exists); never-counted cells stay tier 0/heat
    // 0 (honest zeros). Heat saturates at 63 (EMAP encoding).
    auto to_hex = [](const uint8_t * data, size_t n) {
        static const char * digits = "0123456789abcdef";
        std::string s;
        s.resize(n * 2);
        for (size_t i = 0; i < n; i++) {
            s[2 * i]     = digits[(data[i] >> 4) & 0xF];
            s[2 * i + 1] = digits[data[i] & 0xF];
        }
        return s;
    };
    std::string map_hex;
    std::string hits_hex;
    if (live) {
        std::vector<uint8_t> bytes(cells);
        for (size_t i = 0; i < cells; i++) {
            const uint64_t c = g_counts[i];
            bytes[i] = c == 0 ? 0 : (uint8_t) ((1 << 6) | (c > 63 ? 63 : (int) c));
        }
        map_hex = to_hex(bytes.data(), bytes.size());
        hits_hex = to_hex(g_hits_step.data(), g_hits_step.size());
    } else {
        map_hex.assign(cells * 2, '0');
        hits_hex.assign(((cells + 7) / 8) * 2, '0');
    }

    const json j = {
        {"seq",               (int) g_seq},
        {"rows",              g.rows},
        {"cols",              g.cols},
        {"map",               map_hex},
        {"hits",              hits_hex},
        {"telemetry_enabled", live},
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

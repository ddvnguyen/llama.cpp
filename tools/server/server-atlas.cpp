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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
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

    // hydra fix (layer-40 finding, track t-34d7bdf5d6): with NextN/MTP models,
    // llama-model semantics count the nextn layer INSIDE block_count
    // (n_layer_all = block_count; trunk n_layer = block_count - nextn).
    // The nextn tensors live in blk.<trunk> (e.g. blk.40.nextn.* alongside
    // blk.40.ffn_*_exps), so the last block is the MTP draft layer — built
    // only in the MTP context graph, never in ctx_tgt where the topk hook
    // reads. Classify those rows as nextn_rows (honest: no target-graph
    // routing) instead of trunk rows that read as permanently unrouted.
    const int trunk = (int) block_count - (int) nextn;
    for (int il = 0; il < trunk; il++) {
        if (moe[il]) g.moe_rows.push_back(il);
    }
    for (int il = trunk; il < (int) (block_count + nextn); il++) {
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
// hydra #787 S-C3: EAN state. gxn[] = sum(g*norm) per cell, nsel[] =
// selection count per cell; S = gxn/nsel. Sized lazily like g_counts.
// Mutex = the file mutex.
std::vector<double> g_ean_gxn;
std::vector<uint64_t> g_ean_nsel;
bool g_ean_on = false;
bool g_ean_checked = false;
// hydra #787: per-turn capture. turn_seq++ once per completed completion
// cycle (record_turn, slot-aware). g_turn_base = counts snapshot at the last
// turn boundary; each turn stores the sparse window diff (no new graph
// hooks). g_turn_hits ORs every step bitmap since the last boundary, so a
// turn keeps its window hits even though g_hits_step only holds the latest
// step. Ring capped at kTurnCap; evicted turns 404 honestly. Window-diff
// caveat: with parallel slots the shared counters attribute to whichever
// turn ends first — the slice is a window, not slot-isolated.
uint64_t g_turn_seq = 0;
std::deque<turn_record> g_turns;
std::vector<uint64_t> g_turn_base;
std::vector<uint8_t> g_turn_hits;

std::string hex_of(const uint8_t * data, size_t n) {
    static const char * digits = "0123456789abcdef";
    std::string s;
    s.resize(n * 2);
    for (size_t i = 0; i < n; i++) {
        s[2 * i]     = digits[(data[i] >> 4) & 0xF];
        s[2 * i + 1] = digits[data[i] & 0xF];
    }
    return s;
}
// hydra #788: per-layer expert residency for honest EMAP tier bits. Load
// fact (not probe state): survives reset(). Set once by set_residency()
// from server params at model load; encoders fall back to tier 0 when
// unknown (never guess). g_res_cpu[il]=1 → layer il's experts on CPU.
int g_res_ngl = -1;
int g_res_nlayer = 0;
std::vector<char> g_res_cpu;
bool g_res_ready = false;
// hydra #785: load-time allocation snapshot for /health (sleep-safe copy).
alloc_info g_alloc;
bool g_alloc_ready = false;
// hydra #786 Edge0 S-A2: hidden-state capture state. Sized lazily on first
// capture_hidden (geometry known by then). Mutex = the file mutex.
bool g_cap_on = false;
bool g_cap_checked = false;
std::string g_cap_outdir;
struct cap_entry {
    std::vector<float>  hidden;
    std::vector<int32_t> topk;
};
std::vector<std::vector<cap_entry>> g_cap_data;
int g_cap_k = 0;
int g_cap_embd = 0;
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
    // hydra #787: fold this step into the current turn window. Cheap
    // (bytes = cells/8) and lock-local; cleared at each record_turn.
    if (g_turn_hits.size() != g_hits_step.size()) {
        g_turn_hits.assign(g_hits_step.size(), 0);
    }
    for (size_t i = 0; i < g_hits_step.size(); i++) {
        g_turn_hits[i] |= g_hits_step[i];
    }
}

bool ean_enabled() {
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_ean_checked) {
        g_ean_on = std::getenv("HYDRA_EAN_STATS") != nullptr;
        g_ean_checked = true;
    }
    return g_ean_on;
}

void accumulate_ean(int grid_row, const int32_t * ids, const float * gates, const float * norms, int n, int cols) {
    if (ids == nullptr || gates == nullptr || norms == nullptr || n <= 0 || cols <= 0 || grid_row < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_ean_on || !g_geom) {
        return;
    }
    const size_t cells = (size_t) g_geom->rows * (size_t) g_geom->cols;
    if (g_ean_gxn.size() != cells) {
        g_ean_gxn.assign(cells, 0.0);
        g_ean_nsel.assign(cells, 0);
    }
    if (grid_row >= g_geom->rows) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        const int32_t id = ids[i];
        if (id < 0 || id >= cols || id >= g_geom->cols) {
            continue;
        }
        const float g = gates[i];
        const float v = norms[i];
        if (!(g >= 0.0f) || !(v >= 0.0f)) {
            continue; // NaN/negative guard: never poison the mean
        }
        if (!std::isfinite(g) || !std::isfinite(v)) {
            continue;
        }
        const size_t cell = (size_t) grid_row * (size_t) g_geom->cols + (size_t) id;
        g_ean_gxn[cell] += (double) g * (double) v;
        g_ean_nsel[cell] += 1;
    }
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
    // hydra #787: rebase the turn window too (next delta diffs vs zeros);
    // recorded turn history survives probe resets.
    g_turn_base.clear();
    g_turn_hits.clear();
    // hydra #787 S-C3: EAN joins the per-probe reset (no cross-probe leak).
    g_ean_gxn.clear();
    g_ean_nsel.clear();
    // hydra #786 Edge0 S-A2: capture sidecar joins the per-probe reset.
    for (auto & v : g_cap_data) {
        v.clear();
    }
    g_cap_data.clear();
    g_cap_k = 0;
    g_cap_embd = 0;
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

// hydra #788: match one --n-cpu-moe / --cpu-moe / fit-overflow pattern
// against "blk.<il>.<rest>". Patterns are regex fragments ("blk\\.3\\.ffn_"
// or "blk\\.\\d+\\.ffn_...exps"); the all-layers exps pattern has no digit
// after "blk\\." so it matches every il. Returns cpu iff some pattern hits.
bool res_pat_cpu(const std::string & pat, int il) {
    const char * b = std::strstr(pat.c_str(), "blk");
    if (b == nullptr) return false; // not a layer override — ignore
    const char * bs = std::strchr(b, '\\');
    if (bs == nullptr) return true; // "blk.N..." unescaped — treat as all-layer
    // "blk\\." (literal dot) vs "blk\\d" (digit class): digit class with no
    // literal layer number matches every layer (the --cpu-moe all-exps case).
    if (bs[1] == '.') {
        // literal "blk.<N>.": parse N after the escaped dot.
        const char * p = bs + 2;
        if (!std::isdigit((unsigned char) *p)) return true; // "blk\.<rest>" — all
        return std::atoi(p) == il;
    }
    return true; // "blk\\d..." — layer class, matches all
}

void set_residency(int n_gpu_layers, int n_layer,
                   const std::vector<std::string> & cpu_patterns) {
    std::lock_guard<std::mutex> lock(g_mtx);
    // hydra #788 fix: n_layer arrives as llama_model_n_layer() = trunk only
    // (excludes NextN). Geometry rows may still reference il == trunk (the
    // MTP expert row). Size the table to cover every geometry row so no
    // trunk row falls off the end into the tier-0 fallback.
    int need = n_layer;
    if (g_geom) {
        for (int il : g_geom->moe_rows) need = std::max(need, il + 1);
        for (int il : g_geom->nextn_rows) need = std::max(need, il + 1);
    }
    g_res_ngl = n_gpu_layers;
    g_res_nlayer = need;
    g_res_cpu.assign(need > 0 ? (size_t) need : 0, 0);
    for (int il = 0; il < need; il++) {
        // Rule 1: above the device window → CPU. i_gpu_start mirrors
        // llama-model.cpp:1502 (back-to-front offload; +1 = output layer).
        // The trunk window uses the trunk count; the NextN row inherits the
        // trunk verdict (same weights, llama-model.cpp:2317 filter).
        if (n_gpu_layers >= 0 && n_layer > 0 &&
            il < n_layer + 1 - std::min(n_gpu_layers, n_layer + 1)) {
            g_res_cpu[(size_t) il] = 1;
            continue;
        }
        // Rule 2: a CPU override pattern names this layer's experts.
        for (const auto & pat : cpu_patterns) {
            if (res_pat_cpu(pat, il)) {
                g_res_cpu[(size_t) il] = 1;
                break;
            }
        }
    }
    g_res_ready = need > 0;
}

// hydra #788: shared EMAP encoder (Colibri byte layout verbatim:
// tier=byte>>6, heat=byte&63). Tier is a LOAD fact from set_residency();
// heat is the routing count clamped to 63. Disk unused (tier 0 = unknown,
// honest omit). Applies to every cell: never-routed experts are still
// resident, so their tier is real with heat 0. Callers hold g_mtx.
uint8_t encode_cell(const geometry & g, size_t row, uint64_t count) {
    uint8_t tier = 0;
    const size_t nrows = g.moe_rows.size() + g.nextn_rows.size();
    if (g_res_ready && row < nrows) {
        const size_t nr = g.moe_rows.size();
        const int il = row < nr ? g.moe_rows[row]
                               : g.nextn_rows[row - nr];
        if (il >= 0 && (size_t) il < g_res_cpu.size()) {
            tier = g_res_cpu[(size_t) il] ? 1 : 2;
        }
    }
    return (uint8_t) ((tier << 6) | (count > 63 ? 63 : (int) count));
}

std::optional<std::string> experts_json() {
    if (std::getenv("HYDRA_EXPERT_META") == nullptr) return std::nullopt;
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_geom) return std::nullopt;
    const geometry & g = *g_geom;

    const size_t cells = (size_t) g.rows * (size_t) g.cols;
    const bool live = g_stats_on && g_counts.size() == cells;

    // Colibri EMAP byte encoding verbatim (tier=byte>>6, heat=byte&63).
    // hydra #788: tier is a LOAD fact (encode_cell ← set_residency), heat
    // is the routing count clamped to 63. Every live cell carries its real
    // tier (never-routed still resident, heat 0); OFF state stays honest
    // zeros (tier unknown until a load captures residency).
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
            bytes[i] = encode_cell(g, i / (size_t) g.cols, g_counts[i]);
        }
        map_hex = to_hex(bytes.data(), bytes.size());
        hits_hex = to_hex(g_hits_step.data(), g_hits_step.size());
    } else {
        map_hex.assign(cells * 2, '0');
        hits_hex.assign(((cells + 7) / 8) * 2, '0');
    }

    // hydra #787 S-C3: EAN saliency section. Served only when EAN counted
    // this run (own env HYDRA_EAN_STATS, accumulators sized); otherwise the
    // key is absent (never null-fabricated). Keys are real layer indices
    // "<realLayer>:<expert>" so offline consumers merge without geometry.
    json ean_j = nullptr;
    const bool ean_live = g_ean_on && g_ean_gxn.size() == cells && g_ean_nsel.size() == cells;
    if (ean_live) {
        json sal = json::object();
        uint64_t ean_cells = 0;
        for (size_t r = 0; r < g.moe_rows.size(); ++r) {
            for (int e = 0; e < g.cols; ++e) {
                const size_t cell = r * (size_t) g.cols + (size_t) e;
                const uint64_t n = g_ean_nsel[cell];
                if (n == 0) {
                    continue;
                }
                sal[std::to_string(g.moe_rows[r]) + ":" + std::to_string(e)] =
                    g_ean_gxn[cell] / (double) n;
                ++ean_cells;
            }
        }
        ean_j = {
            {"enabled",      true},
            {"source",       "live-EAN"},
            {"decode_only",  true},
            {"mtp_excluded", true},
            {"cells",        ean_cells},
            {"model",        g.model_hash},
            {"engine_id",    g.engine_id},
            {"saliency",     std::move(sal)},
        };
    }

    json j = {
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
    if (ean_live) {
        j["ean"] = std::move(ean_j);
    }
    return j.dump();
}

// hydra #787 sub-task 1: per-turn capture. Called once per completed
// completion cycle from send_final_response (server-context.cpp), gated on
// the same HYDRA_EXPERT_STATS flag as the counters — OFF = no-op.
void record_turn(int slot, double wall_s, uint64_t prompt_tokens,
                 uint64_t completion_tokens, uint64_t forwards) {
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_stats_on) {
        return;
    }
    if (g_turn_base.size() != g_counts.size()) {
        g_turn_base.assign(g_counts.size(), 0); // first turn / post-reset
    }
    turn_record t;
    t.turn_seq          = ++g_turn_seq;
    t.ts                = (int64_t) std::time(nullptr);
    t.slot              = slot;
    t.cols              = g_geom ? g_geom->cols : 0;
    t.wall_s            = wall_s;
    t.prompt_tokens     = prompt_tokens;
    t.completion_tokens = completion_tokens;
    t.forwards          = forwards;
    for (size_t i = 0; i < g_counts.size(); i++) {
        const uint64_t base = g_turn_base[i];
        const uint64_t cur  = g_counts[i];
        const uint64_t d = cur >= base ? cur - base : cur; // clamp post-reset
        if (d > 0) {
            t.routing.push_back({(uint32_t) i, d});
        }
    }
    g_turn_base = g_counts;
    t.hits = g_turn_hits;
    g_turn_hits.assign(g_hits_step.size(), 0);
    g_turns.push_back(std::move(t));
    while (g_turns.size() > kTurnCap) {
        g_turns.pop_front();
    }
}

std::optional<std::string> profile_json() {
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_stats_on) {
        return std::nullopt;
    }
    json turns = json::array();
    for (const auto & t : g_turns) {
        // Upstream ProfileTurn shape verbatim; phase timings stay 0.0 until
        // a collision-proposal sign-off instruments them (honest zeros —
        // the Profiling tab renders the full wall time as "other").
        turns.push_back({
            {"turn_seq",         t.turn_seq},
            {"ts",               t.ts},
            {"slot",             t.slot},
            {"wall_s",           t.wall_s},
            {"prompt_tokens",    t.prompt_tokens},
            {"completion_tokens", t.completion_tokens},
            {"expert_disk_s",    0.0},
            {"expert_wait_s",    0.0},
            {"expert_matmul_s",  0.0},
            {"attention_s",      0.0},
            {"lm_head_s",        0.0},
            {"forwards",         t.forwards},
        });
    }
    return json({{"seq", g_turn_seq}, {"turns", std::move(turns)}}).dump();
}

std::optional<std::string> turns_json() {
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_stats_on) {
        return std::nullopt;
    }
    json turns = json::array();
    for (const auto & t : g_turns) {
        turns.push_back({
            {"turn_seq",         t.turn_seq},
            {"ts",               t.ts},
            {"slot",             t.slot},
            {"wall_s",           t.wall_s},
            {"prompt_tokens",    t.prompt_tokens},
            {"completion_tokens", t.completion_tokens},
            {"forwards",         t.forwards},
        });
    }
    return json({{"seq", g_turn_seq}, {"turns", std::move(turns)}}).dump();
}

std::optional<std::string> turn_json(uint64_t seq) {
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_stats_on) {
        return std::nullopt;
    }
    const turn_record * hit = nullptr;
    for (const auto & t : g_turns) {
        if (t.turn_seq == seq) {
            hit = &t;
            break;
        }
    }
    if (hit == nullptr) {
        return std::nullopt; // evicted or never recorded — caller 404s
    }
    json routing = json::array();
    for (const auto & c : hit->routing) {
        json cell = {{"count", c.count}};
        if (hit->cols > 0) {
            cell["row"]    = (uint32_t) (c.cell / (uint32_t) hit->cols);
            cell["expert"] = (uint32_t) (c.cell % (uint32_t) hit->cols);
        } else {
            cell["cell"] = c.cell;
        }
        routing.push_back(std::move(cell));
    }
    return json({
        {"turn_seq",         hit->turn_seq},
        {"ts",               hit->ts},
        {"slot",             hit->slot},
        {"wall_s",           hit->wall_s},
        {"prompt_tokens",    hit->prompt_tokens},
        {"completion_tokens", hit->completion_tokens},
        {"forwards",         hit->forwards},
        {"cols",             hit->cols},
        {"routing",          std::move(routing)},
        {"hits",             hex_of(hit->hits.data(), hit->hits.size())},
    }).dump();
}

std::optional<std::string> experts_json_at(uint64_t seq) {
    if (std::getenv("HYDRA_EXPERT_META") == nullptr) {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_stats_on || !g_geom) {
        return std::nullopt;
    }
    const geometry & g = *g_geom;
    const size_t cells = (size_t) g.rows * (size_t) g.cols;
    const turn_record * hit = nullptr;
    for (const auto & t : g_turns) {
        if (t.turn_seq == seq) {
            hit = &t;
            break;
        }
    }
    if (hit == nullptr) {
        return std::nullopt; // evicted or never recorded — caller 404s
    }
    // Cumulative reconstruction from retained sparse deltas (deque is in
    // turn_seq order, so stop at seq). Delta computed client-side by the
    // UI via sparse subtraction vs N-1.
    std::vector<uint64_t> cum(cells, 0);
    for (const auto & t : g_turns) {
        if (t.turn_seq > seq) {
            break;
        }
        for (const auto & c : t.routing) {
            if (c.cell < cells) {
                cum[c.cell] += c.count;
            }
        }
    }
    std::vector<uint8_t> bytes(cells);
    for (size_t i = 0; i < cells; i++) {
        bytes[i] = encode_cell(g, i / (size_t) g.cols, cum[i]); // hydra #788: same load-fact tiers
    }
    const std::string map_hex = hex_of(bytes.data(), bytes.size());
    std::string hits_hex;
    if (!hit->hits.empty()) {
        hits_hex = hex_of(hit->hits.data(), hit->hits.size());
    } else {
        hits_hex.assign(((cells + 7) / 8) * 2, '0');
    }
    const json j = {
        {"seq",               (int) g_seq},
        {"turn_seq",          seq},
        {"rows",              g.rows},
        {"cols",              g.cols},
        {"map",               map_hex},
        {"hits",              hits_hex},
        {"telemetry_enabled", true},
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

// hydra #786 Edge0 S-A2: hidden-state capture for linear-probe prerouter.
// Env-gated (HYDRA_EXPERT_CAPTURE, checked once); OFF = all no-ops.
// Accumulates hidden states + topk per decode step; flushed to per-probe
// sidecar JSONs by flush_sidecar(); cleared by capture_reset().
// State variables live in the hydra_atlas anonymous namespace above.

bool capture_enabled() {
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_cap_checked) {
        g_cap_on = std::getenv("HYDRA_EXPERT_CAPTURE") != nullptr;
        g_cap_checked = true;
    }
    return g_cap_on;
}

void capture_hidden(int grid_row, const float * hidden, int n_embd,
                    const int32_t * topk, int k) {
    if (hidden == nullptr || n_embd <= 0 || grid_row < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_cap_on || !g_geom) {
        return;
    }
    const size_t nrows = g_geom->moe_rows.size() + g_geom->nextn_rows.size();
    if ((size_t) grid_row >= nrows) {
        return;
    }
    if (g_cap_data.size() != nrows) {
        g_cap_data.resize(nrows);
    }
    if (g_cap_k == 0 && k > 0) {
        g_cap_k = k;
    }
    if (g_cap_embd == 0 && n_embd > 0) {
        g_cap_embd = n_embd;
    }
    cap_entry e;
    e.hidden.assign(hidden, hidden + n_embd);
    if (topk != nullptr && k > 0) {
        e.topk.assign(topk, topk + k);
    }
    g_cap_data[(size_t) grid_row].push_back(std::move(e));
}

void set_capture_outdir(const std::string & dir) {
    std::lock_guard<std::mutex> lock(g_mtx);
    g_cap_outdir = dir;
}

std::optional<std::string> flush_sidecar(const std::string & probe_cat,
                                         int probe_idx) {
    std::lock_guard<std::mutex> lock(g_mtx);
    if (!g_cap_on || !g_geom) {
        return std::nullopt;
    }
    if (g_cap_data.empty()) {
        return std::nullopt;
    }
    // Build sidecar JSON matching analyze_edge0.py input contract:
    // {category, idx, layers: {layer_idx: {hidden: [[...]], topk: [[...]]}}}
    json layers = json::object();
    const auto & moe = g_geom->moe_rows;
    for (size_t r = 0; r < moe.size() && r < g_cap_data.size(); ++r) {
        const auto & entries = g_cap_data[r];
        if (entries.empty()) {
            continue;
        }
        json hidden_arr = json::array();
        json topk_arr = json::array();
        for (const auto & e : entries) {
            json h_row = json::array();
            for (float v : e.hidden) {
                h_row.push_back(v);
            }
            hidden_arr.push_back(std::move(h_row));
            json t_row = json::array();
            for (int32_t id : e.topk) {
                t_row.push_back(id);
            }
            topk_arr.push_back(std::move(t_row));
        }
        layers[std::to_string(moe[r])] = {
            {"hidden", std::move(hidden_arr)},
            {"topk",   std::move(topk_arr)},
        };
    }
    if (layers.empty()) {
        return std::nullopt;
    }
    json sidecar = {
        {"category", probe_cat},
        {"idx",      probe_idx},
        {"layers",   std::move(layers)},
    };
    // Write sidecar file
    std::string outdir = g_cap_outdir.empty() ? "." : g_cap_outdir;
    std::error_code ec;
    std::filesystem::create_directories(outdir, ec);
    const std::string fname = outdir + "/" + probe_cat + "_" +
                              std::to_string(probe_idx) + "_sidecar.json";
    std::ofstream ofs(fname);
    if (!ofs.is_open()) {
        return std::nullopt;
    }
    ofs << sidecar.dump();
    ofs.close();
    return fname;
}

void capture_reset() {
    std::lock_guard<std::mutex> lock(g_mtx);
    for (auto & v : g_cap_data) {
        v.clear();
    }
    g_cap_data.clear();
    g_cap_k = 0;
    g_cap_embd = 0;
}

} // namespace hydra_atlas

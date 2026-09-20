#pragma once

// hydra: Colibri expert-atlas read surface (hydra_vortex#771 / fork#133).
// Fork-local file — epic/697-final-verify discipline: reads model geometry
// from the model's own GGUF + public data only; no Store, no sessions, no
// Hydra business logic, no routing-policy change. HTTP + RPC 0x33 consumers
// get the same payload. Env gate: HYDRA_EXPERT_META (surface) — telemetry
// counts (Stage A, HYDRA_EXPERT_STATS) plug into experts_json() when they
// land; until then map/hits are honest zeros with telemetry_enabled=false.

// Forward declaration — the snapshot only needs an opaque handle; the
// implementation (server-atlas.cpp) includes llama.h / llama-ext.h.
struct llama_context;

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace hydra_atlas {

struct geometry {
    std::string engine_id;        // FNV-1a64(arch:name:size), 16 hex chars
    std::string model_hash;       // gguf file name — matches tools/atlas provenance
    std::string arch;             // gguf general.architecture
    int         rows = 0;         // grid rows = moe_rows.size() + nextn_rows.size()
    int         cols = 0;         // n_routed_experts
    int         dense_prefix = 0; // layers before the first routed-expert layer
    std::vector<int> moe_rows;    // real layer idx per trunk grid row
    std::vector<int> nextn_rows;  // trailing NextN/MTP grid rows (MoE only)
    int         n_expert_used = 0;
};

// Parse geometry for the model file. Thread-safe; caches on first success.
bool init(const std::string & model_path);
// Stage A accumulator (HYDRA_EXPERT_STATS). Called once per decode step from
// the decode thread with the routed-expert ids of that step, keyed by grid
// row (trunk moe_rows order; NextN/MTP rows never passed). Thread-safe.
// No-op unless enabled() (env read once at startup).
bool enabled();
void accumulate(int grid_row, const int32_t * ids, int n_ids, int cols);
void end_step(int rows, int cols);
// Trunk geometry snapshot for the Stage A decode hook (copies under lock;
// empty when geometry unavailable). reset() clears counts/seq/hits_step for
// the per-probe reset protocol (confound control); thread-safe.
std::vector<int> trunk_rows();
int grid_cols();
int expert_used();
void reset();
// Allocation snapshot for honest tier/hwinfo reporting on engine /health
// (#785). Captured once at model-load time (sleep-safe: get_health reads the
// stored copy, never ctx_server). Tiers are device/host buffer splits from
// llama_get_memory_breakdown(); when unavailable have_tiers=false and the
// /health handler omits tiers rather than fabricating them.
struct alloc_info {
    bool        have_tiers  = false; // breakdown captured; else omit tiers
    uint64_t    vram_bytes  = 0;     // device-side model + KV + compute
    uint64_t    ram_bytes   = 0;     // host-side buffers
    int         gpus        = 0;     // GPU/IGPU device count
    uint64_t    vram_total  = 0;     // sum of GPU device totals
    uint64_t    ram_total   = 0;     // host total
    uint64_t    ram_avail   = 0;     // host available (best effort)
    int         cores       = 0;     // hardware concurrency
    std::string cpu;                // host CPU model (may be empty)
    std::string gpu;                // first GPU description (may be empty)
};
void snapshot_alloc(const struct llama_context * ctx);
bool health_snapshot(alloc_info & out);
// Stage B payload (Colibri EMAP encoding verbatim: tier=byte>>6, heat=byte&63).
// nullopt when disabled (env unset or geometry unavailable) — callers no-op.
std::optional<std::string> experts_json();

} // namespace hydra_atlas

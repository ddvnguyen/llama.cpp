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

#include <cstddef>
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
// empty when geometry unavailable). reset() clears counts/seq/hits_step +
// EAN state for the per-probe reset protocol (confound control);
// thread-safe.
std::vector<int> trunk_rows();
int grid_cols();
int expert_used();
void reset();
// Stage C EAN accumulator (#787 S-C3, HYDRA_EAN_STATS). Called from the
// decode hook alongside accumulate() with the same expert ids plus the
// per-slot router gate weights and unweighted expert-output L2 norms of
// this step. Saliency per (layer, expert): S = sum(g*norm)/n_sel (REAP
// paper §4 Eq. 9). No-op unless ean_enabled() (env read once at startup).
// Served inside experts_json() under "ean" (saliency keyed by real layer:
// "<realLayer>:<expert>"); absent until counted. Thread-safe.
bool ean_enabled();
void accumulate_ean(int grid_row, const int32_t * ids, const float * gates, const float * norms, int n, int cols);
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
// Per-expert EMAP tier residency (hydra_vortex#788). Captured once at
// model-load time from server params (plain PODs only — no llama/common
// types leak into this header): user n_gpu_layers, trunk layer count
// (llama_model_n_layer), and the CPU-targeted tensor-buffer override
// patterns (--n-cpu-moe / --cpu-moe / fit overflow). Layer experts on
// device → tier 2, CPU-offloaded → tier 1, Disk unused (no expert paging).
// Unknown (e.g. auto-fit ngl<0) → per-row tier 0 (honest omit, never guess).
// Survives reset() (load fact, not probe state); thread-safe.
void set_residency(int n_gpu_layers, int n_layer,
                   const std::vector<std::string> & cpu_patterns);
// Stage B payload (Colibri EMAP encoding verbatim: tier=byte>>6, heat=byte&63).
// nullopt when disabled (env unset or geometry unavailable) — callers no-op.
std::optional<std::string> experts_json();
// Per-turn capture (hydra_vortex#787 sub-task 1). One turn = one completed
// completion cycle, recorded at send_final_response (slot-aware via slot id).
// Ring of the 256 most recent turns + snapshot-diff routing slices of the
// Stage-A counters at turn boundaries (no new graph hooks). Phase timings
// (attention vs expert-matmul vs lm_head) need graph-section events — they
// ship only after a collision proposal is signed off; until then the profile
// turns carry wall_s + forwards with phases as honest zeros (the Profiling
// tab renders the full wall time as "other"). Gated on HYDRA_EXPERT_STATS
// like the counters: OFF = record_turn no-ops, readers return nullopt.
constexpr size_t kTurnCap = 256;
struct turn_cell {
    uint32_t cell  = 0; // gridRow * cols + expert at record time
    uint64_t count = 0; // fires during this turn window
};
struct turn_record {
    uint64_t turn_seq         = 0; // monotonic per engine, starts at 1
    int64_t  ts               = 0; // unix seconds at turn end
    int      slot             = -1;
    int      cols             = 0; // grid cols at record time (decodes cell)
    double   wall_s           = 0.0;
    uint64_t prompt_tokens    = 0;
    uint64_t completion_tokens = 0;
    uint64_t forwards         = 0; // decode forward passes (n_gen_steps)
    std::vector<turn_cell> routing; // sparse fires during this turn window
    std::vector<uint8_t>   hits;    // OR of step bitmaps during this window
};
void record_turn(int slot, double wall_s, uint64_t prompt_tokens,
                 uint64_t completion_tokens, uint64_t forwards);
// Upstream-compatible GET /profile payload: {seq, turns:[ProfileTurn...]}.
// Extra turn_seq/ts/slot fields ride along; unknown phases read as 0.0.
// nullopt unless capture is enabled.
std::optional<std::string> profile_json();
// GET /turns payload: {seq, turns:[{turn_seq, ts, slot, wall_s,
// prompt_tokens, completion_tokens, forwards}...]}. nullopt unless enabled.
std::optional<std::string> turns_json();
// GET /turns/{seq}: full record incl. sparse routing slice ([{row, expert,
// count}]) and hits hex. nullopt unless enabled or seq evicted/unknown.
std::optional<std::string> turn_json(uint64_t seq);
// GET /experts?turn=N: map+hits as of turn N (cumulative reconstruction
// from retained sparse deltas; same EMAP encoding + geometry block as
// experts_json, plus turn_seq). nullopt unless the surface (META) and the
// capture (STATS) are on, geometry is known, and N is still retained.
std::optional<std::string> experts_json_at(uint64_t seq);

} // namespace hydra_atlas

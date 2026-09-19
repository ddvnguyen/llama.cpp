#pragma once

// hydra: Colibri expert-atlas read surface (hydra_vortex#771 / fork#133).
// Fork-local file — epic/697-final-verify discipline: reads model geometry
// from the model's own GGUF + public data only; no Store, no sessions, no
// Hydra business logic, no routing-policy change. HTTP + RPC 0x33 consumers
// get the same payload. Env gate: HYDRA_EXPERT_META (surface) — telemetry
// counts (Stage A, HYDRA_EXPERT_STATS) plug into experts_json() when they
// land; until then map/hits are honest zeros with telemetry_enabled=false.

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

// Stage B payload (Colibri EMAP encoding verbatim: tier=byte>>6, heat=byte&63).
// nullopt when disabled (env unset or geometry unavailable) — callers no-op.
std::optional<std::string> experts_json();

} // namespace hydra_atlas

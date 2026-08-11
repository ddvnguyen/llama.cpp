// Hydra A/B extension seam (epic #610).
//
// server-context.cpp consults this interface at a few well-defined points so
// Hydra-specific behavior can live in a fork-owned file (hydra-server-context.cpp)
// instead of being woven into upstream server-context.cpp. The concrete
// implementation is #include'd at the bottom of server-context.cpp (same
// translation unit) so it can reach server_context_impl's private members via
// the friend declaration on hydra_engine_extension.
//
// A/B toggle: the HYDRA_EXT_MODE env var selects which implementation drives
// Hydra behavior at runtime, so the SAME binary can be A/B tested:
//   HYDRA_EXT_MODE=seam   -> the extension (default since WS5)
//   HYDRA_EXT_MODE=legacy -> the inline Hydra code in server-context.cpp
// Both paths stay compiled; only one is consulted per run. WS4 diffs the same
// scenario through both modes to prove the refactor is behavior-identical.
#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>

struct server_context_impl;
struct server_task;

// True when HYDRA_EXT_MODE is NOT "legacy" (i.e. default = seam since WS5).
inline bool hydra_ext_mode_seam() {
    const char * m = std::getenv("HYDRA_EXT_MODE");
    return !(m && std::strcmp(m, "legacy") == 0);
}

struct server_hydra_extension {
    virtual ~server_hydra_extension() = default;

    // Human-readable name of the active implementation (for A/B logging/tests).
    virtual const char * name() const = 0;

    // Claim a task from process_single_task(). Return true if fully handled
    // (the default dispatch is skipped). The task must NOT be consumed when
    // returning false.
    virtual bool handle_task(server_context_impl & impl, server_task & task) = 0;

    // Called at the top of update_slots(). Return true to skip the default
    // decode loop for this pass (T3 rebuild / CONFIGURE / COMBINED reattach).
    virtual bool pre_loop(server_context_impl & impl) = 0;

    // Called when update_slots() finds batch.n_tokens == 0. Return true if the
    // empty-batch case was fully handled (STATE_GET transfer suppression).
    virtual bool on_empty_batch(server_context_impl & impl) = 0;
};

// Factory. Defined in hydra-server-context.cpp.
std::unique_ptr<server_hydra_extension> hydra_create_extension();

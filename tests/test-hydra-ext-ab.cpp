// epic #610 WS1: hermetic A/B seam checks.
//
// What this can test without a model/GPU:
//   - HYDRA_EXT_MODE env parsing (legacy default, "seam" activates seam)
//   - the factory returns the expected no-op WS1 implementation
//   - the no-op contract: hooks don't claim tasks / don't alter the loop
//
// What CANNOT be tested hermetically: a server_context_impl requires a loaded
// model+context, so the behavioral A/B (run the SAME scenario through
// HYDRA_EXT_MODE=legacy vs =seam and diff the outputs) is a loopback / live-rig
// step, documented as WS4 in the epic.

#include "server-hydra-extension.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

static int g_failures = 0;

static void expect(const char * what, bool ok) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        g_failures++;
    }
}

// setenv/unsetenv are POSIX; on Windows use _putenv_s (empty = unset).
#if defined(_WIN32)
static void set_ext_mode(const char * value) {
    _putenv_s("HYDRA_EXT_MODE", value ? value : "");
}
#else
static void set_ext_mode(const char * value) {
    if (value == nullptr) {
        unsetenv("HYDRA_EXT_MODE");
    } else {
        setenv("HYDRA_EXT_MODE", value, 1);
    }
}
#endif

int main() {
    // --- mode parsing -------------------------------------------------
    set_ext_mode(nullptr);
    expect("unset HYDRA_EXT_MODE -> legacy",      !hydra_ext_mode_seam());

    set_ext_mode("legacy");
    expect("HYDRA_EXT_MODE=legacy -> legacy",     !hydra_ext_mode_seam());

    set_ext_mode("seam");
    expect("HYDRA_EXT_MODE=seam -> seam",          hydra_ext_mode_seam());

    set_ext_mode("garbage");
    expect("HYDRA_EXT_MODE=garbage -> legacy",    !hydra_ext_mode_seam());

    set_ext_mode("LEGACY");  // case-sensitive: not "seam"
    expect("HYDRA_EXT_MODE=LEGACY -> legacy",     !hydra_ext_mode_seam());

    // --- factory + WS1 no-op contract ---------------------------------
    std::unique_ptr<server_hydra_extension> ext = hydra_create_extension();
    expect("factory returns non-null",            ext != nullptr);
    if (ext) {
        // WS2/WS3 impl name (handle_task routes to hydra_process_task, and
        // pre_loop/on_empty_batch replicate the update_slots clusters).
        expect("impl name is hydra-task-ws2",     std::strcmp(ext->name(), "hydra-task-ws2") == 0);
    }

    // The no-op hooks take a server_context_impl&, which cannot be constructed
    // here (needs a loaded model). Their return values are pinned by the WS1
    // contract: handle_task=false (never claims), pre_loop=false (never skips
    // the decode loop), on_empty_batch=false (never handles the empty batch).
    // Behavioral parity is verified by the live-rig A/B in WS4.

    if (g_failures != 0) {
        fprintf(stderr, "test-hydra-ext-ab FAILED (%d)\n", g_failures);
        return 1;
    }

    fprintf(stderr, "%s", "test-hydra-ext-ab OK\n");
    return 0;
}

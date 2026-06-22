// hydra#334: pins llama_hydra_clamp_state_chunk_size's behavior. This value is
// reachable from the network via CONFIGURE (0x40) "state_chunk_size", so the
// clamp bounds must not silently drift.
#include "llama-hydra.h"

#include <cstdio>

static int g_failures = 0;

static void expect_eq(const char * what, size_t actual, size_t expected) {
    if (actual != expected) {
        fprintf(stderr, "FAIL: %s — expected %zu, got %zu\n", what, expected, actual);
        g_failures++;
    }
}

int main() {
    const size_t k_min = 64ull * 1024;
    const size_t k_max = 64ull * 1024 * 1024;
    const size_t k_default = 2ull * 1024 * 1024;

    expect_eq("zero clamps to min",
        llama_hydra_clamp_state_chunk_size(0), k_min);
    expect_eq("below min clamps to min",
        llama_hydra_clamp_state_chunk_size(1024), k_min);
    expect_eq("exactly min passes through",
        llama_hydra_clamp_state_chunk_size(k_min), k_min);
    expect_eq("default (2 MiB) passes through unchanged",
        llama_hydra_clamp_state_chunk_size(k_default), k_default);
    expect_eq("exactly max passes through",
        llama_hydra_clamp_state_chunk_size(k_max), k_max);
    expect_eq("above max clamps to max",
        llama_hydra_clamp_state_chunk_size(k_max + 1), k_max);
    expect_eq("absurdly large clamps to max",
        llama_hydra_clamp_state_chunk_size((size_t)-1), k_max);

    if (g_failures == 0) {
        printf("test-hydra-state-chunk-size: all checks passed\n");
        return 0;
    }
    fprintf(stderr, "test-hydra-state-chunk-size: %d check(s) failed\n", g_failures);
    return 1;
}

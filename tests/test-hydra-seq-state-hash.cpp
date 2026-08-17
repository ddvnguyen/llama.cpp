// hydra#470: pins llama_state_seq_hash's byte count against
// llama_state_seq_get_size. The PREFILL M2 wire path pre-computes an XXH3
// hash of the seq state ([4B magic][4B seq_id] + KV state) BEFORE streaming,
// then the server guards `hashed != state_size`. Two bugs killed every M2
// request:
//   1. llama_io_write_hash::n_bytes() was a stub returning 0 — state_seq_hash
//      always reported 0 bytes hashed.
//   2. The server guard added sizeof(uint32_t)+sizeof(llama_seq_id) on top of
//      state_size even though llama_state_seq_get_size ALREADY counts the
//      8-byte header (llama_io_write_dummy) — so hashed(8+raw) never matched
//      the expected (16+raw).
// This test asserts the invariant both fixes rely on: hash byte count == size
// byte count for a live context with seeded KV cells (both == 8 + raw state).
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-hydra.h"
// XXH_IMPLEMENTATION compiles the xxhash implementation into this TU — the
// llama lib only declares these symbols (same pattern as server-context.cpp).
#define XXH_STATIC_LINKING_ONLY
#define XXH_IMPLEMENTATION
#include "../vendor/xxhash/xxhash.h"

#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    common_params params;

    params.sampling.seed = 1234;
    params.kv_unified = true;
    params.n_parallel = 1;
    params.n_ctx = 256;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    GGML_UNUSED(model);

    // Seed a few KV cells for seq 0 (2 tokens: a past token + a logits token).
    const std::vector<llama_token> tokens = { 1, 1 };
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        common_batch_add(batch, tokens[i], (int32_t) i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode seed tokens\n", __func__);
        llama_batch_free(batch);
        return 1;
    }
    llama_batch_free(batch);

    const size_t state_size = llama_state_seq_get_size(ctx, 0);
    if (state_size <= 8) {
        fprintf(stderr, "FAIL: state_size (%zu) must exceed the 8-byte wire header "
                        "(magic + seq_id) — KV cells were not seeded\n", state_size);
        return 1;
    }

    XXH3_state_t * hst = XXH3_createState();
    if (hst == nullptr) {
        fprintf(stderr, "FAIL: XXH3_createState returned null\n");
        return 1;
    }
    XXH3_64bits_reset(hst);

    const size_t hashed = llama_state_seq_hash(ctx, 0, hst);
    XXH3_freeState(hst);

    // The #470 invariant: the hash pass feeds exactly the bytes get_size counts
    // ([4B magic][4B seq_id] + KV state). Before fix 1, hashed was 0; before
    // fix 2 the server expected 8 + state_size, so 8+raw never matched 16+raw.
    if (hashed != state_size) {
        fprintf(stderr, "FAIL: llama_state_seq_hash returned %zu B but "
                        "llama_state_seq_get_size returned %zu B (expected equal)\n",
                hashed, state_size);
        return 1;
    }

    fprintf(stderr, "%s : PASS — hash %zu B == size %zu B (incl. 8-byte header)\n",
            __func__, hashed, state_size);
    return 0;
}
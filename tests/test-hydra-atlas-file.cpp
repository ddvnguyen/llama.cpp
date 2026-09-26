// hydra task-16ec332378: engine-hosted Stage-C atlas artifact routes
// (GET /experts.json + /expert-ranks.json). CPU-only unit test for
// hydra_atlas::atlas_file()/atlas_json(): synthetic 2-row MoE GGUF + the
// two shipped artifacts in a temp dir, exercising the engine-id refusal
// discipline end to end:
//   200 happy path (exact 16-hex FNV geometry id AND the
//        "$arch:$basename" short form from tools/atlas provenance),
//   404 artifact missing / wrong engine_id / provenance missing,
//   500 artifact present but not valid JSON.
// Refusal contract: a 404 body is an error JSON — never another model's
// artifact bytes, never a misleading 200.
#include "server-atlas.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <sys/stat.h>

namespace {

int failures = 0;

#define ATLAS_CHECK(cond, msg) do {                                     \
        if (cond) {                                                     \
            printf("  ok: %s\n", msg);                                  \
        } else {                                                        \
            printf("FAIL: %s (line %d)\n", msg, __LINE__);              \
            failures++;                                                 \
        }                                                               \
    } while (0)

std::string slurp(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void spit(const std::string & path, const std::string & body) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << body;
}

// Minimal MoE GGUF: metadata (arch, block_count, expert_count,
// expert_used_count) + one "blk.0.ffn_gate_exps" tensor so load_impl() sees
// a routed-expert layer. Tensor data is 16 zero bytes; only names matter
// for geometry.
bool write_moe_gguf(const std::string & path) {
    struct gguf_context * gguf = gguf_init_empty();
    if (!gguf) return false;
    gguf_set_val_str(gguf, "general.architecture", "testmoe38");
    gguf_set_val_u32(gguf, "testmoe38.block_count", 2);
    gguf_set_val_u32(gguf, "testmoe38.expert_count", 8);
    gguf_set_val_u32(gguf, "testmoe38.expert_used_count", 2);

    struct ggml_init_params ip = { /*mem_size*/ 1024*1024, /*mem_buffer*/ nullptr, /*no_alloc*/ false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { gguf_free(gguf); return false; }
    const int64_t ne[2] = { 4, 4 }; // tiny dense tensor (n_expert dim not enforced here)
    struct ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
    ggml_set_name(t, "blk.0.ffn_gate_exps");
    gguf_add_tensor(gguf, t);

    const bool ok = gguf_write_to_file(gguf, path.c_str(), /*only_meta*/ false);
    ggml_free(ctx);
    gguf_free(gguf);
    return ok;
}

// Expected geometry engine id, recomputed independently (FNV-1a64 over
// "arch:basename:summed_size", 16 hex) — mirrors server-atlas.cpp:186.
std::string expected_engine_id(const std::string & model_path, uint64_t total_size) {
    const std::string base = model_path.substr(model_path.find_last_of("/") + 1);
    const std::string s = "testmoe38:" + base + ":" + std::to_string(total_size);
    uint64_t h = 1469598103934665603ULL;
    for (const unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
    return buf;
}

} // namespace

int main() {
    const std::string dir = "/tmp/hydra-atlas-test-16ec332378";
    ::mkdir(dir.c_str(), 0755);
    const std::string model_path = dir + "/model.gguf";
    if (!write_moe_gguf(model_path)) {
        printf("FAIL: could not write synthetic MoE gguf\n");
        return 1;
    }
    const std::string model_id = expected_engine_id(model_path, slurp(model_path).size());

    // Artifact dir ≠ model dir: exercise the HYDRA_EXPERT_ATLAS override
    // (the model-dir shipping path needs no env at all).
    const std::string adir = dir + "/atlas";
    ::mkdir(adir.c_str(), 0755);
    setenv("HYDRA_EXPERT_ATLAS", adir.c_str(), 1);

    const std::string experts_path = adir + "/experts.json";
    const std::string ranks_path   = adir + "/expert-ranks.json";
    const std::string prov = "\"provenance\":{\"engine_id\":\"" + model_id + "\",\"commit\":\"test\"}";
    const std::string ranks_ok = "{\"version\":1," + prov + ",\"layers\":{}}";

    // ---- case 1+2: happy paths (exact FNV id on experts.json, short
    // "arch:basename" form on expert-ranks.json — both must serve 0/200)
    {
        std::string body = "{\"categories\":[\"coding\"]," + prov + ",\"experts\":{}}";
        spit(experts_path, body);
        std::string ranks = ranks_ok;
        spit(ranks_path, ranks);

        setenv("HYDRA_EXPERT_META", "1", 1); // unrelated gate; artifacts are env-free
        ATLAS_CHECK(hydra_atlas::init(model_path), "geometry init (synthetic MoE gguf)");

        std::string path, out;
        int st = hydra_atlas::atlas_file("experts", path, out);
        ATLAS_CHECK(st == 0, "case 1: exact FNV id serves the artifact (st=0)");
        ATLAS_CHECK(path == experts_path, "case 1: resolved path is the artifact path");
        ATLAS_CHECK(out == body, "case 1: body served byte-identical");
        ATLAS_CHECK(out.find("\"categories\"") != std::string::npos,
                    "case 1: served body is the artifact, not an error object");

        st = hydra_atlas::atlas_json("ranks", out);
        ATLAS_CHECK(st == 0, "case 2: short arch:basename id serves ranks (st=0)");
        ATLAS_CHECK(out == ranks_ok, "case 2: ranks body served byte-identical");
    }
    // ---- case 3: artifact missing → 404 with error body (never 200, never
    // leftover bytes from a previous serve)
    {
        std::remove(experts_path.c_str());
        std::string path, out;
        const int st = hydra_atlas::atlas_file("experts", path, out);
        ATLAS_CHECK(st == 404, "case 3: missing artifact → 404");
        ATLAS_CHECK(out.find("atlas artifact not found") != std::string::npos,
                    "case 3: 404 body names the missing artifact");
        ATLAS_CHECK(out.find("categories") == std::string::npos,
                    "case 3: no artifact bytes leak on 404");
    }

    // ---- case 4: wrong engine_id → 404 refusal (another model's atlas must
    // never be served, engine-id refusal discipline)
    {
        const std::string foreign =
            "{\"categories\":[],\"provenance\":{\"engine_id\":\"othermodel\"},\"experts\":{}}";
        spit(experts_path, foreign);
        std::string path, out;
        const int st = hydra_atlas::atlas_file("experts", path, out);
        ATLAS_CHECK(st == 404, "case 4: wrong engine_id → 404 (refused)");
        ATLAS_CHECK(out.find("refused") != std::string::npos,
                    "case 4: refusal body says 'refused'");
        ATLAS_CHECK(out.find("othermodel") != std::string::npos,
                    "case 4: refusal names the offending id");
        ATLAS_CHECK(out.find("categories") == std::string::npos,
                    "case 4: foreign artifact bytes never served");
    }

    // ---- case 5: provenance missing → 404 (provenance is mandatory per
    // #1078; an unlabeled artifact is treated as foreign)
    {
        spit(experts_path, "{\"categories\":[],\"experts\":{}}");
        std::string path, out;
        const int st = hydra_atlas::atlas_file("experts", path, out);
        ATLAS_CHECK(st == 404, "case 5: missing provenance → 404 (refused)");
        ATLAS_CHECK(out.find("refused") != std::string::npos,
                    "case 5: refusal body explains the missing id");
    }

    // ---- case 6: corrupt JSON → 500 (artifact exists but unreadable; a
    // 404 would lie, a 200 would poison the client)
    {
        spit(experts_path, "{ not json ");
        std::string path, out;
        const int st = hydra_atlas::atlas_file("experts", path, out);
        ATLAS_CHECK(st == 500, "case 6: corrupt artifact → 500");
        ATLAS_CHECK(out.find("not valid JSON") != std::string::npos,
                    "case 6: 500 body says 'not valid JSON'");
    }

    // ---- case 7: the two tiers are independent — ranks missing while
    // experts.json is fine still 404s ranks and vice versa
    {
        std::string body = "{\"categories\":[]," + prov + ",\"experts\":{}}";
        spit(experts_path, body);
        std::remove(ranks_path.c_str());
        std::string path, out;
        int st = hydra_atlas::atlas_json("ranks", out);
        ATLAS_CHECK(st == 404, "case 7a: ranks missing → 404 while experts ok");
        spit(ranks_path, ranks_ok);
        std::remove(experts_path.c_str());
        st = hydra_atlas::atlas_file("experts", path, out);
        ATLAS_CHECK(st == 404, "case 7b: experts missing → 404 while ranks ok");
    }

    // ---- case 8: unknown kind → clean 404-shaped failure (defensive;
    // the HTTP layer only passes "experts"/"ranks")
    {
        std::string path, out;
        const int st = hydra_atlas::atlas_file("bogus", path, out);
        ATLAS_CHECK(st == 404, "case 8: unknown kind → 404 (no artifact location)");
    }

    // ---- hydra F2/F3 (architect package d-9981fa1092): hits-window
    // semantics + bit-length heat, driven through the public accumulator
    // API on the synthetic MoE model (1 grid row, 8 expert cols).
    {
        auto hex_popcount = [](const std::string & hex) -> int {
            int bits = 0;
            for (size_t i = 0; i + 1 < hex.size(); i += 2) {
                const int b = std::stoi(hex.substr(i, 2), nullptr, 16);
                for (int m = 0; m < 8; m++) bits += (b >> m) & 1;
            }
            return bits;
        };
        auto json_string_field = [](const std::string & j, const char * key) -> std::string {
            const std::string pat = "\"" + std::string(key) + "\":\"";
            const size_t p = j.find(pat);
            if (p == std::string::npos) return "";
            const size_t s = p + pat.size();
            const size_t e = j.find('"', s);
            return e == std::string::npos ? "" : j.substr(s, e - s);
        };

        setenv("HYDRA_EXPERT_STATS", "1", 1);
        ATLAS_CHECK(hydra_atlas::init(model_path), "F2/F3: stats init (synthetic MoE gguf)");
        // The real decode hook gates every call on enabled(); the lazy
        // one-shot env read lives there, so the test must arm it explicitly.
        ATLAS_CHECK(hydra_atlas::enabled(), "F2/F3: stats gate armed (lazy env read)");

        // F2: /experts hits = CURRENT TURN window. Step {1,2} → popcount 2.
        hydra_atlas::begin_step();
        { const int32_t ids[2] = {1, 2}; hydra_atlas::accumulate(0, ids, 2, 8); }
        hydra_atlas::end_step(1, 8);
        std::string body = hydra_atlas::experts_json().value_or("");
        ATLAS_CHECK(!body.empty(), "F2: experts_json serves with stats on");
        ATLAS_CHECK(hex_popcount(json_string_field(body, "hits")) == 2,
                    "F2: window hits = this step's bits (2)");

        // Second step {5} ORs into the same turn window → 3 distinct cells.
        hydra_atlas::begin_step();
        { const int32_t ids[1] = {5}; hydra_atlas::accumulate(0, ids, 1, 8); }
        hydra_atlas::end_step(1, 8);
        body = hydra_atlas::experts_json().value_or("");
        ATLAS_CHECK(hex_popcount(json_string_field(body, "hits")) == 3,
                    "F2: window ORs steps within the turn (3)");

        // Turn closes → served hits return to 0. The pre-F2 sticky bitmap
        // kept serving 3 here forever (the bug).
        hydra_atlas::record_turn(0, 0.1, 1, 1, 1);
        body = hydra_atlas::experts_json().value_or("");
        ATLAS_CHECK(hex_popcount(json_string_field(body, "hits")) == 0,
                    "F2: record_turn clears the served window (no sticky hits)");

        // F2 invariant: per-step per-row popcount <= k * n_out. Three top-k
        // outputs of k=2 over {0..3} → 4 distinct cells, never more than
        // 3*2=6 and never lifetime-accumulated (begin_step zeroes).
        hydra_atlas::begin_step();
        for (int o = 0; o < 3; o++) {
            const int32_t ids[2] = { (int32_t) o, (int32_t) (o + 1) };
            hydra_atlas::accumulate(0, ids, 2, 8);
        }
        hydra_atlas::end_step(1, 8);
        body = hydra_atlas::experts_json().value_or("");
        ATLAS_CHECK(hex_popcount(json_string_field(body, "hits")) == 4,
                    "F2: step popcount bounded by k*n_out (4 <= 6), not sticky-accumulated");
        hydra_atlas::record_turn(0, 0.1, 1, 1, 1);

        // F3: heat = bit-length(count), upstream c/telemetry.h emap_emit
        // parity. Fresh counts via reset(): reps over cells 1..4 → counts
        // 1,3,4,70 → heat 1,2,3,7 (pre-F3 linear-clamp would say 1,3,4,63).
        hydra_atlas::reset();
        const int want[4] = {1, 2, 3, 7};
        const int reps[4] = {1, 3, 4, 70};
        for (int c = 0; c < 4; c++) {
            hydra_atlas::begin_step();
            for (int r = 0; r < reps[c]; r++) {
                const int32_t ids[1] = { (int32_t) (c + 1) };
                hydra_atlas::accumulate(0, ids, 1, 8);
            }
            hydra_atlas::end_step(1, 8);
        }
        body = hydra_atlas::experts_json().value_or("");
        const std::string map = json_string_field(body, "map");
        bool heat_ok = map.size() >= 16; // 8 cells → 16 hex chars
        for (int c = 0; c < 4 && heat_ok; c++) {
            const int byte = std::stoi(map.substr(2 * (c + 1), 2), nullptr, 16);
            heat_ok = (byte & 63) == want[c];
        }
        ATLAS_CHECK(heat_ok,
                    "F3: heat = bit-length(count) at counts 1/3/4/70 (emap_emit parity)");
        hydra_atlas::record_turn(0, 0.1, 1, 1, 1);
    }

    printf(failures ? "\nFAILED %d checks\n" : "\nall ok\n", failures);
    return failures ? 1 : 0;
}

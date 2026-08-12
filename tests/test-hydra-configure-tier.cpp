// hydra#406: pin the tiered CONFIGURE response shape. The Coordinator's
// WorkerSchedulerService reads these fields to confirm exactly what took
// effect (tier + params_applied + deferred_keys) — drift here is a
// wire-incompatible API change. Pure header-only test, no model needed.
//
//   {"success": true,  "tier": "T1", "params_applied": {"sampling.temp": 0.5}, "deferred_keys": []}
//   {"success": true,  "tier": "T2", "params_applied": {"state_chunk_size": 2097152}, "deferred_keys": ["n_ctx"]}
//   {"success": true,  "tier": "T3", "params_applied": {"sampling.temp": 0.5}, "deferred_keys": ["n_ctx", "override_tensor"]}
//   {"success": false, "tier": "T1", "error": "...", ...}
//
// Backward compat: a request with state_chunk_size_applied set must still
// echo the hydra#334 field so the existing detection logic keeps working.
//
// hydra#470 (no silent drops): the Coordinator's EngineConfig emits 28 keys
// (see the contract table below); EVERY one must be either a special
// T1/T2/T3 key, or a T4 generic key appliable via llama.cpp's own arg table,
// or explicitly denied with a visible warning. Zero keys may be silently
// ignored. The response also carries unrecognized_keys / rejected_keys so
// the Coordinator can see the rejects on the wire.
#include "../tools/server/server-context.h"
#include "speculative.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static int g_failures = 0;

static void expect_eq(const char * what, const std::string & actual, const std::string & expected) {
    if (actual != expected) {
        fprintf(stderr, "FAIL: %s\n  expected: %s\n  actual:   %s\n", what, expected.c_str(), actual.c_str());
        g_failures++;
    }
}

static void expect_contains(const char * what, const std::string & haystack, const std::string & needle) {
    if (haystack.find(needle) == std::string::npos) {
        fprintf(stderr, "FAIL: %s — '%s' not in '%s'\n", what, needle.c_str(), haystack.c_str());
        g_failures++;
    }
}

static void expect_not_contains(const char * what, const std::string & haystack, const std::string & needle) {
    if (haystack.find(needle) != std::string::npos) {
        fprintf(stderr, "FAIL: %s — '%s' should not be in '%s'\n", what, needle.c_str(), haystack.c_str());
        g_failures++;
    }
}

int main() {
    // Case 1: T1 only — sampling.temp applied, no deferred keys
    {
        server_task_result_hydra_engine res;
        res.op         = 0x40;  // CONFIGURE
        res.rpc_status = 0x00;  // OK
        res.success    = true;
        res.tier       = "T1";
        res.params_applied["sampling.temp"] = 0.5;
        res.deferred_keys = {};

        const std::string j = res.to_json().dump();
        expect_contains("T1 has success=true",   j, "\"success\":true");
        expect_contains("T1 has tier=T1",        j, "\"tier\":\"T1\"");
        expect_contains("T1 has sampling.temp",  j, "\"sampling.temp\":0.5");
        expect_contains("T1 has params_applied",j, "\"params_applied\":");
        expect_contains("T1 has deferred_keys",  j, "\"deferred_keys\":[]");
        expect_not_contains("T1 no error",       j, "\"error\"");
    }

    // Case 2: T2 — state_chunk_size applied, n_ctx deferred
    {
        server_task_result_hydra_engine res;
        res.op         = 0x40;
        res.rpc_status = 0x00;
        res.success    = true;
        res.tier       = "T2";
        res.state_chunk_size_applied = 2097152;
        res.params_applied["state_chunk_size"] = (uint64_t) 2097152;
        res.deferred_keys = {"n_ctx"};

        const std::string j = res.to_json().dump();
        expect_contains("T2 has tier=T2",         j, "\"tier\":\"T2\"");
        expect_contains("T2 has state_chunk_size",j, "\"state_chunk_size_applied\":2097152");
        expect_contains("T2 has n_ctx deferred",  j, "\"n_ctx\"");
        expect_contains("T2 has params_applied",  j, "\"state_chunk_size\":2097152");
    }

    // Case 3: T3 — sampling applied, n_ctx + override_tensor deferred
    {
        server_task_result_hydra_engine res;
        res.op         = 0x40;
        res.rpc_status = 0x00;
        res.success    = true;
        res.tier       = "T3";
        res.params_applied["sampling.temp"] = 0.5;
        res.deferred_keys = {"n_ctx", "override_tensor"};

        const std::string j = res.to_json().dump();
        expect_contains("T3 has tier=T3",                 j, "\"tier\":\"T3\"");
        expect_contains("T3 has sampling.temp applied",   j, "\"sampling.temp\":0.5");
        expect_contains("T3 has n_ctx deferred",          j, "\"n_ctx\"");
        expect_contains("T3 has override_tensor deferred",j, "\"override_tensor\"");
    }

    // Case 4: Failure path — success=false, error present, tier still set
    {
        server_task_result_hydra_engine res;
        res.op         = 0x40;
        res.rpc_status = 0x02;  // ERROR
        res.success    = false;
        res.tier       = "T1";
        res.error      = "drain timeout";

        const std::string j = res.to_json().dump();
        expect_contains("fail has success=false", j, "\"success\":false");
        expect_contains("fail has error",         j, "\"error\":\"drain timeout\"");
        expect_contains("fail has tier=T1",       j, "\"tier\":\"T1\"");
    }

    // Case 5: Backward compat — legacy state_chunk_size call only, no tier
    // (this case is what the existing hydra#334 emit produced — the new
    // to_json() must NOT add a "tier" key when tier is empty, to keep
    // pre-#406 callers parsing unchanged output.)
    {
        server_task_result_hydra_engine res;
        res.op         = 0x40;
        res.rpc_status = 0x00;
        res.success    = true;
        // tier intentionally empty (legacy path)
        res.state_chunk_size_applied = 2097152;

        const std::string j = res.to_json().dump();
        expect_contains("legacy has state_chunk_size_applied", j, "\"state_chunk_size_applied\":2097152");
        expect_not_contains("legacy has no tier key",          j, "\"tier\"");
        expect_not_contains("legacy has no params_applied key",j, "\"params_applied\"");
        expect_not_contains("legacy has no deferred_keys key", j, "\"deferred_keys\"");
    }

    // Case 6: Non-CONFIGURE op must not emit tier/params_applied/deferred_keys
    {
        server_task_result_hydra_engine res;
        res.op         = 0x41;  // INFO
        res.rpc_status = 0x00;
        res.success    = true;
        // even if the fields are set, non-CONFIGURE ops must skip them
        res.tier = "T1";
        res.params_applied["x"] = 1;
        res.deferred_keys = {"y"};

        const std::string j = res.to_json().dump();
        expect_not_contains("INFO has no tier key",          j, "\"tier\"");
        expect_not_contains("INFO has no params_applied key",j, "\"params_applied\"");
        expect_not_contains("INFO has no deferred_keys key", j, "\"deferred_keys\"");
    }

    // Case 7 (hydra#470): the 28-key Coordinator contract — zero silent drops.
    // Every key EngineConfig.ToHydraConfigDict() emits must be either
    //   (a) a special T1/T2/T3 key, or
    //   (b) a T4 generic key that is APPLIABLE via llama.cpp's own arg
    //       table (snake_case → kebab-case pass-through), or
    //   (c) explicitly DENIED (reported as rejected_keys with a warning).
    // UNRECOGNIZED (no arg-table entry) is a contract violation.
    {
        const std::vector<std::pair<std::string, json>> contract = {
            {"cache_prompt",         true},
            {"cache_reuse",          512},
            {"cache_type_k",         "f16"},
            {"cache_type_v",         "f16"},
            {"chat_template_kwargs", R"({"enable_thinking":true})"},
            {"cont_batching",        true},
            {"context_shift",        true},
            {"fit",                  true},
            {"flash_attn",           true},
            {"jinja",                true},
            {"kv_unified",           true},
            {"model_path",           "/tmp/hydra-test.gguf"},
            {"n_cpu_moe",            0},
            {"n_ctx",                8192},
            {"n_gpu_layers",         99},
            {"override_tensor",      "blk.*.ffn_*_exps.weight=CPU"},
            {"reasoning",            true},
            {"rope_scale",           2.0},
            {"rope_scaling",         "yarn"},
            {"rpc_servers",          json::array({"192.168.122.21:9504"})},
            {"spec_draft_ngl",       12},
            {"spec_draft_n_max",     8},
            {"spec_draft_p_min",     0.1},
            {"spec_type",            "draft-mtp"},
            {"split_mode",           "layer"},
            {"tensor_split",         json::array({0.5, 0.5})},
            {"ubatch_size",          512},
            {"yarn_orig_ctx",        8192},
        };

        size_t n_special = 0, n_appliable = 0, n_denied = 0;
        for (const auto & kv : contract) {
            const int tier = hydra_classify_config_key(kv.first);
            if (tier >= 1 && tier <= 3) {
                n_special++;  // (a)
                continue;
            }
            const hydra_generic_key_status st = hydra_classify_generic_key(kv.first);
            if (st == hydra_generic_key_status::APPLIABLE) {
                n_appliable++;  // (b)
            } else if (st == hydra_generic_key_status::DENIED) {
                n_denied++;     // (c)
            } else {
                fprintf(stderr, "FAIL: contract key '%s' is neither special, appliable, nor denied\n",
                        kv.first.c_str());
                g_failures++;
            }
        }
        if (n_special + n_appliable + n_denied != contract.size()) {
            fprintf(stderr, "FAIL: contract classification mismatch (%zu special + %zu appliable + %zu denied != %zu)\n",
                    n_special, n_appliable, n_denied, contract.size());
            g_failures++;
        }
        // Pin the split so a future classifier/denylist drift is caught:
        // 12 special (5 T2: cache_type_k/v, n_ctx, rope_scale, rope_scaling;
        // 7 T3: model_path, n_cpu_moe, n_gpu_layers, override_tensor,
        // rpc_servers, split_mode, tensor_split), 15 appliable, 1 denied (fit).
        if (n_special != 12 || n_appliable != 15 || n_denied != 1) {
            fprintf(stderr, "FAIL: contract split drifted (special=%zu appliable=%zu denied=%zu; expected 12/15/1)\n",
                    n_special, n_appliable, n_denied);
            g_failures++;
        }

        // Every appliable generic key must also accept its contract value
        // through the arg-table handler (exercises the JSON→handler
        // conversion for bool/int/float/string/array).
        common_params p;
        for (const auto & kv : contract) {
            if (hydra_classify_config_key(kv.first) <= 3) {
                continue;  // special keys have their own apply path
            }
            if (hydra_classify_generic_key(kv.first) != hydra_generic_key_status::APPLIABLE) {
                continue;  // denied keys are reported, not applied
            }
            if (!hydra_apply_generic_key(p, kv.first, kv.second)) {
                fprintf(stderr, "FAIL: contract key '%s' is appliable but its value was rejected\n",
                        kv.first.c_str());
                g_failures++;
            }
        }
    }

    // Case 8 (hydra#470): the regression that started this — spec_type must
    // ACTUALLY land in params.speculative.types (it used to be classified
    // unknown and silently dropped, so dense MTP never activated).
    {
        common_params p;
        if (!hydra_apply_generic_key(p, "spec_type", json("draft-mtp"))) {
            fprintf(stderr, "FAIL: spec_type apply returned false\n");
            g_failures++;
        }
        const bool has_mtp = std::find(p.speculative.types.begin(),
                                       p.speculative.types.end(),
                                       COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != p.speculative.types.end();
        if (!has_mtp) {
            fprintf(stderr, "FAIL: spec_type=\"draft-mtp\" did not land in params.speculative.types\n");
            g_failures++;
        }
        // CONFIGURE is absolute state: re-applying must not accumulate
        // duplicate entries (the CLI --spec-type handler appends).
        if (!hydra_apply_generic_key(p, "spec_type", json("draft-mtp"))) {
            fprintf(stderr, "FAIL: spec_type re-apply returned false\n");
            g_failures++;
        }
        const size_t n_mtp = std::count(p.speculative.types.begin(),
                                        p.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
        if (n_mtp != 1) {
            fprintf(stderr, "FAIL: spec_type re-apply accumulated %zu draft-mtp entries (expected 1)\n", n_mtp);
            g_failures++;
        }
    }

    // Case 9 (hydra#470): the CONFIGURE response carries unrecognized_keys
    // and rejected_keys so the Coordinator can see rejected generic keys on
    // the wire — never a silent drop.
    {
        server_task_result_hydra_engine res;
        res.op         = 0x40;
        res.rpc_status = 0x00;
        res.success    = true;
        res.tier       = "T4";
        res.params_applied["sampling.temp"] = 0.5;
        res.deferred_keys    = {"spec_type"};
        res.unrecognized_keys = {"totally_bogus_param"};
        res.rejected_keys     = {"fit"};

        const std::string j = res.to_json().dump();
        expect_contains("T4 has tier=T4",                  j, "\"tier\":\"T4\"");
        expect_contains("T4 has unrecognized_keys",        j, "\"unrecognized_keys\":[\"totally_bogus_param\"]");
        expect_contains("T4 has rejected_keys",            j, "\"rejected_keys\":[\"fit\"]");
        expect_contains("T4 has spec_type deferred",       j, "\"spec_type\"");
    }

    // Case 10 (hydra#470): the status classifier itself — a denylisted key
    // is DENIED (never applied), a key with no arg-table entry is
    // UNRECOGNIZED, and a plain unknown key is not silently accepted.
    {
        if (hydra_classify_generic_key("fit") != hydra_generic_key_status::DENIED) {
            fprintf(stderr, "FAIL: 'fit' must be DENIED at reload\n");
            g_failures++;
        }
        if (hydra_classify_generic_key("definitely_not_a_llama_arg") != hydra_generic_key_status::UNRECOGNIZED) {
            fprintf(stderr, "FAIL: unknown key must be UNRECOGNIZED\n");
            g_failures++;
        }
        common_params p;
        if (hydra_apply_generic_key(p, "fit", json(true))) {
            fprintf(stderr, "FAIL: denied key 'fit' must not be applied\n");
            g_failures++;
        }
        if (hydra_apply_generic_key(p, "definitely_not_a_llama_arg", json(true))) {
            fprintf(stderr, "FAIL: unrecognized key must not be applied\n");
            g_failures++;
        }
    }

    if (g_failures == 0) {
        printf("test-hydra-configure-tier: all checks passed\n");
        return 0;
    }
    fprintf(stderr, "test-hydra-configure-tier: %d check(s) failed\n", g_failures);
    return 1;
}

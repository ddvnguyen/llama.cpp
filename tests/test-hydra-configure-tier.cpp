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
#include "../tools/server/server-task.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>

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

    if (g_failures == 0) {
        printf("test-hydra-configure-tier: all checks passed\n");
        return 0;
    }
    fprintf(stderr, "test-hydra-configure-tier: %d check(s) failed\n", g_failures);
    return 1;
}

#include <cstdio>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s (line %d): %s\n", #cond, __LINE__, msg); \
            return 1; \
        } \
    } while (0)

// Replicate the capability bitfield derivation logic from
// llama_model_get_capabilities_bitfield() in src/llama-model.cpp.
// This tests the logic in isolation without needing to load a real GGUF
// or construct an abstract llama_model instance.
static uint32_t derive_capabilities(
    const std::unordered_map<std::string, std::string> & gguf_kv,
    const std::vector<std::string> & gguf_tags)
{
    uint32_t caps = 0;

    // REASONING (bit 2): check general.tags for "reasoning"
    for (const auto & tag : gguf_tags) {
        std::string lower = tag;
        for (auto & c : lower) c = (char)tolower((unsigned char)c);
        if (lower.find("reasoning") != std::string::npos) {
            caps |= 0x04;
            break;
        }
    }

    // TOOL_USE (bit 3) and CODE (bit 4): explicit tag sets
    static const std::unordered_set<std::string> kCodeTags = {
        "code", "coding", "coder", "codes",
    };
    static const std::unordered_set<std::string> kToolTags = {
        "tool", "tools", "tool-use", "function-calling", "function_calling",
    };

    // Display name: prefer general.base_model.0.name, fall back to general.name
    const std::string display_name = [&]() {
        const auto & base_it = gguf_kv.find("general.base_model.0.name");
        if (base_it != gguf_kv.end() && !base_it->second.empty()) {
            return base_it->second;
        }
        const auto & name_it = gguf_kv.find("general.name");
        if (name_it != gguf_kv.end() && !name_it->second.empty()) {
            return name_it->second;
        }
        return std::string();
    }();

    for (const auto & tag : gguf_tags) {
        std::string lower = tag;
        for (auto & c : lower) c = (char)tolower((unsigned char)c);
        if (kToolTags.count(lower)) {
            caps |= 0x08;
        }
        if (kCodeTags.count(lower)) {
            caps |= 0x10;
        }
    }
    // Also check display name for code/tool hints
    {
        std::string lower_name = display_name;
        for (auto & c : lower_name) c = (char)tolower((unsigned char)c);
        // Check against original (non-lowered) display_name for CamelCase boundaries
        auto has_word = [&display_name, &lower_name](const std::unordered_set<std::string> & keywords) -> bool {
            for (const auto & kw : keywords) {
                size_t pos = 0;
                while ((pos = lower_name.find(kw, pos)) != std::string::npos) {
                    bool word_start = (pos == 0)
                        || !isalnum((unsigned char)lower_name[pos - 1])
                        || isupper((unsigned char)display_name[pos - 1]);
                    bool word_end   = (pos + kw.size() >= lower_name.size())
                        || !isalnum((unsigned char)lower_name[pos + kw.size()])
                        || isupper((unsigned char)display_name[pos + kw.size()]);
                    if (word_start && word_end) return true;
                    pos++;
                }
            }
            return false;
        };
        if (has_word(kCodeTags)) {
            caps |= 0x10;
        }
        if (has_word(kToolTags)) {
            caps |= 0x08;
        }
    }

    return caps;
}

int main() {
    fprintf(stderr, "=== test-model-identity (logic-only) ===\n");
    int passed = 0;

    // ── Test 1: Empty → no bits ──
    {
        uint32_t caps = derive_capabilities({}, {});
        TEST_ASSERT(caps == 0, "empty → 0");
        fprintf(stderr, "  PASS #1: empty → 0\n"); passed++;
    }

    // ── Test 2: "decoder-only" should NOT set CODE ──
    {
        uint32_t caps = derive_capabilities({}, {"text-generation", "decoder-only"});
        TEST_ASSERT((caps & 0x10) == 0, "'decoder-only' → no CODE");
        fprintf(stderr, "  PASS #2: 'decoder-only' → no CODE\n"); passed++;
    }

    // ── Test 3: "encoder-only" should NOT set CODE ──
    {
        uint32_t caps = derive_capabilities({}, {"encoder-only"});
        TEST_ASSERT((caps & 0x10) == 0, "'encoder-only' → no CODE");
        fprintf(stderr, "  PASS #3: 'encoder-only' → no CODE\n"); passed++;
    }

    // ── Test 4: "autoencoder" should NOT set CODE ──
    {
        uint32_t caps = derive_capabilities({}, {"autoencoder"});
        TEST_ASSERT((caps & 0x10) == 0, "'autoencoder' → no CODE");
        fprintf(stderr, "  PASS #4: 'autoencoder' → no CODE\n"); passed++;
    }

    // ── Test 5: "encode" should NOT set CODE ──
    {
        uint32_t caps = derive_capabilities({}, {"encode"});
        TEST_ASSERT((caps & 0x10) == 0, "'encode' → no CODE");
        fprintf(stderr, "  PASS #5: 'encode' → no CODE\n"); passed++;
    }

    // ── Test 6: "decode" should NOT set CODE ──
    {
        uint32_t caps = derive_capabilities({}, {"decode"});
        TEST_ASSERT((caps & 0x10) == 0, "'decode' → no CODE");
        fprintf(stderr, "  PASS #6: 'decode' → no CODE\n"); passed++;
    }

    // ── Test 7: "code" tag → CODE ──
    {
        uint32_t caps = derive_capabilities({}, {"code"});
        TEST_ASSERT((caps & 0x10) != 0, "'code' → CODE");
        fprintf(stderr, "  PASS #7: 'code' → CODE\n"); passed++;
    }

    // ── Test 8: "coding" tag → CODE ──
    {
        uint32_t caps = derive_capabilities({}, {"coding"});
        TEST_ASSERT((caps & 0x10) != 0, "'coding' → CODE");
        fprintf(stderr, "  PASS #8: 'coding' → CODE\n"); passed++;
    }

    // ── Test 9: "coder" tag → CODE ──
    {
        uint32_t caps = derive_capabilities({}, {"coder"});
        TEST_ASSERT((caps & 0x10) != 0, "'coder' → CODE");
        fprintf(stderr, "  PASS #9: 'coder' → CODE\n"); passed++;
    }

    // ── Test 10: "reasoning" → REASONING ──
    {
        uint32_t caps = derive_capabilities({}, {"reasoning"});
        TEST_ASSERT((caps & 0x04) != 0, "'reasoning' → REASONING");
        fprintf(stderr, "  PASS #10: 'reasoning' → REASONING\n"); passed++;
    }

    // ── Test 11: "tool" → TOOL_USE ──
    {
        uint32_t caps = derive_capabilities({}, {"tool"});
        TEST_ASSERT((caps & 0x08) != 0, "'tool' → TOOL_USE");
        fprintf(stderr, "  PASS #11: 'tool' → TOOL_USE\n"); passed++;
    }

    // ── Test 12: "multi-tool" → NO TOOL_USE (not exact match) ──
    {
        uint32_t caps = derive_capabilities({}, {"multi-tool"});
        TEST_ASSERT((caps & 0x08) == 0, "'multi-tool' → no TOOL_USE");
        fprintf(stderr, "  PASS #12: 'multi-tool' → no TOOL_USE\n"); passed++;
    }

    // ── Test 13: "function-calling" → TOOL_USE ──
    {
        uint32_t caps = derive_capabilities({}, {"function-calling"});
        TEST_ASSERT((caps & 0x08) != 0, "'function-calling' → TOOL_USE");
        fprintf(stderr, "  PASS #13: 'function-calling' → TOOL_USE\n"); passed++;
    }

    // ── Test 14: Display name "Qwopus Coder" → CODE (word-boundary match) ──
    {
        uint32_t caps = derive_capabilities(
            {{"general.base_model.0.name", "Qwopus Coder"}}, {});
        TEST_ASSERT((caps & 0x10) != 0, "name 'Qwopus Coder' → CODE");
        fprintf(stderr, "  PASS #14: name 'Qwopus Coder' → CODE\n"); passed++;
    }

    // ── Test 15: Display name "Qwopus Decoder" → NO CODE ──
    {
        uint32_t caps = derive_capabilities(
            {{"general.base_model.0.name", "Qwopus Decoder"}}, {});
        TEST_ASSERT((caps & 0x10) == 0, "name 'Qwopus Decoder' → no CODE");
        fprintf(stderr, "  PASS #15: name 'Qwopus Decoder' → no CODE\n"); passed++;
    }

    // ── Test 16: Empty base_model.0.name falls back to general.name ──
    {
        uint32_t caps = derive_capabilities(
            {{"general.base_model.0.name", ""},
             {"general.name", "My-Coder-Model"}},
            {"code"});
        TEST_ASSERT((caps & 0x10) != 0, "empty base_model fallback → CODE from tag");
        fprintf(stderr, "  PASS #16: empty base_model fallback → CODE\n"); passed++;
    }

    // ── Test 17: Case-insensitive tag matching ──
    {
        uint32_t caps = derive_capabilities({}, {"CODE"});
        TEST_ASSERT((caps & 0x10) != 0, "'CODE' (uppercase) → CODE");
        fprintf(stderr, "  PASS #17: 'CODE' uppercase → CODE\n"); passed++;
    }

    // ── Test 18: Unrelated tags → no bits ──
    {
        uint32_t caps = derive_capabilities({},
            {"text-generation", "causal-lm", "transformer"});
        TEST_ASSERT(caps == 0, "unrelated tags → 0");
        fprintf(stderr, "  PASS #18: unrelated tags → 0\n"); passed++;
    }

    // ── Test 19: Both CODE and TOOL_USE tags → both bits ──
    {
        uint32_t caps = derive_capabilities({}, {"code", "tool"});
        TEST_ASSERT((caps & 0x10) != 0, "code → CODE");
        TEST_ASSERT((caps & 0x08) != 0, "tool → TOOL_USE");
        fprintf(stderr, "  PASS #19: code+tool → both bits\n"); passed++;
    }

    // ── Test 20: "codes" (plural) → CODE ──
    {
        uint32_t caps = derive_capabilities({}, {"codes"});
        TEST_ASSERT((caps & 0x10) != 0, "'codes' → CODE");
        fprintf(stderr, "  PASS #20: 'codes' → CODE\n"); passed++;
    }

    // ── Test 21: "tools" (plural) → TOOL_USE ──
    {
        uint32_t caps = derive_capabilities({}, {"tools"});
        TEST_ASSERT((caps & 0x08) != 0, "'tools' → TOOL_USE");
        fprintf(stderr, "  PASS #21: 'tools' → TOOL_USE\n"); passed++;
    }

    // ── Test 22: "function_calling" (underscore) → TOOL_USE ──
    {
        uint32_t caps = derive_capabilities({}, {"function_calling"});
        TEST_ASSERT((caps & 0x08) != 0, "'function_calling' → TOOL_USE");
        fprintf(stderr, "  PASS #22: 'function_calling' → TOOL_USE\n"); passed++;
    }

    // ── Test 23: "tool-use" (hyphenated) → TOOL_USE ──
    {
        uint32_t caps = derive_capabilities({}, {"tool-use"});
        TEST_ASSERT((caps & 0x08) != 0, "'tool-use' → TOOL_USE");
        fprintf(stderr, "  PASS #23: 'tool-use' → TOOL_USE\n"); passed++;
    }

    // ── Test 24: Real-world: Qwopus non-MTP with "reasoning" tag ──
    {
        uint32_t caps = derive_capabilities(
            {{"general.base_model.0.name", "Qwen3.6 35B A3B"}},
            {"text-generation", "reasoning"});
        TEST_ASSERT((caps & 0x04) != 0, "reasoning tag → REASONING");
        TEST_ASSERT((caps & 0x10) == 0, "no code tag → no CODE");
        TEST_ASSERT((caps & 0x08) == 0, "no tool tag → no TOOL_USE");
        fprintf(stderr, "  PASS #24: Qwopus non-MTP realistic tags\n"); passed++;
    }

    // ── Test 25: Real-world: Coder model with "code" tag ──
    {
        uint32_t caps = derive_capabilities(
            {{"general.base_model.0.name", "Qwopus3.6 27B Coder"}},
            {"code", "reasoning"});
        TEST_ASSERT((caps & 0x10) != 0, "code tag → CODE");
        TEST_ASSERT((caps & 0x04) != 0, "reasoning tag → REASONING");
        fprintf(stderr, "  PASS #25: Coder model realistic tags\n"); passed++;
    }

    // ── Test 26: Display name "Autoencoder Model" → NO CODE ──
    {
        uint32_t caps = derive_capabilities(
            {{"general.base_model.0.name", "Autoencoder Model"}}, {});
        TEST_ASSERT((caps & 0x10) == 0, "name 'Autoencoder Model' → no CODE");
        fprintf(stderr, "  PASS #26: name 'Autoencoder Model' → no CODE\n"); passed++;
    }

    // ── Test 27: Display name "Encoder-Decoder" → NO CODE ──
    {
        uint32_t caps = derive_capabilities(
            {{"general.base_model.0.name", "Encoder-Decoder"}}, {});
        TEST_ASSERT((caps & 0x10) == 0, "name 'Encoder-Decoder' → no CODE");
        fprintf(stderr, "  PASS #27: name 'Encoder-Decoder' → no CODE\n"); passed++;
    }

    // ── Test 28: Display name "CodeLlama-7B" → CODE (word-boundary: "code" at start) ──
    {
        uint32_t caps = derive_capabilities(
            {{"general.base_model.0.name", "CodeLlama-7B"}}, {});
        TEST_ASSERT((caps & 0x10) != 0, "name 'CodeLlama-7B' → CODE");
        fprintf(stderr, "  PASS #28: name 'CodeLlama-7B' → CODE\n"); passed++;
    }

    fprintf(stderr, "=== all %d tests passed ===\n", passed);
    return 0;
}

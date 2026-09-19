// E0 measurement foundation: pin-file loader shared by every consult path.
//
// One format, one parser, two call sites (CUDA backend now, CPU consult when
// E1 creates one). Header-only and dependency-free so both translation units
// include it without new link edges. The loader never exits; it reports the
// first failure as text and the CALLER hard-fails (GGML_ABORT), which keeps
// the abort idiom local to each TU and the parser unit-testable.
//
// Format: "L <il> <expert ids...>" per line, "#" comments, blank lines skip.
// Any other unparseable line is a load error, never a silent skip.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct hydra_pin_set {
    std::vector<std::vector<int32_t>> layers; // layers[il] = deduped expert ids
    int64_t n_pins  = 0; // validated pins kept
    int64_t n_dups  = 0; // exact duplicates removed (counted, not fatal)
    int64_t n_lines = 0; // data lines parsed
};

// Loads pins from path. Returns true on success; on false, err carries the
// reason and the caller MUST hard-fail (an armed-but-unreadable run is the
// failure this exists to prevent). Zero pins kept is also an error.
static bool hydra_pins_load(const char * path, hydra_pin_set & out, std::string & err) {
    FILE * f = fopen(path, "r");
    if (!f) {
        err = std::string("HYDRA_PIN_FILE set but fopen failed: ") + path;
        return false;
    }
    char line[8192];
    int64_t lineno = 0;
    while (fgets(line, sizeof(line), f)) {
        ++lineno;
        if (!strchr(line, '\n') && !feof(f)) {
            err = "line " + std::to_string(lineno) + " exceeds 8 KiB (truncated, refusing to guess)";
            fclose(f);
            return false;
        }
        // comments (leading whitespace tolerated), blanks, and CRLF endings
        // tolerate cross-machine authoring; anything else must parse or fail
        const char * head = line;
        while (*head == ' ' || *head == '\t') {
            ++head;
        }
        if (*head == '#' || *head == '\n' || *head == '\r' || *head == '\0') {
            continue;
        }
        // whitespace-only lines skip like blanks
        {
            bool blank = true;
            for (const char * c = line; *c && *c != '\n'; ++c) {
                if (*c != ' ' && *c != '\t' && *c != '\r') { blank = false; break; }
            }
            if (blank) {
                continue;
            }
        }
        int il = -1;
        int nch = 0;
        if (sscanf(line, "L %d%n", &il, &nch) != 1 || il < 0) {
            err = "line " + std::to_string(lineno) + " is not \"L <il> <ids...>\"";
            fclose(f);
            return false;
        }
        if ((size_t) il >= out.layers.size()) {
            out.layers.resize((size_t) il + 1);
        }
        const char * p = line + nch;
        bool any = false;
        while (*p && *p != '\n' && *p != '#') {
            int e = -1;
            int adv = 0;
            if (sscanf(p, "%d%n", &e, &adv) != 1) {
                // only whitespace may trail the id list
                bool tail_blank = true;
                for (const char * c = p; *c && *c != '\n' && *c != '#'; ++c) {
                    if (*c != ' ' && *c != '\t' && *c != '\r') { tail_blank = false; break; }
                }
                if (!tail_blank) {
                    err = "line " + std::to_string(lineno) + " has a non-integer pin entry";
                    fclose(f);
                    return false;
                }
                break;
            }
            if (e < 0) {
                err = "line " + std::to_string(lineno) + " has a negative expert id";
                fclose(f);
                return false;
            }
            bool dup = false;
            for (int32_t kept : out.layers[(size_t) il]) {
                if (kept == e) { dup = true; break; }
            }
            if (dup) {
                ++out.n_dups;
            } else {
                out.layers[(size_t) il].push_back(e);
                ++out.n_pins;
            }
            any = true;
            p += adv;
        }
        if (!any) {
            err = "line " + std::to_string(lineno) + " names a layer with zero pins";
            fclose(f);
            return false;
        }
        ++out.n_lines;
    }
    fclose(f);
    if (out.n_pins == 0) {
        err = std::string("HYDRA_PIN_FILE set but zero pins loaded from ") + path;
        return false;
    }
    return true;
}

// Range-check against the layer's expert count. n_expert is known only once
// the model is loaded, so E1 calls this at attach time, not at file load.
// Returns true when every pin indexes a real expert.
[[maybe_unused]] static bool hydra_pins_validate_range(const hydra_pin_set & s, int64_t n_expert, std::string & err) {
    for (size_t il = 0; il < s.layers.size(); ++il) {
        for (int32_t e : s.layers[il]) {
            if ((int64_t) e >= n_expert) {
                err = "layer " + std::to_string(il) + " pins expert " + std::to_string(e) +
                      " but the layer has only " + std::to_string(n_expert) + " experts";
                return false;
            }
        }
    }
    return true;
}

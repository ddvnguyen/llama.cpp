#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// M-Perf.9 (#289): compute SHA-256 of a file on disk and write the
// 64-character lowercase hex string + NUL terminator into `buf`. `buf_size`
// must be at least 65 bytes. Returns 64 on success, -1 on error (file not
// found, read failure, or invalid arguments).
//
// This is a self-contained SHA-256 — no external crypto library. The cost
// is one full file read at the loader's disk bandwidth (a 25 GB GGUF
// takes ~1-2 seconds on NVMe), paid once per llama_model_load_from_file.
int llama_hash_file_sha256(const char * path, char * buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

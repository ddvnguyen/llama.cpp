// M-Perf.9 (#289): SHA-256 file hashing for model identity.
//
// Computes the SHA-256 of a GGUF file's contents and returns it as a
// 64-character lowercase hex string. Used by llama_model_load_from_file to
// stamp each loaded llama_model with its source-file hash, so the engine can
// report model_hash in slot META responses and the Coordinator can detect
// cross-model restores.
//
// Implementation is a self-contained SHA-256 (no OpenSSL dependency, no
// libcrypto). Adapted from the SHA-256 reference algorithm in
// examples/gguf-hash/gguf-hash.cpp — small enough to inline and keeps the
// library dependency-free.

#include "llama-hash.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

// SHA-256 round constants (first 32 bits of the fractional parts of the
// cube roots of the first 64 primes).
static const uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

struct sha256_ctx {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  buffer[64];
    size_t   buflen;
};

static inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

static void sha256_init(sha256_ctx * ctx) {
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
    ctx->bitlen = 0;
    ctx->buflen = 0;
}

static void sha256_compress(sha256_ctx * ctx, const uint8_t * block) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i*4]   << 24) |
               ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] <<  8) |
               ((uint32_t)block[i*4+3]);
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19)  ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    uint32_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + SHA256_K[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e;
        e = d + t1;
        d = c; c = b; b = a;
        a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_update(sha256_ctx * ctx, const uint8_t * data, size_t len) {
    while (len > 0) {
        size_t take = 64 - ctx->buflen;
        if (take > len) take = len;
        memcpy(ctx->buffer + ctx->buflen, data, take);
        ctx->buflen += take;
        data += take;
        len  -= take;
        if (ctx->buflen == 64) {
            sha256_compress(ctx, ctx->buffer);
            ctx->bitlen += 512;
            ctx->buflen = 0;
        }
    }
}

static void sha256_final(sha256_ctx * ctx, uint8_t out[32]) {
    ctx->bitlen += (uint64_t)ctx->buflen * 8;
    ctx->buffer[ctx->buflen++] = 0x80;
    if (ctx->buflen > 56) {
        while (ctx->buflen < 64) ctx->buffer[ctx->buflen++] = 0;
        sha256_compress(ctx, ctx->buffer);
        ctx->buflen = 0;
    }
    while (ctx->buflen < 56) ctx->buffer[ctx->buflen++] = 0;
    for (int i = 7; i >= 0; i--) {
        ctx->buffer[ctx->buflen++] = (uint8_t)(ctx->bitlen >> (i * 8));
    }
    sha256_compress(ctx, ctx->buffer);
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(ctx->state[i] >> 24);
        out[i*4+1] = (uint8_t)(ctx->state[i] >> 16);
        out[i*4+2] = (uint8_t)(ctx->state[i] >>  8);
        out[i*4+3] = (uint8_t)(ctx->state[i]);
    }
}

} // anonymous namespace

int llama_hash_file_sha256(const char * path, char * buf, size_t buf_size) {
    if (!path || !buf || buf_size < 65) return -1;

    std::ifstream f(path, std::ios::binary);
    if (!f) return -1;

    sha256_ctx ctx;
    sha256_init(&ctx);

    std::vector<uint8_t> chunk(64 * 1024);
    while (f) {
        f.read(reinterpret_cast<char *>(chunk.data()), chunk.size());
        std::streamsize n = f.gcount();
        if (n > 0) {
            sha256_update(&ctx, chunk.data(), (size_t)n);
        }
    }

    uint8_t digest[32];
    sha256_final(&ctx, digest);

    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        buf[i*2]     = hex[(digest[i] >> 4) & 0xF];
        buf[i*2 + 1] = hex[digest[i]        & 0xF];
    }
    buf[64] = '\0';
    return 64;
}

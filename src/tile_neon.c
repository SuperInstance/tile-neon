/* tile_neon.c — NEON-accelerated tile ops (with scalar fallback)
 *
 * Compiles on:
 *   ARM64:  -march=armv8-a+simd -DUSE_NEON   → NEON path
 *   x86:    -DNO_NEON                             → scalar path
 */

#include "tile_neon.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>

/* ================================================================== */
/* BLAKE2b — simplified, NEON-optional                                 */
/* ================================================================== */

/* BLAKE2b IV */
static const uint64_t blake2b_iv[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

/* Sigma table for permutation */
static const uint8_t blake2b_sigma[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
    {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4},
    { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8},
    { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13},
    { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9},
    {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11},
    {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10},
    { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0},
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
};

static inline uint64_t rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

static inline void blake2b_g(uint64_t *v, int a, int b, int c, int d,
                             uint64_t x, uint64_t y) {
    v[a] += v[b] + x;
    v[d] = rotr64(v[d] ^ v[a], 32);
    v[c] += v[d];
    v[b] = rotr64(v[b] ^ v[c], 24);
    v[a] += v[b] + y;
    v[d] = rotr64(v[d] ^ v[a], 16);
    v[c] += v[d];
    v[b] = rotr64(v[b] ^ v[c], 63);
}

static void blake2b_compress(uint64_t h[8], const uint8_t block[128],
                             uint64_t counter, int is_final) {
    uint64_t v[16];
    memcpy(v, h, 64);
    memcpy(v + 8, blake2b_iv, 64);
    v[12] ^= counter;
    v[13] ^= (counter >> 63) >> 1; /* high bits */
    if (is_final) v[14] = ~v[14];

    uint64_t m[16];
    for (int i = 0; i < 16; i++)
        m[i] = *(const uint64_t *)(block + i * 8);

    for (int round = 0; round < 12; round++) {
        const uint8_t *s = blake2b_sigma[round];
        blake2b_g(v, 0, 4,  8, 12, m[s[ 0]], m[s[ 1]]);
        blake2b_g(v, 1, 5,  9, 13, m[s[ 2]], m[s[ 3]]);
        blake2b_g(v, 2, 6, 10, 14, m[s[ 4]], m[s[ 5]]);
        blake2b_g(v, 3, 7, 11, 15, m[s[ 6]], m[s[ 7]]);
        blake2b_g(v, 0, 5, 10, 15, m[s[ 8]], m[s[ 9]]);
        blake2b_g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        blake2b_g(v, 2, 7,  8, 13, m[s[12]], m[s[13]]);
        blake2b_g(v, 3, 4,  9, 14, m[s[14]], m[s[15]]);
    }

    for (int i = 0; i < 8; i++)
        h[i] ^= v[i] ^ v[i + 8];
}

void tile_blake2b(const uint8_t *msg, size_t msg_len,
                  const uint8_t *key, size_t key_len,
                  uint8_t out[TILE_HASH_LEN]) {
    uint64_t h[8];
    memcpy(h, blake2b_iv, 64);
    h[0] ^= 0x01010000 | (key_len << 8) | TILE_HASH_LEN;

    uint8_t block[TILE_BLOCK_BYTES];
    memset(block, 0, sizeof(block));

    if (key_len > 0) {
        memcpy(block, key, key_len);
    }

    /* If there's a key, the first block is the key pad */
    size_t offset = 0;
    uint64_t counter = 0;

    if (key_len > 0) {
        blake2b_compress(h, block, counter++, 0);
        offset = 0;
    }

    /* Process full blocks */
    while (offset + TILE_BLOCK_BYTES <= msg_len) {
        memcpy(block, msg + offset, TILE_BLOCK_BYTES);
        blake2b_compress(h, block, counter++, 0);
        offset += TILE_BLOCK_BYTES;
    }

    /* Last block */
    memset(block, 0, sizeof(block));
    size_t rem = msg_len - offset;
    if (rem > 0) memcpy(block, msg + offset, rem);
    blake2b_compress(h, block, counter, 1);

    memcpy(out, h, TILE_HASH_LEN);
}

void tile_blake2b_batch(const uint8_t *msgs, size_t msg_len,
                        size_t count, uint8_t *outs) {
    for (size_t i = 0; i < count; i++) {
        tile_blake2b(msgs + i * msg_len, msg_len,
                     NULL, 0,
                     outs + i * TILE_HASH_LEN);
    }
}

/* ================================================================== */
/* Embedding — position-aware with optional NEON                       */
/* ================================================================== */

/* Simple deterministic mapping: hash bytes → floats in [-1,1].
   Positional encoding uses sin/cos on position index.               */

static inline float hash_byte_to_float(uint8_t b) {
    /* map [0,255] → [-1, 1] */
    return (float)b / 127.5f - 1.0f;
}

void tile_embed(const uint8_t hash[TILE_HASH_LEN],
                int pos,
                float out[TILE_VEC_DIM]) {
    const float pos_scale = 1.0f / 10000.0f;

#if USE_NEON
    /* NEON path: process 4 floats at a time */
    int i = 0;
    for (; i + 3 < TILE_VEC_DIM; i += 4) {
        /* Mix hash bytes cyclically */
        uint8_t b[4];
        b[0] = hash[(i + 0) % TILE_HASH_LEN] ^ (uint8_t)(pos & 0xFF);
        b[1] = hash[(i + 1) % TILE_HASH_LEN] ^ (uint8_t)((pos >> 3) & 0xFF);
        b[2] = hash[(i + 2) % TILE_HASH_LEN] ^ (uint8_t)((pos >> 7) & 0xFF);
        b[3] = hash[(i + 3) % TILE_HASH_LEN] ^ (uint8_t)((pos >> 11) & 0xFF);

        float vals[4];
        for (int j = 0; j < 4; j++) {
            float dim_freq = pos_scale * powf(10.0f, (float)(i + j) / (float)TILE_VEC_DIM * 4.0f);
            float pos_enc = sinf((float)pos * dim_freq);
            vals[j] = hash_byte_to_float(b[j]) + pos_enc * 0.1f;
        }
        float32x4_t v = vld1q_f32(vals);
        vst1q_f32(out + i, v);
    }
    /* Remainder */
    for (; i < TILE_VEC_DIM; i++) {
        uint8_t b = hash[i % TILE_HASH_LEN] ^ (uint8_t)(pos & 0xFF);
        float dim_freq = pos_scale * powf(10.0f, (float)i / (float)TILE_VEC_DIM * 4.0f);
        float pos_enc = sinf((float)pos * dim_freq);
        out[i] = hash_byte_to_float(b) + pos_enc * 0.1f;
    }
#else
    for (int i = 0; i < TILE_VEC_DIM; i++) {
        uint8_t b = hash[i % TILE_HASH_LEN] ^ (uint8_t)(pos & 0xFF);
        float dim_freq = pos_scale * powf(10.0f, (float)i / (float)TILE_VEC_DIM * 4.0f);
        float pos_enc = sinf((float)pos * dim_freq);
        out[i] = hash_byte_to_float(b) + pos_enc * 0.1f;
    }
#endif
}

void tile_embed_batch(const uint8_t *hashes, size_t count,
                      const int *positions,
                      float *out) {
    for (size_t i = 0; i < count; i++) {
        tile_embed(hashes + i * TILE_HASH_LEN,
                   positions ? positions[i] : (int)i,
                   out + i * TILE_VEC_DIM);
    }
}

/* ================================================================== */
/* Full gate pipeline                                                  */
/* ================================================================== */

void tile_gate_pipeline(const uint8_t *msgs, size_t msg_len,
                        size_t n_msgs,
                        const float *index_mat, size_t index_rows,
                        size_t top_k,
                        size_t *out_indices,
                        float  *out_scores) {
    /* 1. Hash all messages */
    uint8_t *hashes = (uint8_t *)malloc(n_msgs * TILE_HASH_LEN);
    tile_blake2b_batch(msgs, msg_len, n_msgs, hashes);

    /* 2. Embed all hashes */
    float *embeddings = (float *)malloc(n_msgs * TILE_VEC_DIM * sizeof(float));
    tile_embed_batch(hashes, n_msgs, NULL, embeddings);

    /* 3. Search using first embedding as query against index */
    if (n_msgs > 0 && index_rows > 0) {
        tile_cosine_search(index_mat, index_rows,
                           embeddings, top_k,
                           out_indices, out_scores);
    }

    free(hashes);
    free(embeddings);
}

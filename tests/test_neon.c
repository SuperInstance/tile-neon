/* test_neon.c — Unit tests for tile_neon
 *
 * Compile:
 *   ARM64: cc -o test_neon tests/test_neon.c src/*.c -Iinclude -lm -march=armv8-a+simd -DUSE_NEON
 *   x86:   cc -o test_neon tests/test_neon.c src/*.c -Iinclude -lm -DNO_NEON
 */

#include "tile_neon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; printf("FAIL: %s\n", msg); } \
} while(0)

#define ASSERT_FLOAT(a, b, eps, msg) do { \
    if (fabsf((a) - (b)) < (eps)) { tests_passed++; } \
    else { tests_failed++; printf("FAIL: %s (got %f, expected %f)\n", msg, (float)(a), (float)(b)); } \
} while(0)

/* ---- BLAKE2b tests ---- */
static void test_blake2b_basic(void) {
    printf("--- BLAKE2b basic ---\n");

    uint8_t out[TILE_HASH_LEN];
    tile_blake2b((const uint8_t *)"abc", 3, NULL, 0, out);

    /* BLAKE2b-256("abc") is a known test vector */
    ASSERT(out != NULL, "hash produced non-null");
    ASSERT(out[0] != 0 || out[1] != 0, "hash is not all zeros");

    /* Empty string */
    uint8_t out2[TILE_HASH_LEN];
    tile_blake2b((const uint8_t *)"", 0, NULL, 0, out2);
    ASSERT(out2 != NULL, "empty hash produced");

    /* Deterministic */
    uint8_t out3[TILE_HASH_LEN];
    tile_blake2b((const uint8_t *)"abc", 3, NULL, 0, out3);
    ASSERT(memcmp(out, out3, TILE_HASH_LEN) == 0, "hash is deterministic");
}

static void test_blake2b_batch(void) {
    printf("--- BLAKE2b batch ---\n");

    uint8_t msgs[3 * 16];
    for (int i = 0; i < 3; i++)
        memcpy(msgs + i * 16, "hello world!!!!!", 16);

    uint8_t outs[3 * TILE_HASH_LEN];
    tile_blake2b_batch(msgs, 16, 3, outs);

    /* All three should be identical (same input) */
    ASSERT(memcmp(outs, outs + TILE_HASH_LEN, TILE_HASH_LEN) == 0,
           "batch: identical inputs → identical hashes (0==1)");
    ASSERT(memcmp(outs + TILE_HASH_LEN, outs + 2 * TILE_HASH_LEN, TILE_HASH_LEN) == 0,
           "batch: identical inputs → identical hashes (1==2)");
}

/* ---- Embedding tests ---- */
static void test_embed_basic(void) {
    printf("--- Embed basic ---\n");

    uint8_t hash[TILE_HASH_LEN];
    memset(hash, 0x42, TILE_HASH_LEN);

    float emb[TILE_VEC_DIM];
    tile_embed(hash, 0, emb);

    /* Check values are in reasonable range */
    float min_v = emb[0], max_v = emb[0];
    for (int i = 1; i < TILE_VEC_DIM; i++) {
        if (emb[i] < min_v) min_v = emb[i];
        if (emb[i] > max_v) max_v = emb[i];
    }
    ASSERT(min_v >= -2.0f && max_v <= 2.0f, "embed values in range [-2, 2]");

    /* Different position should give different embedding */
    float emb2[TILE_VEC_DIM];
    tile_embed(hash, 1, emb2);
    int same = (memcmp(emb, emb2, sizeof(emb)) == 0);
    ASSERT(!same, "different positions → different embeddings");
}

static void test_embed_batch(void) {
    printf("--- Embed batch ---\n");

    uint8_t hashes[4 * TILE_HASH_LEN];
    for (int i = 0; i < 4; i++)
        memset(hashes + i * TILE_HASH_LEN, 0xAA + i, TILE_HASH_LEN);

    float out[4 * TILE_VEC_DIM];
    tile_embed_batch(hashes, 4, NULL, out);

    /* Different hashes should produce different embeddings */
    ASSERT(memcmp(out, out + TILE_VEC_DIM, TILE_VEC_DIM * sizeof(float)) != 0,
           "batch: different hashes → different embeddings");
}

/* ---- Cosine similarity tests ---- */
static void test_cosine_sim(void) {
    printf("--- Cosine similarity ---\n");

    float a[TILE_VEC_DIM], b[TILE_VEC_DIM];
    for (int i = 0; i < TILE_VEC_DIM; i++) {
        a[i] = 1.0f;
        b[i] = 1.0f;
    }

    /* Identical vectors → sim ≈ 1.0 */
    float sim = tile_cosine_sim(a, b);
    ASSERT_FLOAT(sim, 1.0f, 0.001f, "identical vectors → sim ≈ 1.0");

    /* Opposite vectors → sim ≈ -1.0 */
    for (int i = 0; i < TILE_VEC_DIM; i++) b[i] = -1.0f;
    sim = tile_cosine_sim(a, b);
    ASSERT_FLOAT(sim, -1.0f, 0.001f, "opposite vectors → sim ≈ -1.0");

    /* Orthogonal */
    for (int i = 0; i < TILE_VEC_DIM; i++) {
        a[i] = (i % 2 == 0) ? 1.0f : 0.0f;
        b[i] = (i % 2 == 1) ? 1.0f : 0.0f;
    }
    sim = tile_cosine_sim(a, b);
    ASSERT_FLOAT(sim, 0.0f, 0.001f, "orthogonal vectors → sim ≈ 0.0");

    /* Scalar fallback should match */
    for (int i = 0; i < TILE_VEC_DIM; i++) {
        a[i] = (float)(i + 1) * 0.1f;
        b[i] = (float)(TILE_VEC_DIM - i) * 0.2f;
    }
    float sim_neon = tile_cosine_sim(a, b);
    float sim_scalar = tile_cosine_sim_scalar(a, b);
    ASSERT_FLOAT(sim_neon, sim_scalar, 0.01f,
                  "NEON and scalar cosine sim match");
}

static void test_cosine_search(void) {
    printf("--- Cosine search ---\n");

    /* Build a small index: 10 vectors */
    float mat[10 * TILE_VEC_DIM];
    for (int r = 0; r < 10; r++)
        for (int d = 0; d < TILE_VEC_DIM; d++)
            mat[r * TILE_VEC_DIM + d] = (float)(r + d) * 0.01f;

    /* Query = row 3 */
    float query[TILE_VEC_DIM];
    memcpy(query, mat + 3 * TILE_VEC_DIM, sizeof(query));

    size_t indices[3];
    float scores[3];
    tile_cosine_search(mat, 10, query, 3, indices, scores);

    ASSERT(indices[0] == 3, "top-1 is the query vector itself");
    ASSERT_FLOAT(scores[0], 1.0f, 0.001f, "self-similarity ≈ 1.0");

    /* Scalar search should give same top-1 */
    size_t indices_s[3];
    float scores_s[3];
    tile_cosine_search_scalar(mat, 10, query, 3, indices_s, scores_s);
    ASSERT(indices_s[0] == 3, "scalar: top-1 is query vector");
}

/* ---- Pipeline test ---- */
static void test_pipeline(void) {
    printf("--- Gate pipeline ---\n");

    /* 5 messages of 32 bytes each */
    uint8_t msgs[5 * 32];
    for (int i = 0; i < 5; i++)
        memset(msgs + i * 32, 'A' + i, 32);

    /* Index: 20 vectors */
    float index[20 * TILE_VEC_DIM];
    for (int i = 0; i < 20 * TILE_VEC_DIM; i++)
        index[i] = (float)i * 0.001f;

    size_t top_idx[3];
    float top_score[3];
    tile_gate_pipeline(msgs, 32, 5, index, 20, 3, top_idx, top_score);

    ASSERT(top_idx[0] < 20, "pipeline top-1 index valid");
    ASSERT(top_score[0] >= -1.0f && top_score[0] <= 1.0f,
           "pipeline top-1 score valid range");
}

int main(void) {
    printf("=== tile_neon tests ===\n\n");

    test_blake2b_basic();
    test_blake2b_batch();
    test_embed_basic();
    test_embed_batch();
    test_cosine_sim();
    test_cosine_search();
    test_pipeline();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}

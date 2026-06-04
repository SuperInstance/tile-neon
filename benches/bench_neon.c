/* bench_neon.c — Benchmarks: NEON vs scalar
 *
 * Measures:
 *   1. BLAKE2b: 1K hashes throughput
 *   2. Embed: 64-dim embedding generation
 *   3. Cosine search: 1K, 10K, 100K vectors
 *   4. Full gate pipeline: hash → embed → search
 *
 * Compile:
 *   ARM64: cc -O2 -o bench_neon benches/bench_neon.c src/*.c -Iinclude -lm -march=armv8-a+simd -DUSE_NEON
 *   x86:   cc -O2 -o bench_neon benches/bench_neon.c src/*.c -Iinclude -lm -DNO_NEON
 */

#include "tile_neon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Generate random float vector */
static void rand_vec(float *v, int dim) {
    for (int i = 0; i < dim; i++)
        v[i] = (float)rand() / (float)RAND_MAX - 0.5f;
}

/* ---- 1. BLAKE2b throughput ---- */
static void bench_blake2b(void) {
    printf("\n=== BLAKE2b: 1K hashes ===\n");

    const size_t msg_len = 64;
    const size_t count = 1024;

    uint8_t *msgs = (uint8_t *)malloc(count * msg_len);
    for (size_t i = 0; i < count * msg_len; i++)
        msgs[i] = (uint8_t)(rand() & 0xFF);

    uint8_t *outs = (uint8_t *)malloc(count * TILE_HASH_LEN);

    double t0 = now_sec();
    tile_blake2b_batch(msgs, msg_len, count, outs);
    double t1 = now_sec();

    double elapsed = t1 - t0;
    double per_hash_us = (elapsed / count) * 1e6;
    double throughput = count / elapsed;

    printf("  %zu hashes in %.3f ms\n", count, elapsed * 1000.0);
    printf("  %.1f µs/hash\n", per_hash_us);
    printf("  %.0f hashes/sec\n", throughput);

    free(msgs);
    free(outs);
}

/* ---- 2. Embedding generation ---- */
static void bench_embed(void) {
    printf("\n=== Embed: 64-dim generation ===\n");

    const size_t count = 4096;

    uint8_t *hashes = (uint8_t *)malloc(count * TILE_HASH_LEN);
    for (size_t i = 0; i < count * TILE_HASH_LEN; i++)
        hashes[i] = (uint8_t)(rand() & 0xFF);

    float *embeddings = (float *)malloc(count * TILE_VEC_DIM * sizeof(float));

    double t0 = now_sec();
    tile_embed_batch(hashes, count, NULL, embeddings);
    double t1 = now_sec();

    double elapsed = t1 - t0;
    double per_embed_us = (elapsed / count) * 1e6;

    printf("  %zu embeddings in %.3f ms\n", count, elapsed * 1000.0);
    printf("  %.1f µs/embedding\n", per_embed_us);

    free(hashes);
    free(embeddings);
}

/* ---- 3. Cosine search at scale ---- */
static void bench_cosine_search(size_t n_rows, const char *label) {
    printf("\n=== Cosine search: %s vectors (top-10) ===\n", label);

    const size_t k = 10;

    float *mat = (float *)malloc(n_rows * TILE_VEC_DIM * sizeof(float));
    float query[TILE_VEC_DIM];

    for (size_t i = 0; i < n_rows * TILE_VEC_DIM; i++)
        mat[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    rand_vec(query, TILE_VEC_DIM);

    size_t *indices = (size_t *)malloc(k * sizeof(size_t));
    float *scores = (float *)malloc(k * sizeof(float));

    /* NEON (or scalar via #define) */
    double t0 = now_sec();
    tile_cosine_search(mat, n_rows, query, k, indices, scores);
    double t1 = now_sec();

    double elapsed_neon = t1 - t0;

    /* Scalar fallback */
    size_t *indices_s = (size_t *)malloc(k * sizeof(size_t));
    float *scores_s = (float *)malloc(k * sizeof(float));

    double t2 = now_sec();
    tile_cosine_search_scalar(mat, n_rows, query, k, indices_s, scores_s);
    double t3 = now_sec();

    double elapsed_scalar = t3 - t2;

    printf("  %-8s %.3f ms  (%.0f vecs/sec)\n",
#if USE_NEON
           "NEON",
#else
           "NEON*",
#endif
           elapsed_neon * 1000.0,
           n_rows / elapsed_neon);
    printf("  %-8s %.3f ms  (%.0f vecs/sec)\n",
           "Scalar", elapsed_scalar * 1000.0, n_rows / elapsed_scalar);

    if (elapsed_scalar > 0) {
        printf("  Speedup: %.2fx\n", elapsed_scalar / elapsed_neon);
    }

    free(mat);
    free(indices);
    free(scores);
    free(indices_s);
    free(scores_s);
}

/* ---- 4. Full pipeline ---- */
static void bench_pipeline(void) {
    printf("\n=== Full gate pipeline: hash → embed → search ===\n");

    const size_t n_msgs = 256;
    const size_t msg_len = 64;
    const size_t index_rows = 10000;
    const size_t top_k = 10;

    uint8_t *msgs = (uint8_t *)malloc(n_msgs * msg_len);
    for (size_t i = 0; i < n_msgs * msg_len; i++)
        msgs[i] = (uint8_t)(rand() & 0xFF);

    float *index = (float *)malloc(index_rows * TILE_VEC_DIM * sizeof(float));
    for (size_t i = 0; i < index_rows * TILE_VEC_DIM; i++)
        index[i] = (float)rand() / (float)RAND_MAX - 0.5f;

    size_t *top_idx = (size_t *)malloc(top_k * sizeof(size_t));
    float *top_score = (float *)malloc(top_k * sizeof(float));

    double t0 = now_sec();
    tile_gate_pipeline(msgs, msg_len, n_msgs, index, index_rows,
                       top_k, top_idx, top_score);
    double t1 = now_sec();

    double elapsed = t1 - t0;

    printf("  %zu msgs × %zu-index search in %.3f ms\n",
           n_msgs, index_rows, elapsed * 1000.0);
    printf("  %.1f µs/msg\n", (elapsed / n_msgs) * 1e6);

    free(msgs);
    free(index);
    free(top_idx);
    free(top_score);
}

int main(void) {
    srand(42);

    printf("tile_neon benchmarks\n");
#if USE_NEON
    printf("Backend: ARM NEON\n");
#else
    printf("Backend: Scalar (NO_NEON)\n");
#endif
    printf("Vector dim: %d\n", TILE_VEC_DIM);

    bench_blake2b();
    bench_embed();
    bench_cosine_search(1000, "1K");
    bench_cosine_search(10000, "10K");
    bench_cosine_search(100000, "100K");
    bench_pipeline();

    printf("\nDone.\n");
    return 0;
}

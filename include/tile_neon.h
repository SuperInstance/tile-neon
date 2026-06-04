#ifndef TILE_NEON_H
#define TILE_NEON_H

#include <stdint.h>
#include <stddef.h>

/* Detect NEON availability */
#if defined(__aarch64__) && !defined(NO_NEON)
#define USE_NEON 1
#include <arm_neon.h>
#else
#define USE_NEON 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Configuration                                                       */
/* ------------------------------------------------------------------ */

#define TILE_VEC_DIM       64       /* embedding dimensionality          */
#define TILE_HASH_LEN      32       /* BLAKE2b output length (bytes)     */
#define TILE_BLOCK_BYTES   128      /* BLAKE2b block size                */

/* ------------------------------------------------------------------ */
/* BLAKE2b (optionally NEON-accelerated)                               */
/* ------------------------------------------------------------------ */

void tile_blake2b(const uint8_t *msg, size_t msg_len,
                  const uint8_t *key, size_t key_len,
                  uint8_t out[TILE_HASH_LEN]);

/* Batch: hash N messages, each of the same length */
void tile_blake2b_batch(const uint8_t *msgs, size_t msg_len,
                        size_t count, uint8_t *outs);

/* ------------------------------------------------------------------ */
/* Position-aware embedding                                            */
/* ------------------------------------------------------------------ */

/* Generate a TILE_VEC_DIM embedding from a hash digest.
   pos is the token position index (for positional encoding).         */
void tile_embed(const uint8_t hash[TILE_HASH_LEN],
                int pos,
                float out[TILE_VEC_DIM]);

/* Batch embed N hashes */
void tile_embed_batch(const uint8_t *hashes, size_t count,
                      const int *positions,
                      float *out);  /* count * TILE_VEC_DIM */

/* ------------------------------------------------------------------ */
/* Cosine similarity search                                            */
/* ------------------------------------------------------------------ */

/* Single cosine similarity between two TILE_VEC_DIM vectors */
float tile_cosine_sim(const float *a, const float *b);

/* Search: find the top-k vectors in a flat matrix.
   mat  = rows * TILE_VEC_DIM  (row-major)
   query = TILE_VEC_DIM
   out_indices, out_scores = pre-allocated arrays of size k           */
void tile_cosine_search(const float *mat, size_t rows,
                        const float *query,
                        size_t k,
                        size_t *out_indices,
                        float *out_scores);

/* Scalar fallback versions (always available for benchmarking) */
float tile_cosine_sim_scalar(const float *a, const float *b);
void  tile_cosine_search_scalar(const float *mat, size_t rows,
                                const float *query,
                                size_t k,
                                size_t *out_indices,
                                float *out_scores);

/* ------------------------------------------------------------------ */
/* Full gate pipeline: hash -> embed -> search                         */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t  hash[TILE_HASH_LEN];
    float    embedding[TILE_VEC_DIM];
} tile_entry_t;

void tile_gate_pipeline(const uint8_t *msgs, size_t msg_len,
                        size_t n_msgs,
                        const float *index_mat, size_t index_rows,
                        size_t top_k,
                        size_t *out_indices,
                        float  *out_scores);

#ifdef __cplusplus
}
#endif

#endif /* TILE_NEON_H */

/* search_scalar.c — Pure scalar fallback for comparison benchmarking */

#include "tile_neon.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>

static float dot_scalar(const float *a, const float *b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; i++) s += a[i] * b[i];
    return s;
}

static float norm_scalar(const float *a, int dim) {
    return sqrtf(dot_scalar(a, a, dim));
}

float tile_cosine_sim_scalar(const float *a, const float *b) {
    float d = dot_scalar(a, b, TILE_VEC_DIM);
    float na = norm_scalar(a, TILE_VEC_DIM);
    float nb = norm_scalar(b, TILE_VEC_DIM);
    float denom = na * nb;
    return denom > 0.0f ? d / denom : 0.0f;
}

void tile_cosine_search_scalar(const float *mat, size_t rows,
                               const float *query,
                               size_t k,
                               size_t *out_indices,
                               float *out_scores) {
    float *scores = (float *)malloc(rows * sizeof(float));
    float query_norm = norm_scalar(query, TILE_VEC_DIM);

    for (size_t r = 0; r < rows; r++) {
        const float *row = mat + r * TILE_VEC_DIM;
        float d = dot_scalar(row, query, TILE_VEC_DIM);
        float n = norm_scalar(row, TILE_VEC_DIM);
        float denom = n * query_norm;
        scores[r] = denom > 0.0f ? d / denom : 0.0f;
    }

    /* Top-k selection */
    for (size_t i = 0; i < k; i++) {
        float best = -FLT_MAX;
        size_t best_idx = 0;
        for (size_t j = 0; j < rows; j++) {
            if (scores[j] > best) {
                best = scores[j];
                best_idx = j;
            }
        }
        out_indices[i] = best_idx;
        out_scores[i] = best;
        scores[best_idx] = -FLT_MAX;
    }

    free(scores);
}

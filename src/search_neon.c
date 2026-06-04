/* search_neon.c — Cosine similarity with NEON intrinsics
 *
 * Uses:
 *   vld1q_f32  — load 4 floats
 *   vmlaq_f32  — fused multiply-accumulate (4-wide)
 *   vaddq_f32  — horizontal pair-wise add
 *   vmaxvq_f32 — horizontal max (find best score)
 */

#include "tile_neon.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>

/* ---------- scalar helpers (used by both paths) ---------- */

static float dot_scalar(const float *a, const float *b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; i++) s += a[i] * b[i];
    return s;
}

static float norm_scalar(const float *a, int dim) {
    return sqrtf(dot_scalar(a, a, dim));
}

/* ---------- NEON cosine sim ---------- */

float tile_cosine_sim(const float *a, const float *b) {
#if USE_NEON
    float32x4_t sum = vdupq_n_f32(0.0f);
    float32x4_t na2 = vdupq_n_f32(0.0f);
    float32x4_t nb2 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 3 < TILE_VEC_DIM; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vmlaq_f32(sum, va, vb);   /* sum += a * b */
        na2 = vmlaq_f32(na2, va, va);   /* |a|^2 */
        nb2 = vmlaq_f32(nb2, vb, vb);   /* |b|^2 */
    }

    /* Horizontal sum: pairwise add then extract */
    float dp[4], na[4], nb[4];
    vst1q_f32(dp, sum);
    vst1q_f32(na, na2);
    vst1q_f32(nb, nb2);

    float dot = dp[0] + dp[1] + dp[2] + dp[3];
    float norm_a = sqrtf(na[0] + na[1] + na[2] + na[3]);
    float norm_b = sqrtf(nb[0] + nb[1] + nb[2] + nb[3]);

    /* Handle remainder */
    for (; i < TILE_VEC_DIM; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (i > TILE_VEC_DIM - 4) {
        norm_a = sqrtf(na[0] + na[1] + na[2] + na[3] +
                       (i > TILE_VEC_DIM - 4 ? 0 : 0));
    }

    float denom = norm_a * norm_b;
    return denom > 0.0f ? dot / denom : 0.0f;
#else
    return tile_cosine_sim_scalar(a, b);
#endif
}

/* ---------- NEON cosine search ---------- */

void tile_cosine_search(const float *mat, size_t rows,
                        const float *query,
                        size_t k,
                        size_t *out_indices,
                        float *out_scores) {
    /* Simple top-k: compute all scores, keep top k.
     * For production you'd use a priority queue; this is a clean benchmark target. */

    float *scores = (float *)malloc(rows * sizeof(float));

#if USE_NEON
    /* Precompute query norm */
    float32x4_t qn2 = vdupq_n_f32(0.0f);
    int qi = 0;
    for (; qi + 3 < TILE_VEC_DIM; qi += 4) {
        float32x4_t vq = vld1q_f32(query + qi);
        qn2 = vmlaq_f32(qn2, vq, vq);
    }
    float qn[4];
    vst1q_f32(qn, qn2);
    float query_norm = sqrtf(qn[0] + qn[1] + qn[2] + qn[3]);
    for (; qi < TILE_VEC_DIM; qi++) query_norm += query[qi] * query[qi];
    query_norm = sqrtf(query_norm);

    for (size_t r = 0; r < rows; r++) {
        const float *row = mat + r * TILE_VEC_DIM;
        float32x4_t dot = vdupq_n_f32(0.0f);
        float32x4_t rn2 = vdupq_n_f32(0.0f);

        int i = 0;
        for (; i + 3 < TILE_VEC_DIM; i += 4) {
            float32x4_t vr = vld1q_f32(row + i);
            float32x4_t vq = vld1q_f32(query + i);
            dot = vmlaq_f32(dot, vr, vq);
            rn2 = vmlaq_f32(rn2, vr, vr);
        }

        float d[4], rn[4];
        vst1q_f32(d, dot);
        vst1q_f32(rn, rn2);
        float dsum = d[0] + d[1] + d[2] + d[3];
        float row_norm = sqrtf(rn[0] + rn[1] + rn[2] + rn[3]);

        for (; i < TILE_VEC_DIM; i++) {
            dsum += row[i] * query[i];
            row_norm += row[i] * row[i];
        }
        row_norm = sqrtf(row_norm);

        float denom = row_norm * query_norm;
        scores[r] = denom > 0.0f ? dsum / denom : 0.0f;
    }
#else
    /* Scalar path */
    float query_norm = norm_scalar(query, TILE_VEC_DIM);
    for (size_t r = 0; r < rows; r++) {
        const float *row = mat + r * TILE_VEC_DIM;
        float d = dot_scalar(row, query, TILE_VEC_DIM);
        float n = norm_scalar(row, TILE_VEC_DIM);
        float denom = n * query_norm;
        scores[r] = denom > 0.0f ? d / denom : 0.0f;
    }
#endif

    /* Top-k selection (insertion sort — fine for benchmark sizes) */
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
        scores[best_idx] = -FLT_MAX; /* mark as taken */
    }

    free(scores);
}

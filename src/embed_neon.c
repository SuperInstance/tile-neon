/* embed_neon.c — Position-aware embedding with NEON intrinsics
 *
 * Uses vld1q_f32 / vst1q_f32 for 4-wide load/store,
 * vmulq_f32 / vaddq_f32 for vector arithmetic.
 */

#include "tile_neon.h"
#include <math.h>
#include <string.h>

static inline float hash_byte_to_float(uint8_t b) {
    return (float)b / 127.5f - 1.0f;
}

void tile_embed_neon(const uint8_t hash[TILE_HASH_LEN],
                     int pos,
                     float out[TILE_VEC_DIM]) {
    const float pos_scale = 1.0f / 10000.0f;

#if USE_NEON
    int32x4_t vpos = vdupq_n_s32(pos);
    int i = 0;

    for (; i + 3 < TILE_VEC_DIM; i += 4) {
        /* Compute dimensional frequencies */
        float freqs[4];
        for (int j = 0; j < 4; j++) {
            freqs[j] = pos_scale * powf(10.0f, (float)(i + j) / (float)TILE_VEC_DIM * 4.0f);
        }

        float32x4_t vfreq = vld1q_f32(freqs);

        /* pos * freq for each dimension */
        float pf[4];
        for (int j = 0; j < 4; j++) pf[j] = (float)pos * freqs[j];
        float32x4_t vpf = vld1q_f32(pf);

        /* sin(pos * freq) — compute scalar, load as vector */
        float sin_vals[4];
        for (int j = 0; j < 4; j++) sin_vals[j] = sinf(pf[j]);
        float32x4_t vsin = vld1q_f32(sin_vals);

        /* Hash contribution */
        float hash_vals[4];
        for (int j = 0; j < 4; j++) {
            uint8_t b = hash[(i + j) % TILE_HASH_LEN] ^ (uint8_t)(pos & 0xFF);
            hash_vals[j] = hash_byte_to_float(b);
        }
        float32x4_t vhash = vld1q_f32(hash_vals);

        /* Scale positional encoding */
        float32x4_t v01 = vdupq_n_f32(0.1f);
        float32x4_t vpos_enc = vmulq_f32(vsin, v01);

        /* out = hash + 0.1 * sin(pos * freq) */
        float32x4_t vout = vaddq_f32(vhash, vpos_enc);
        vst1q_f32(out + i, vout);
    }

    /* Handle remainder */
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

void tile_embed_batch_neon(const uint8_t *hashes, size_t count,
                           const int *positions,
                           float *out) {
    for (size_t i = 0; i < count; i++) {
        tile_embed_neon(hashes + i * TILE_HASH_LEN,
                        positions ? positions[i] : (int)i,
                        out + i * TILE_VEC_DIM);
    }
}

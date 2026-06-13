# tile-neon — ARM NEON SIMD Tile Operations for Edge Deployment

BLAKE2b hashing, position-aware embedding, and cosine similarity search — all with NEON-accelerated vector paths and scalar fallbacks for cross-platform development.

## What It Does

Three core operations for hash-based tile processing on ARM64, using 128-bit NEON SIMD intrinsics for 4-wide float operations:

1. **BLAKE2b Hashing** — Full BLAKE2b implementation with NEON-accelerated batch word scheduling
2. **Position-Aware Embedding** — Generates 64-dim float vectors from hash digests using sinusoidal positional encoding
3. **Cosine Similarity Search** — Vectorized dot product and norm computation with top-K selection

Each operation has a NEON path (`USE_NEON`) and a scalar path (`NO_NEON`) for benchmarking and x86 development.

## Architecture

| File | Purpose | Key Intrinsics |
|------|---------|----------------|
| `src/tile_neon.c` | Core BLAKE2b, pipeline orchestration | Scalar (portable C) |
| `src/hash_blake2b.c` | Batch BLAKE2b with NEON loads | `vld1q_u8` |
| `src/embed_neon.c` | Position-aware embedding | `vld1q_f32`, `vst1q_f32`, `vmulq_f32`, `vaddq_f32` |
| `src/search_neon.c` | Cosine sim + search | `vmlaq_f32` (FMA), `vaddq_f32`, `vmaxvq_f32` |
| `src/search_scalar.c` | Pure scalar fallback | — |

### NEON Intrinsics Reference

| Intrinsic | Operation | Width |
|-----------|-----------|-------|
| `vld1q_f32` / `vst1q_f32` | Load/store 4 floats | 128-bit |
| `vmlaq_f32` | Fused multiply-accumulate | 4-wide |
| `vaddq_f32` / `vmulq_f32` | Vector add/multiply | 4-wide |
| `vmaxvq_f32` | Horizontal max | 4-wide |
| `vld1q_u8` | Byte load for hashing | 128-bit |
| `vdupq_n_f32` | Broadcast scalar to vector | 128-bit |

## Key Types (C API)

```c
// Configuration (tile_neon.h)
#define TILE_VEC_DIM     64    // Embedding dimensionality
#define TILE_HASH_LEN    32    // BLAKE2b output length (bytes)
#define TILE_BLOCK_BYTES 128   // BLAKE2b block size

// BLAKE2b
void tile_blake2b(const uint8_t *msg, size_t msg_len,
                  const uint8_t *key, size_t key_len,
                  uint8_t out[TILE_HASH_LEN]);
void tile_blake2b_batch(const uint8_t *msgs, size_t msg_len,
                        size_t count, uint8_t *outs);

// Embedding
void tile_embed(const uint8_t hash[TILE_HASH_LEN], int pos,
                float out[TILE_VEC_DIM]);
void tile_embed_batch(const uint8_t *hashes, size_t count,
                      const int *positions, float *out);

// Search
float tile_cosine_sim(const float *a, const float *b);
void tile_cosine_search(const float *mat, size_t rows,
                        const float *query, size_t k,
                        size_t *out_indices, float *out_scores);

// Scalar fallbacks (for benchmarking)
float tile_cosine_sim_scalar(const float *a, const float *b);
void tile_cosine_search_scalar(const float *mat, size_t rows,
                               const float *query, size_t k,
                               size_t *out_indices, float *out_scores);

// Full pipeline: hash → embed → search
void tile_gate_pipeline(const uint8_t *msgs, size_t msg_len, size_t n_msgs,
                        const float *index_mat, size_t index_rows,
                        size_t top_k, size_t *out_indices, float *out_scores);
```

## Installation

### Building

**x86 development (scalar path):**
```bash
make              # Defaults to -DNO_NEON
make test         # Build + run tests
make bench        # Build + run benchmarks
```

**ARM64 production (NEON path):**
```bash
make NEON=1       # -march=armv8-a+simd -DUSE_NEON
make test NEON=1
make bench NEON=1
```

### Conditional Compilation

- `__aarch64__` + no `NO_NEON` defined → NEON path
- `NO_NEON` defined → scalar path (works on x86 for development)

## Usage

```c
#include "tile_neon.h"

// Hash a message
uint8_t hash[TILE_HASH_LEN];
tile_blake2b((const uint8_t *)"hello", 5, NULL, 0, hash);

// Embed at position 0
float vec[TILE_VEC_DIM];
tile_embed(hash, 0, vec);

// Search a database of 10K vectors
float db[10000 * TILE_VEC_DIM];
size_t indices[10];
float scores[10];
tile_cosine_search(db, 10000, vec, 10, indices, scores);

// Full pipeline: hash → embed → search in one call
tile_gate_pipeline(msgs, msg_len, n_msgs, db, db_rows, 10, indices, scores);
```

## Benchmarks

- **BLAKE2b**: 1K hashes, throughput (hashes/sec)
- **Embed**: 64-dim embedding generation (µs/embedding)
- **Cosine search**: 1K / 10K / 100K vectors, top-10 (NEON vs scalar)
- **Full pipeline**: hash → embed → search end-to-end

## Cross-Platform

The scalar fallback compiles cleanly on x86 with `-DNO_NEON`, enabling development and benchmarking on standard hardware before deploying to ARM64 targets.

## License

MIT

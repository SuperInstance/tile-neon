# tile-neon

ARM NEON SIMD tile operations for edge deployment.

BLAKE2b hashing, position-aware embedding, and cosine similarity search — all with NEON-accelerated paths and scalar fallbacks.

## Building

### x86 (development, scalar path)

```bash
make          # defaults to -DNO_NEON
make test     # build + run tests
make bench    # build + run benchmarks
```

### ARM64 (production, NEON path)

```bash
make NEON=1          # -march=armv8-a+simd -DUSE_NEON
make test NEON=1
make bench NEON=1
```

## Architecture

| File | Purpose |
|------|---------|
| `src/tile_neon.c` | Core BLAKE2b, embedding, pipeline — NEON/scalar dual path |
| `src/hash_blake2b.c` | BLAKE2b batch with NEON-accelerated word scheduling |
| `src/embed_neon.c` | Position-aware embedding using `vld1q_f32` / `vst1q_f32` |
| `src/search_neon.c` | Cosine similarity using `vmlaq_f32`, `vaddq_f32`, `vmaxvq_f32` |
| `src/search_scalar.c` | Pure scalar fallback for benchmarking comparison |

## NEON Intrinsics Used

| Intrinsic | Operation | Width |
|-----------|-----------|-------|
| `vld1q_f32` / `vst1q_f32` | Load/store 4 floats | 128-bit |
| `vmlaq_f32` | Fused multiply-accumulate | 4-wide |
| `vaddq_f32` / `vmulq_f32` | Vector add/multiply | 4-wide |
| `vmaxvq_f32` | Horizontal max | 4-wide |
| `vld1q_u8` | Byte operations for hashing | 128-bit |

## Benchmarks

- **BLAKE2b**: 1K hashes, throughput (hashes/sec)
- **Embed**: 64-dim embedding generation (µs/embedding)
- **Cosine search**: 1K / 10K / 100K vectors, top-10 (NEON vs scalar)
- **Full pipeline**: hash → embed → search end-to-end

## Conditional Compilation

- `__aarch64__` + no `NO_NEON` → NEON path
- `NO_NEON` defined → scalar path (works on x86 for development)

## License

MIT

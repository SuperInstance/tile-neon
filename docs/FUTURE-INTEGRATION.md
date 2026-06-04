# Future Integration: tile-neon

## Current State
ARM NEON SIMD tile operations for edge deployment. BLAKE2b hashing, position-aware embedding, and cosine similarity search with NEON-accelerated paths and scalar fallbacks. Targets ARM64 (Jetson, Raspberry Pi).

## Integration Opportunities

### With JetsonClaw1 edge rooms
tile-neon provides the SIMD acceleration for rooms running on Jetson hardware. Jetson's ARM cores benefit from NEON vectorization: 128-bit SIMD processes 4 floats per cycle. The 10M hash/sec, 1M embed/sec, 10B comparison/sec benchmarks apply to edge rooms.

### With construct-core Layer 1
Layer 1 (SyncConstruct, no_std + alloc) runs on ARM targets. tile-neon's SIMD kernels accelerate Layer 1's query operations: fast hashing for skill lookup, fast embedding for similarity search, fast cosine for neighbor finding.

### With room-as-codespace edge deployment
When a room yokes out from Codespace to Jetson (via codespace-edge-rd), tile-neon provides the computational acceleration. The same room logic that ran on x86 in the Codespace now runs on ARM with NEON optimization.

## Dormant Ideas Now Unlockable
NEON optimization was for tile-specific operations. Now it accelerates the entire room computation pipeline on edge hardware. Every ARM device in the fleet benefits.

## Potential in Mature Systems
Every ARM device (Jetson, Raspberry Pi, mobile) runs tile-neon for room computation. The fleet extends to edge hardware with near-desktop performance through SIMD optimization.

## Cross-Pollination Ideas
- **tile-cuda**: CUDA counterpart for NVIDIA GPUs
- **tile-opencl**: OpenCL counterpart for non-NVIDIA GPUs
- **lever-runner-carapace**: Carapace's hashing benefits from NEON acceleration

## Dependencies for Next Steps
- Integration with ternary-cell tick on ARM
- Testing on Jetson Orin Nano hardware
- Room-specific NEON kernel configurations

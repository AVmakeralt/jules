//===- PolyhedralOptPass.h - Polyhedral (Affine) Optimization Pass ----------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the polyhedral (affine) optimization pass for the
// Jules compiler. This pass is the "Polly" of the AOT pipeline — it
// treats loops as multidimensional geometric spaces (integer polyhedra)
// and restructures them for maximum hardware utilization.
//
// The pass performs three core transformations:
//
//   1. Loop Tiling: Chop large matrix operations into cache-friendly tiles
//      (e.g., 32x32 blocks that fit in L1/L2 cache)
//
//   2. Loop Skewing: Shift iteration domains to expose parallelism in
//      seemingly sequential loops with data dependencies
//
//   3. Loop Fusion: Merge adjacent loops (e.g., matmul + ReLU) into a
//      single fused kernel to eliminate temporary buffer overhead
//
// Because Jules is AOT-compiled and purely functional, its structures
// are perfectly suited for polyhedral modeling — no pointer aliasing,
// no unstructured jumps, and clean dataflow.
//
// Pipeline integration:
//
//   [Custom MLIR Dialect]
//           │
//           ▼ (AOT optimization)
//   [MLIR Affine & MemRef Dialect]
//           │
//           ▼ (Tiling, Fusion, Skewing)
//   [Hyper-Optimized Affine Loops]
//           │
//           ▼
//   [Lowering to XLA / LLVM]
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_POLYHEDRAL_OPT_PASS_H
#define JULES_PASSES_POLYHEDRAL_OPT_PASS_H

#include <memory>
#include <cstdint>

namespace mlir {
class Pass;
}

namespace jules {

/// Create the polyhedral (affine) optimization pass.
/// This pass converts Jules loops to affine dialect, applies tiling,
/// fusion, and skewing, and lowers back for optimal cache locality
/// and SIMD execution.
std::unique_ptr<mlir::Pass> createPolyhedralOptPass();

/// Create the polyhedral optimization pass with a specific cache size.
/// The cache size is used to determine optimal tile sizes.
std::unique_ptr<mlir::Pass> createPolyhedralOptPass(uint64_t cacheSizeBytes);

} // namespace jules

#endif // JULES_PASSES_POLYHEDRAL_OPT_PASS_H

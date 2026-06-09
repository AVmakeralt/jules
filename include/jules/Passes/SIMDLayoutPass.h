//===- SIMDLayoutPass.h - SIMD Vectorization Layout Optimization Pass -------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SIMD vectorization layout optimization pass.
//
// Since Jules targets XLA via StableHLO, we don't write raw x86 AVX-512
// or ARM Neon assembly loops ourselves. Instead, this pass structures
// the MLIR code so that XLA's backend can instantly recognize and emit
// parallel SIMD hardware instructions.
//
// The pass ensures:
//   1. Row-major, strictly aligned memory allocations (dense literals
//      or aligned memref allocations)
//   2. Contiguous memory layout for element-wise operations
//   3. Proper vector dialect annotations for XLA to map to SIMD loops
//   4. Memory layout hints for optimal cache line utilization
//
// When XLA sees a contiguous stablehlo.add on a tensor<32x784xf32>,
// it maps this to a fused vector loop, automatically grouping operations
// into the hardware's native SIMD width (e.g., 16 floats at once using
// a single 512-bit vector register).
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_SIMD_LAYOUT_PASS_H
#define JULES_PASSES_SIMD_LAYOUT_PASS_H

#include <memory>

namespace mlir {
class Pass;
}

namespace jules {

/// Create the SIMD vectorization layout optimization pass.
/// This pass structures MLIR code for optimal XLA SIMD generation.
std::unique_ptr<mlir::Pass> createSIMDLayoutPass();

} // namespace jules

#endif // JULES_PASSES_SIMD_LAYOUT_PASS_H

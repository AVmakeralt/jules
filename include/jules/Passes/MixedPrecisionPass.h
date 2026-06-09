//===- MixedPrecisionPass.h - Mixed Precision Optimization Pass -------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the mixed precision optimization pass for the Jules
// compiler. The pass inserts CastOp operations to downcast compute-heavy
// operations (matmul, convolutions) to lower precision (bf16 or fp8) while
// keeping accumulation in f32 for numerical correctness.
//
// The pass performs the following transformations:
//
//   1. MatMulOp: Downcast inputs to target precision (bf16/fp8),
//      accumulate in f32, cast result back to original type
//
//   2. Activation ops (Relu, Sigmoid, Tanh): Downcast to lower precision
//      for standalone ops; keep f32 when following a matmul accumulation
//
//   3. Elementwise binary ops (Add, Sub, Mul, Div): Cast mixed operands
//      to the lower precision, stay in lower precision if both already are
//
//   4. Reduction ops (Sum, Mean): Always accumulate in f32 to avoid
//      precision loss; cast input up before reduction, result back after
//
// Configuration:
//   - targetComputePrecision: "bf16" or "fp8" (default "bf16")
//   - accumulationPrecision: always "f32" (hardcoded for correctness)
//   - minOpCountForMixedPrecision: only apply to functions with >= N ops
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_MIXED_PRECISION_PASS_H
#define JULES_PASSES_MIXED_PRECISION_PASS_H

#include <memory>
#include <string>

namespace mlir {
class Pass;
}

namespace jules {

/// Create the mixed precision optimization pass with default settings
/// (target precision = bf16, min op count = 5).
std::unique_ptr<mlir::Pass> createMixedPrecisionPass();

/// Create the mixed precision optimization pass with a specific target
/// precision ("bf16" or "fp8").
std::unique_ptr<mlir::Pass> createMixedPrecisionPass(std::string targetPrecision);

} // namespace jules

#endif // JULES_PASSES_MIXED_PRECISION_PASS_H

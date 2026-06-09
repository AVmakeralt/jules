//===- QuantizePass.h - Quantization Pass ----------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Quantization pass for the Jules MLIR dialect.
//
// The pass inserts fake-quantization nodes for inference optimization.
// Fake quantization simulates the effect of int8 quantization during training
// (quantization-aware training) by quantizing to int8 and dequantizing back
// in the forward pass.
//
// During inference mode, fake_quant ops are replaced with actual int8
// CastOp + MatMulInt8 for efficient execution.
//
// Options:
//   - quantizeWeights (default: true)   — insert fake_quant for matmul weights
//   - quantizeActivations (default: false) — insert fake_quant for relu outputs
//   - numBits (default: 8)              — quantization bit width
//   - perChannel (default: true)        — per-channel quantization for weights
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_QUANTIZE_PASS_H
#define JULES_PASSES_QUANTIZE_PASS_H

#include <memory>

namespace mlir {
class Pass;
}

namespace jules {

/// Create the Quantization pass with default options.
std::unique_ptr<mlir::Pass> createQuantizePass();

/// Create the Quantization pass with explicit options.
std::unique_ptr<mlir::Pass> createQuantizePass(
    bool quantizeWeights,
    bool quantizeActivations,
    unsigned numBits,
    bool perChannel);

} // namespace jules

#endif // JULES_PASSES_QUANTIZE_PASS_H

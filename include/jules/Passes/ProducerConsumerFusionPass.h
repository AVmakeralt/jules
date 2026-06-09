//===- ProducerConsumerFusionPass.h - Producer-Consumer Fusion Pass --------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Producer-Consumer Fusion pass for the Jules MLIR
// dialect. This pass identifies chains of fusible operations (elementwise
// chains, matmul+activation, reduction+elementwise) and wraps them into
// fusion clusters to minimize kernel launch overhead and intermediate
// buffer allocation.
//
// Fusion strategies:
//
//   1. Elementwise chain fusion: sequences of pure elementwise ops
//      (add, sub, mul, div, neg, relu, sigmoid, tanh) connected by
//      def-use edges where each intermediate result has exactly one user.
//
//   2. Matmul + activation fusion: MatMulOp followed immediately by
//      ReluOp/SigmoidOp/TanhOp, fused into a single kernel.
//
//   3. Elementwise + reduction fusion: e.g. add → mul → sum becomes
//      one fusion cluster so the reducer can consume elementwise results
//      without materializing intermediates.
//
// When the StableHLO dialect is loaded, the pass emits stablehlo.fusion
// operations. Otherwise it falls back to a jules.fusion wrapper with a
// jules.fusion_group attribute for later lowering.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_PRODUCER_CONSUMER_FUSION_PASS_H
#define JULES_PASSES_PRODUCER_CONSUMER_FUSION_PASS_H

#include <memory>

namespace mlir {
class Pass;
}

namespace jules {

/// Create the Producer-Consumer Fusion pass.
/// This pass fuses elementwise chains, matmul+activation patterns, and
/// elementwise+reduction sequences into fusion clusters for reduced kernel
/// launch overhead and eliminated intermediate buffers.
std::unique_ptr<mlir::Pass> createProducerConsumerFusionPass();

} // namespace jules

#endif // JULES_PASSES_PRODUCER_CONSUMER_FUSION_PASS_H

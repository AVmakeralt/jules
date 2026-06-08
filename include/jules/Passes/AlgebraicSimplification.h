//===- AlgebraicSimplification.h - Algebraic Simplification Pass -----------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the algebraic simplification pass. This pass handles
// mathematical identities and simplifications:
//
//   - x + 0 -> x,  0 + x -> x           (additive identity)
//   - x * 1 -> x,  1 * x -> x           (multiplicative identity)
//   - x * 0 -> 0,  0 * x -> 0           (annihilation)
//   - x - 0 -> x                          (subtraction identity)
//   - x / 1 -> x                          (division identity)
//   - x ^ 0 -> 1,  x ^ 1 -> x           (power identity)
//   - neg(neg(x)) -> x                    (double negation)
//   - add(x, neg(y)) -> sub(x, y)        (negate-to-subtract)
//   - sub(x, x) -> 0                      (self-subtraction)
//   - matmul(x, identity) -> x            (identity matmul)
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_ALGEBRAICSIMPLIFICATION_H
#define JULES_PASSES_ALGEBRAICSIMPLIFICATION_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>

namespace jules {

/// Create the algebraic simplification pass.
std::unique_ptr<mlir::Pass> createAlgebraicSimplificationPass();

/// Populate algebraic simplification patterns.
void populateAlgebraicSimplificationPatterns(mlir::RewritePatternSet &patterns);

} // namespace jules

#endif // JULES_PASSES_ALGEBRAICSIMPLIFICATION_H

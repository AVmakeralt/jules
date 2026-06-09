//===- AutodiffPruning.h - Autodiff Pruning Pass ---------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the autodiff pruning pass. After the autodiff engine
// generates the backward pass, it often leaves behind redundant operations:
//
//   - Adding zero gradients: add(x, 0) -> x
//   - Multiplying by one gradients: mul(x, 1) -> x
//   - Zero gradient through unused paths
//   - Negate(negate(x)) -> x from double negation in sub gradient
//   - Redundant transposes from matmul gradients
//   - Unused gradient accumulation from dead paths
//
// This pass cleans up the autodiff output before graph collapsing
// and StableHLO lowering.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_AUTODIFFPRUNING_H
#define JULES_PASSES_AUTODIFFPRUNING_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>

namespace jules {

/// Create the autodiff pruning pass.
std::unique_ptr<mlir::Pass> createAutodiffPruningPass();

/// Populate autodiff pruning patterns.
void populateAutodiffPruningPatterns(mlir::RewritePatternSet &patterns);

} // namespace jules

#endif // JULES_PASSES_AUTODIFFPRUNING_H

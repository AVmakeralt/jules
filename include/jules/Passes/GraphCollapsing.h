//===- GraphCollapsing.h - Graph Collapsing Pass ---------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the graph collapsing passes for the Jules MLIR dialect.
// Graph collapsing aggressively prunes redundant operations before they reach
// XLA, including:
//
//   - Transpose(transpose(x)) -> x          (identity collapse)
//   - Reshape(reshape(x, s1), s2) -> reshape(x, s2)  (reshape fusion)
//   - Slice(slice(x, ...), ...) -> slice(x, merged)   (slice fusion)
//   - add(x, zeros) -> x                    (additive identity)
//   - mul(x, ones) -> x                     (multiplicative identity)
//   - concat(split(x, ...), ...) -> x       (split/concat roundtrip)
//   - add(x, neg(x)) -> zeros               (cancellation)
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_GRAPHCOLLAPSING_H
#define JULES_PASSES_GRAPHCOLLAPSING_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>

namespace jules {

/// Create the graph collapsing pass.
std::unique_ptr<mlir::Pass> createGraphCollapsingPass();

/// Populate graph collapsing patterns into a RewritePatternSet.
void populateGraphCollapsingPatterns(mlir::RewritePatternSet &patterns);

} // namespace jules

#endif // JULES_PASSES_GRAPHCOLLAPSING_H

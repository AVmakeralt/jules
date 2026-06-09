//===- FusedAutodiffPass.h - Fused Autodiff + Pruning + Collapsing ---------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Fused Automatic Differentiation pass — the "killer
// feature" of the Jules compiler. Unlike conventional autodiff implementations
// (e.g., JAX) which build a bloated backward pass and then optimize it in
// separate passes, FusedAutodiff performs ALL of the following in a single
// topological walk:
//
//   Phase 1: Forward Analysis
//     - Walk the forward graph from output to input
//     - Build a map of which ops are "needed" (transitive deps of output)
//     - Identify which forward op results are used in the backward pass
//
//   Phase 2: Selective Backward Graph Construction (Dead Gradient Pruning)
//     - For each needed forward op, compute ONLY required adjoints
//     - If an adjoint would be zero (e.g., grad of a constant), don't create it
//     - If a gradient contribution would be multiplied by zero, skip it
//     - If a gradient is only used by a dead consumer, skip it
//
//   Phase 3: On-the-fly Graph Collapsing
//     - As backward ops are built, immediately apply simplifications:
//       * 0 * x -> 0 (zero gradient * anything = zero)
//       * x + 0 -> x (no need to accumulate zero gradients)
//       * 1 * x -> x (identity gradient)
//       * transpose(transpose(x)) -> x
//     - The backward pass is already simplified as it's constructed
//
//   Phase 4: Result
//     - The backward pass is typically 1.5-3x the forward pass, not 5-10x
//     - No intermediate "bloated" IR is ever created
//
// This is the key differentiator from JAX/PyTorch: by fusing autodiff,
// pruning, and collapsing into one pass, we avoid the intermediate IR bloat
// that JAX suffers from (where the backward pass is 5-10x the forward pass
// size before optimization).
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_FUSEDAUTODIFFPASS_H
#define JULES_PASSES_FUSEDAUTODIFFPASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <vector>

namespace jules {

/// Create the fused autodiff pass.
/// This pass combines autodiff, dead gradient pruning, graph collapsing,
/// and algebraic simplification into a single walk over the forward graph.
std::unique_ptr<mlir::Pass> createFusedAutodiffPass();

} // namespace jules

#endif // JULES_PASSES_FUSEDAUTODIFFPASS_H

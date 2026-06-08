//===- Passes.h - Jules MLIR Passes ----------------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MLIR transformation passes for the Jules compiler:
//
//   - AutodiffPass: Reverse-mode automatic differentiation at the MLIR level
//   - JulesToStableHLOLoweringPass: Lower Jules dialect to StableHLO
//   - ShapeInferencePass: Resolve symbolic dimensions
//   - GraphCollapsingPass: Aggressive structural pruning
//   - AlgebraicSimplificationPass: Mathematical identity simplification
//   - AutodiffPruningPass: Clean up after autodiff
//   - PGOPass: Profile-guided shape specialization
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_PASSES_H
#define JULES_PASSES_PASSES_H

#include <memory>

namespace mlir {
class Pass;
class Operation;
} // namespace mlir

namespace jules {

class Profiler;

/// Create the Autodiff pass.
std::unique_ptr<mlir::Pass> createAutodiffPass();

/// Create the shape-inference pass.
std::unique_ptr<mlir::Pass> createShapeInferencePass();

/// Create the Jules-to-StableHLO lowering pass.
std::unique_ptr<mlir::Pass> createJulesToStableHLOLoweringPass();

/// Create the graph collapsing pass.
std::unique_ptr<mlir::Pass> createGraphCollapsingPass();

/// Create the algebraic simplification pass.
std::unique_ptr<mlir::Pass> createAlgebraicSimplificationPass();

/// Create the autodiff pruning pass.
std::unique_ptr<mlir::Pass> createAutodiffPruningPass();

/// Create the PGO-informed optimization pass.
std::unique_ptr<mlir::Pass> createPGOPass(const Profiler &profiler);

/// Register all Jules passes with the MLIR pass pipeline.
void registerJulesPasses();

} // namespace jules

#endif // JULES_PASSES_PASSES_H

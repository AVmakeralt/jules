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

/// Create the Autodiff pass. This pass walks the MLIR module, finds
/// jules.grad operations, and expands them into the forward + backward
/// computation graph using reverse-mode automatic differentiation.
std::unique_ptr<mlir::Pass> createAutodiffPass();

/// Create the shape-inference pass. This pass resolves symbolic dimension
/// variables in tensor types based on the constraints imposed by operations.
std::unique_ptr<mlir::Pass> createShapeInferencePass();

/// Create the Jules-to-StableHLO lowering pass. This converts each Jules
/// operation to its StableHLO equivalent, enabling XLA compilation.
std::unique_ptr<mlir::Pass> createJulesToStableHLOLoweringPass();

/// Register all Jules passes with the MLIR pass pipeline.
void registerJulesPasses();

} // namespace jules

#endif // JULES_PASSES_PASSES_H

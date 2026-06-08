//===- ShapePolymorphismPass.h - Shape Polymorphism Pass -------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Shape Polymorphism pass for the Jules MLIR dialect.
//
// Unlike traditional shape inference that lowers all unknown dimensions to
// ShapedType::kDynamic, this pass carries symbolic dimension constraints
// through the pipeline. This enables XLA to specialize on symbolic dimensions
// rather than treating every dynamic dimension as fully opaque.
//
// Algorithm:
//   1. Walk the module looking for functions with dynamic dimensions
//   2. For each function argument with ? dimensions, check for named
//      dimension attributes (e.g., jules.dim_name = "Batch")
//   3. Build a symbolic dimension equality graph (which dims must be equal)
//   4. Propagate symbolic shape constraints through each operation:
//      - MatMulOp: [B, I] × [I, O] → [B, O] with B and I constrained
//      - AddOp: same shape → dimensions are pairwise equal
//      - ReshapeOp: total elements must be equal
//   5. Insert shape.assuming ops around the function body to assert constraints
//   6. After assertion, XLA can specialize on the symbolic dimensions
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_SHAPEPOLYMORPHISM_PASS_H
#define JULES_PASSES_SHAPEPOLYMORPHISM_PASS_H

#include <memory>

namespace mlir {
class Pass;
}

namespace jules {

/// Create the Shape Polymorphism pass.
/// This pass carries symbolic dimension constraints through the pipeline
/// instead of lowering everything to ShapedType::kDynamic, enabling XLA
/// to specialize on symbolic dimensions.
std::unique_ptr<mlir::Pass> createShapePolymorphismPass();

} // namespace jules

#endif // JULES_PASSES_SHAPEPOLYMORPHISM_PASS_H

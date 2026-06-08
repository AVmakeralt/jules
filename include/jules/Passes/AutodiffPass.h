//===- AutodiffPass.h - Jules Autodiff MLIR Pass ---------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the reverse-mode automatic differentiation pass for the
// Jules MLIR dialect. The pass:
//
//   1. Identifies jules.grad operations in the module
//   2. Traces the forward computation graph from the differentiated function
//   3. Constructs the backward (adjoint) graph in reverse topological order
//   4. Replaces the grad op with the computed gradient value
//
// The pass operates entirely at the MLIR level, enabling optimization of the
// forward+backward graph before lowering to StableHLO (e.g. kernel fusion).
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_AUTODIFFPASS_H
#define JULES_PASSES_AUTODIFFPASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"

#include <unordered_map>
#include <vector>

namespace jules {

/// Implementation of the reverse-mode autodiff algorithm on Jules MLIR ops.
class AutodiffEngine {
public:
  /// Differentiate the computation that produces \p output with respect to
  /// \p input, inserting the backward graph using \p builder.
  /// Returns the gradient Value.
  mlir::Value differentiate(mlir::Value output, mlir::Value input,
                            mlir::OpBuilder &builder);

private:
  /// Trace the forward graph to find all operations between input and output.
  std::vector<mlir::Operation *> traceForwardGraph(mlir::Value output,
                                                     mlir::Value input);

  /// For each operation in the forward graph, compute the vector-Jacobian
  /// product (VJP) given the incoming adjoint from the output side.
  void computeAdjoint(mlir::Operation *op, mlir::Value incomingAdjoint,
                      mlir::OpBuilder &builder);

  /// Map from forward Value to its adjoint (gradient) Value.
  std::unordered_map<mlir::Value, mlir::Value> adjoints_;

  /// The seed adjoint (typically 1.0 for the loss output).
  mlir::Value seed_;
};

} // namespace jules

#endif // JULES_PASSES_AUTODIFFPASS_H

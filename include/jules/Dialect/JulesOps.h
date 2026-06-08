//===- JulesOps.h - Jules MLIR Operations ----------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operations in the Jules MLIR dialect. Each operation
// corresponds to a construct in the Jules language:
//
//   jules.add             - element-wise addition
//   jules.mul             - element-wise multiplication
//   jules.matmul          - matrix multiplication
//   jules.sub             - element-wise subtraction
//   jules.div             - element-wise division
//   jules.pow             - power
//   jules.neg             - negation
//   jules.relu            - ReLU activation
//   jules.sigmoid         - sigmoid activation
//   jules.tanh            - tanh activation
//   jules.mean            - mean reduction
//   jules.sum             - sum reduction
//   jules.zeros           - zero-filled tensor
//   jules.ones            - one-filled tensor
//   jules.random          - random tensor
//   jules.cast            - type cast
//   jules.grad            - automatic differentiation
//   jules.transpose       - tensor transpose
//   jules.reshape         - reshape tensor
//   jules.concat          - tensor concatenation
//   jules.slice           - tensor slicing
//   jules.select          - conditional select
//   jules.cmp             - comparison
//   jules.log             - element-wise natural logarithm
//   jules.pad             - tensor padding
//   jules.broadcast_in_dim - broadcast with dimension mapping
//   jules.reduce          - reduction with body region
//   jules.while           - while loop with carried variables
//   jules.parallel        - data-parallel execution primitive
//   jules.extern_kernel   - external kernel call
//   jules.return          - function return
//   jules.constant        - scalar/tensor constant
//   jules.func            - function definition
//   jules.call            - function call
//
//===----------------------------------------------------------------------===//

#ifndef JULES_DIALECT_JULESOPS_H
#define JULES_DIALECT_JULESOPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// TableGen'd operation declarations
#include "jules/Dialect/JulesOps.h.inc"

#endif // JULES_DIALECT_JULESOPS_H

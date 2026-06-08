//===- JulesTypes.h - Jules MLIR Types -------------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares custom types for the Jules MLIR dialect. While most
// tensor types use MLIR's built-in RankedTensorType, Jules also defines:
//
//   - SymbolicDimType: a dimension that is a named symbolic variable
//     (e.g. "Batch", "Features") which gets resolved during shape inference
//
//===----------------------------------------------------------------------===//

#ifndef JULES_DIALECT_JULESTYPES_H
#define JULES_DIALECT_JULESTYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"

// TableGen'd type declarations
#include "jules/Dialect/JulesOpsTypes.h.inc"

#endif // JULES_DIALECT_JULESTYPES_H

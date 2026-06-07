//===- JulesDialect.h - Jules MLIR Dialect ---------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Jules MLIR Dialect. The dialect provides operations
// and types that correspond one-to-one with the Jules language AST, enabling
// analysis and transformation at the MLIR level before lowering to StableHLO.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_DIALECT_JULESDIALECT_H
#define JULES_DIALECT_JULESDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"

// Forward declare TableGen'd include
#include "jules/Dialect/JulesOpsDialect.h.inc"

#endif // JULES_DIALECT_JULESDIALECT_H

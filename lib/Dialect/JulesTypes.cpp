//===- JulesTypes.cpp - Jules MLIR Types Implementation --------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements custom types for the Jules dialect, including
// the BF16, FP8E4M3, and FP8E5M2 types for mixed precision computation.
//
//===----------------------------------------------------------------------===//

#include "jules/Dialect/JulesTypes.h"
#include "jules/Dialect/JulesDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace jules;

#define GET_TYPEDEF_CLASSES
#include "jules/Dialect/JulesOpsTypes.cpp.inc"

// ── BF16Type Implementation ─────────────────────────────────────────────────

BF16Type BF16Type::get(MLIRContext *ctx) {
  // Use the base class's getUnchecked to create or retrieve the uniqued
  // BF16Type instance. Since TypeStorage is a singleton (no parameters),
  // this is a simple lookup in the MLIR context's type uniquer.
  return Base::get(ctx);
}

// ── FP8E4M3Type Implementation ───────────────────────────────────────────────

FP8E4M3Type FP8E4M3Type::get(MLIRContext *ctx) {
  return Base::get(ctx);
}

// ── FP8E5M2Type Implementation ───────────────────────────────────────────────

FP8E5M2Type FP8E5M2Type::get(MLIRContext *ctx) {
  return Base::get(ctx);
}

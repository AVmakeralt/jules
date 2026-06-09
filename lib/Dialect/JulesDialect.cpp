//===- JulesDialect.cpp - Jules MLIR Dialect Implementation ----------------===//
//
// This file implements the Jules MLIR dialect. The dialect is registered
// with the MLIR context and provides the namespace "jules" for all ops
// and types.
//
//===----------------------------------------------------------------------===//

#include "jules/Dialect/JulesDialect.h"
#include "jules/Dialect/JulesOps.h"
#include "jules/Dialect/JulesTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

// Include the full op class declarations (needed for addOperations<> below)
#define GET_OP_CLASSES
#include "jules/Dialect/JulesOps.h.inc"

// Include the definitions for the dialect class (constructor, destructor, etc.)
#include "jules/Dialect/JulesOpsDialect.cpp.inc"

// ── Dialect initialization ─────────────────────────────────────────────────

void jules::JulesDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "jules/Dialect/JulesOps.cpp.inc"
  >();

  // Register custom types for mixed precision support.
  addTypes<BF16Type, FP8E4M3Type, FP8E5M2Type>();

  // Also register any TableGen'd types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "jules/Dialect/JulesOpsTypes.cpp.inc"
      >();
}

// ── Parse / Print for custom types ──────────────────────────────────────────

Type jules::JulesDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword)) return Type();

  // Parse custom Jules types for mixed precision.
  if (keyword == "bf16") {
    return BF16Type::get(getContext());
  }
  if (keyword == "fp8_e4m3") {
    return FP8E4M3Type::get(getContext());
  }
  if (keyword == "fp8_e5m2") {
    return FP8E5M2Type::get(getContext());
  }

  parser.emitError(parser.getNameLoc(), "unknown Jules type: " + keyword);
  return Type();
}

void jules::JulesDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (type.isa<BF16Type>()) {
    os << "bf16";
  } else if (type.isa<FP8E4M3Type>()) {
    os << "fp8_e4m3";
  } else if (type.isa<FP8E5M2Type>()) {
    os << "fp8_e5m2";
  } else {
    llvm_unreachable("unknown Jules type to print");
  }
}

// ── Dialect registration hook ───────────────────────────────────────────────

namespace mlir {
namespace jules {
  // The dialect is auto-registered through the TableGen'd
  // JulesDialect::initialize() and the #include above.
} // namespace jules
} // namespace mlir

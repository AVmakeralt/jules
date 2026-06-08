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

#define GET_DIALECT_CLASSES
#include "jules/Dialect/JulesOps.cpp.inc"

#include "jules/Dialect/JulesOpsDialect.cpp.inc"

// ── Dialect initialization ─────────────────────────────────────────────────

void jules::JulesDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "jules/Dialect/JulesOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "jules/Dialect/JulesOpsTypes.cpp.inc"
      >();
}

// ── Parse / Print for custom types ──────────────────────────────────────────

Type jules::JulesDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword)) return Type();

  // We use MLIR's built-in RankedTensorType for tensor types, so
  // we only need to parse any custom types we define here.
  parser.emitError(parser.getNameLoc(), "unknown Jules type: " + keyword);
  return Type();
}

void jules::JulesDialect::printType(Type type, DialectAsmPrinter &os) const {
  llvm_unreachable("no custom Jules types to print");
}

// ── Dialect registration hook ───────────────────────────────────────────────

namespace mlir {
namespace jules {
  // The dialect is auto-registered through the TableGen'd
  // JulesDialect::initialize() and the #include above.
} // namespace jules
} // namespace mlir

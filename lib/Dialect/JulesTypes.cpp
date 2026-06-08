//===- JulesTypes.cpp - Jules MLIR Types Implementation --------------------===//
//
// This file implements any custom types for the Jules dialect. Most tensor
// types use MLIR's built-in RankedTensorType, so this file is minimal.
//
//===----------------------------------------------------------------------===//

#include "jules/Dialect/JulesTypes.h"
#include "jules/Dialect/JulesDialect.h"

using namespace mlir;
using namespace jules;

#define GET_TYPEDEF_CLASSES
#include "jules/Dialect/JulesOpsTypes.cpp.inc"

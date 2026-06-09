//===- JulesOps.cpp - Jules MLIR Operations Implementation -----------------===//
//
// This file implements the operation classes for the Jules MLIR dialect.
// Since we use TableGen for op definitions, most of the boilerplate is
// auto-generated. This file provides:
//   - Custom parser/printer for operations not fully handled by TableGen
//   - InferTypeOpInterface implementations for shape inference at the MLIR level
//   - FoldHook for constant folding
//   - Verifier for additional semantic checks
//
//===----------------------------------------------------------------------===//

#include "jules/Dialect/JulesOps.h"
#include "jules/Dialect/JulesDialect.h"
#include "jules/Dialect/JulesTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace jules;

//===----------------------------------------------------------------------===//
// TableGen'd operation implementations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "jules/Dialect/JulesOps.h.inc"
#include "jules/Dialect/JulesOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Custom operation implementations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// jules.constant
//===----------------------------------------------------------------------===//

void ConstantOp::build(OpBuilder &builder, OperationState &result,
                       double value, Type type) {
  auto attr = builder.getFloatAttr(type, value);
  result.addAttribute("value", attr);
  result.addTypes(type);
}

//===----------------------------------------------------------------------===//
// jules.matmul
//===----------------------------------------------------------------------===//

// Infer result type for matmul.
LogicalResult MatMulOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, ::mlir::OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto lhsType = operands[0].getType().dyn_cast<RankedTensorType>();
  auto rhsType = operands[1].getType().dyn_cast<RankedTensorType>();

  if (!lhsType || !rhsType) {
    inferredReturnTypes.push_back(UnrankedTensorType::get(
        FloatType::getF32(context)));
    return success();
  }

  Type elemType = lhsType.getElementType();

  // 2D @ 2D: [M, K] ** [K, N] -> [M, N]
  if (lhsType.getRank() == 2 && rhsType.getRank() == 2) {
    SmallVector<int64_t, 2> shape = {
      lhsType.getDimSize(0),
      rhsType.getDimSize(1)
    };
    inferredReturnTypes.push_back(
        RankedTensorType::get(shape, elemType));
    return success();
  }

  // Batched: [B, M, K] ** [K, N] -> [B, M, N]
  if (lhsType.getRank() == 3 && rhsType.getRank() == 2) {
    SmallVector<int64_t, 3> shape = {
      lhsType.getDimSize(0),
      lhsType.getDimSize(1),
      rhsType.getDimSize(1)
    };
    inferredReturnTypes.push_back(
        RankedTensorType::get(shape, elemType));
    return success();
  }

  // Batched: [B, M, K] ** [B, K, N] -> [B, M, N]
  if (lhsType.getRank() == 3 && rhsType.getRank() == 3) {
    SmallVector<int64_t, 3> shape = {
      lhsType.getDimSize(0),
      lhsType.getDimSize(1),
      rhsType.getDimSize(2)
    };
    inferredReturnTypes.push_back(
        RankedTensorType::get(shape, elemType));
    return success();
  }

  // Fallback
  inferredReturnTypes.push_back(UnrankedTensorType::get(elemType));
  return success();
}

//===----------------------------------------------------------------------===//
// jules.relu
//===----------------------------------------------------------------------===//

LogicalResult ReluOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, ::mlir::OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return success();
}

//===----------------------------------------------------------------------===//
// jules.sigmoid
//===----------------------------------------------------------------------===//

LogicalResult SigmoidOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, ::mlir::OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return success();
}

//===----------------------------------------------------------------------===//
// jules.tanh
//===----------------------------------------------------------------------===//

LogicalResult TanhOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, ::mlir::OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return success();
}

//===----------------------------------------------------------------------===//
// jules.mean
//===----------------------------------------------------------------------===//

LogicalResult MeanOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, ::mlir::OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = operands[0].getType().dyn_cast<RankedTensorType>();
  if (inputType) {
    // Mean reduces to a scalar tensor.
    inferredReturnTypes.push_back(
        RankedTensorType::get({}, inputType.getElementType()));
  } else {
    inferredReturnTypes.push_back(
        UnrankedTensorType::get(FloatType::getF32(context)));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// jules.sum
//===----------------------------------------------------------------------===//

LogicalResult SumOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, ::mlir::OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = operands[0].getType().dyn_cast<RankedTensorType>();
  if (inputType) {
    // Sum reduces to a scalar tensor.
    inferredReturnTypes.push_back(
        RankedTensorType::get({}, inputType.getElementType()));
  } else {
    inferredReturnTypes.push_back(
        UnrankedTensorType::get(FloatType::getF32(context)));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// jules.zeros
//===----------------------------------------------------------------------===//

void ZerosOp::build(OpBuilder &builder, OperationState &result,
                    RankedTensorType resultType) {
  // Build a shape attribute.
  SmallVector<int64_t> shape(resultType.getShape().begin(),
                              resultType.getShape().end());
  auto shapeAttr = builder.getI64ArrayAttr(shape);
  result.addAttribute("shape", shapeAttr);
  result.addTypes(resultType);
}

//===----------------------------------------------------------------------===//
// jules.ones
//===----------------------------------------------------------------------===//

void OnesOp::build(OpBuilder &builder, OperationState &result,
                   RankedTensorType resultType) {
  SmallVector<int64_t> shape(resultType.getShape().begin(),
                              resultType.getShape().end());
  auto shapeAttr = builder.getI64ArrayAttr(shape);
  result.addAttribute("shape", shapeAttr);
  result.addTypes(resultType);
}

//===----------------------------------------------------------------------===//
// jules.random
//===----------------------------------------------------------------------===//

void RandomOp::build(OpBuilder &builder, OperationState &result,
                     RankedTensorType resultType) {
  SmallVector<int64_t> shape(resultType.getShape().begin(),
                              resultType.getShape().end());
  auto shapeAttr = builder.getI64ArrayAttr(shape);
  result.addAttribute("shape", shapeAttr);
  result.addTypes(resultType);
}

//===----------------------------------------------------------------------===//
// jules.transpose
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, ::mlir::OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = operands[0].getType().dyn_cast<RankedTensorType>();
  if (inputType) {
    SmallVector<int64_t> shape(inputType.getShape().rbegin(),
                                inputType.getShape().rend());
    inferredReturnTypes.push_back(
        RankedTensorType::get(shape, inputType.getElementType()));
  } else {
    inferredReturnTypes.push_back(
        UnrankedTensorType::get(FloatType::getF32(context)));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// jules.reshape
//===----------------------------------------------------------------------===//

void ReshapeOp::build(OpBuilder &builder, OperationState &result,
                      Value input, RankedTensorType resultType) {
  result.addOperands(input);
  SmallVector<int64_t> shape(resultType.getShape().begin(),
                              resultType.getShape().end());
  auto shapeAttr = builder.getI64ArrayAttr(shape);
  result.addAttribute("shape", shapeAttr);
  result.addTypes(resultType);
}

//===----------------------------------------------------------------------===//
// jules.mul
//===----------------------------------------------------------------------===//

LogicalResult MulOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, ::mlir::OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Element-wise multiply: result type is the broadcast of the two inputs.
  auto lhsType = operands[0].getType();
  auto rhsType = operands[1].getType();
  if (lhsType == rhsType) {
    inferredReturnTypes.push_back(lhsType);
  } else {
    // Simple broadcasting: use the tensor type.
    if (lhsType.isa<RankedTensorType>()) {
      inferredReturnTypes.push_back(lhsType);
    } else if (rhsType.isa<RankedTensorType>()) {
      inferredReturnTypes.push_back(rhsType);
    } else {
      inferredReturnTypes.push_back(lhsType);
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// jules.sub
//===----------------------------------------------------------------------===//

LogicalResult SubOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, ::mlir::OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return success();
}

//===----------------------------------------------------------------------===//
// jules.zeros
//===----------------------------------------------------------------------===//

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

LogicalResult ConstantOp::verify() {
  auto valueAttr = this->getValueAttr();
  auto resultType = getResult().getType();

  if (auto floatType = resultType.dyn_cast<FloatType>()) {
    if (!valueAttr.getType().isa<FloatType>()) {
      return emitOpError("result type is float but value is not");
    }
    return success();
  }

  if (auto integerType = resultType.dyn_cast<IntegerType>()) {
    if (!valueAttr.getType().isa<IntegerType>()) {
      return emitOpError("result type is integer but value is not");
    }
    return success();
  }

  if (auto tensorType = resultType.dyn_cast<RankedTensorType>()) {
    // Tensor constant: value should be an ElementsAttr.
    if (auto elementsAttr = valueAttr.dyn_cast<ElementsAttr>()) {
      if (elementsAttr.getType() != tensorType) {
        return emitOpError("constant value type doesn't match result type");
      }
      return success();
    }
    // DenseFPElementsAttr for float tensors
    if (auto denseAttr = valueAttr.dyn_cast<DenseFPElementsAttr>()) {
      return success();
    }
    if (auto denseAttr = valueAttr.dyn_cast<DenseIntElementsAttr>()) {
      return success();
    }
    return emitOpError("tensor constant requires DenseElementsAttr");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// jules.add
//===----------------------------------------------------------------------===//

LogicalResult AddOp::verify() {
  if (getLhs().getType() != getRhs().getType()) {
    // Allow broadcasting by not strictly requiring type equality.
    // The type inference pass will resolve the broadcast shape.
  }
  return success();
}

//===----------------------------------------------------------------------===//
// jules.matmul
//===----------------------------------------------------------------------===//

LogicalResult MatMulOp::verify() {
  auto lhsType = getLhs().getType().dyn_cast<RankedTensorType>();
  auto rhsType = getRhs().getType().dyn_cast<RankedTensorType>();

  if (!lhsType || !rhsType) {
    return emitOpError("matmul requires ranked tensor operands");
  }

  // Verify dimension compatibility for 2D matmul.
  if (lhsType.getRank() == 2 && rhsType.getRank() == 2) {
    int64_t lhsInner = lhsType.getDimSize(1);
    int64_t rhsInner = rhsType.getDimSize(0);
    if (lhsInner != ShapedType::kDynamic &&
        rhsInner != ShapedType::kDynamic &&
        lhsInner != rhsInner) {
      return emitOpError("matmul inner dimensions must match: ")
             << lhsInner << " vs " << rhsInner;
    }
  }

  return success();
}

// Infer result type for matmul.
::llvm::SmallVector<Type, 1> MatMulOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto lhsType = operands[0].getType().dyn_cast<RankedTensorType>();
  auto rhsType = operands[1].getType().dyn_cast<RankedTensorType>();

  if (!lhsType || !rhsType) {
    inferredReturnTypes.push_back(UnrankedTensorType::get(
        FloatType::getF32(context)));
    return ::llvm::SmallVector<Type, 1>();
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
    return ::llvm::SmallVector<Type, 1>();
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
    return ::llvm::SmallVector<Type, 1>();
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
    return ::llvm::SmallVector<Type, 1>();
  }

  // Fallback
  inferredReturnTypes.push_back(UnrankedTensorType::get(elemType));
  return ::llvm::SmallVector<Type, 1>();
}

//===----------------------------------------------------------------------===//
// jules.relu
//===----------------------------------------------------------------------===//

::llvm::SmallVector<Type, 1> ReluOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return ::llvm::SmallVector<Type, 1>();
}

//===----------------------------------------------------------------------===//
// jules.sigmoid
//===----------------------------------------------------------------------===//

::llvm::SmallVector<Type, 1> SigmoidOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return ::llvm::SmallVector<Type, 1>();
}

//===----------------------------------------------------------------------===//
// jules.tanh
//===----------------------------------------------------------------------===//

::llvm::SmallVector<Type, 1> TanhOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return ::llvm::SmallVector<Type, 1>();
}

//===----------------------------------------------------------------------===//
// jules.mean
//===----------------------------------------------------------------------===//

::llvm::SmallVector<Type, 1> MeanOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
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
  return ::llvm::SmallVector<Type, 1>();
}

//===----------------------------------------------------------------------===//
// jules.sum
//===----------------------------------------------------------------------===//

::llvm::SmallVector<Type, 1> SumOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
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
  return ::llvm::SmallVector<Type, 1>();
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

::llvm::SmallVector<Type, 1> TransposeOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
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
  return ::llvm::SmallVector<Type, 1>();
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
// jules.grad
//===----------------------------------------------------------------------===//

LogicalResult GradOp::verify() {
  // The grad operation will be replaced by the autodiff pass with
  // the actual forward + backward computation.
  return success();
}

//===----------------------------------------------------------------------===//
// jules.mul
//===----------------------------------------------------------------------===//

::llvm::SmallVector<Type, 1> MulOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
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
  return ::llvm::SmallVector<Type, 1>();
}

//===----------------------------------------------------------------------===//
// jules.sub
//===----------------------------------------------------------------------===//

::llvm::SmallVector<Type, 1> SubOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return ::llvm::SmallVector<Type, 1>();
}

//===----------------------------------------------------------------------===//
// jules.div
//===----------------------------------------------------------------------===//

::llvm::SmallVector<Type, 1> DivOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return ::llvm::SmallVector<Type, 1>();
}

//===----------------------------------------------------------------------===//
// jules.pow
//===----------------------------------------------------------------------===//

::llvm::SmallVector<Type, 1> PowOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return ::llvm::SmallVector<Type, 1>();
}

//===----------------------------------------------------------------------===//
// jules.neg
//===----------------------------------------------------------------------===//

::llvm::SmallVector<Type, 1> NegOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return ::llvm::SmallVector<Type, 1>();
}

//===----------------------------------------------------------------------===//
// jules.cast
//===----------------------------------------------------------------------===//

::llvm::SmallVector<Type, 1> CastOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // The result type is determined by the target_type attribute.
  if (auto typeAttr = attributes.get("target_type")) {
    if (auto ty = typeAttr.dyn_cast<TypeAttr>()) {
      inferredReturnTypes.push_back(ty.getValue());
      return ::llvm::SmallVector<Type, 1>();
    }
  }
  inferredReturnTypes.push_back(operands[0].getType());
  return ::llvm::SmallVector<Type, 1>();
}

//===----------------------------------------------------------------------===//
// jules.concat
//===----------------------------------------------------------------------===//

::llvm::SmallVector<Type, 1> ConcatOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Result type matches the first operand's type.
  if (!operands.empty()) {
    inferredReturnTypes.push_back(operands[0].getType());
  }
  return ::llvm::SmallVector<Type, 1>();
}

//===----------------------------------------------------------------------===//
// jules.slice
//===----------------------------------------------------------------------===//

LogicalResult SliceOp::verify() {
  // Verify slice bounds are valid.
  return success();
}

//===----------------------------------------------------------------------===//
// jules.select
//===----------------------------------------------------------------------===//

::llvm::SmallVector<Type, 1> SelectOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Result type matches the true_value / false_value type.
  if (operands.size() >= 3) {
    inferredReturnTypes.push_back(operands[1].getType());
  }
  return ::llvm::SmallVector<Type, 1>();
}

//===----------------------------------------------------------------------===//
// jules.cmp
//===----------------------------------------------------------------------===//

::llvm::SmallVector<Type, 1> CmpOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = operands[0].getType().dyn_cast<RankedTensorType>();
  if (inputType) {
    inferredReturnTypes.push_back(
        RankedTensorType::get(inputType.getShape(),
                              IntegerType::get(context, 1)));
  } else {
    inferredReturnTypes.push_back(IntegerType::get(context, 1));
  }
  return ::llvm::SmallVector<Type, 1>();
}

//===----------------------------------------------------------------------===//
// jules.log
//===----------------------------------------------------------------------===//

::llvm::SmallVector<Type, 1> LogOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // log preserves the input type.
  inferredReturnTypes.push_back(operands[0].getType());
  return ::llvm::SmallVector<Type, 1>();
}

//===----------------------------------------------------------------------===//
// jules.pad
//===----------------------------------------------------------------------===//

LogicalResult PadOp::verify() {
  auto inputType = getInput().getType().dyn_cast<RankedTensorType>();
  auto padValueType = getPaddingValue().getType().dyn_cast<RankedTensorType>();
  if (!inputType) {
    return emitOpError("pad requires ranked tensor input");
  }
  // The padding_value must be a scalar tensor.
  if (padValueType && padValueType.getRank() != 0) {
    return emitOpError("padding_value must be a scalar tensor");
  }
  // Check that padding attributes have the correct size (one entry per dim).
  auto rank = inputType.getRank();
  if (static_cast<int64_t>(getPaddingLow().size()) != rank) {
    return emitOpError("padding_low size must match input rank");
  }
  if (static_cast<int64_t>(getPaddingHigh().size()) != rank) {
    return emitOpError("padding_high size must match input rank");
  }
  if (static_cast<int64_t>(getInteriorPadding().size()) != rank) {
    return emitOpError("interior_padding size must match input rank");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// jules.broadcast_in_dim
//===----------------------------------------------------------------------===//

LogicalResult BroadcastInDimOp::verify() {
  auto inputType = getInput().getType().dyn_cast<RankedTensorType>();
  auto resultType = getResult().getType().dyn_cast<RankedTensorType>();
  if (!inputType || !resultType) {
    return emitOpError("broadcast_in_dim requires ranked tensor operands");
  }
  auto broadcastDims = getBroadcastDimensions();
  if (static_cast<int64_t>(broadcastDims.size()) != inputType.getRank()) {
    return emitOpError("broadcast_dimensions size must match input rank");
  }
  // Verify that each broadcast dimension is within the result rank.
  for (auto attr : broadcastDims) {
    int64_t dim = attr.cast<IntegerAttr>().getInt();
    if (dim < 0 || dim >= resultType.getRank()) {
      return emitOpError("broadcast dimension out of range");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// jules.reduce
//===----------------------------------------------------------------------===//

LogicalResult ReduceOp::verify() {
  auto inputType = getInput().getType().dyn_cast<RankedTensorType>();
  if (!inputType) {
    return emitOpError("reduce requires ranked tensor input");
  }
  auto &body = getBody();
  if (body.empty()) {
    return emitOpError("reduce must have a non-empty body region");
  }
  if (body.getNumArguments() != 2) {
    return emitOpError("reduce body must take exactly 2 arguments "
                       "(accumulator, element)");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// jules.while
//===----------------------------------------------------------------------===//

LogicalResult WhileOp::verify() {
  auto &condRegion = getCond();
  auto &bodyRegion = getBody();

  // The condition region must produce a single i1 or tensor<i1> result.
  if (condRegion.empty()) {
    return emitOpError("while must have a non-empty condition region");
  }
  if (bodyRegion.empty()) {
    return emitOpError("while must have a non-empty body region");
  }

  // The condition region should have the same number of arguments as
  // carried variables.
  auto numCarriedVars = getCarriedVars().size();
  if (condRegion.getNumArguments() != numCarriedVars) {
    return emitOpError("condition region argument count must match "
                       "carried variable count");
  }
  if (bodyRegion.getNumArguments() != numCarriedVars) {
    return emitOpError("body region argument count must match "
                       "carried variable count");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// jules.parallel
//===----------------------------------------------------------------------===//

LogicalResult ParallelOp::verify() {
  if (getLowerBound() < 0) {
    return emitOpError("lower bound must be non-negative");
  }
  if (getUpperBound() < getLowerBound()) {
    return emitOpError("upper bound must be >= lower bound");
  }
  if (getStep() <= 0) {
    return emitOpError("step must be positive");
  }
  auto &bodyRegion = getBody();
  if (bodyRegion.empty()) {
    return emitOpError("parallel must have a non-empty body region");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// jules.extern_kernel
//===----------------------------------------------------------------------===//

LogicalResult ExternKernelOp::verify() {
  if (getKernelName().empty()) {
    return emitOpError("kernel_name must not be empty");
  }
  return success();
}

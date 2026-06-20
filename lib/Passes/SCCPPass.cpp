//===- SCCPPass.cpp - Sparse Conditional Constant Propagation Implementation-===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the SCCP pass for the Jules MLIR dialect.
// It uses a lattice-based worklist algorithm to discover and propagate
// constant values throughout the program.
//
// Lattice values:
//   Top (unknown)    → No information yet
//   Constant(value)  → Known to be this specific value
//   Bottom (overdefined) → Known to be non-constant
//
// The algorithm:
//   1. Initialize all values to Top
//   2. Mark function arguments and non-constant ops as Bottom
//   3. Propagate constant values through operations
//   4. If a value transitions from Top → Constant or Constant → Constant',
//      add its users to the worklist
//   5. Continue until the worklist is empty (fixed point)
//   6. Replace all Constant values with literal constants
//
// Enhanced with tensor constant folding:
//   - TensorLatticeValue supports both scalar and tensor (DenseFPElementsAttr)
//     constant values, enabling entire MLP forward passes to be evaluated at
//     compile time when weights are fixed.
//   - Element-wise tensor operations (Add, Mul, etc.) are folded when both
//     operands are constant tensors.
//   - ZerosOp/OnesOp produce constant tensors directly.
//   - ReshapeOp/TransposeOp with constant inputs produce constant outputs.
//   - A maxConstevalElements limit prevents compile-time explosion from
//     evaluating very large tensor constants.
//
//===----------------------------------------------------------------------===//

#include "jules/Passes/SCCPPass.h"
#include "jules/Dialect/JulesOps.h"
#include "jules/Dialect/JulesDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"

#include <cmath>
#include <limits>

using namespace mlir;
using namespace jules;

namespace {

// ── Tensor Lattice Value ────────────────────────────────────────────────────

/// A lattice value for SCCP analysis that supports both scalar and tensor
/// constants. The lattice has three states:
///   Top     – no information yet (may become constant)
///   Constant – known to be a specific scalar or tensor value
///   Bottom  – known to be non-constant (overdefined)
///
/// For tensor constants, we store a DenseFPElementsAttr which holds the
/// full element data. This enables consteval of entire tensor subgraphs
/// (e.g., MLP forward passes with fixed weights).
class TensorLatticeValue {
public:
  enum class Kind { Top, Constant, Bottom };

  /// Default constructor: Top (unknown).
  TensorLatticeValue() : kind_(Kind::Top) {}

  /// Construct a Bottom (overdefined) lattice value.
  static TensorLatticeValue getBottom() {
    TensorLatticeValue val;
    val.kind_ = Kind::Bottom;
    return val;
  }

  /// Construct a Top (unknown) lattice value.
  static TensorLatticeValue getTop() {
    TensorLatticeValue val;
    val.kind_ = Kind::Top;
    return val;
  }

  /// Construct a scalar constant lattice value.
  TensorLatticeValue(double val, Type type)
      : kind_(Kind::Constant), scalarValue_(val), isScalar_(true),
        scalarType_(type) {}

  /// Construct a scalar constant lattice value without an explicit type
  /// (used by evaluateScalarOperation where the result type is implied by
  /// the op's result type, set later by the caller if needed).
  TensorLatticeValue(double val)
      : kind_(Kind::Constant), scalarValue_(val), isScalar_(true) {}

  /// Construct a tensor constant lattice value from DenseFPElementsAttr.
  TensorLatticeValue(DenseFPElementsAttr tensorValue)
      : kind_(Kind::Constant), tensorValue_(tensorValue), isScalar_(false) {}

  // ── Accessors ──────────────────────────────────────────────────────────

  Kind getKind() const { return kind_; }
  bool isTop() const { return kind_ == Kind::Top; }
  bool isConstant() const { return kind_ == Kind::Constant; }
  bool isBottom() const { return kind_ == Kind::Bottom; }

  /// Returns true if this is a tensor constant (not scalar).
  bool isTensorConstant() const {
    return kind_ == Kind::Constant && !isScalar_ && tensorValue_.has_value();
  }

  /// Returns true if this is a scalar constant.
  bool isScalarConstant() const {
    return kind_ == Kind::Constant && isScalar_;
  }

  /// Get the tensor constant value. Asserts isTensorConstant().
  DenseFPElementsAttr getTensorValue() const {
    assert(isTensorConstant() && "not a tensor constant");
    return *tensorValue_;
  }

  /// Get the scalar constant value. Asserts isScalarConstant().
  double getScalarValue() const {
    assert(isScalarConstant() && "not a scalar constant");
    return scalarValue_;
  }

  /// Get the scalar type (for scalar constants). May be null for tensor constants.
  Type getScalarType() const {
    assert(isScalarConstant() && "not a scalar constant");
    return scalarType_;
  }

  /// Get the number of elements in a tensor constant. Returns 1 for scalars.
  int64_t getNumElements() const {
    if (isScalarConstant())
      return 1;
    if (isTensorConstant()) {
      auto shapedType = tensorValue_->getType().dyn_cast<ShapedType>();
      return shapedType ? shapedType.getNumElements() : 0;
    }
    return 0;
  }

  /// Meet another lattice value into this one.
  /// Returns true if this value changed.
  bool meet(const TensorLatticeValue &other) {
    if (kind_ == Kind::Bottom)
      return false;

    // Other is Bottom → this becomes Bottom.
    if (other.kind_ == Kind::Bottom) {
      bool changed = kind_ != Kind::Bottom;
      kind_ = Kind::Bottom;
      return changed;
    }

    // Other is Top → no information to add.
    if (other.kind_ == Kind::Top)
      return false;

    // Other is Constant.
    if (kind_ == Kind::Top) {
      // Top meets Constant → becomes that constant.
      kind_ = Kind::Constant;
      isScalar_ = other.isScalar_;
      scalarValue_ = other.scalarValue_;
      tensorValue_ = other.tensorValue_;
      scalarType_ = other.scalarType_;
      return true;
    }

    // Both are Constant. They must agree; otherwise → Bottom.
    if (isScalar_ != other.isScalar_) {
      // Scalar vs tensor mismatch → Bottom.
      kind_ = Kind::Bottom;
      return true;
    }

    if (isScalar_) {
      // Both scalar: check if they have the same value.
      if (scalarValue_ != other.scalarValue_) {
        kind_ = Kind::Bottom;
        return true;
      }
      return false;
    }

    // Both tensor: check element-wise equality.
    if (tensorValue_.has_value() && other.tensorValue_.has_value()) {
      if (*tensorValue_ == *other.tensorValue_)
        return false;
    }
    // Tensor values disagree or one is missing → Bottom.
    kind_ = Kind::Bottom;
    return true;
  }

  /// Print for debugging.
  void print(raw_ostream &os) const {
    switch (kind_) {
    case Kind::Top:
      os << "Top";
      break;
    case Kind::Bottom:
      os << "Bottom";
      break;
    case Kind::Constant:
      if (isScalar_) {
        os << "Scalar(" << scalarValue_ << ")";
      } else if (tensorValue_.has_value()) {
        auto shapedType = tensorValue_->getType().dyn_cast<ShapedType>();
        os << "Tensor(";
        if (shapedType)
          os << shapedType.getNumElements() << " elems";
        os << ")";
      } else {
        os << "Constant(malformed)";
      }
      break;
    }
  }

private:
  Kind kind_ = Kind::Top;
  std::optional<DenseFPElementsAttr> tensorValue_;
  double scalarValue_ = 0.0;
  bool isScalar_ = true;
  Type scalarType_;
};

// ── Helper: Element-wise tensor arithmetic ──────────────────────────────────

/// Apply a binary function element-wise to two DenseFPElementsAttr values.
/// Both must have the same shape (or one may be a scalar splat).
/// Returns a new DenseFPElementsAttr with the result.
template <typename BinaryFn>
DenseFPElementsAttr elementWiseBinary(DenseFPElementsAttr lhs,
                                       DenseFPElementsAttr rhs,
                                       RankedTensorType resultType,
                                       BinaryFn &&fn) {
  SmallVector<APFloat, 64> resultValues;
  resultValues.reserve(lhs.getNumElements());

  auto lhsIt = lhs.begin();
  auto rhsIt = rhs.begin();

  // If one operand is a splat (single element broadcast), handle it specially.
  bool lhsSplat = lhs.isSplat();
  bool rhsSplat = rhs.isSplat();

  if (lhsSplat && rhsSplat) {
    // Both splat: result is a splat.
    APFloat result = fn(*lhsIt, *rhsIt);
    auto attr = DenseFPElementsAttr::get(resultType, result);
    return attr;
  }

  if (lhsSplat) {
    APFloat lhsVal = *lhsIt;
    for (auto rhsVal : rhs) {
      resultValues.push_back(fn(lhsVal, rhsVal));
    }
  } else if (rhsSplat) {
    APFloat rhsVal = *rhsIt;
    for (auto lhsVal : lhs) {
      resultValues.push_back(fn(lhsVal, rhsVal));
    }
  } else {
    // Neither splat: element-wise iteration.
    for (size_t i = 0, e = lhs.getNumElements(); i < e; ++i) {
      resultValues.push_back(fn(lhsIt[i], rhsIt[i]));
    }
  }

  return DenseFPElementsAttr::get(resultType, resultValues);
}

/// Apply a unary function element-wise to a DenseFPElementsAttr.
template <typename UnaryFn>
DenseFPElementsAttr elementWiseUnary(DenseFPElementsAttr input,
                                      RankedTensorType resultType,
                                      UnaryFn &&fn) {
  SmallVector<APFloat, 64> resultValues;
  resultValues.reserve(input.getNumElements());

  if (input.isSplat()) {
    APFloat result = fn(*input.begin());
    return DenseFPElementsAttr::get(resultType, result);
  }

  for (auto val : input) {
    resultValues.push_back(fn(val));
  }

  return DenseFPElementsAttr::get(resultType, resultValues);
}

/// Make a DenseFPElementsAttr filled with a single value, for a given
/// RankedTensorType.
DenseFPElementsAttr makeSplatTensor(RankedTensorType type, APFloat value) {
  return DenseFPElementsAttr::get(type, value);
}

// ── SCCP Pass ───────────────────────────────────────────────────────────────

struct SCCPPass : public PassWrapper<SCCPPass, OperationPass<ModuleOp>> {
  /// Maximum number of elements in a tensor to consteval.
  /// Tensors larger than this will remain as Bottom to avoid
  /// compile-time memory and CPU explosion. Stored as plain member
  /// (cl::opt is non-copyable, which would make SCCPPass non-copyable).
  unsigned maxConstevalElements = 1024;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    builder_ = std::make_shared<OpBuilder>(&getContext());

    // Run SCCP on each function in the module.
    module.walk([&](func::FuncOp funcOp) {
      runOnFunction(funcOp);
    });
  }

  void runOnFunction(func::FuncOp funcOp) {
    llvm::DenseMap<Value, TensorLatticeValue> lattice;
    llvm::SetVector<Value> worklist;

    // ── Initialize ────────────────────────────────────────────────────────
    //
    // Function arguments are Bottom (unknown at compile time).
    // Operations with all-constant inputs start as their computed constant.

    // Mark function arguments as Bottom.
    for (auto arg : funcOp.getArguments()) {
      lattice[arg] = TensorLatticeValue::getBottom();
    }

    // Initialize worklist with all operation results.
    funcOp.walk([&](Operation *op) {
      for (auto result : op->getResults()) {
        worklist.insert(result);
      }
    });

    // ── Propagate ─────────────────────────────────────────────────────────
    //
    // Process the worklist until empty (fixed point).

    while (!worklist.empty()) {
      Value val = worklist.pop_back_val();
      auto &currentLattice = lattice[val];

      // Compute the new lattice value for this value.
      TensorLatticeValue newLattice = computeLattice(val, lattice);

      // Meet the new value into the current lattice.
      if (currentLattice.meet(newLattice)) {
        // Value changed — add all users to the worklist.
        for (auto *user : val.getUsers()) {
          for (auto result : user->getResults()) {
            worklist.insert(result);
          }
        }
      }
    }

    // ── Replace ───────────────────────────────────────────────────────────
    //
    // Replace all values that resolved to constants with literal constants.

    OpBuilder builder(funcOp.getContext());
    funcOp.walk([&](Operation *op) {
      // Skip if no results or if it's a terminator.
      if (op->getNumResults() == 0) return;
      if (op->hasTrait<OpTrait::IsTerminator>()) return;

      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        auto result = op->getResult(i);
        auto it = lattice.find(result);
        if (it == lattice.end()) continue;

        if (it->second.isScalarConstant()) {
          // Replace this result with a scalar constant.
          double constVal = it->second.getScalarValue();
          Type resultType = result.getType();
          Type attrType = resultType;

          // If the result is a scalar tensor, create the constant with the
          // element type so the ConstantOp verifier is happy.
          if (auto tensorType = resultType.dyn_cast<RankedTensorType>()) {
            if (tensorType.getRank() == 0) {
              attrType = tensorType.getElementType();
            }
          }

          builder.setInsertionPoint(op);
          auto constOp = builder.create<ConstantOp>(op->getLoc(), builder.getFloatAttr(
                  attrType.isa<FloatType>() ? attrType : builder.getF32Type(),
                  constVal), attrType.isa<FloatType>() ? attrType : builder.getF32Type());

          result.replaceAllUsesWith(constOp.getResult());
        } else if (it->second.isTensorConstant()) {
          // Replace this result with a tensor constant.
          DenseFPElementsAttr tensorVal = it->second.getTensorValue();
          auto tensorType = result.getType().dyn_cast<RankedTensorType>();
          if (!tensorType) continue;

          // Ensure the tensor value type matches the result type.
          // If they differ (e.g. due to shape inference), reshape the attribute.
          DenseFPElementsAttr finalVal = tensorVal;
          if (tensorVal.getType() != tensorType) {
            // Try to reshape the attribute to match the expected type.
            if (tensorVal.getNumElements() ==
                tensorType.getNumElements()) {
              // Reshape the values.
              SmallVector<APFloat, 64> elements;
              elements.reserve(tensorVal.getNumElements());
              for (auto elem : tensorVal) {
                elements.push_back(elem);
              }
              finalVal = DenseFPElementsAttr::get(tensorType, elements);
            } else {
              // Cannot reconcile — skip replacement.
              continue;
            }
          }

          builder.setInsertionPoint(op);
          auto constOp = builder.create<ConstantOp>(op->getLoc(), finalVal, finalVal.getType());
          result.replaceAllUsesWith(constOp.getResult());
        }
      }
    });
  }

  /// Check whether a tensor constant is within the consteval budget.
  bool withinConstevalBudget(const TensorLatticeValue &val) const {
    int64_t numElements = val.getNumElements();
    if (numElements < 0) return false; // dynamic size
    return static_cast<uint64_t>(numElements) <= maxConstevalElements;
  }

  /// Check whether a RankedTensorType is within the consteval budget.
  bool withinConstevalBudget(RankedTensorType type) const {
    if (!type.hasStaticShape()) return false;
    int64_t numElements = type.getNumElements();
    return numElements >= 0 &&
           static_cast<uint64_t>(numElements) <= maxConstevalElements;
  }

  /// Compute the lattice value for a value based on its defining operation.
  TensorLatticeValue computeLattice(
      Value val,
      const llvm::DenseMap<Value, TensorLatticeValue> &lattice) {
    auto *defOp = val.getDefiningOp();
    if (!defOp) {
      // Function argument — unknown.
      return TensorLatticeValue::getBottom();
    }

    // ── ConstantOp ────────────────────────────────────────────────────────
    if (auto constOp = dyn_cast<ConstantOp>(defOp)) {
      auto valueAttr = constOp.getValueAttr();

      // Scalar float constant.
      if (auto floatAttr = valueAttr.dyn_cast<FloatAttr>()) {
        return TensorLatticeValue(
            floatAttr.getValue().convertToDouble(),
            floatAttr.getType());
      }

      // Scalar integer constant.
      if (auto intAttr = valueAttr.dyn_cast<IntegerAttr>()) {
        return TensorLatticeValue(
            static_cast<double>(intAttr.getInt()),
            intAttr.getType());
      }

      // Tensor constant (DenseFPElementsAttr).
      if (auto denseFP = valueAttr.dyn_cast<DenseFPElementsAttr>()) {
        auto shapedType = denseFP.getType().dyn_cast<RankedTensorType>();
        if (shapedType && withinConstevalBudget(shapedType)) {
          // If it's a 0-d tensor (scalar tensor), extract the scalar.
          if (shapedType.getRank() == 0 && denseFP.getNumElements() == 1) {
            // MLIR 19 returns APFloat by value from iterator; use *it instead of it->
            auto floatVal = *denseFP.begin();
            double scalarVal = floatVal.convertToDouble();
            return TensorLatticeValue(scalarVal, shapedType.getElementType());
          }
          return TensorLatticeValue(denseFP);
        }
        // Too large or unranked → Bottom.
        if (shapedType && !withinConstevalBudget(shapedType)) {
          return TensorLatticeValue::getBottom();
        }
      }

      // DenseIntElementsAttr tensor constant.
      if (auto denseInt = valueAttr.dyn_cast<DenseIntElementsAttr>()) {
        auto shapedType = denseInt.getType().dyn_cast<RankedTensorType>();
        if (shapedType && shapedType.getRank() == 0 &&
            denseInt.getNumElements() == 1) {
          // MLIR 19 returns APInt by value from iterator; use *it instead of it->
          auto intVal = *denseInt.begin();
          double scalarVal = static_cast<double>(intVal.getSExtValue());
          return TensorLatticeValue(scalarVal, shapedType.getElementType());
        }
        // For now, int tensors → Bottom (we focus on float tensor folding).
      }
    }

    // ── ZerosOp: always produces a constant tensor of zeros ───────────────
    if (auto zerosOp = dyn_cast<ZerosOp>(defOp)) {
      auto resultType =
          zerosOp.getResult().getType().dyn_cast<RankedTensorType>();
      if (resultType && withinConstevalBudget(resultType)) {
        // If 0-d tensor, produce scalar 0.0.
        if (resultType.getRank() == 0) {
          return TensorLatticeValue(0.0, resultType.getElementType());
        }
        APFloat zero(0.0);
        bool zero_loses;
        zero.convert(resultType.getElementType()
                         .cast<FloatType>()
                         .getFloatSemantics(),
                     APFloat::rmNearestTiesToEven, &zero_loses);
        return TensorLatticeValue(makeSplatTensor(resultType, zero));
      }
      // Fallback: scalar 0.
      return TensorLatticeValue(0.0, defOp->getResult(0).getType());
    }

    // ── OnesOp: always produces a constant tensor of ones ─────────────────
    if (auto onesOp = dyn_cast<OnesOp>(defOp)) {
      auto resultType =
          onesOp.getResult().getType().dyn_cast<RankedTensorType>();
      if (resultType && withinConstevalBudget(resultType)) {
        if (resultType.getRank() == 0) {
          return TensorLatticeValue(1.0, resultType.getElementType());
        }
        APFloat one(1.0);
        bool one_loses;
        one.convert(resultType.getElementType()
                        .cast<FloatType>()
                        .getFloatSemantics(),
                    APFloat::rmNearestTiesToEven, &one_loses);
        return TensorLatticeValue(makeSplatTensor(resultType, one));
      }
      return TensorLatticeValue(1.0, defOp->getResult(0).getType());
    }

    // ── ReshapeOp with constant input ─────────────────────────────────────
    if (auto reshapeOp = dyn_cast<ReshapeOp>(defOp)) {
      auto inputIt = lattice.find(reshapeOp.getInput());
      if (inputIt != lattice.end() && inputIt->second.isConstant()) {
        auto resultType =
            reshapeOp.getResult().getType().dyn_cast<RankedTensorType>();
        if (!resultType || !withinConstevalBudget(resultType))
          return TensorLatticeValue::getBottom();

        if (inputIt->second.isTensorConstant()) {
          DenseFPElementsAttr inputTensor = inputIt->second.getTensorValue();
          // Reshape: flatten all values and reshape to new type.
          SmallVector<APFloat, 64> elements;
          elements.reserve(inputTensor.getNumElements());
          for (auto elem : inputTensor) {
            elements.push_back(elem);
          }
          auto reshaped = DenseFPElementsAttr::get(resultType, elements);
          return TensorLatticeValue(reshaped);
        }
        if (inputIt->second.isScalarConstant()) {
          // Scalar input → splat the scalar into the reshape target.
          APFloat val(inputIt->second.getScalarValue());
          bool val_loses;
          val.convert(resultType.getElementType()
                          .cast<FloatType>()
                          .getFloatSemantics(),
                      APFloat::rmNearestTiesToEven, &val_loses);
          return TensorLatticeValue(makeSplatTensor(resultType, val));
        }
      }
      // If the input isn't constant, fall through to generic handling.
      return propagateGeneric(defOp, lattice);
    }

    // ── TransposeOp with constant input ───────────────────────────────────
    if (auto transposeOp = dyn_cast<TransposeOp>(defOp)) {
      auto inputIt = lattice.find(transposeOp.getInput());
      if (inputIt != lattice.end() && inputIt->second.isConstant()) {
        auto resultType =
            transposeOp.getResult().getType().dyn_cast<RankedTensorType>();
        if (!resultType || !withinConstevalBudget(resultType))
          return TensorLatticeValue::getBottom();

        if (inputIt->second.isTensorConstant()) {
          DenseFPElementsAttr inputTensor = inputIt->second.getTensorValue();
          auto inputType =
              inputTensor.getType().dyn_cast<RankedTensorType>();
          if (!inputType) return TensorLatticeValue::getBottom();

          // For 2D transpose: read row-major, write column-major.
          int64_t rows = inputType.getDimSize(0);
          int64_t cols = inputType.getDimSize(1);

          // Flatten input values.
          SmallVector<APFloat, 64> inputValues;
          inputValues.reserve(inputTensor.getNumElements());
          for (auto elem : inputTensor) {
            inputValues.push_back(elem);
          }

          // Transpose: output[i][j] = input[j][i]
          SmallVector<APFloat, 64> transposedValues;
          transposedValues.reserve(inputValues.size());
          for (int64_t j = 0; j < cols; ++j) {
            for (int64_t i = 0; i < rows; ++i) {
              transposedValues.push_back(inputValues[i * cols + j]);
            }
          }

          auto transposed =
              DenseFPElementsAttr::get(resultType, transposedValues);
          return TensorLatticeValue(transposed);
        }

        if (inputIt->second.isScalarConstant()) {
          // Scalar stays scalar.
          return inputIt->second;
        }
      }
      return propagateGeneric(defOp, lattice);
    }

    // ── Generic operation handling ─────────────────────────────────────────
    return propagateGeneric(defOp, lattice);
  }

  /// Generic lattice propagation: check if all operands are constant,
  /// then try to evaluate.
  TensorLatticeValue propagateGeneric(
      Operation *op,
      const llvm::DenseMap<Value, TensorLatticeValue> &lattice) {
    // Gather operand lattice values.
    SmallVector<TensorLatticeValue, 4> operandLattices;
    bool allConst = true;
    bool anyBottom = false;

    for (auto operand : op->getOperands()) {
      auto it = lattice.find(operand);
      if (it == lattice.end() || it->second.isBottom()) {
        anyBottom = true;
        allConst = false;
        operandLattices.push_back(TensorLatticeValue::getBottom());
      } else if (it->second.isTop()) {
        allConst = false;
        operandLattices.push_back(TensorLatticeValue::getTop());
      } else {
        operandLattices.push_back(it->second);
      }
    }

    if (anyBottom) {
      return TensorLatticeValue::getBottom();
    }

    if (!allConst) {
      return TensorLatticeValue::getTop();
    }

    // All operands are constant — try to evaluate the operation.
    return evaluateOperation(op, operandLattices);
  }

  /// Evaluate an operation given constant operand lattice values.
  /// This handles both scalar and tensor constant folding.
  TensorLatticeValue evaluateOperation(
      Operation *op,
      const SmallVector<TensorLatticeValue, 4> &operands) {

    // ── Try tensor-level evaluation first ─────────────────────────────────
    //
    // If any operand is a tensor constant, we attempt element-wise tensor
    // folding. If that fails or exceeds the budget, we fall through to
    // scalar evaluation.

    // Determine if we should attempt tensor evaluation.
    bool hasTensorOperand = false;
    for (auto &operand : operands) {
      if (operand.isTensorConstant()) {
        hasTensorOperand = true;
        break;
      }
    }

    if (hasTensorOperand) {
      auto result = evaluateTensorOperation(op, operands);
      if (result.isConstant()) {
        // Check budget before returning.
        if (withinConstevalBudget(result)) {
          return result;
        }
        // Exceeded budget — don't consteval.
        return TensorLatticeValue::getBottom();
      }
      // If tensor evaluation returned non-constant, try scalar fallback.
    }

    // ── Scalar evaluation ─────────────────────────────────────────────────
    // All operands must be scalar constants for this path.
    SmallVector<double, 4> scalarOps;
    for (auto &operand : operands) {
      if (!operand.isScalarConstant()) {
        // Mixed scalar/tensor that we couldn't handle → Bottom.
        return TensorLatticeValue::getBottom();
      }
      scalarOps.push_back(operand.getScalarValue());
    }

    return evaluateScalarOperation(op, scalarOps);
  }

  /// Evaluate a tensor operation with at least one tensor operand.
  TensorLatticeValue evaluateTensorOperation(
      Operation *op,
      const SmallVector<TensorLatticeValue, 4> &operands) {

    // ── AddOp with two tensor/scalar constants ────────────────────────────
    if (isa<AddOp>(op) && operands.size() >= 2) {
      return evaluateBinaryTensorOp(
          operands[0], operands[1], op,
          [](const APFloat &a, const APFloat &b) -> APFloat {
            APFloat result = a;
            result.add(b, APFloat::rmNearestTiesToEven);
            return result;
          });
    }

    // ── SubOp with two tensor/scalar constants ────────────────────────────
    if (isa<SubOp>(op) && operands.size() >= 2) {
      return evaluateBinaryTensorOp(
          operands[0], operands[1], op,
          [](const APFloat &a, const APFloat &b) -> APFloat {
            APFloat result = a;
            result.subtract(b, APFloat::rmNearestTiesToEven);
            return result;
          });
    }

    // ── MulOp with tensor + scalar (broadcast multiply) ───────────────────
    if (isa<MulOp>(op) && operands.size() >= 2) {
      return evaluateBinaryTensorOp(
          operands[0], operands[1], op,
          [](const APFloat &a, const APFloat &b) -> APFloat {
            APFloat result = a;
            result.multiply(b, APFloat::rmNearestTiesToEven);
            return result;
          });
    }

    // ── DivOp with two tensor/scalar constants ────────────────────────────
    if (isa<DivOp>(op) && operands.size() >= 2) {
      return evaluateBinaryTensorOp(
          operands[0], operands[1], op,
          [](const APFloat &a, const APFloat &b) -> APFloat {
            if (b.isZero())
              return APFloat::getNaN(b.getSemantics());
            APFloat result = a;
            result.divide(b, APFloat::rmNearestTiesToEven);
            return result;
          });
    }

    // ── NegOp with tensor constant ────────────────────────────────────────
    if (isa<NegOp>(op) && operands.size() >= 1) {
      if (operands[0].isTensorConstant()) {
        auto resultType =
            op->getResult(0).getType().dyn_cast<RankedTensorType>();
        if (!resultType) return TensorLatticeValue::getBottom();

        auto result = elementWiseUnary(
            operands[0].getTensorValue(), resultType,
            [](const APFloat &v) -> APFloat {
              APFloat result = v;
              result.changeSign();
              return result;
            });
        return TensorLatticeValue(result);
      }
    }

    // ── ReluOp with tensor constant ───────────────────────────────────────
    if (isa<ReluOp>(op) && operands.size() >= 1) {
      if (operands[0].isTensorConstant()) {
        auto resultType =
            op->getResult(0).getType().dyn_cast<RankedTensorType>();
        if (!resultType) return TensorLatticeValue::getBottom();

        APFloat zero(0.0);
        bool zero_loses;
        zero.convert(resultType.getElementType()
                         .cast<FloatType>()
                         .getFloatSemantics(),
                     APFloat::rmNearestTiesToEven, &zero_loses);
        auto result = elementWiseUnary(
            operands[0].getTensorValue(), resultType,
            [&zero](const APFloat &v) -> APFloat {
              return v < zero ? zero : v;
            });
        return TensorLatticeValue(result);
      }
    }

    // ── SigmoidOp with tensor constant ────────────────────────────────────
    if (isa<SigmoidOp>(op) && operands.size() >= 1) {
      if (operands[0].isTensorConstant()) {
        auto resultType =
            op->getResult(0).getType().dyn_cast<RankedTensorType>();
        if (!resultType) return TensorLatticeValue::getBottom();

        auto result = elementWiseUnary(
            operands[0].getTensorValue(), resultType,
            [](const APFloat &v) -> APFloat {
              double d = v.convertToDouble();
              double sigmoid = 1.0 / (1.0 + std::exp(-d));
              APFloat result(sigmoid);
              return result;
            });
        return TensorLatticeValue(result);
      }
    }

    // ── TanhOp with tensor constant ───────────────────────────────────────
    if (isa<TanhOp>(op) && operands.size() >= 1) {
      if (operands[0].isTensorConstant()) {
        auto resultType =
            op->getResult(0).getType().dyn_cast<RankedTensorType>();
        if (!resultType) return TensorLatticeValue::getBottom();

        auto result = elementWiseUnary(
            operands[0].getTensorValue(), resultType,
            [](const APFloat &v) -> APFloat {
              double d = v.convertToDouble();
              double tanhVal = std::tanh(d);
              return APFloat(tanhVal);
            });
        return TensorLatticeValue(result);
      }
    }

    // ── MeanOp with tensor constant ───────────────────────────────────────
    if (isa<MeanOp>(op) && operands.size() >= 1) {
      if (operands[0].isTensorConstant()) {
        DenseFPElementsAttr inputTensor = operands[0].getTensorValue();
        auto inputType = inputTensor.getType().dyn_cast<RankedTensorType>();
        if (!inputType) return TensorLatticeValue::getBottom();

        double sum = 0.0;
        int64_t count = 0;
        for (auto elem : inputTensor) {
          sum += elem.convertToDouble();
          ++count;
        }
        if (count == 0) return TensorLatticeValue::getBottom();
        double mean = sum / static_cast<double>(count);

        auto resultType =
            op->getResult(0).getType().dyn_cast<RankedTensorType>();
        if (resultType && resultType.getRank() == 0) {
          // Scalar tensor result.
          return TensorLatticeValue(mean,
                                     resultType.getElementType());
        }
        return TensorLatticeValue(mean, builder_->getF32Type());
      }
    }

    // ── SumOp with tensor constant ────────────────────────────────────────
    if (isa<SumOp>(op) && operands.size() >= 1) {
      if (operands[0].isTensorConstant()) {
        DenseFPElementsAttr inputTensor = operands[0].getTensorValue();
        double sum = 0.0;
        for (auto elem : inputTensor) {
          sum += elem.convertToDouble();
        }

        auto resultType =
            op->getResult(0).getType().dyn_cast<RankedTensorType>();
        if (resultType && resultType.getRank() == 0) {
          return TensorLatticeValue(sum,
                                     resultType.getElementType());
        }
        return TensorLatticeValue(sum, builder_->getF32Type());
      }
    }

    // ── PowOp with tensor constants ───────────────────────────────────────
    if (isa<PowOp>(op) && operands.size() >= 2) {
      return evaluateBinaryTensorOp(
          operands[0], operands[1], op,
          [](const APFloat &a, const APFloat &b) -> APFloat {
            double da = a.convertToDouble();
            double db = b.convertToDouble();
            return APFloat(std::pow(da, db));
          });
    }

    // Unknown tensor operation → Bottom.
    return TensorLatticeValue::getBottom();
  }

  /// Evaluate a binary tensor/scalar operation, handling broadcasting.
  TensorLatticeValue evaluateBinaryTensorOp(
      const TensorLatticeValue &lhs, const TensorLatticeValue &rhs,
      Operation *op,
      llvm::function_ref<APFloat(const APFloat &, const APFloat &)> fn) {

    auto resultType =
        op->getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!resultType) return TensorLatticeValue::getBottom();

    // Case 1: Both tensor constants.
    if (lhs.isTensorConstant() && rhs.isTensorConstant()) {
      auto result = elementWiseBinary(lhs.getTensorValue(),
                                       rhs.getTensorValue(),
                                       resultType, fn);
      return TensorLatticeValue(result);
    }

    // Case 2: Tensor + scalar (broadcast scalar to tensor shape).
    if (lhs.isTensorConstant() && rhs.isScalarConstant()) {
      auto elemType = resultType.getElementType().cast<FloatType>();
      APFloat scalarVal(rhs.getScalarValue());
      bool scalarVal_loses;
      scalarVal.convert(elemType.getFloatSemantics(),
                        APFloat::rmNearestTiesToEven, &scalarVal_loses);
      auto rhsTensor = makeSplatTensor(resultType, scalarVal);
      auto result = elementWiseBinary(lhs.getTensorValue(),
                                       rhsTensor, resultType, fn);
      return TensorLatticeValue(result);
    }

    // Case 3: Scalar + tensor (broadcast scalar to tensor shape).
    if (lhs.isScalarConstant() && rhs.isTensorConstant()) {
      auto elemType = resultType.getElementType().cast<FloatType>();
      APFloat scalarVal(lhs.getScalarValue());
      bool scalarVal_loses;
      scalarVal.convert(elemType.getFloatSemantics(),
                        APFloat::rmNearestTiesToEven, &scalarVal_loses);
      auto lhsTensor = makeSplatTensor(resultType, scalarVal);
      auto result = elementWiseBinary(lhsTensor,
                                       rhs.getTensorValue(),
                                       resultType, fn);
      return TensorLatticeValue(result);
    }

    // Case 4: Both scalar but at least one is actually a scalar
    // (shouldn't reach here in the tensor path).
    return TensorLatticeValue::getBottom();
  }

  /// Evaluate a scalar operation given constant scalar operands.
  TensorLatticeValue evaluateScalarOperation(
      Operation *op,
      const llvm::SmallVector<double, 4> &operands) {

    if (isa<AddOp>(op) && operands.size() >= 2) {
      return TensorLatticeValue(operands[0] + operands[1]);
    }
    if (isa<SubOp>(op) && operands.size() >= 2) {
      return TensorLatticeValue(operands[0] - operands[1]);
    }
    if (isa<MulOp>(op) && operands.size() >= 2) {
      return TensorLatticeValue(operands[0] * operands[1]);
    }
    if (isa<DivOp>(op) && operands.size() >= 2) {
      if (operands[1] == 0.0) {
        return TensorLatticeValue::getBottom(); // Division by zero
      }
      return TensorLatticeValue(operands[0] / operands[1]);
    }
    if (isa<NegOp>(op) && operands.size() >= 1) {
      return TensorLatticeValue(-operands[0]);
    }
    if (isa<ReluOp>(op) && operands.size() >= 1) {
      return TensorLatticeValue(std::max(0.0, operands[0]));
    }
    if (isa<SigmoidOp>(op) && operands.size() >= 1) {
      return TensorLatticeValue(1.0 / (1.0 + std::exp(-operands[0])));
    }
    if (isa<TanhOp>(op) && operands.size() >= 1) {
      return TensorLatticeValue(std::tanh(operands[0]));
    }
    if (isa<PowOp>(op) && operands.size() >= 2) {
      return TensorLatticeValue(std::pow(operands[0], operands[1]));
    }
    if (isa<MeanOp>(op) && operands.size() >= 1) {
      return TensorLatticeValue(operands[0]); // Scalar mean = itself
    }
    if (isa<SumOp>(op) && operands.size() >= 1) {
      return TensorLatticeValue(operands[0]); // Scalar sum = itself
    }

    // Unknown operation — conservatively assume Bottom.
    return TensorLatticeValue::getBottom();
  }

private:
  /// Builder used for creating constants during replacement.
  /// Lazily created in runOnOperation because getContext() is not available
  /// at member-initialization time. Uses shared_ptr (not unique_ptr) so the
  /// pass remains copyable (PassWrapper requires copy).
  std::shared_ptr<OpBuilder> builder_;
};

} // anonymous namespace

std::unique_ptr<Pass> jules::createSCCPPass() {
  return std::make_unique<SCCPPass>();
}

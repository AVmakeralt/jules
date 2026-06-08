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
//===----------------------------------------------------------------------===//

#include "jules/Passes/SCCPPass.h"
#include "jules/Dialect/JulesOps.h"
#include "jules/Dialect/JulesDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace jules;

namespace {

// ── Lattice Value ───────────────────────────────────────────────────────────

/// A lattice value for SCCP analysis.
class LatticeValue {
public:
  enum class Kind { Top, Constant, Bottom };

  LatticeValue() : kind_(Kind::Top) {}
  LatticeValue(Kind kind) : kind_(kind) {}
  LatticeValue(double value) : kind_(Kind::Constant), constValue_(value) {}

  Kind getKind() const { return kind_; }

  bool isTop() const { return kind_ == Kind::Top; }
  bool isConstant() const { return kind_ == Kind::Constant; }
  bool isBottom() const { return kind_ == Kind::Bottom; }

  double getConstantValue() const {
    assert(isConstant() && "not a constant");
    return constValue_;
  }

  /// Meet another lattice value into this one.
  /// Returns true if this value changed.
  bool meet(const LatticeValue &other) {
    if (kind_ == Kind::Bottom) return false;

    if (other.kind_ == Kind::Bottom) {
      bool changed = kind_ != Kind::Bottom;
      kind_ = Kind::Bottom;
      return changed;
    }

    if (other.kind_ == Kind::Top) return false;

    if (kind_ == Kind::Top) {
      bool changed = kind_ != Kind::Constant;
      kind_ = Kind::Constant;
      constValue_ = other.constValue_;
      return changed;
    }

    // Both are constants. If they agree, no change. Otherwise, go to Bottom.
    if (constValue_ != other.constValue_) {
      kind_ = Kind::Bottom;
      return true;
    }
    return false;
  }

private:
  Kind    kind_;
  double  constValue_ = 0.0;
};

// ── SCCP Pass ───────────────────────────────────────────────────────────────

struct SCCPPass : public PassWrapper<SCCPPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Run SCCP on each function in the module.
    module.walk([&](func::FuncOp funcOp) {
      runOnFunction(funcOp);
    });
  }

  void runOnFunction(func::FuncOp funcOp) {
    llvm::DenseMap<Value, LatticeValue> lattice;
    llvm::SetVector<Value> worklist;

    // ── Initialize ────────────────────────────────────────────────────────
    //
    // Function arguments are Bottom (unknown at compile time).
    // Operations with all-constant inputs start as their computed constant.

    // Mark function arguments as Bottom.
    for (auto arg : funcOp.getArguments()) {
      lattice[arg] = LatticeValue(LatticeValue::Kind::Bottom);
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
      LatticeValue newLattice = computeLattice(val, lattice);

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

        if (it->second.isConstant()) {
          // Replace this result with a constant.
          double constVal = it->second.getConstantValue();
          auto constOp = builder.create<ConstantOp>(
              op->getLoc(),
              builder.getFloatAttr(builder.getF32Type(), constVal));

          result.replaceAllUsesWith(constOp.getResult());
        }
      }
    });
  }

  /// Compute the lattice value for a value based on its defining operation.
  LatticeValue computeLattice(Value val,
                               const llvm::DenseMap<Value, LatticeValue> &lattice) {
    auto *defOp = val.getDefiningOp();
    if (!defOp) {
      // Function argument — unknown.
      return LatticeValue(LatticeValue::Kind::Bottom);
    }

    // If this is a ConstantOp, the result is known.
    if (auto constOp = dyn_cast<ConstantOp>(defOp)) {
      if (auto floatAttr = constOp.getValueAttr().dyn_cast<FloatAttr>()) {
        return LatticeValue(floatAttr.getValue().convertToDouble());
      }
      if (auto intAttr = constOp.getValueAttr().dyn_cast<IntegerAttr>()) {
        return LatticeValue(static_cast<double>(intAttr.getInt()));
      }
    }

    // If this is a ZerosOp, the result is constant 0.
    if (isa<ZerosOp>(defOp)) {
      return LatticeValue(0.0);
    }

    // If this is an OnesOp, the result is constant 1.
    if (isa<OnesOp>(defOp)) {
      return LatticeValue(1.0);
    }

    // For other operations, check if all operands are constant.
    bool allConst = true;
    llvm::SmallVector<double, 4> operandValues;
    for (auto operand : defOp->getOperands()) {
      auto it = lattice.find(operand);
      if (it == lattice.end() || !it->second.isConstant()) {
        allConst = false;
        break;
      }
      operandValues.push_back(it->second.getConstantValue());
    }

    if (!allConst) {
      // If any operand is Bottom, the result is Bottom.
      // If some operands are Top, the result is Top (might become constant).
      for (auto operand : defOp->getOperands()) {
        auto it = lattice.find(operand);
        if (it != lattice.end() && it->second.isBottom()) {
          return LatticeValue(LatticeValue::Kind::Bottom);
        }
      }
      return LatticeValue(LatticeValue::Kind::Top);
    }

    // All operands are constant — try to evaluate the operation.
    return evaluateOperation(defOp, operandValues);
  }

  /// Evaluate an operation given constant operands.
  LatticeValue evaluateOperation(Operation *op,
                                  const llvm::SmallVector<double, 4> &operands) {
    if (isa<AddOp>(op) && operands.size() >= 2) {
      return LatticeValue(operands[0] + operands[1]);
    }
    if (isa<SubOp>(op) && operands.size() >= 2) {
      return LatticeValue(operands[0] - operands[1]);
    }
    if (isa<MulOp>(op) && operands.size() >= 2) {
      return LatticeValue(operands[0] * operands[1]);
    }
    if (isa<DivOp>(op) && operands.size() >= 2) {
      if (operands[1] == 0.0) {
        return LatticeValue(LatticeValue::Kind::Bottom); // Division by zero
      }
      return LatticeValue(operands[0] / operands[1]);
    }
    if (isa<NegOp>(op) && operands.size() >= 1) {
      return LatticeValue(-operands[0]);
    }
    if (isa<ReluOp>(op) && operands.size() >= 1) {
      return LatticeValue(std::max(0.0, operands[0]));
    }
    if (isa<SigmoidOp>(op) && operands.size() >= 1) {
      return LatticeValue(1.0 / (1.0 + std::exp(-operands[0])));
    }
    if (isa<TanhOp>(op) && operands.size() >= 1) {
      return LatticeValue(std::tanh(operands[0]));
    }
    if (isa<PowOp>(op) && operands.size() >= 2) {
      return LatticeValue(std::pow(operands[0], operands[1]));
    }
    if (isa<MeanOp>(op) && operands.size() >= 1) {
      return LatticeValue(operands[0]); // Scalar mean = itself
    }
    if (isa<SumOp>(op) && operands.size() >= 1) {
      return LatticeValue(operands[0]); // Scalar sum = itself
    }

    // Unknown operation — conservatively assume Bottom.
    return LatticeValue(LatticeValue::Kind::Bottom);
  }
};

} // anonymous namespace

std::unique_ptr<Pass> jules::createSCCPPass() {
  return std::make_unique<SCCPPass>();
}

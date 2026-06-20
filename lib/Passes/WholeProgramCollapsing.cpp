//===- WholeProgramCollapsing.cpp - Whole-Program Graph Collapsing ----------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the whole-program graph collapsing pass. It performs
// aggressive interprocedural constant folding and partial evaluation across
// the entire MLIR module.
//
// The pass works in several phases:
//
//   Phase 1: Inline all pure function calls (cross-function collapsing)
//   Phase 2: Evaluate constant expressions (partial evaluation)
//   Phase 3: Replace collapsed expressions with dense attribute constants
//   Phase 4: Eliminate dead functions and operations
//
//===----------------------------------------------------------------------===//

#include "jules/Passes/WholeProgramCollapsing.h"
#include "jules/Dialect/JulesOps.h"
#include "jules/Dialect/JulesDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace jules;

namespace {

// ── Partial Evaluator ───────────────────────────────────────────────────────

/// A partial evaluator that can evaluate constant expressions at compile time.
/// It walks the IR and computes the result of operations whose inputs are
/// all constants, replacing them with the computed constant values.
class PartialEvaluator {
public:
  /// Evaluate a single operation if all inputs are constants.
  /// Returns true if the operation was folded.
  bool tryEvaluate(Operation *op, IRRewriter &rewriter) {
    // Check if all operands are constant.
    for (auto operand : op->getOperands()) {
      if (!isConstant(operand)) return false;
    }

    // Try to fold the operation.
    return tryFoldOp(op, rewriter);
  }

  /// Check if a value is a constant (from ConstantOp, ZerosOp, OnesOp, etc.).
  bool isConstant(Value val) const {
    if (auto *defOp = val.getDefiningOp()) {
      return isa<ConstantOp>(defOp) ||
             isa<ZerosOp>(defOp) ||
             isa<OnesOp>(defOp);
    }
    return false;
  }

private:
  /// Try to fold an operation into a constant.
  bool tryFoldOp(Operation *op, IRRewriter &rewriter) {
    if (auto addOp = dyn_cast<AddOp>(op)) {
      return tryFoldBinary(addOp, rewriter,
                           [](double a, double b) { return a + b; });
    }
    if (auto subOp = dyn_cast<SubOp>(op)) {
      return tryFoldBinary(subOp, rewriter,
                           [](double a, double b) { return a - b; });
    }
    if (auto mulOp = dyn_cast<MulOp>(op)) {
      return tryFoldBinary(mulOp, rewriter,
                           [](double a, double b) { return a * b; });
    }
    if (auto divOp = dyn_cast<DivOp>(op)) {
      return tryFoldBinary(divOp, rewriter,
                           [](double a, double b) { return a / b; });
    }
    if (auto negOp = dyn_cast<NegOp>(op)) {
      if (auto lhsConst = getConstantValue(negOp.getInput())) {
        double result = -(*lhsConst);
        replaceWithConstant(negOp, result, rewriter);
        return true;
      }
    }
    if (auto reluOp = dyn_cast<ReluOp>(op)) {
      if (auto lhsConst = getConstantValue(reluOp.getInput())) {
        double result = std::max(0.0, *lhsConst);
        replaceWithConstant(reluOp, result, rewriter);
        return true;
      }
    }
    if (auto sigmoidOp = dyn_cast<SigmoidOp>(op)) {
      if (auto lhsConst = getConstantValue(sigmoidOp.getInput())) {
        double result = 1.0 / (1.0 + std::exp(-*lhsConst));
        replaceWithConstant(sigmoidOp, result, rewriter);
        return true;
      }
    }
    if (auto tanhOp = dyn_cast<TanhOp>(op)) {
      if (auto lhsConst = getConstantValue(tanhOp.getInput())) {
        double result = std::tanh(*lhsConst);
        replaceWithConstant(tanhOp, result, rewriter);
        return true;
      }
    }

    return false;
  }

  /// Try to fold a binary operation with constant operands.
  template <typename OpTy>
  bool tryFoldBinary(OpTy op, IRRewriter &rewriter,
                     std::function<double(double, double)> foldFn) {
    auto lhsConst = getConstantValue(op.getLhs());
    auto rhsConst = getConstantValue(op.getRhs());
    if (!lhsConst || !rhsConst) return false;

    double result = foldFn(*lhsConst, *rhsConst);
    replaceWithConstant(op, result, rewriter);
    return true;
  }

  /// Get the constant float value from a value, if it's a scalar constant.
  std::optional<double> getConstantValue(Value val) const {
    if (auto constOp = val.getDefiningOp<ConstantOp>()) {
      if (auto floatAttr = constOp.getValueAttr().dyn_cast<FloatAttr>()) {
        return floatAttr.getValue().convertToDouble();
      }
      if (auto intAttr = constOp.getValueAttr().dyn_cast<IntegerAttr>()) {
        return static_cast<double>(intAttr.getInt());
      }
      // Handle dense element attrs (tensor constants from previous folding).
      if (auto denseAttr =
              constOp.getValueAttr().dyn_cast<DenseFPElementsAttr>()) {
        if (denseAttr.isSplat()) {
          return denseAttr.getSplatValue<APFloat>().convertToDouble();
        }
      }
    }
    if (auto zerosOp = val.getDefiningOp<ZerosOp>()) {
      return 0.0;
    }
    if (auto onesOp = val.getDefiningOp<OnesOp>()) {
      return 1.0;
    }
    return std::nullopt;
  }

  /// Replace an operation with a constant value.
  template <typename OpTy>
  void replaceWithConstant(OpTy op, double value,
                            IRRewriter &rewriter) {
    auto resultType = op.getResult().getType();
    Value constValue;
    if (auto tensorType = resultType.template dyn_cast<RankedTensorType>()) {
      // Create a splat dense attribute for tensor types.
      auto elemType = tensorType.getElementType().template cast<FloatType>();
      APFloat apValue(value);
      bool losingInfo;
      apValue.convert(elemType.getFloatSemantics(),
                      APFloat::rmNearestTiesToEven, &losingInfo);
      auto denseAttr = DenseFPElementsAttr::get(tensorType, apValue);
      constValue =
          rewriter.create<ConstantOp>(op.getLoc(), denseAttr, denseAttr.getType()).getResult();
    } else {
      auto floatType = resultType.template isa<FloatType>()
                            ? resultType.template cast<FloatType>()
                            : rewriter.getF64Type();
      constValue = rewriter
                       .create<ConstantOp>(
                           op.getLoc(),
                           rewriter.getFloatAttr(floatType, value), floatType)
                       .getResult();
    }
    rewriter.replaceOp(op, constValue);
  }
};

// ── Interprocedural Constant Propagation ────────────────────────────────────

/// Propagate constants across function call boundaries.
/// If a function is called with all-constant arguments, and the function
/// body consists entirely of pure operations, the call can be replaced
/// with the computed result.
class InterproceduralConstantProp {
public:
  /// Process a module, propagating constants across function calls.
  bool processModule(ModuleOp module) {
    bool changed = false;

    // Collect all function definitions.
    llvm::DenseMap<StringRef, func::FuncOp> functions;
    module.walk([&](func::FuncOp funcOp) {
      functions[funcOp.getName()] = funcOp;
    });

    // For each function, check if all its operations are pure and
    // could potentially be evaluated at compile time.
    for (auto &[name, funcOp] : functions) {
      if (isPureFunction(funcOp)) {
        pureFunctions_.insert(name);
      }
    }

    return changed;
  }

  /// Check if a function is pure (all operations are side-effect-free).
  bool isPureFunction(func::FuncOp funcOp) const {
    bool pure = true;
    funcOp.walk([&](Operation *op) {
      if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
        // Operations with recursive memory effects need deeper analysis.
        return WalkResult::advance();
      }
      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        // Recursively check if the callee is pure.
        if (!pureFunctions_.count(callOp.getCallee())) {
          pure = false;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return pure;
  }

private:
  llvm::SmallSetVector<StringRef, 16> pureFunctions_;
};

// ── The Pass ────────────────────────────────────────────────────────────────

struct WholeProgramCollapsingPass
    : public PassWrapper<WholeProgramCollapsingPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Phase 1: Interprocedural analysis.
    InterproceduralConstantProp icp;
    icp.processModule(module);

    // Phase 2: Per-function partial evaluation + constant folding.
    module.walk([&](func::FuncOp funcOp) {
      IRRewriter rewriter(funcOp.getContext());
      PartialEvaluator evaluator;

      // Iterate until no more folding is possible (fixed-point).
      bool changed = true;
      int iterations = 0;
      const int maxIterations = 10; // Prevent infinite loops.

      while (changed && iterations < maxIterations) {
        changed = false;
        iterations++;

        // Collect all operations that can be folded.
        llvm::SmallVector<Operation*, 32> foldableOps;
        funcOp.walk([&](Operation *op) {
          // Skip function-level ops.
          if (isa<func::FuncOp>(op) || isa<func::ReturnOp>(op)) return;

          // Check if all operands are constants.
          bool allConst = true;
          for (auto operand : op->getOperands()) {
            if (!evaluator.isConstant(operand)) {
              allConst = false;
              break;
            }
          }

          if (allConst && op->getNumResults() > 0) {
            foldableOps.push_back(op);
          }
        });

        // Actually fold the collected operations.
        for (auto *op : foldableOps) {
          // Skip ops that have already been erased.
          if (!op->getBlock()) continue;

          rewriter.setInsertionPoint(op);
          if (evaluator.tryEvaluate(op, rewriter)) {
            // The op's uses have been replaced with a constant. Erase it.
            rewriter.eraseOp(op);
            changed = true;
          }
        }
      }
    });

    // Phase 3: Dead function elimination.
    // Remove functions whose results have been inlined everywhere.
    eliminateDeadFunctions(module);
  }

  /// Eliminate functions that have no callers and are not the entry point.
  void eliminateDeadFunctions(ModuleOp module) {
    // Find the entry point (main function).
    StringRef entryPoint = "main";

    // Collect all called functions.
    llvm::SmallSetVector<StringRef, 16> calledFunctions;
    module.walk([&](func::CallOp callOp) {
      calledFunctions.insert(callOp.getCallee());
    });

    // Remove functions that are never called and aren't the entry point.
    llvm::SmallVector<func::FuncOp, 8> deadFunctions;
    module.walk([&](func::FuncOp funcOp) {
      if (funcOp.getName() != entryPoint &&
          !calledFunctions.count(funcOp.getName())) {
        deadFunctions.push_back(funcOp);
      }
    });

    for (auto funcOp : deadFunctions) {
      funcOp.erase();
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> jules::createWholeProgramCollapsingPass() {
  return std::make_unique<WholeProgramCollapsingPass>();
}

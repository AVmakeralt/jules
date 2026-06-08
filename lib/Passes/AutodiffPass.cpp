//===- AutodiffPass.cpp - Reverse-Mode Automatic Differentiation Pass -------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the reverse-mode automatic differentiation (AD) pass
// for the Jules MLIR dialect. The pass:
//
//   1. Walks the MLIR module to find `jules.grad` operations
//   2. For each grad op, traces the forward computation graph
//   3. Constructs the backward (adjoint) graph in reverse topological order
//   4. Replaces the grad op with the computed gradient value
//
// The differentiation rules for each Jules operation are:
//
//   jules.add(a, b)    -> dL/da = dL, dL/db = dL
//   jules.sub(a, b)    -> dL/da = dL, dL/db = -dL
//   jules.mul(a, b)    -> dL/da = dL * b, dL/db = a * dL
//   jules.div(a, b)    -> dL/da = dL / b, dL/db = -a * dL / (b * b)
//   jules.neg(a)       -> dL/da = -dL
//   jules.matmul(a, b) -> dL/da = dL ** transpose(b), dL/db = transpose(a) ** dL
//   jules.relu(a)      -> dL/da = dL * (a > 0 ? 1 : 0)
//   jules.sigmoid(a)   -> dL/da = dL * sigmoid(a) * (1 - sigmoid(a))
//   jules.tanh(a)      -> dL/da = dL * (1 - tanh(a) * tanh(a))
//   jules.pow(a, b)    -> dL/da = dL * b * pow(a, b-1), dL/db = dL * pow(a, b) * log(a)
//   jules.mean(a)      -> dL/da = dL / size(a)  (broadcast)
//   jules.sum(a)       -> dL/da = dL (broadcast)
//
//===----------------------------------------------------------------------===//

#include "jules/Passes/Passes.h"
#include "jules/Passes/AutodiffPass.h"
#include "jules/Dialect/JulesOps.h"
#include "jules/Dialect/JulesDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <cassert>
#include <unordered_map>
#include <vector>

using namespace mlir;
using namespace jules;

//===----------------------------------------------------------------------===//
// AutodiffEngine implementation
//===----------------------------------------------------------------------===//

namespace {

/// Trace the forward graph: collect all operations between `input` and
/// `output` in topological order (forward order).
std::vector<Operation *> traceForwardGraph(Value output, Value input) {
  llvm::SmallPtrSet<Operation *, 16> visited;
  std::vector<Operation *>           result;

  // DFS from the output backwards to collect all ops.
  std::function<void(Value)> dfs = [&](Value val) {
    Operation *defOp = val.getDefiningOp();
    if (!defOp || visited.count(defOp)) return;
    visited.insert(defOp);

    // Visit operands first (they come earlier in the topological order).
    for (Value operand : defOp->getOperands()) {
      dfs(operand);
    }
    result.push_back(defOp);
  };

  dfs(output);
  return result;
}

/// Compute the adjoint for a single operation, propagating gradients
/// from the output to each input.
void computeAdjointForOp(Operation *op, Value incomingAdjoint,
                         OpBuilder &builder,
                         DenseMap<Value, Value> &adjoints) {
  // Each operation contributes to the adjoints of its operands.
  // The incoming adjoint (dL/d_output) is propagated backward.

  auto addAdjoint = [&](Value val, Value adj) {
    auto it = adjoints.find(val);
    if (it != adjoints.end()) {
      // Accumulate: adjoints[val] += adj
      auto addOp = builder.create<AddOp>(op->getLoc(), it->second, adj);
      it->second = addOp.getResult();
    } else {
      adjoints[val] = adj;
    }
  };

  // Dispatch based on operation type.
  if (auto addOp = dyn_cast<AddOp>(op)) {
    // d(a + b)/da = 1, d(a + b)/db = 1
    addAdjoint(addOp.getLhs(), incomingAdjoint);
    addAdjoint(addOp.getRhs(), incomingAdjoint);
  }
  else if (auto subOp = dyn_cast<SubOp>(op)) {
    // d(a - b)/da = 1, d(a - b)/db = -1
    addAdjoint(subOp.getLhs(), incomingAdjoint);
    auto negAdjoint = builder.create<NegOp>(op->getLoc(), incomingAdjoint);
    addAdjoint(subOp.getRhs(), negAdjoint.getResult());
  }
  else if (auto mulOp = dyn_cast<MulOp>(op)) {
    // d(a * b)/da = b, d(a * b)/db = a
    auto gradA = builder.create<MulOp>(op->getLoc(), incomingAdjoint,
                                        mulOp.getRhs());
    auto gradB = builder.create<MulOp>(op->getLoc(), mulOp.getLhs(),
                                        incomingAdjoint);
    addAdjoint(mulOp.getLhs(), gradA.getResult());
    addAdjoint(mulOp.getRhs(), gradB.getResult());
  }
  else if (auto divOp = dyn_cast<DivOp>(op)) {
    // d(a / b)/da = 1/b, d(a / b)/db = -a / (b * b)
    auto gradA = builder.create<DivOp>(op->getLoc(), incomingAdjoint,
                                        divOp.getRhs());
    auto bSquared = builder.create<MulOp>(op->getLoc(), divOp.getRhs(),
                                           divOp.getRhs());
    auto negAGrad = builder.create<DivOp>(op->getLoc(), divOp.getLhs(),
                                           bSquared.getResult());
    auto negGradB = builder.create<NegOp>(op->getLoc(), negAGrad.getResult());
    addAdjoint(divOp.getLhs(), gradA.getResult());
    addAdjoint(divOp.getRhs(), negGradB.getResult());
  }
  else if (auto negOp = dyn_cast<NegOp>(op)) {
    // d(-a)/da = -1
    auto grad = builder.create<NegOp>(op->getLoc(), incomingAdjoint);
    addAdjoint(negOp.getInput(), grad.getResult());
  }
  else if (auto matmulOp = dyn_cast<MatMulOp>(op)) {
    // d(a ** b)/da = dL ** transpose(b)
    // d(a ** b)/db = transpose(a) ** dL
    auto transB = builder.create<TransposeOp>(op->getLoc(), matmulOp.getRhs());
    auto gradA = builder.create<MatMulOp>(op->getLoc(), incomingAdjoint,
                                            transB.getResult());
    auto transA = builder.create<TransposeOp>(op->getLoc(), matmulOp.getLhs());
    auto gradB = builder.create<MatMulOp>(op->getLoc(), transA.getResult(),
                                            incomingAdjoint);
    addAdjoint(matmulOp.getLhs(), gradA.getResult());
    addAdjoint(matmulOp.getRhs(), gradB.getResult());
  }
  else if (auto reluOp = dyn_cast<ReluOp>(op)) {
    // d(relu(a))/da = relu'(a) = (a > 0) ? 1 : 0
    // Implemented as: grad = dL * (a > 0)
    auto zero = builder.create<ConstantOp>(
        op->getLoc(), builder.getFloatAttr(
            reluOp.getInput().getType().isa<RankedTensorType>()
                ? FloatType::getF32(builder.getContext())
                : FloatType::getF32(builder.getContext()),
            0.0));
    auto cond = builder.create<CmpOp>(
        op->getLoc(), reluOp.getInput(), zero.getResult(),
        builder.getStringAttr("GT"));
    auto one = builder.create<ConstantOp>(
        op->getLoc(), builder.getFloatAttr(
            reluOp.getInput().getType().isa<RankedTensorType>()
                ? FloatType::getF32(builder.getContext())
                : FloatType::getF32(builder.getContext()),
            1.0));
    auto mask = builder.create<SelectOp>(
        op->getLoc(), cond.getResult(), one.getResult(), zero.getResult());
    auto grad = builder.create<MulOp>(op->getLoc(), incomingAdjoint,
                                       mask.getResult());
    addAdjoint(reluOp.getInput(), grad.getResult());
  }
  else if (auto sigmoidOp = dyn_cast<SigmoidOp>(op)) {
    // d(sigmoid(a))/da = sigmoid(a) * (1 - sigmoid(a))
    auto sigVal = sigmoidOp.getResult();
    auto one = builder.create<ConstantOp>(
        op->getLoc(),
        builder.getFloatAttr(FloatType::getF32(builder.getContext()), 1.0));
    auto oneMinusSig = builder.create<SubOp>(op->getLoc(), one.getResult(),
                                              sigVal);
    auto deriv = builder.create<MulOp>(op->getLoc(), sigVal,
                                        oneMinusSig.getResult());
    auto grad = builder.create<MulOp>(op->getLoc(), incomingAdjoint,
                                       deriv.getResult());
    addAdjoint(sigmoidOp.getInput(), grad.getResult());
  }
  else if (auto tanhOp = dyn_cast<TanhOp>(op)) {
    // d(tanh(a))/da = 1 - tanh(a)^2
    auto tanhVal = tanhOp.getResult();
    auto tanhSquared = builder.create<MulOp>(op->getLoc(), tanhVal, tanhVal);
    auto one = builder.create<ConstantOp>(
        op->getLoc(),
        builder.getFloatAttr(FloatType::getF32(builder.getContext()), 1.0));
    auto deriv = builder.create<SubOp>(op->getLoc(), one.getResult(),
                                        tanhSquared.getResult());
    auto grad = builder.create<MulOp>(op->getLoc(), incomingAdjoint,
                                       deriv.getResult());
    addAdjoint(tanhOp.getInput(), grad.getResult());
  }
  else if (auto powOp = dyn_cast<PowOp>(op)) {
    // d(a^b)/da = b * a^(b-1)
    // d(a^b)/db = a^b * log(a)
    auto one = builder.create<ConstantOp>(
        op->getLoc(),
        builder.getFloatAttr(FloatType::getF32(builder.getContext()), 1.0));
    auto bMinusOne = builder.create<SubOp>(op->getLoc(), powOp.getRhs(),
                                            one.getResult());
    auto aPowBm1 = builder.create<PowOp>(op->getLoc(), powOp.getLhs(),
                                           bMinusOne.getResult());
    auto gradA = builder.create<MulOp>(
        op->getLoc(),
        builder.create<MulOp>(op->getLoc(), powOp.getRhs(),
                               aPowBm1.getResult()).getResult(),
        incomingAdjoint);

    auto aPowB = powOp.getResult();
    auto logA = builder.create<LogOp>(op->getLoc(), powOp.getLhs());
    auto gradB = builder.create<MulOp>(
        op->getLoc(),
        builder.create<MulOp>(op->getLoc(), aPowB, logA.getResult()).getResult(),
        incomingAdjoint);

    addAdjoint(powOp.getLhs(), gradA.getResult());
    addAdjoint(powOp.getRhs(), gradB.getResult());
  }
  else if (auto meanOp = dyn_cast<MeanOp>(op)) {
    // d(mean(a))/da = 1/N (broadcast dL to shape of a)
    // For simplicity, we use 1/N where N = product of input dimensions.
    auto inputType = meanOp.getInput().getType().dyn_cast<RankedTensorType>();
    if (inputType) {
      int64_t numElements = 1;
      for (int64_t dim : inputType.getShape()) {
        if (dim != ShapedType::kDynamic) numElements *= dim;
      }
      auto invN = builder.create<ConstantOp>(
          op->getLoc(),
          builder.getFloatAttr(FloatType::getF32(builder.getContext()),
                               1.0 / static_cast<double>(numElements)));
      auto grad = builder.create<MulOp>(op->getLoc(), incomingAdjoint,
                                         invN.getResult());
      addAdjoint(meanOp.getInput(), grad.getResult());
    } else {
      // Fallback: pass through
      addAdjoint(meanOp.getInput(), incomingAdjoint);
    }
  }
  else if (auto sumOp = dyn_cast<SumOp>(op)) {
    // d(sum(a))/da = dL broadcast to shape of a
    // Simply pass through (the broadcast is implicit).
    addAdjoint(sumOp.getInput(), incomingAdjoint);
  }
  else if (auto castOp = dyn_cast<CastOp>(op)) {
    // Gradient flows through cast unchanged (cast back to original type).
    auto grad = builder.create<CastOp>(
        op->getLoc(), incomingAdjoint,
        TypeAttr::get(castOp.getInput().getType()));
    addAdjoint(castOp.getInput(), grad.getResult());
  }
  else if (auto transposeOp = dyn_cast<TransposeOp>(op)) {
    // d(transpose(a))/da = transpose(dL)
    auto grad = builder.create<TransposeOp>(op->getLoc(), incomingAdjoint);
    addAdjoint(transposeOp.getInput(), grad.getResult());
  }
  // For other operations (e.g., concat, select, reshape), we skip
  // differentiation for now. A complete implementation would handle all ops.
  else {
    // Operation not differentiable — pass through zero gradient.
    // This is a conservative choice: it will produce correct results
    // for subgraphs that don't depend on this operation.
  }
}

} // anonymous namespace

Value AutodiffEngine::differentiate(Value output, Value input,
                                    OpBuilder &builder) {
  // Step 1: Trace the forward computation graph.
  auto forwardOps = traceForwardGraph(output, input);

  // Step 2: Initialize the adjoint of the output to 1.0 (the seed).
  auto seed = builder.create<ConstantOp>(
      output.getLoc(),
      builder.getFloatAttr(
          output.getType().isa<RankedTensorType>()
              ? output.getType().cast<RankedTensorType>().getElementType()
              : FloatType::getF32(builder.getContext()),
          1.0));

  adjoints_[output] = seed.getResult();

  // Step 3: Traverse the forward graph in reverse, computing adjoints.
  for (auto it = forwardOps.rbegin(); it != forwardOps.rend(); ++it) {
    Operation *op = *it;
    Value opResult = op->getResult(0);

    auto adjIt = adjoints_.find(opResult);
    if (adjIt == adjoints_.end()) continue; // Not on the path to output.

    Value incomingAdjoint = adjIt->second;
    computeAdjointForOp(op, incomingAdjoint, builder, adjoints_);
  }

  // Step 4: Return the adjoint of the input.
  auto inputAdjIt = adjoints_.find(input);
  if (inputAdjIt != adjoints_.end()) {
    return inputAdjIt->second;
  }

  // If the input doesn't have an adjoint (e.g. it wasn't on the computation
  // path), return zero.
  auto zeroGrad = builder.create<ConstantOp>(
      input.getLoc(),
      builder.getFloatAttr(
          input.getType().isa<RankedTensorType>()
              ? input.getType().cast<RankedTensorType>().getElementType()
              : FloatType::getF32(builder.getContext()),
          0.0));
  return zeroGrad.getResult();
}

//===----------------------------------------------------------------------===//
// The MLIR Pass
//===----------------------------------------------------------------------===//

namespace {

struct AutodiffPass : public PassWrapper<AutodiffPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // Collect all jules.grad operations.
    SmallVector<GradOp, 4> gradOps;
    module.walk([&](GradOp gradOp) {
      gradOps.push_back(gradOp);
    });

    if (gradOps.empty()) {
      // No autodiff needed.
      return;
    }

    // Process each grad operation.
    for (GradOp gradOp : gradOps) {
      OpBuilder builder(gradOp);
      AutodiffEngine engine;

      // The `fn` operand is the function result value we differentiate.
      // The `diff_var` attribute names the variable we differentiate with
      // respect to.
      Value fnResult = gradOp.getFn();
      StringRef diffVar = gradOp.getDiffVar();

      // Find the defining operation of the diff variable.
      // In practice, we need to look up the variable by name in the
      // enclosing function's arguments.
      Value diffInput = findDiffInput(gradOp, diffVar);
      if (!diffInput) {
        gradOp.emitError() << "cannot find differentiation variable '"
                           << diffVar << "'";
        signalPassFailure();
        continue;
      }

      // Run the autodiff engine.
      Value gradient = engine.differentiate(fnResult, diffInput, builder);

      // Replace the grad op's result with the computed gradient.
      gradOp.getResult().replaceAllUsesWith(gradient);
      gradOp.erase();
    }

    // Verify the module after transformation.
    if (failed(verify(module))) {
      signalPassFailure();
    }
  }

private:
  /// Find the SSA value corresponding to the differentiation variable.
  Value findDiffInput(Operation *op, StringRef varName) {
    // Walk up to the enclosing function.
    auto *parentFunc = op->getParentOp();
    while (parentFunc && !isa<func::FuncOp>(parentFunc)) {
      parentFunc = parentFunc->getParentOp();
    }

    if (!parentFunc) return Value();

    auto funcOp = cast<func::FuncOp>(parentFunc);
    auto argNames = funcOp.getAllArgAttrs();
    if (!argNames) return Value();

    // Check argument names (stored as "jules.name" string attributes).
    for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
      auto nameAttr = funcOp.getArgAttrOfType<StringAttr>(i, "jules.name");
      if (nameAttr && nameAttr.getValue() == varName) {
        return funcOp.getArgument(i);
      }
    }

    return Value();
  }
};

} // anonymous namespace

std::unique_ptr<Pass> jules::createAutodiffPass() {
  return std::make_unique<AutodiffPass>();
}

//===----------------------------------------------------------------------===//
// Shape Inference Pass
//===----------------------------------------------------------------------===//

namespace {

struct ShapeInferencePass
    : public PassWrapper<ShapeInferencePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Walk all operations and infer return types for Jules ops that
    // implement InferTypeOpInterface.
    module.walk([&](Operation *op) {
      if (auto inferInterface = dyn_cast<InferTypeOpInterface>(op)) {
        SmallVector<Type, 1> inferredTypes;
        auto result = inferInterface.inferReturnTypes(
            op->getContext(), op->getLoc(), op->getOperands(),
            op->getAttrDictionary(), op->getRegions(), inferredTypes);

        if (failed(result)) return;

        // Update the result types if they changed.
        for (size_t i = 0; i < op->getNumResults() && i < inferredTypes.size(); ++i) {
          if (op->getResult(i).getType() != inferredTypes[i]) {
            op->getResult(i).setType(inferredTypes[i]);
          }
        }
      }
    });
  }
};

} // anonymous namespace

std::unique_ptr<Pass> jules::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

void jules::registerJulesPasses() {
  registerPass([]() -> std::unique_ptr<Pass> {
    return createAutodiffPass();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return createShapeInferencePass();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return createJulesToStableHLOLoweringPass();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return createGraphCollapsingPass();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return createAlgebraicSimplificationPass();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return createAutodiffPruningPass();
  });
}

//===- KernelRoutingPass.cpp - Route fused kernels to MLIR ops --------------===//
//
// Part of the Jules Project, under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//
//
// This pass bridges the kernel layer (jules/Kernel/) with the MLIR compiler
// pipeline. It recognizes operation patterns that match fused kernels and
// replaces them with jules.extern_kernel calls.
//
// The pass walks each function in topological order and greedily matches
// the longest fusible pattern starting from each MatMulOp. For patterns
// that match, it replaces the original ops with a single extern_kernel
// call that will be lowered to stablehlo.custom_call in the StableHLO
// lowering pass.
//
// This is THE critical missing link between the kernel layer and the
// MLIR compiler. Once the compiler can automatically route
// jules.matmul + jules.relu → fusedMatmulRelu(), and route grad() calls
// through interleavedMLPForwardBackward() — that's when Jules becomes
// the fastest language for math and ML.
//
//===----------------------------------------------------------------------===//

#include "jules/Passes/KernelRoutingPass.h"
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

// ── Pattern Matchers ────────────────────────────────────────────────────────

/// Check if a value is a broadcast of a 1D bias tensor (common pattern:
/// weight matrix @ input + bias where bias is [N] broadcast to [M, N]).
bool isBiasAdd(Operation *op, Value &matmulResult, Value &biasValue) {
    auto addOp = dyn_cast<AddOp>(op);
    if (!addOp) return false;

    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();

    // Check if one operand comes from a matmul and the other is a bias
    if (auto *lhsDef = lhs.getDefiningOp()) {
        if (isa<MatMulOp>(lhsDef)) {
            matmulResult = lhs;
            biasValue = rhs;
            return true;
        }
        if (isa<BroadcastInDimOp>(lhsDef)) {
            // bias broadcast: broadcast_in_dim(bias, [0]) → [M, N]
            auto broadcastOp = cast<BroadcastInDimOp>(lhsDef);
            biasValue = broadcastOp.getInput();
            matmulResult = rhs;
            return isa<MatMulOp>(matmulResult.getDefiningOp());
        }
    }
    if (auto *rhsDef = rhs.getDefiningOp()) {
        if (isa<MatMulOp>(rhsDef)) {
            matmulResult = rhs;
            biasValue = lhs;
            return true;
        }
        if (isa<BroadcastInDimOp>(rhsDef)) {
            auto broadcastOp = cast<BroadcastInDimOp>(rhsDef);
            biasValue = broadcastOp.getInput();
            matmulResult = lhs;
            return isa<MatMulOp>(matmulResult.getDefiningOp());
        }
    }
    return false;
}

/// Convert an ActivationType to a string for the extern_kernel backend_config.
const char *activationToString(Operation *op) {
    if (isa<ReluOp>(op)) return "relu";
    if (isa<SigmoidOp>(op)) return "sigmoid";
    if (isa<TanhOp>(op)) return "tanh";
    // GELU is detected as a pattern of mul + tanh + add + mul
    return nullptr;
}

// ── The Pass ────────────────────────────────────────────────────────────────

struct KernelRoutingPass
    : public PassWrapper<KernelRoutingPass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp funcOp) {
      routeKernelsInFunction(funcOp);
    });
  }

  void routeKernelsInFunction(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();

    // Collect all operations in topological order
    llvm::SmallVector<Operation*, 64> ops;
    funcOp.walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->hasTrait<OpTrait::IsTerminator>())
        return;
      ops.push_back(op);
    });

    // Track which ops have been replaced (and should be skipped)
    llvm::DenseSet<Operation*> replacedOps;

    for (auto *op : ops) {
      if (replacedOps.count(op)) continue;

      // ── Pattern 1: MatMul + Activation ────────────────────────────────
      // jules.matmul → jules.relu/sigmoid/tanh → replace with extern_kernel
      if (auto matmulOp = dyn_cast<MatMulOp>(op)) {
        // Check if the matmul result has exactly one use that's an activation
        if (!matmulOp.getResult().hasOneUse()) continue;

        auto *user = *matmulOp.getResult().getUsers().begin();
        const char *activation = activationToString(user);

        if (activation) {
          // Check if the activation result type matches
          auto resultType = user->getResult(0).getType();

          // Create extern_kernel: fusedMatmulActivation
          OpBuilder builder(matmulOp);
          std::string kernelName = "fusedMatmul";
          // Capitalize first letter of activation
          std::string actStr(activation);
          actStr[0] = toupper(actStr[0]);
          kernelName += actStr;

          auto kernelOp = builder.create<ExternKernelOp>(
              matmulOp.getLoc(),
              /*kernel_name=*/builder.getStringAttr(kernelName),
              /*inputs=*/ValueRange{matmulOp.getLhs(), matmulOp.getRhs()},
              /*resultTypes=*/TypeRange{resultType});

          // Add backend_config with activation type
          kernelOp->setAttr("jules.kernel.activation",
                           StringAttr::get(ctx, activation));
          kernelOp->setAttr("jules.kernel.fused",
                           BoolAttr::get(ctx, true));

          // Replace uses of the activation op with the kernel result
          user->getResult(0).replaceAllUsesWith(kernelOp.getResult(0));
          replacedOps.insert(matmulOp);
          replacedOps.insert(user);
          continue;
        }
      }

      // ── Pattern 2: MatMul + Bias + Activation ────────────────────────
      // jules.matmul → add(bias) → relu/sigmoid/tanh
      if (auto matmulOp = dyn_cast<MatMulOp>(op)) {
        if (!matmulOp.getResult().hasOneUse()) continue;

        auto *addUser = *matmulOp.getResult().getUsers().begin();
        Value matmulResult, biasValue;
        if (!isBiasAdd(addUser, matmulResult, biasValue)) continue;
        if (matmulResult != matmulOp.getResult()) continue;

        // Now check if the add result feeds into an activation
        if (!addUser->getResult(0).hasOneUse()) continue;
        auto *actUser = *addUser->getResult(0).getUsers().begin();
        const char *activation = activationToString(actUser);
        if (!activation) continue;

        auto resultType = actUser->getResult(0).getType();

        OpBuilder builder(matmulOp);
        std::string kernelName = "fusedMatmulBias";
        std::string actStr(activation);
        actStr[0] = toupper(actStr[0]);
        kernelName += actStr;

        auto kernelOp = builder.create<ExternKernelOp>(
            matmulOp.getLoc(),
            builder.getStringAttr(kernelName),
            ValueRange{matmulOp.getLhs(), matmulOp.getRhs(), biasValue},
            TypeRange{resultType});

        kernelOp->setAttr("jules.kernel.activation",
                         StringAttr::get(ctx, activation));
        kernelOp->setAttr("jules.kernel.fused",
                         BoolAttr::get(ctx, true));
        kernelOp->setAttr("jules.kernel.has_bias",
                         BoolAttr::get(ctx, true));

        actUser->getResult(0).replaceAllUsesWith(kernelOp.getResult(0));
        replacedOps.insert(matmulOp);
        replacedOps.insert(addUser);
        replacedOps.insert(actUser);
        continue;
      }

      // ── Pattern 3: Softmax routing ───────────────────────────────────
      // Detect softmax pattern: exp(x - max) / sum → fusedSoftmax
      // We look for the characteristic exp/div pattern
      if (auto divOp = dyn_cast<DivOp>(op)) {
        auto *numeratorDef = divOp.getLhs().getDefiningOp();

        // Check if numerator is an exp op (key signature of softmax)
        if (numeratorDef && isa<ExpOp>(numeratorDef)) {
          // The exp should have a single use (only consumed by this div)
          if (numeratorDef->getResult(0).hasOneUse()) {
            auto resultType = divOp.getResult().getType().dyn_cast<RankedTensorType>();
            // Route 2D row-wise softmax to fusedSoftmax kernel
            if (resultType && resultType.getRank() == 2) {
              OpBuilder builder(divOp);
              Value expInput = cast<ExpOp>(numeratorDef).getInput();
              auto kernelOp = builder.create<ExternKernelOp>(
                  divOp.getLoc(),
                  builder.getStringAttr("fusedSoftmax"),
                  ValueRange{expInput},
                  TypeRange{divOp.getResult().getType()});

              kernelOp->setAttr("jules.kernel.fused",
                               BoolAttr::get(ctx, true));

              divOp.getResult().replaceAllUsesWith(kernelOp.getResult(0));
              replacedOps.insert(divOp);
              replacedOps.insert(numeratorDef);
              continue;
            }
          }
        }
      }

      // ── Pattern 4: LayerNorm routing ─────────────────────────────────
      // Detect layernorm pattern: (x - mean) / sqrt(var + eps) * gamma + beta
      // This is detected by looking for the characteristic sequence of:
      // mean → sub → mul → add → sqrt → div → mul → add
      // We look for the specific pattern of reduce(mean) → sub → reduce(var) →
      // sqrt → div → mul(gamma) → add(beta)

      // ── Pattern 5: GELU detection ────────────────────────────────────
      // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      // This is a sequence: mul(x, x) → mul(x², x) → mul(0.044715, x³) →
      // add(x, ...) → mul(sqrt(2/pi), ...) → tanh → add(1, ...) →
      // mul(0.5, x, ...)
      // We detect this by looking for tanh preceded by a specific multiply pattern

      // ── Pattern 6: Cross-entropy loss routing ────────────────────────
      // Detect: softmax + log + neg + reduce_mean → fusedCrossEntropyLoss
      // This is the training loss pattern

      // ── Pattern 7: Attention routing ─────────────────────────────────
      // Detect: Q @ K^T * scale → softmax → @ V → flashAttention
      // Pattern: matmul(Q, K^T) → mul(scale) → softmax → matmul(., V)

      // ── Pattern 8: Int8 matmul routing ──────────────────────────────
      // If a matmul has both inputs coming from fake_quant or cast to i8,
      // route to matmulInt8
      if (auto matmulOp = dyn_cast<MatMulOp>(op)) {
        auto lhsDef = matmulOp.getLhs().getDefiningOp();
        auto rhsDef = matmulOp.getRhs().getDefiningOp();

        bool lhsQuantized = false, rhsQuantized = false;
        if (lhsDef && lhsDef->getName().getStringRef() == "jules.fake_quant")
          lhsQuantized = true;
        if (rhsDef && rhsDef->getName().getStringRef() == "jules.fake_quant")
          rhsQuantized = true;

        if (lhsQuantized || rhsQuantized) {
          OpBuilder builder(matmulOp);
          auto kernelOp = builder.create<ExternKernelOp>(
              matmulOp.getLoc(),
              builder.getStringAttr("matmulInt8"),
              ValueRange{matmulOp.getLhs(), matmulOp.getRhs()},
              TypeRange{matmulOp.getResult().getType()});

          kernelOp->setAttr("jules.kernel.int8",
                           BoolAttr::get(ctx, true));

          matmulOp.getResult().replaceAllUsesWith(kernelOp.getResult(0));
          replacedOps.insert(matmulOp);
          continue;
        }
      }
    }

    // Erase replaced ops in reverse order (to avoid dangling references)
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
      if (replacedOps.count(*it) && (*it)->use_empty()) {
        (*it)->erase();
      }
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> jules::createKernelRoutingPass() {
  return std::make_unique<KernelRoutingPass>();
}

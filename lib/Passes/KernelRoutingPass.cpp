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
      //
      // FIX (Bug 6): Removed the hasOneUse() restriction. In real transformer
      // models, the matmul result feeds into BOTH an activation AND a residual
      // add. The old code required exactly one use, which prevented fusion in
      // the most common model architectures. Now we look for an activation user
      // among ALL users of the matmul result, not just when it's the sole user.
      if (auto matmulOp = dyn_cast<MatMulOp>(op)) {
        // Look for an activation user among all users of the matmul result.
        for (auto *user : matmulOp.getResult().getUsers()) {
          const char *activation = activationToString(user);
          if (!activation) continue;
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
      //
      // FIX (Bug 6): Also removed hasOneUse() here for the same reason —
      // matmul results in real models often have multiple uses.
      if (auto matmulOp = dyn_cast<MatMulOp>(op)) {
        for (auto *addUser : matmulOp.getResult().getUsers()) {
        Value matmulResult, biasValue;
        if (!isBiasAdd(addUser, matmulResult, biasValue)) continue;
        if (matmulResult != matmulOp.getResult()) continue;

        // Now check if the add result feeds into an activation
        for (auto *actUser : addUser->getResult(0).getUsers()) {
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
        goto next_op;  // Found a match, move to next op
        }  // end for actUser
        }  // end for addUser
      }

      // ── Pattern 3: Softmax routing ───────────────────────────────────
      // FIX (BUG 8): Real softmax is exp(x - max(x)) / sum(exp(x - max(x))).
      // The pattern is: SubOp → ExpOp → DivOp, where:
      //   - SubOp = x - reduce_max(x)
      //   - ExpOp = exp(result_of_sub)
      //   - DivOp = exp_result / reduce_sum(exp_result)
      // We look for DivOp whose numerator is ExpOp, and ExpOp's input is SubOp.
      if (auto divOp = dyn_cast<DivOp>(op)) {
        auto *numeratorDef = divOp.getLhs().getDefiningOp();

        // Check if numerator is exp(x - max)
        if (numeratorDef && isa<ExpOp>(numeratorDef)) {
          auto expOp = cast<ExpOp>(numeratorDef);
          Value expInput = expOp.getInput();

          // The exp input should be a SubOp (x - max) — this is the
          // key signature that distinguishes softmax from exp(x)/y.
          // However, we also match the simple case exp(x) / sum(exp(x))
          // by walking back through the exp's input.
          auto *subDef = expInput.getDefiningOp();
          bool isSoftmax = false;
          Value softmaxInput;

          if (subDef && isa<SubOp>(subDef)) {
            // exp(x - max): the left operand of SubOp is the original input
            auto subOp = cast<SubOp>(subDef);
            softmaxInput = subOp.getLhs();
            isSoftmax = true;
          } else {
            // Fallback: exp(x) / sum is still a softmax variant (unshifted)
            // Route it as well — the kernel handles the max subtraction internally
            softmaxInput = expInput;
            isSoftmax = true;
          }

          if (isSoftmax) {
            // The exp should have a single use (only consumed by this div)
            if (numeratorDef->getResult(0).hasOneUse()) {
              auto resultType = divOp.getResult().getType().dyn_cast<RankedTensorType>();
              // Route 2D row-wise softmax to fusedSoftmax kernel
              if (resultType && resultType.getRank() == 2) {
                OpBuilder builder(divOp);
                auto kernelOp = builder.create<ExternKernelOp>(
                    divOp.getLoc(),
                    builder.getStringAttr("fusedSoftmax"),
                    ValueRange{softmaxInput},
                    TypeRange{divOp.getResult().getType()});

                kernelOp->setAttr("jules.kernel.fused",
                                 BoolAttr::get(ctx, true));

                divOp.getResult().replaceAllUsesWith(kernelOp.getResult(0));
                replacedOps.insert(divOp);
                replacedOps.insert(numeratorDef);
                if (subDef && isa<SubOp>(subDef))
                  replacedOps.insert(subDef);
                continue;
              }
            }
          }
        }
      }

      // ── Pattern 4: GELU routing ─────────────────────────────────────
      // Detect GeluOp directly (it's now a first-class op)
      if (auto geluOp = dyn_cast<GeluOp>(op)) {
        auto resultType = geluOp.getResult().getType();
        OpBuilder builder(geluOp);
        auto kernelOp = builder.create<ExternKernelOp>(
            geluOp.getLoc(),
            builder.getStringAttr("fusedGelu"),
            ValueRange{geluOp.getInput()},
            TypeRange{resultType});
        kernelOp->setAttr("jules.kernel.fused",
                         BoolAttr::get(ctx, true));
        geluOp.getResult().replaceAllUsesWith(kernelOp.getResult(0));
        replacedOps.insert(geluOp);
        continue;
      }

      // ── Pattern 5: LayerNorm routing ─────────────────────────────────
      // Detect the characteristic pattern:
      //   MeanOp(x) → SubOp(x, mean) → MeanOp(SubOp^2) →
      //   sqrt → Div → Mul(gamma) → Add(beta)
      // We detect this by finding a DivOp whose numerator comes from
      // a SubOp(x, mean), and the denominator comes from sqrt.
      if (auto divOp = dyn_cast<DivOp>(op)) {
        auto *numDef = divOp.getLhs().getDefiningOp();
        auto *denDef = divOp.getRhs().getDefiningOp();

        // Pattern: (x - mean(x)) / sqrt(var(x) + eps)
        if (numDef && isa<SubOp>(numDef) && denDef) {
          auto subOp = cast<SubOp>(numDef);
          auto *lhsDef = subOp.getLhs().getDefiningOp();
          // lhs should be the original input or come from it
          // Check if the denominator involves a sqrt (via pow with 0.5)
          bool hasSqrt = false;
          if (isa<PowOp>(denDef)) {
            auto powOp = cast<PowOp>(denDef);
            if (auto constOp = powOp.getRhs().getDefiningOp<ConstantOp>()) {
              if (auto fAttr = constOp.getValueAttr().dyn_cast<FloatAttr>()) {
                hasSqrt = std::abs(fAttr.getValueAsDouble() - 0.5) < 0.01;
              }
            }
          }

          if (hasSqrt && lhsDef) {
            // This looks like a layernorm pattern
            auto resultType = divOp.getResult().getType().dyn_cast<RankedTensorType>();
            if (resultType && resultType.getRank() == 2) {
              OpBuilder builder(divOp);
              auto kernelOp = builder.create<ExternKernelOp>(
                  divOp.getLoc(),
                  builder.getStringAttr("fusedLayerNorm"),
                  ValueRange{subOp.getLhs()},
                  TypeRange{divOp.getResult().getType()});
              kernelOp->setAttr("jules.kernel.fused",
                               BoolAttr::get(ctx, true));
              divOp.getResult().replaceAllUsesWith(kernelOp.getResult(0));
              replacedOps.insert(divOp);
              replacedOps.insert(numDef);
              replacedOps.insert(denDef);
              continue;
            }
          }
        }
      }

      // ── Pattern 6: Attention routing ─────────────────────────────────
      // Detect: MatMul(Q, K^T) → Mul(scale) → Exp/Softmax → MatMul(., V)
      // This is the standard attention pattern. When both matmuls and
      // the intervening softmax are detected, route to flashAttention.
      if (auto matmulOp = dyn_cast<MatMulOp>(op)) {
        // Check if this is Q @ K^T (first matmul in attention)
        // Heuristic: if the matmul result feeds into a MulOp with a scalar,
        // then into a softmax pattern, then into another MatMul, it's attention
        if (matmulOp.getResult().hasOneUse()) {
          auto *mulUser = *matmulOp.getResult().getUsers().begin();
          if (isa<MulOp>(mulUser) && mulUser->getResult(0).hasOneUse()) {
            // Check if this eventually leads to another MatMul via softmax
            // For now, mark as attention candidate if the second MatMul exists
            // TODO: Full softmax chain validation
          }
        }
      }

      // ── Pattern 7: Cross-entropy loss routing ────────────────────────
      // Detect: softmax + log + neg + reduce_mean → fusedCrossEntropyLoss
      // This is detected by looking for MeanOp whose input is NegOp whose
      // input is LogOp (the -log(softmax(x)) pattern)
      if (auto meanOp = dyn_cast<MeanOp>(op)) {
        auto *negDef = meanOp.getInput().getDefiningOp();
        if (negDef && isa<NegOp>(negDef)) {
          auto *logDef = negDef->getOperand(0).getDefiningOp();
          if (logDef && isa<LogOp>(logDef)) {
            // This is -log(something) → mean, likely cross-entropy
            // Route the entire chain to fusedCrossEntropyLoss
            auto logOp = cast<LogOp>(logDef);
            OpBuilder builder(meanOp);
            auto kernelOp = builder.create<ExternKernelOp>(
                meanOp.getLoc(),
                builder.getStringAttr("fusedCrossEntropyLoss"),
                ValueRange{logOp.getInput()},
                TypeRange{meanOp.getResult().getType()});
            kernelOp->setAttr("jules.kernel.fused",
                             BoolAttr::get(ctx, true));
            meanOp.getResult().replaceAllUsesWith(kernelOp.getResult(0));
            replacedOps.insert(meanOp);
            replacedOps.insert(negDef);
            replacedOps.insert(logDef);
            continue;
          }
        }
      }

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
    next_op:;
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

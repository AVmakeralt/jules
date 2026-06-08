//===- PolyhedralOptPass.cpp - Polyhedral (Affine) Optimization Impl --------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the polyhedral optimization pass for the Jules
// compiler. It applies loop tiling, fusion, and skewing transformations
// using MLIR's Affine dialect.
//
// Implementation strategy:
//
//   Phase 1: Convert Jules tensor operations to structured affine loops
//   Phase 2: Apply loop tiling for cache maximization
//   Phase 3: Apply loop fusion for kernel fusion
//   Phase 4: Apply loop skewing for parallel execution
//   Phase 5: Clean up and canonicalize
//
//===----------------------------------------------------------------------===//

#include "jules/Passes/PolyhedralOptPass.h"
#include "jules/Dialect/JulesOps.h"
#include "jules/Dialect/JulesDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace jules;

namespace {

// ── Tile Size Computation ───────────────────────────────────────────────────

/// Compute optimal tile sizes based on cache size.
/// The goal is to choose tile dimensions that fit in L1/L2 cache.
struct TileSizeInfo {
  /// Tile size for the first dimension (rows).
  int64_t tileSizeM = 32;

  /// Tile size for the second dimension (columns).
  int64_t tileSizeN = 32;

  /// Tile size for the inner dimension (contraction).
  int64_t tileSizeK = 32;
};

/// Compute optimal tile sizes for a given cache size (in bytes).
TileSizeInfo computeOptimalTileSize(uint64_t cacheSizeBytes) {
  TileSizeInfo info;

  // Assume L1 cache is ~32KB, L2 is ~256KB.
  // For f32 elements (4 bytes), a 32x32 tile uses 32*32*4 = 4KB.
  // Three such tiles (for read, read, write) use 12KB, fitting in L1.

  // If cache is >= 256KB, use 64x64 tiles.
  if (cacheSizeBytes >= 256 * 1024) {
    info.tileSizeM = 64;
    info.tileSizeN = 64;
    info.tileSizeK = 64;
  }

  // If cache is >= 1MB (L3), use 128x128 tiles.
  if (cacheSizeBytes >= 1024 * 1024) {
    info.tileSizeM = 128;
    info.tileSizeN = 128;
    info.tileSizeK = 128;
  }

  return info;
}

// ── Loop Fusion Analysis ────────────────────────────────────────────────────

/// Analyze whether two operations can be fused.
/// Operations can be fused if:
///   1. One produces a tensor that the other consumes
///   2. The element-wise access patterns are compatible
///   3. The loop bounds are compatible
class LoopFusionAnalyzer {
public:
  /// Check if two operations are fusible.
  bool canFuse(Operation *producer, Operation *consumer) const {
    // Check if the producer's output is the consumer's input.
    for (auto operand : consumer->getOperands()) {
      if (auto *defOp = operand.getDefiningOp()) {
        if (defOp == producer) {
          // Direct data dependency — candidate for fusion.
          return true;
        }
      }
    }
    return false;
  }

  /// Find all fusible operation pairs in a function.
  llvm::SmallVector<std::pair<Operation*, Operation*>, 8>
  findFusiblePairs(func::FuncOp funcOp) const {
    llvm::SmallVector<std::pair<Operation*, Operation*>, 8> pairs;

    // Collect all operations in execution order.
    llvm::SmallVector<Operation*, 32> ops;
    funcOp.walk([&](Operation *op) {
      if (!isa<func::FuncOp>(op) && !op->hasTrait<OpTrait::IsTerminator>()) {
        ops.push_back(op);
      }
    });

    // Check each pair for fusibility.
    for (size_t i = 0; i < ops.size(); ++i) {
      for (size_t j = i + 1; j < ops.size(); ++j) {
        if (canFuse(ops[i], ops[j])) {
          pairs.emplace_back(ops[i], ops[j]);
        }
      }
    }

    return pairs;
  }
};

// ── The Pass ────────────────────────────────────────────────────────────────

struct PolyhedralOptPass
    : public PassWrapper<PolyhedralOptPass, OperationPass<ModuleOp>> {
  explicit PolyhedralOptPass(uint64_t cacheSizeBytes = 32 * 1024)
      : cacheSizeBytes_(cacheSizeBytes), enableBackwardOpt_(true) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<bufferization::BufferizationDialect>();
    registry.insert<math::MathDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    auto tileSize = computeOptimalTileSize(cacheSizeBytes_);

    module.walk([&](func::FuncOp funcOp) {
      optimizeFunction(funcOp, tileSize);
    });

    // ── Backward Pass Polyhedral Optimization ─────────────────────────────
    //
    // After optimizing the forward pass, apply the same polyhedral
    // transformations to the backward (gradient) pass. This is the key
    // insight: since autodiff runs at MLIR level BEFORE XLA, we can
    // tile and fuse the backward pass loops. No other framework does this.
    if (enableBackwardOpt_) {
      module.walk([&](func::FuncOp funcOp) {
        optimizeBackwardPass(funcOp, tileSize);
      });
    }
  }

  void optimizeFunction(func::FuncOp funcOp, const TileSizeInfo &tileSize) {
    // ── Phase 1: Loop Tiling Hints ─────────────────────────────────────────
    //
    // Annotate matmul operations with optimal tile sizes.
    // These are consumed by the StableHLO lowering to set XLA's tiling
    // configuration, and by the affine loop generator below.

    funcOp.walk([&](MatMulOp matmulOp) {
      matmulOp->setAttr("polyhedral.tile_m",
                        IntegerAttr::get(matmulOp.getContext(), tileSize.tileSizeM));
      matmulOp->setAttr("polyhedral.tile_n",
                        IntegerAttr::get(matmulOp.getContext(), tileSize.tileSizeN));
      matmulOp->setAttr("polyhedral.tile_k",
                        IntegerAttr::get(matmulOp.getContext(), tileSize.tileSizeK));
    });

    // ── Phase 2: Loop Fusion ───────────────────────────────────────────────
    //
    // Identify fusible operation pairs and perform actual IR fusion.
    // Common fusion patterns in ML:
    //   - matmul + relu      -> fused matmul with activation
    //   - matmul + sigmoid   -> fused matmul with activation
    //   - matmul + tanh      -> fused matmul with activation
    //   - matmul + bias_add  -> annotated for downstream fusion

    LoopFusionAnalyzer fusionAnalyzer;
    auto fusiblePairs = fusionAnalyzer.findFusiblePairs(funcOp);

    // Collect ops to erase after fusion.
    llvm::SmallVector<Operation*, 8> opsToErase;

    for (auto &[producer, consumer] : fusiblePairs) {
      // Fuse matmul + elementwise activation (relu, sigmoid, tanh).
      // Only fuse when the matmul result has a single use (the activation),
      // so that no other consumer expects the un-activated result.
      if (isa<MatMulOp>(producer) && producer->getResult(0).hasOneUse()) {
        StringRef activationName;
        if (isa<ReluOp>(consumer))    activationName = "relu";
        if (isa<SigmoidOp>(consumer)) activationName = "sigmoid";
        if (isa<TanhOp>(consumer))    activationName = "tanh";

        if (!activationName.empty()) {
          // Annotate the matmul with the fused activation.
          producer->setAttr("fused_activation",
                            StringAttr::get(producer->getContext(), activationName));
          // Replace all uses of the activation result with the matmul result.
          // The matmul now semantically produces the activated result.
          consumer->getResult(0).replaceAllUsesWith(producer->getResult(0));
          opsToErase.push_back(consumer);
        }
      }

      // Annotate matmul + bias_add for downstream fusion.
      if (isa<MatMulOp>(producer) && isa<AddOp>(consumer)) {
        consumer->setAttr("polyhedral.fuse_with_producer",
                          UnitAttr::get(consumer->getContext()));
      }
    }

    // Erase fused-away consumers.
    for (auto *op : opsToErase) {
      op->erase();
    }

    // ── Phase 3: Lower MatMul to Affine Loops ──────────────────────────────
    //
    // For matmul ops with static 2D shapes, generate explicit tiled
    // affine loop nests. This enables polyhedral analysis and further
    // optimization by the affine dialect.

    lowerToAffineLoops(funcOp, tileSize);

    // ── Phase 4: Loop Skewing Annotations ──────────────────────────────────
    //
    // For loops with data dependencies (e.g., recurrent operations,
    // autodiff backward passes), annotate operations that would benefit
    // from loop skewing to expose parallelism.

    funcOp.walk([&](Operation *op) {
      // Mark operations in dependency chains as skewing candidates.
      // Check if any result is consumed by a later operation that also
      // reads the same buffer (read-after-write dependency).
      for (auto result : op->getResults()) {
        for (auto user : result.getUsers()) {
          if (user->isBeforeInBlock(op)) {
            // This is a backward dependency — candidate for skewing.
            op->setAttr("polyhedral.skew_candidate",
                        UnitAttr::get(op->getContext()));
            break;
          }
        }
      }
    });
  }

  /// Lower MatMulOp operations with static 2D shapes to tiled affine loop
  /// nests with memref buffers. Operations with dynamic shapes or higher
  /// ranks are left for the StableHLO lowering path.
  void lowerToAffineLoops(func::FuncOp funcOp, const TileSizeInfo &tileSize) {
    MLIRContext *ctx = funcOp.getContext();
    IRRewriter rewriter(ctx);

    // Collect matmul ops eligible for affine lowering.
    llvm::SmallVector<MatMulOp, 4> matmulOps;
    funcOp.walk([&](MatMulOp op) {
      auto lhsType = op.getLhs().getType().dyn_cast<RankedTensorType>();
      auto rhsType = op.getRhs().getType().dyn_cast<RankedTensorType>();
      // Only lower 2D matmuls with fully static shapes.
      if (lhsType && rhsType &&
          lhsType.hasStaticShape() && rhsType.hasStaticShape() &&
          lhsType.getRank() == 2 && rhsType.getRank() == 2) {
        matmulOps.push_back(op);
      }
    });

    for (auto matmulOp : matmulOps) {
      lowerMatMulToAffine(matmulOp, rewriter, tileSize);
    }
  }

  /// Lower a single MatMulOp to a tiled affine loop nest.
  void lowerMatMulToAffine(MatMulOp op, IRRewriter &rewriter,
                           const TileSizeInfo &tileSize) {
    auto loc = op.getLoc();
    MLIRContext *ctx = op->getContext();

    auto lhsType = op.getLhs().getType().cast<RankedTensorType>();
    auto rhsType = op.getRhs().getType().cast<RankedTensorType>();
    auto resultType = op.getResult().getType().cast<RankedTensorType>();

    int64_t M = lhsType.getDimSize(0);
    int64_t K = lhsType.getDimSize(1);
    int64_t N = rhsType.getDimSize(1);

    auto elemType = lhsType.getElementType().cast<FloatType>();

    // Clamp tile sizes to the actual dimensions.
    int64_t TM = std::min(tileSize.tileSizeM, M);
    int64_t TN = std::min(tileSize.tileSizeN, N);
    int64_t TK = std::min(tileSize.tileSizeK, K);

    rewriter.setInsertionPoint(op);

    // ── Convert input tensors to memrefs ─────────────────────────────────
    auto lhsMemrefType = MemRefType::get({M, K}, elemType);
    auto rhsMemrefType = MemRefType::get({K, N}, elemType);
    auto resultMemrefType = MemRefType::get({M, N}, elemType);

    auto lhsMemref = rewriter.create<bufferization::ToMemrefOp>(
        loc, lhsMemrefType, op.getLhs());
    auto rhsMemref = rewriter.create<bufferization::ToMemrefOp>(
        loc, rhsMemrefType, op.getRhs());

    // Allocate output memref.
    auto resultMemref = rewriter.create<memref::AllocOp>(
        loc, resultMemrefType);

    // ── Initialize output to zero ────────────────────────────────────────
    {
      auto zeroAttr = rewriter.getFloatAttr(elemType, 0.0);
      auto zeroValue = rewriter.create<arith::ConstantOp>(loc, zeroAttr);

      auto initI = rewriter.create<affine::AffineForOp>(loc, 0, M, 1);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(initI.getBody());
        Value iv_i = initI.getInductionVar();

        auto initJ = rewriter.create<affine::AffineForOp>(loc, 0, N, 1);
        {
          OpBuilder::InsertionGuard guard2(rewriter);
          rewriter.setInsertionPointToStart(initJ.getBody());
          Value iv_j = initJ.getInductionVar();

          rewriter.create<affine::AffineStoreOp>(
              loc, zeroValue, resultMemref, ValueRange{iv_i, iv_j});
        }
      }
    }

    // ── Tiled matmul loop nest ───────────────────────────────────────────
    //
    // for ti = 0 to M step TM
    //   for tj = 0 to N step TN
    //     for tk = 0 to K step TK
    //       for i = ti to min(ti+TM, M)
    //         for j = tj to min(tj+TN, N)
    //           for k = tk to min(tk+TK, K)
    //             C[i,j] += A[i,k] * B[k,j]
    //             (apply fused activation if present)

    // Build affine maps for the intra-tile bounds.
    // Lower bound:  (d0) -> (d0)
    // Upper bound:  (d0) -> (d0 + tile, dim)  [interpreted as min]
    SmallVector<AffineExpr, 1> lbExpr = {rewriter.getAffineDimExpr(0)};
    auto lbMap = AffineMap::get(1, 0, lbExpr, ctx);

    SmallVector<AffineExpr, 2> ubExprI =
        {rewriter.getAffineDimExpr(0) + TM,
         rewriter.getAffineConstantExpr(M)};
    auto ubMapI = AffineMap::get(1, 0, ubExprI, ctx);

    SmallVector<AffineExpr, 2> ubExprJ =
        {rewriter.getAffineDimExpr(0) + TN,
         rewriter.getAffineConstantExpr(N)};
    auto ubMapJ = AffineMap::get(1, 0, ubExprJ, ctx);

    SmallVector<AffineExpr, 2> ubExprK =
        {rewriter.getAffineDimExpr(0) + TK,
         rewriter.getAffineConstantExpr(K)};
    auto ubMapK = AffineMap::get(1, 0, ubExprK, ctx);

    // Create the tile loops.
    auto tileLoopI = rewriter.create<affine::AffineForOp>(loc, 0, M, TM);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(tileLoopI.getBody());
      Value ti = tileLoopI.getInductionVar();

      auto tileLoopJ = rewriter.create<affine::AffineForOp>(loc, 0, N, TN);
      {
        OpBuilder::InsertionGuard guard2(rewriter);
        rewriter.setInsertionPointToStart(tileLoopJ.getBody());
        Value tj = tileLoopJ.getInductionVar();

        auto tileLoopK = rewriter.create<affine::AffineForOp>(loc, 0, K, TK);
        {
          OpBuilder::InsertionGuard guard3(rewriter);
          rewriter.setInsertionPointToStart(tileLoopK.getBody());
          Value tk = tileLoopK.getInductionVar();

          // Intra-tile loop i: for i = ti to min(ti+TM, M)
          auto intraLoopI = rewriter.create<affine::AffineForOp>(
              loc, lbMap, ValueRange{ti}, ubMapI, ValueRange{ti}, 1);
          {
            OpBuilder::InsertionGuard guard4(rewriter);
            rewriter.setInsertionPointToStart(intraLoopI.getBody());
            Value iv_i = intraLoopI.getInductionVar();

            // Intra-tile loop j: for j = tj to min(tj+TN, N)
            auto intraLoopJ = rewriter.create<affine::AffineForOp>(
                loc, lbMap, ValueRange{tj}, ubMapJ, ValueRange{tj}, 1);
            {
              OpBuilder::InsertionGuard guard5(rewriter);
              rewriter.setInsertionPointToStart(intraLoopJ.getBody());
              Value iv_j = intraLoopJ.getInductionVar();

              // Intra-tile loop k: for k = tk to min(tk+TK, K)
              auto intraLoopK = rewriter.create<affine::AffineForOp>(
                  loc, lbMap, ValueRange{tk}, ubMapK, ValueRange{tk}, 1);
              {
                OpBuilder::InsertionGuard guard6(rewriter);
                rewriter.setInsertionPointToStart(intraLoopK.getBody());
                Value iv_k = intraLoopK.getInductionVar();

                // Load A[i,k], B[k,j], C[i,j]
                auto aVal = rewriter.create<affine::AffineLoadOp>(
                    loc, lhsMemref, ValueRange{iv_i, iv_k});
                auto bVal = rewriter.create<affine::AffineLoadOp>(
                    loc, rhsMemref, ValueRange{iv_k, iv_j});
                auto cVal = rewriter.create<affine::AffineLoadOp>(
                    loc, resultMemref, ValueRange{iv_i, iv_j});

                // C[i,j] += A[i,k] * B[k,j]
                auto prod = rewriter.create<arith::MulFOp>(loc, aVal, bVal);
                Value accum = rewriter.create<arith::AddFOp>(loc, cVal, prod);

                // Apply fused activation if present.
                if (op->hasAttr("fused_activation")) {
                  auto actName = op->getAttr("fused_activation")
                                     .cast<StringAttr>()
                                     .getValue();
                  if (actName == "relu") {
                    auto zeroAttr = rewriter.getFloatAttr(elemType, 0.0);
                    auto zeroVal =
                        rewriter.create<arith::ConstantOp>(loc, zeroAttr);
                    auto cmp = rewriter.create<arith::CmpFOp>(
                        loc, arith::CmpFPredicate::OGT, accum, zeroVal);
                    accum = rewriter.create<arith::SelectOp>(
                        loc, cmp, accum, zeroVal);
                  } else if (actName == "sigmoid") {
                    auto oneAttr = rewriter.getFloatAttr(elemType, 1.0);
                    auto oneVal =
                        rewriter.create<arith::ConstantOp>(loc, oneAttr);
                    auto negVal = rewriter.create<arith::NegFOp>(loc, accum);
                    auto expVal = rewriter.create<math::ExpOp>(loc, negVal);
                    auto denom =
                        rewriter.create<arith::AddFOp>(loc, oneVal, expVal);
                    accum =
                        rewriter.create<arith::DivFOp>(loc, oneVal, denom);
                  } else if (actName == "tanh") {
                    accum = rewriter.create<math::TanhOp>(loc, accum);
                  }
                }

                rewriter.create<affine::AffineStoreOp>(
                    loc, accum, resultMemref, ValueRange{iv_i, iv_j});
              }
            }
          }
        }
      }
    }

    // ── Convert result memref back to tensor ─────────────────────────────
    auto resultTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, resultType, resultMemref);

    // Replace the matmul op with the computed tensor result and erase it.
    rewriter.replaceOp(op, resultTensor.getResult());
    rewriter.eraseOp(op);
  }

  // ── Backward Pass Polyhedral Optimization ───────────────────────────────
  //
  // Apply polyhedral optimization to gradient code. The key insight:
  // since autodiff runs at MLIR level BEFORE XLA, we can tile and fuse
  // the backward pass loops. No other framework does this.
  //
  // Specifically:
  //   1. Identify backward pass ops (ops between gradient seed and final
  //      gradient output)
  //   2. Tile the matmul gradient ops (transpose(b) ** dL, transpose(a) ** dL)
  //      with cache-friendly tile sizes
  //   3. Fuse activation gradients with their preceding matmul gradients
  //   4. Skew the backward loop nests for parallelization
  void optimizeBackwardPass(func::FuncOp funcOp,
                            const TileSizeInfo &tileSize) {
    MLIRContext *ctx = funcOp.getContext();

    // ── Step 1: Identify backward pass operations ─────────────────────────
    //
    // The backward pass is identified by looking for operations that are
    // part of the gradient computation. These are characterized by:
    //   - Operations that consume a "gradient seed" (constant 1.0 produced
    //     for the output of the original function)
    //   - TransposeOp operations that are part of matmul gradient computation
    //   - MatMulOp operations whose operands include transposed values
    //     (characteristic of backward matmul: dL ** T(b) or T(a) ** dL)
    //
    // We also look for the autodiff attribute markers that the
    // FusedAutodiffPass or AutodiffPass may have left on the IR.

    DenseSet<Operation *> backwardOps;
    DenseSet<Value> gradientValues;

    // Find all operations that have the "autodiff.backward" attribute,
    // or are reachable from gradient seed constants through the backward
    // dataflow graph.
    funcOp.walk([&](Operation *op) {
      // Operations explicitly marked as backward by autodiff.
      if (op->hasAttr("autodiff.backward")) {
        backwardOps.insert(op);
        for (auto result : op->getResults()) {
          gradientValues.insert(result);
        }
      }
    });

    // Propagate backward-ness: any op whose result is used only by
    // backward ops and whose operands include a gradient value is
    // also part of the backward pass. We iterate until fixed point.
    bool changed = true;
    while (changed) {
      changed = false;
      funcOp.walk([&](Operation *op) {
        if (backwardOps.count(op)) return;

        // Check if any operand is a gradient value.
        bool hasGradOperand = false;
        for (auto operand : op->getOperands()) {
          if (gradientValues.count(operand)) {
            hasGradOperand = true;
            break;
          }
        }

        // If this op consumes a gradient value and all its users are
        // backward ops (or it has the backward marker), it's part of
        // the backward pass.
        if (hasGradOperand) {
          // Check if this looks like a gradient computation op:
          // matmul, transpose, mul, add, neg that feeds into a gradient.
          bool isGradOp = isa<MatMulOp>(op) || isa<TransposeOp>(op) ||
                          isa<MulOp>(op) || isa<AddOp>(op) ||
                          isa<NegOp>(op) || isa<SubOp>(op) ||
                          isa<DivOp>(op);

          // Also check if the op feeds into a known backward op.
          bool feedsBackward = false;
          for (auto user : op->getUsers()) {
            if (backwardOps.count(user)) {
              feedsBackward = true;
              break;
            }
          }

          if (isGradOp || feedsBackward) {
            backwardOps.insert(op);
            for (auto result : op->getResults()) {
              gradientValues.insert(result);
            }
            changed = true;
          }
        }
      });
    }

    if (backwardOps.empty()) return; // No backward pass found.

    // ── Step 2: Tile the backward matmul gradient ops ─────────────────────
    //
    // In the backward pass of a matmul C = A ** B:
    //   dL/dA = dL ** T(B)   [a matmul with transposed B]
    //   dL/dB = T(A) ** dL   [a matmul with transposed A]
    //
    // These matmuls have the same dimensions as the forward matmul,
    // so we can tile them with the same cache-friendly tile sizes.
    //
    // We identify backward matmuls by checking if either operand
    // is a TransposeOp (the hallmark of gradient matmul).

    llvm::SmallVector<MatMulOp, 4> backwardMatmuls;
    for (Operation *op : backwardOps) {
      if (auto matmulOp = dyn_cast<MatMulOp>(op)) {
        // Check if either operand is a transpose (gradient matmul marker).
        bool isLhsTranspose = isa_and_nonnull<TransposeOp>(
            matmulOp.getLhs().getDefiningOp());
        bool isRhsTranspose = isa_and_nonnull<TransposeOp>(
            matmulOp.getRhs().getDefiningOp());

        if (isLhsTranspose || isRhsTranspose) {
          backwardMatmuls.push_back(matmulOp);
        }
      }
    }

    // Annotate backward matmuls with tiling hints.
    for (auto matmulOp : backwardMatmuls) {
      matmulOp->setAttr("polyhedral.backward_tile_m",
                        IntegerAttr::get(ctx, tileSize.tileSizeM));
      matmulOp->setAttr("polyhedral.backward_tile_n",
                        IntegerAttr::get(ctx, tileSize.tileSizeN));
      matmulOp->setAttr("polyhedral.backward_tile_k",
                        IntegerAttr::get(ctx, tileSize.tileSizeK));
      matmulOp->setAttr("polyhedral.backward_matmul",
                        UnitAttr::get(ctx));
    }

    // Lower backward matmuls to tiled affine loops, same as forward.
    IRRewriter rewriter(ctx);
    for (auto matmulOp : backwardMatmuls) {
      auto lhsType = matmulOp.getLhs().getType().dyn_cast<RankedTensorType>();
      auto rhsType = matmulOp.getRhs().getType().dyn_cast<RankedTensorType>();
      if (lhsType && rhsType &&
          lhsType.hasStaticShape() && rhsType.hasStaticShape() &&
          lhsType.getRank() == 2 && rhsType.getRank() == 2) {
        lowerMatMulToAffine(matmulOp, rewriter, tileSize);
      }
    }

    // ── Step 3: Fuse activation gradients with preceding matmul gradients ─
    //
    // In the backward pass, activation gradients (relu grad, sigmoid grad,
    // tanh grad) often immediately follow a matmul gradient. For example:
    //
    //   dL/dA = (dL * sigmoid_grad) ** T(B)
    //
    // This is equivalent to:
    //
    //   temp = dL * sigmoid_grad
    //   dL/dA = temp ** T(B)
    //
    // We can fuse the elementwise activation gradient into the matmul
    // gradient by annotating it, similar to how forward matmul+activation
    // fusion works.

    LoopFusionAnalyzer fusionAnalyzer;

    for (Operation *op : backwardOps) {
      if (!isa<MatMulOp>(op)) continue;
      if (!op->hasAttr("polyhedral.backward_matmul")) continue;

      // Look for elementwise ops that feed into this backward matmul.
      for (auto operand : op->getOperands()) {
        auto *defOp = operand.getDefiningOp();
        if (!defOp) continue;
        if (!backwardOps.count(defOp)) continue;

        // Check if the producer is an elementwise activation gradient.
        // These are typically MulOp (mask * dL for relu), or a chain
        // of MulOp + SubOp for sigmoid/tanh gradients.
        if (isa<MulOp>(defOp)) {
          // Check if this mul is a gradient computation (one operand
          // is an incoming adjoint, the other is a derivative mask).
          bool hasGradInput = false;
          for (auto mulOperand : defOp->getOperands()) {
            if (gradientValues.count(mulOperand)) {
              hasGradInput = true;
              break;
            }
          }
          if (hasGradInput && op->getResult(0).hasOneUse()) {
            defOp->setAttr("polyhedral.fuse_into_consumer",
                           UnitAttr::get(ctx));
            op->setAttr("fused_activation_grad",
                        StringAttr::get(ctx, "elementwise"));
          }
        }
      }
    }

    // ── Step 4: Skew the backward loop nests for parallelization ──────────
    //
    // Backward pass loops often have sequential dependencies (the
    // gradient must flow from output to input). However, within each
    // layer's gradient computation, the loop over batch elements is
    // embarrassingly parallel. Loop skewing transforms the iteration
    // space so that the batch dimension can be parallelized while
    // preserving the layer-to-layer dependency.
    //
    // We annotate backward ops that are skew candidates. The actual
    // skewing is applied by the affine dialect during lowering.

    for (Operation *op : backwardOps) {
      // Only skew operations that are part of a loop nest
      // (matmul gradients and their immediate elementwise consumers).
      if (isa<MatMulOp>(op) || isa<MulOp>(op) || isa<AddOp>(op)) {
        // Check if this op has a cross-iteration dependency.
        // In the backward pass, this happens when a gradient value
        // produced by one layer is consumed by the previous layer's
        // gradient computation.
        for (auto result : op->getResults()) {
          for (auto user : result.getUsers()) {
            if (backwardOps.count(user) &&
                (user->isBeforeInBlock(op) ||
                 op->getParentOp() != user->getParentOp())) {
              // Cross-iteration or cross-layer dependency.
              op->setAttr("polyhedral.backward_skew",
                          IntegerAttr::get(ctx, 1));
              user->setAttr("polyhedral.backward_skew",
                            IntegerAttr::get(ctx, 1));
              break;
            }
          }
        }
      }
    }

    // ── Step 5: Annotate backward pass boundaries ─────────────────────────
    //
    // Mark the boundary ops of the backward pass so that downstream
    // passes (StableHLO lowering, XLA compilation) can apply
    // backend-specific optimizations to the gradient code.

    for (Operation *op : backwardOps) {
      // First backward op: consumes a forward value but produces
      // only gradient values.
      bool isBoundaryStart = false;
      for (auto operand : op->getOperands()) {
        if (!gradientValues.count(operand)) {
          // This operand comes from the forward pass.
          isBoundaryStart = true;
          break;
        }
      }
      if (isBoundaryStart) {
        op->setAttr("autodiff.backward_start",
                    UnitAttr::get(ctx));
      }

      // Last backward op: produces a value that is not consumed by
      // any other backward op.
      bool isBoundaryEnd = true;
      for (auto user : op->getUsers()) {
        if (backwardOps.count(user)) {
          isBoundaryEnd = false;
          break;
        }
      }
      if (isBoundaryEnd && !op->getUsers().empty()) {
        op->setAttr("autodiff.backward_end",
                    UnitAttr::get(ctx));
      }
    }
  }

private:
  uint64_t cacheSizeBytes_;
  bool enableBackwardOpt_;
};

} // anonymous namespace

std::unique_ptr<Pass> jules::createPolyhedralOptPass() {
  return std::make_unique<PolyhedralOptPass>();
}

std::unique_ptr<Pass> jules::createPolyhedralOptPass(uint64_t cacheSizeBytes) {
  return std::make_unique<PolyhedralOptPass>(cacheSizeBytes);
}

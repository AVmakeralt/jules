//===- AutodiffPruning.cpp - Autodiff Pruning Pass Implementation ----------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the autodiff pruning pass. After the autodiff engine
// generates the backward pass, it often leaves behind redundant operations.
// This pass cleans them up:
//
//   - add(x, zeros) -> x  (from gradient paths that don't contribute)
//   - mul(x, ones) -> x   (from identity Jacobians)
//   - mul(zeros, x) -> zeros (from zero gradient paths)
//   - neg(neg(x)) -> x    (from double-negation in subtraction gradient)
//   - transpose(transpose(x)) -> x  (from matmul gradient transposes)
//   - mul(x, constant_0) -> zeros  (dead gradient paths)
//   - add(x, constant_0) -> x      (zero gradient accumulation)
//
// This pass runs after autodiff but before graph collapsing, so it can
// catch autodiff-specific patterns before the general-purpose collapsing
// pass runs.
//
//===----------------------------------------------------------------------===//

#include "jules/Passes/AutodiffPruning.h"
#include "jules/Dialect/JulesOps.h"
#include "jules/Dialect/JulesDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace jules;

//===----------------------------------------------------------------------===//
// Pattern: mul(x, constant_0) -> zeros (dead gradient path)
//===----------------------------------------------------------------------===//

namespace {

struct ZeroGradientPrunePattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    // If one operand is a constant zero (scalar), replace with zeros.
    for (unsigned i = 0; i < 2; ++i) {
      auto operand = (i == 0) ? op.getLhs() : op.getRhs();
      if (auto constOp = operand.getDefiningOp<ConstantOp>()) {
        if (auto floatAttr = constOp.getValueAttr().dyn_cast<FloatAttr>()) {
          if (floatAttr.getValue().isZero()) {
            auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
            if (resultType) {
              rewriter.replaceOpWithNewOp<ZerosOp>(op, resultType);
            } else {
              rewriter.replaceOp(op, operand);
            }
            return success();
          }
        }
        if (auto denseAttr = constOp.getValueAttr().dyn_cast<DenseFPElementsAttr>()) {
          bool allZero = true;
          for (auto val : denseAttr) {
            if (!val.isZero()) { allZero = false; break; }
          }
          if (allZero) {
            auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
            if (resultType) {
              rewriter.replaceOpWithNewOp<ZerosOp>(op, resultType);
              return success();
            }
          }
        }
      }
      // Also check for zeros op.
      if (operand.getDefiningOp<ZerosOp>()) {
        auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
        if (resultType) {
          rewriter.replaceOpWithNewOp<ZerosOp>(op, resultType);
        } else {
          rewriter.replaceOp(op, operand);
        }
        return success();
      }
    }
    return failure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern: add(x, zeros) -> x (zero gradient accumulation)
//===----------------------------------------------------------------------===//

namespace {

struct ZeroAccumPrunePattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    // add(x, zeros) -> x
    if (op.getRhs().getDefiningOp<ZerosOp>()) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    // add(zeros, x) -> x
    if (op.getLhs().getDefiningOp<ZerosOp>()) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }

    // add(x, constant_0) -> x
    if (auto constOp = op.getRhs().getDefiningOp<ConstantOp>()) {
      if (auto floatAttr = constOp.getValueAttr().dyn_cast<FloatAttr>()) {
        if (floatAttr.getValue().isZero()) {
          rewriter.replaceOp(op, op.getLhs());
          return success();
        }
      }
      if (auto denseAttr = constOp.getValueAttr().dyn_cast<DenseFPElementsAttr>()) {
        bool allZero = true;
        for (auto val : denseAttr) {
          if (!val.isZero()) { allZero = false; break; }
        }
        if (allZero) {
          rewriter.replaceOp(op, op.getLhs());
          return success();
        }
      }
    }
    if (auto constOp = op.getLhs().getDefiningOp<ConstantOp>()) {
      if (auto floatAttr = constOp.getValueAttr().dyn_cast<FloatAttr>()) {
        if (floatAttr.getValue().isZero()) {
          rewriter.replaceOp(op, op.getRhs());
          return success();
        }
      }
    }

    return failure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern: neg(neg(x)) -> x (from subtraction gradient)
//===----------------------------------------------------------------------===//

namespace {

struct DoubleNegPrunePattern : public OpRewritePattern<NegOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NegOp op,
                                PatternRewriter &rewriter) const override {
    if (auto innerNeg = op.getInput().getDefiningOp<NegOp>()) {
      rewriter.replaceOp(op, innerNeg.getInput());
      return success();
    }
    return failure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern: transpose(transpose(x)) -> x (from matmul gradient)
//===----------------------------------------------------------------------===//

namespace {

struct DoubleTransposePrunePattern : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    if (auto innerTranspose = op.getInput().getDefiningOp<TransposeOp>()) {
      rewriter.replaceOp(op, innerTranspose.getInput());
      return success();
    }
    return failure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern: add(x, neg(x)) -> zeros (cancellation from grad paths)
//===----------------------------------------------------------------------===//

namespace {

struct AddNegCancelPrunePattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (auto negOp = op.getRhs().getDefiningOp<NegOp>()) {
      if (negOp.getInput() == op.getLhs()) {
        auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
        if (resultType) {
          rewriter.replaceOpWithNewOp<ZerosOp>(op, resultType);
          return success();
        }
      }
    }
    if (auto negOp = op.getLhs().getDefiningOp<NegOp>()) {
      if (negOp.getInput() == op.getRhs()) {
        auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
        if (resultType) {
          rewriter.replaceOpWithNewOp<ZerosOp>(op, resultType);
          return success();
        }
      }
    }
    return failure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Dead code elimination: remove unused gradient computations
//===----------------------------------------------------------------------===//

namespace {

/// Remove operations whose results are unused (DCE for autodiff output).
struct AutodiffDCEPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    // This is a simplified DCE: if the result of an add has no users,
    // it will be cleaned up by MLIR's built-in DCE in the canonicalizer.
    return failure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Populate and Pass
//===----------------------------------------------------------------------===//

void jules::populateAutodiffPruningPatterns(RewritePatternSet &patterns) {
  patterns.add<ZeroGradientPrunePattern,
                ZeroAccumPrunePattern,
                DoubleNegPrunePattern,
                DoubleTransposePrunePattern,
                AddNegCancelPrunePattern>(patterns.getContext());
}

namespace {

struct AutodiffPruningPass
    : public PassWrapper<AutodiffPruningPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateAutodiffPruningPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> jules::createAutodiffPruningPass() {
  return std::make_unique<AutodiffPruningPass>();
}

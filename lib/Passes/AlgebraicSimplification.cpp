//===- AlgebraicSimplification.cpp - Algebraic Simplification Pass ----------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the algebraic simplification pass. It handles
// mathematical identities that survive graph collapsing:
//
//   - x * 0 -> 0,  0 * x -> 0           (annihilation)
//   - x - 0 -> x                          (subtraction identity)
//   - x / 1 -> x                          (division identity)
//   - x ^ 0 -> 1,  x ^ 1 -> x           (power identity)
//   - neg(neg(x)) -> x                    (double negation)
//   - add(x, neg(y)) -> sub(x, y)        (negate-to-subtract)
//   - relu(constant_0) -> 0               (dead activation)
//   - sigmoid(constant) -> constant        (constant folding)
//
//===----------------------------------------------------------------------===//

#include "jules/Passes/AlgebraicSimplification.h"
#include "jules/Dialect/JulesOps.h"
#include "jules/Dialect/JulesDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace jules;

//===----------------------------------------------------------------------===//
// Pattern: x * 0 -> 0,  0 * x -> 0  (annihilation)
//===----------------------------------------------------------------------===//

namespace {

struct MulZeroPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    // x * 0 -> zeros
    if (auto constOp = op.getRhs().getDefiningOp<ConstantOp>()) {
      if (auto floatAttr = constOp.getValueAttr().dyn_cast<FloatAttr>()) {
        if (floatAttr.getValue().isZero()) {
          auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
          if (resultType) {
            rewriter.replaceOpWithNewOp<ZerosOp>(op, resultType);
          } else {
            rewriter.replaceOp(op, op.getRhs());
          }
          return success();
        }
      }
    }
    // 0 * x -> zeros
    if (auto constOp = op.getLhs().getDefiningOp<ConstantOp>()) {
      if (auto floatAttr = constOp.getValueAttr().dyn_cast<FloatAttr>()) {
        if (floatAttr.getValue().isZero()) {
          auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
          if (resultType) {
            rewriter.replaceOpWithNewOp<ZerosOp>(op, resultType);
          } else {
            rewriter.replaceOp(op, op.getLhs());
          }
          return success();
        }
      }
    }
    // x * zeros -> zeros
    if (op.getRhs().getDefiningOp<ZerosOp>()) {
      auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
      if (resultType) {
        rewriter.replaceOpWithNewOp<ZerosOp>(op, resultType);
      } else {
        rewriter.replaceOp(op, op.getRhs());
      }
      return success();
    }
    // zeros * x -> zeros
    if (op.getLhs().getDefiningOp<ZerosOp>()) {
      auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
      if (resultType) {
        rewriter.replaceOpWithNewOp<ZerosOp>(op, resultType);
      } else {
        rewriter.replaceOp(op, op.getLhs());
      }
      return success();
    }

    return failure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern: x - 0 -> x
//===----------------------------------------------------------------------===//

namespace {

struct SubZeroPattern : public OpRewritePattern<SubOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter &rewriter) const override {
    if (auto constOp = op.getRhs().getDefiningOp<ConstantOp>()) {
      if (auto floatAttr = constOp.getValueAttr().dyn_cast<FloatAttr>()) {
        if (floatAttr.getValue().isZero()) {
          rewriter.replaceOp(op, op.getLhs());
          return success();
        }
      }
    }
    if (op.getRhs().getDefiningOp<ZerosOp>()) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    return failure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern: x / 1 -> x
//===----------------------------------------------------------------------===//

namespace {

struct DivOnePattern : public OpRewritePattern<DivOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DivOp op,
                                PatternRewriter &rewriter) const override {
    if (auto constOp = op.getRhs().getDefiningOp<ConstantOp>()) {
      if (auto floatAttr = constOp.getValueAttr().dyn_cast<FloatAttr>()) {
        if (floatAttr.getValue().isExactlyValue(1.0)) {
          rewriter.replaceOp(op, op.getLhs());
          return success();
        }
      }
    }
    if (op.getRhs().getDefiningOp<OnesOp>()) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    return failure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern: x ^ 0 -> 1,  x ^ 1 -> x
//===----------------------------------------------------------------------===//

namespace {

struct PowerIdentityPattern : public OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter &rewriter) const override {
    if (auto constOp = op.getRhs().getDefiningOp<ConstantOp>()) {
      if (auto floatAttr = constOp.getValueAttr().dyn_cast<FloatAttr>()) {
        double val = floatAttr.getValue().convertToDouble();
        if (val == 0.0) {
          // x ^ 0 -> 1
          auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
          if (resultType) {
            rewriter.replaceOpWithNewOp<OnesOp>(op, resultType);
          } else {
            auto one = rewriter.create<ConstantOp>(
                op.getLoc(), rewriter.getFloatAttr(
                    op.getResult().getType(), 1.0));
            rewriter.replaceOp(op, one.getResult());
          }
          return success();
        }
        if (val == 1.0) {
          // x ^ 1 -> x
          rewriter.replaceOp(op, op.getLhs());
          return success();
        }
      }
    }
    return failure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern: neg(neg(x)) -> x  (double negation)
//===----------------------------------------------------------------------===//

namespace {

struct DoubleNegPattern : public OpRewritePattern<NegOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NegOp op,
                                PatternRewriter &rewriter) const override {
    if (auto innerNeg = op.getInput().getDefiningOp<NegOp>()) {
      // neg(neg(x)) -> x
      rewriter.replaceOp(op, innerNeg.getInput());
      return success();
    }
    return failure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern: add(x, neg(y)) -> sub(x, y)
//===----------------------------------------------------------------------===//

namespace {

struct AddNegToSubPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (auto negOp = op.getRhs().getDefiningOp<NegOp>()) {
      // x + (-y) -> x - y
      auto sub = rewriter.create<SubOp>(op.getLoc(), op.getLhs(),
                                          negOp.getInput());
      rewriter.replaceOp(op, sub.getResult());
      return success();
    }
    return failure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern: relu(zeros) -> zeros
//===----------------------------------------------------------------------===//

namespace {

struct ReluZerosPattern : public OpRewritePattern<ReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInput().getDefiningOp<ZerosOp>()) {
      // relu(0) = 0
      rewriter.replaceOp(op, op.getInput());
      return success();
    }
    // relu(constant_negative) -> zeros
    if (auto constOp = op.getInput().getDefiningOp<ConstantOp>()) {
      if (auto floatAttr = constOp.getValueAttr().dyn_cast<FloatAttr>()) {
        if (floatAttr.getValue().isNegative()) {
          auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
          if (resultType) {
            rewriter.replaceOpWithNewOp<ZerosOp>(op, resultType);
            return success();
          }
        }
      }
    }
    return failure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern: mean(zeros) -> 0
//===----------------------------------------------------------------------===//

namespace {

struct MeanZerosPattern : public OpRewritePattern<MeanOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MeanOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInput().getDefiningOp<ZerosOp>()) {
      // mean(0) = 0
      auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
      if (resultType) {
        rewriter.replaceOpWithNewOp<ZerosOp>(op, resultType);
        return success();
      }
    }
    return failure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern: sum(zeros) -> 0
//===----------------------------------------------------------------------===//

namespace {

struct SumZerosPattern : public OpRewritePattern<SumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SumOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInput().getDefiningOp<ZerosOp>()) {
      auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
      if (resultType) {
        rewriter.replaceOpWithNewOp<ZerosOp>(op, resultType);
        return success();
      }
    }
    return failure();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Populate and Pass
//===----------------------------------------------------------------------===//

void jules::populateAlgebraicSimplificationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MulZeroPattern,
                SubZeroPattern,
                DivOnePattern,
                PowerIdentityPattern,
                DoubleNegPattern,
                AddNegToSubPattern,
                ReluZerosPattern,
                MeanZerosPattern,
                SumZerosPattern>(patterns.getContext());
}

namespace {

struct AlgebraicSimplificationPass
    : public PassWrapper<AlgebraicSimplificationPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateAlgebraicSimplificationPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> jules::createAlgebraicSimplificationPass() {
  return std::make_unique<AlgebraicSimplificationPass>();
}

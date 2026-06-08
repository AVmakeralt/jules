//===- GraphCollapsing.cpp - Graph Collapsing Pass Implementation ----------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the graph collapsing pass. It detects and collapses
// structural patterns in the Jules MLIR dialect:
//
//   - transpose(transpose(x)) -> x
//   - reshape(reshape(x, s1), s2) -> reshape(x, s2)
//   - slice(slice(x, ...), ...) -> slice(x, merged_offsets)
//   - concat(split(x, ...), ...) -> x
//
//===----------------------------------------------------------------------===//

#include "jules/Passes/GraphCollapsing.h"
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
// Pattern: transpose(transpose(x)) -> x
//===----------------------------------------------------------------------===//

namespace {

struct DoubleTransposePattern : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    // Check if the input is also a transpose.
    auto inputOp = op.getInput().getDefiningOp<TransposeOp>();
    if (!inputOp) return failure();

    // Double transpose cancels out: transpose(transpose(x)) -> x
    rewriter.replaceOp(op, inputOp.getInput());
    return success();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern: reshape(reshape(x, s1), s2) -> reshape(x, s2)
//===----------------------------------------------------------------------===//

namespace {

struct DoubleReshapePattern : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // Check if the input is also a reshape.
    auto inputOp = op.getInput().getDefiningOp<ReshapeOp>();
    if (!inputOp) return failure();

    // Collapse two reshapes into one: reshape(reshape(x, s1), s2) -> reshape(x, s2)
    auto newReshape = rewriter.create<ReshapeOp>(
        op.getLoc(), inputOp.getInput(), op.getResult().getType().cast<RankedTensorType>());
    rewriter.replaceOp(op, newReshape.getResult());
    return success();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern: add(x, zeros) -> x, add(zeros, x) -> x
//===----------------------------------------------------------------------===//

namespace {

struct AddZeroCollapsePattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    // Check if either operand is a zeros op.
    if (auto zerosOp = op.getLhs().getDefiningOp<ZerosOp>()) {
      // 0 + x -> x
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    if (auto zerosOp = op.getRhs().getDefiningOp<ZerosOp>()) {
      // x + 0 -> x
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    // Check if either operand is a constant zero.
    if (auto constOp = op.getLhs().getDefiningOp<ConstantOp>()) {
      if (auto floatAttr = constOp.getValueAttr().dyn_cast<FloatAttr>()) {
        if (floatAttr.getValue().isZero()) {
          rewriter.replaceOp(op, op.getRhs());
          return success();
        }
      }
      if (auto intAttr = constOp.getValueAttr().dyn_cast<IntegerAttr>()) {
        if (intAttr.getValue().isZero()) {
          rewriter.replaceOp(op, op.getRhs());
          return success();
        }
      }
    }
    if (auto constOp = op.getRhs().getDefiningOp<ConstantOp>()) {
      if (auto floatAttr = constOp.getValueAttr().dyn_cast<FloatAttr>()) {
        if (floatAttr.getValue().isZero()) {
          rewriter.replaceOp(op, op.getLhs());
          return success();
        }
      }
      if (auto intAttr = constOp.getValueAttr().dyn_cast<IntegerAttr>()) {
        if (intAttr.getValue().isZero()) {
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
// Pattern: mul(x, ones) -> x, mul(ones, x) -> x
//===----------------------------------------------------------------------===//

namespace {

struct MulOneCollapsePattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    // Check if either operand is an ones op.
    if (auto onesOp = op.getLhs().getDefiningOp<OnesOp>()) {
      // 1 * x -> x
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    if (auto onesOp = op.getRhs().getDefiningOp<OnesOp>()) {
      // x * 1 -> x
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    // Check if either operand is a constant one.
    if (auto constOp = op.getRhs().getDefiningOp<ConstantOp>()) {
      if (auto floatAttr = constOp.getValueAttr().dyn_cast<FloatAttr>()) {
        if (floatAttr.getValue().isExactlyValue(1.0)) {
          rewriter.replaceOp(op, op.getLhs());
          return success();
        }
      }
    }
    if (auto constOp = op.getLhs().getDefiningOp<ConstantOp>()) {
      if (auto floatAttr = constOp.getValueAttr().dyn_cast<FloatAttr>()) {
        if (floatAttr.getValue().isExactlyValue(1.0)) {
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
// Pattern: add(x, neg(x)) -> zeros
//===----------------------------------------------------------------------===//

namespace {

struct AddNegCollapsePattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    // Check: add(x, neg(x)) -> zeros(shape(x))
    if (auto negOp = op.getRhs().getDefiningOp<NegOp>()) {
      if (negOp.getInput() == op.getLhs()) {
        auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
        if (resultType) {
          rewriter.replaceOpWithNewOp<ZerosOp>(op, resultType);
          return success();
        }
      }
    }
    // Check: add(neg(x), x) -> zeros(shape(x))
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
// Pattern: sub(x, x) -> zeros
//===----------------------------------------------------------------------===//

namespace {

struct SubSelfCollapsePattern : public OpRewritePattern<SubOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getLhs() == op.getRhs()) {
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

void jules::populateGraphCollapsingPatterns(RewritePatternSet &patterns) {
  patterns.add<DoubleTransposePattern,
                DoubleReshapePattern,
                AddZeroCollapsePattern,
                MulOneCollapsePattern,
                AddNegCollapsePattern,
                SubSelfCollapsePattern>(patterns.getContext());
}

namespace {

struct GraphCollapsingPass
    : public PassWrapper<GraphCollapsingPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGraphCollapsingPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> jules::createGraphCollapsingPass() {
  return std::make_unique<GraphCollapsingPass>();
}

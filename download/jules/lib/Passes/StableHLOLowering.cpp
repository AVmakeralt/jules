//===- StableHLOLowering.cpp - Jules to StableHLO Lowering -----------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the lowering pass from the Jules MLIR dialect to the
// StableHLO dialect. Each Jules operation is converted to its StableHLO
// equivalent using MLIR's dialect conversion framework.
//
//===----------------------------------------------------------------------===//

#include "jules/Passes/Passes.h"
#include "jules/Passes/StableHLOLowering.h"
#include "jules/Dialect/JulesOps.h"
#include "jules/Dialect/JulesDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace jules;

//===----------------------------------------------------------------------===//
// Type converter: Jules types -> StableHLO types
//===----------------------------------------------------------------------===//

TypeConverter jules::getJulesToStableHLOTypeConverter() {
  TypeConverter converter;

  // Identity conversion for most types.
  converter.addConversion([](Type type) { return type; });

  // Jules uses the same RankedTensorType as StableHLO, so no conversion
  // needed for tensor types. The element types (f32, f64, i32, i64) are
  // also standard MLIR types.

  // If we had custom Jules types, we'd add converters here.

  return converter;
}

//===----------------------------------------------------------------------===//
// Conversion patterns for each Jules operation
//===----------------------------------------------------------------------===//

namespace {

// ── AddOp ──────────────────────────────────────────────────────────────────

struct AddOpLowering : public OpConversionPattern<AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(AddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // jules.add -> stablehlo.add
    rewriter.replaceOpWithNewOp<stablehlo::AddOp>(
        op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

// ── SubOp ──────────────────────────────────────────────────────────────────

struct SubOpLowering : public OpConversionPattern<SubOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::SubtractOp>(
        op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

// ── MulOp ──────────────────────────────────────────────────────────────────

struct MulOpLowering : public OpConversionPattern<MulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(MulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::MulOp>(
        op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

// ── DivOp ──────────────────────────────────────────────────────────────────

struct DivOpLowering : public OpConversionPattern<DivOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(DivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::DivOp>(
        op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

// ── PowOp ──────────────────────────────────────────────────────────────────

struct PowOpLowering : public OpConversionPattern<PowOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(PowOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::PowOp>(
        op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

// ── NegOp ──────────────────────────────────────────────────────────────────

struct NegOpLowering : public OpConversionPattern<NegOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(NegOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::NegOp>(
        op, adaptor.getInput());
    return success();
  }
};

// ── MatMulOp ───────────────────────────────────────────────────────────────

struct MatMulOpLowering : public OpConversionPattern<MatMulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(MatMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // jules.matmul -> stablehlo.dot_general
    //
    // For 2D matmul [M, K] ** [K, N]:
    //   dot_general with lhs_batching_dims={}, rhs_batching_dims={},
    //   lhs_contracting_dims={1}, rhs_contracting_dims={0}
    //
    // For batched matmul [B, M, K] ** [B, K, N]:
    //   dot_general with lhs_batching_dims={0}, rhs_batching_dims={0},
    //   lhs_contracting_dims={2}, rhs_contracting_dims={1}

    auto lhsType = adaptor.getLhs().getType().dyn_cast<RankedTensorType>();
    auto rhsType = adaptor.getRhs().getType().dyn_cast<RankedTensorType>();

    if (!lhsType || !rhsType) {
      return rewriter.notifyMatchFailure(op, "requires ranked tensor operands");
    }

    // Determine batching and contracting dimensions.
    SmallVector<int64_t, 2> lhsBatchDims;
    SmallVector<int64_t, 2> rhsBatchDims;
    SmallVector<int64_t, 1> lhsContractDims;
    SmallVector<int64_t, 1> rhsContractDims;

    if (lhsType.getRank() == 2 && rhsType.getRank() == 2) {
      // [M, K] ** [K, N]
      lhsContractDims.push_back(1);
      rhsContractDims.push_back(0);
    } else if (lhsType.getRank() == 3 && rhsType.getRank() == 2) {
      // [B, M, K] ** [K, N]
      lhsBatchDims.push_back(0);
      lhsContractDims.push_back(2);
      rhsContractDims.push_back(0);
    } else if (lhsType.getRank() == 3 && rhsType.getRank() == 3) {
      // [B, M, K] ** [B, K, N]
      lhsBatchDims.push_back(0);
      rhsBatchDims.push_back(0);
      lhsContractDims.push_back(2);
      rhsContractDims.push_back(1);
    } else if (lhsType.getRank() == 1 && rhsType.getRank() == 1) {
      // [K] ** [K] -> dot product
      lhsContractDims.push_back(0);
      rhsContractDims.push_back(0);
    } else if (lhsType.getRank() == 2 && rhsType.getRank() == 1) {
      // [M, K] ** [K] -> [M]
      lhsContractDims.push_back(1);
      rhsContractDims.push_back(0);
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported matmul shape");
    }

    // Build the dimension numbers attribute.
    auto dotDimNums = stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(),
        lhsBatchDims, rhsBatchDims,
        lhsContractDims, rhsContractDims);

    rewriter.replaceOpWithNewOp<stablehlo::DotGeneralOp>(
        op, op.getResult().getType(),
        adaptor.getLhs(), adaptor.getRhs(), dotDimNums,
        /*precision_config=*/ArrayAttr());

    return success();
  }
};

// ── ReluOp ─────────────────────────────────────────────────────────────────

struct ReluOpLowering : public OpConversionPattern<ReluOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // jules.relu(x) -> stablehlo.max(stablehlo.constant(0), x)
    auto inputType = adaptor.getInput().getType().dyn_cast<RankedTensorType>();
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "requires ranked tensor input");
    }

    // Create a zero constant with the same shape.
    auto zeroAttr = DenseFPElementsAttr::get(
        inputType, ArrayRef<APFloat>(APFloat::getZero(APFloat::IEEEsingle())));
    auto zero = rewriter.create<stablehlo::ConstantOp>(op.getLoc(), zeroAttr);

    rewriter.replaceOpWithNewOp<stablehlo::MaxOp>(
        op, zero, adaptor.getInput());
    return success();
  }
};

// ── SigmoidOp ──────────────────────────────────────────────────────────────

struct SigmoidOpLowering : public OpConversionPattern<SigmoidOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SigmoidOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::LogisticOp>(
        op, adaptor.getInput());
    return success();
  }
};

// ── TanhOp ─────────────────────────────────────────────────────────────────

struct TanhOpLowering : public OpConversionPattern<TanhOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TanhOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::TanhOp>(
        op, adaptor.getInput());
    return success();
  }
};

// ── MeanOp ─────────────────────────────────────────────────────────────────

struct MeanOpLowering : public OpConversionPattern<MeanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(MeanOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // jules.mean -> stablehlo.reduce + stablehlo.divide
    auto inputType = adaptor.getInput().getType().dyn_cast<RankedTensorType>();
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "requires ranked tensor input");
    }

    // Compute total number of elements for the divisor.
    int64_t numElements = 1;
    SmallVector<int64_t, 4> reduceDims;
    for (int64_t i = 0; i < inputType.getRank(); ++i) {
      int64_t dimSize = inputType.getDimSize(i);
      if (dimSize != ShapedType::kDynamic) {
        numElements *= dimSize;
      }
      reduceDims.push_back(i);
    }

    // Create the reduction: add all elements together.
    auto zeroAttr = DenseFPElementsAttr::get(
        RankedTensorType::get({}, inputType.getElementType()),
        ArrayRef<APFloat>(APFloat::getZero(APFloat::IEEEsingle())));
    auto zero = rewriter.create<stablehlo::ConstantOp>(op.getLoc(), zeroAttr);

    auto reduceOp = rewriter.create<stablehlo::ReduceOp>(
        op.getLoc(),
        /*resultTypes=*/TypeRange{RankedTensorType::get({}, inputType.getElementType())},
        /*inputs=*/ValueRange{adaptor.getInput()},
        /*initValues=*/ValueRange{zero},
        /*dimensions=*/DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(reduceDims.size())},
                                   rewriter.getI64Type()),
            reduceDims));

    // Build the reducer region (add).
    {
      auto &block = reduceOp.getBody().emplaceBlock();
      auto blockArgType = RankedTensorType::get({}, inputType.getElementType());
      block.addArgument(blockArgType, op.getLoc());
      block.addArgument(blockArgType, op.getLoc());
      auto addOp = rewriter.create<stablehlo::AddOp>(
          op.getLoc(), block.getArgument(0), block.getArgument(1));
      rewriter.create<stablehlo::ReturnOp>(op.getLoc(), addOp.getResult());
    }

    // Divide by number of elements.
    auto divisorAttr = DenseFPElementsAttr::get(
        RankedTensorType::get({}, inputType.getElementType()),
        ArrayRef<APFloat>(APFloat(static_cast<double>(numElements))));
    auto divisor = rewriter.create<stablehlo::ConstantOp>(op.getLoc(), divisorAttr);

    rewriter.replaceOpWithNewOp<stablehlo::DivOp>(
        op, reduceOp.getResult(0), divisor);
    return success();
  }
};

// ── SumOp ──────────────────────────────────────────────────────────────────

struct SumOpLowering : public OpConversionPattern<SumOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SumOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto inputType = adaptor.getInput().getType().dyn_cast<RankedTensorType>();
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "requires ranked tensor input");
    }

    SmallVector<int64_t, 4> reduceDims;
    for (int64_t i = 0; i < inputType.getRank(); ++i) {
      reduceDims.push_back(i);
    }

    auto zeroAttr = DenseFPElementsAttr::get(
        RankedTensorType::get({}, inputType.getElementType()),
        ArrayRef<APFloat>(APFloat::getZero(APFloat::IEEEsingle())));
    auto zero = rewriter.create<stablehlo::ConstantOp>(op.getLoc(), zeroAttr);

    auto reduceOp = rewriter.create<stablehlo::ReduceOp>(
        op.getLoc(),
        TypeRange{RankedTensorType::get({}, inputType.getElementType())},
        ValueRange{adaptor.getInput()},
        ValueRange{zero},
        DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(reduceDims.size())},
                                   rewriter.getI64Type()),
            reduceDims));

    {
      auto &block = reduceOp.getBody().emplaceBlock();
      auto blockArgType = RankedTensorType::get({}, inputType.getElementType());
      block.addArgument(blockArgType, op.getLoc());
      block.addArgument(blockArgType, op.getLoc());
      auto addOp = rewriter.create<stablehlo::AddOp>(
          op.getLoc(), block.getArgument(0), block.getArgument(1));
      rewriter.create<stablehlo::ReturnOp>(op.getLoc(), addOp.getResult());
    }

    rewriter.replaceOp(op, reduceOp.getResult(0));
    return success();
  }
};

// ── ZerosOp ────────────────────────────────────────────────────────────────

struct ZerosOpLowering : public OpConversionPattern<ZerosOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ZerosOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "requires ranked tensor result");
    }

    auto zeroAttr = DenseFPElementsAttr::get(
        resultType,
        ArrayRef<APFloat>(APFloat::getZero(
            resultType.getElementType().cast<FloatType>().getFloatSemantics())));
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, zeroAttr);
    return success();
  }
};

// ── OnesOp ─────────────────────────────────────────────────────────────────

struct OnesOpLowering : public OpConversionPattern<OnesOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(OnesOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "requires ranked tensor result");
    }

    auto oneAttr = DenseFPElementsAttr::get(
        resultType,
        ArrayRef<APFloat>(APFloat(
            resultType.getElementType().cast<FloatType>().getFloatSemantics(),
            1)));
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, oneAttr);
    return success();
  }
};

// ── CastOp ─────────────────────────────────────────────────────────────────

struct CastOpLowering : public OpConversionPattern<CastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(CastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(
        op, adaptor.getInput(), op.getTargetType());
    return success();
  }
};

// ── TransposeOp ────────────────────────────────────────────────────────────

struct TransposeOpLowering : public OpConversionPattern<TransposeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto inputType = adaptor.getInput().getType().dyn_cast<RankedTensorType>();
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "requires ranked tensor input");
    }

    // Transpose permutation: reverse the dimensions.
    SmallVector<int64_t, 4> permutation;
    for (int64_t i = inputType.getRank() - 1; i >= 0; --i) {
      permutation.push_back(i);
    }

    auto permAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<int64_t>(permutation.size())},
                               rewriter.getI64Type()),
        permutation);

    rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(
        op, adaptor.getInput(), permAttr);
    return success();
  }
};

// ── ReshapeOp ──────────────────────────────────────────────────────────────

struct ReshapeOpLowering : public OpConversionPattern<ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "requires ranked tensor result");
    }

    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(
        op, adaptor.getInput(), resultType);
    return success();
  }
};

// ── ConcatOp ───────────────────────────────────────────────────────────────

struct ConcatOpLowering : public OpConversionPattern<ConcatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ConcatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
        op, adaptor.getInputs(), adaptor.getAxis());
    return success();
  }
};

// ── SelectOp ───────────────────────────────────────────────────────────────

struct SelectOpLowering : public OpConversionPattern<SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SelectOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(
        op, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    return success();
  }
};

// ── CmpOp ──────────────────────────────────────────────────────────────────

struct CmpOpLowering : public OpConversionPattern<CmpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(CmpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Map Jules comparison direction to StableHLO comparison direction.
    auto direction = stablehlo::ComparisonDirection::EQ;
    StringRef dirStr = adaptor.getComparisonDirection();

    if (dirStr == "EQ")       direction = stablehlo::ComparisonDirection::EQ;
    else if (dirStr == "NE")  direction = stablehlo::ComparisonDirection::NE;
    else if (dirStr == "LT")  direction = stablehlo::ComparisonDirection::LT;
    else if (dirStr == "GT")  direction = stablehlo::ComparisonDirection::GT;
    else if (dirStr == "LE")  direction = stablehlo::ComparisonDirection::LE;
    else if (dirStr == "GE")  direction = stablehlo::ComparisonDirection::GE;

    rewriter.replaceOpWithNewOp<stablehlo::CompareOp>(
        op, adaptor.getLhs(), adaptor.getRhs(),
        stablehlo::ComparisonDirectionAttr::get(rewriter.getContext(), direction),
        stablehlo::ComparisonTypeAttr::get(rewriter.getContext(),
                                            stablehlo::ComparisonType::NOTYPE));
    return success();
  }
};

// ── ConstantOp ─────────────────────────────────────────────────────────────

struct ConstantOpLowering : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Jules constant -> StableHLO constant.
    // The value attribute is already an MLIR attribute, so we can directly
    // create a stablehlo.constant.
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, adaptor.getValue());
    return success();
  }
};

// ── RandomOp ────────────────────────────────────────────────────────────────

struct RandomOpLowering : public OpConversionPattern<RandomOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(RandomOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // jules.random -> stablehlo.rng
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "requires ranked tensor result");
    }

    // Create a constant seed state: tensor<2xui64> with values [0, 1].
    auto seedType = RankedTensorType::get({2}, rewriter.getIntegerType(64));
    SmallVector<APInt, 2> seedValues = {APInt(64, 0), APInt(64, 1)};
    auto seedAttr = DenseIntElementsAttr::get(seedType, seedValues);
    auto seed = rewriter.create<stablehlo::ConstantOp>(op.getLoc(), seedAttr);

    // Create a shape constant from the result type's shape.
    SmallVector<int64_t, 4> shapeValues;
    for (int64_t dim : resultType.getShape()) {
      shapeValues.push_back(dim);
    }
    auto shapeType = RankedTensorType::get(
        {static_cast<int64_t>(shapeValues.size())}, rewriter.getI64Type());
    auto shapeAttr = DenseIntElementsAttr::get(shapeType, shapeValues);
    auto shape = rewriter.create<stablehlo::ConstantOp>(op.getLoc(), shapeAttr);

    // Create stablehlo.rng with UNIFORM distribution.
    auto rngDistribution = stablehlo::RngDistributionAttr::get(
        rewriter.getContext(), stablehlo::RngDistribution::UNIFORM);

    rewriter.replaceOpWithNewOp<stablehlo::RngOp>(
        op, op.getResult().getType(), seed, shape, rngDistribution);
    return success();
  }
};

// ── SliceOp ─────────────────────────────────────────────────────────────────

struct SliceOpLowering : public OpConversionPattern<SliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SliceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // jules.slice -> stablehlo.slice
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "requires ranked tensor result");
    }

    // Extract the I64ArrayAttr values as SmallVector<int64_t>.
    auto extractI64Array = [](ArrayAttr arr) -> SmallVector<int64_t, 4> {
      SmallVector<int64_t, 4> result;
      for (auto attr : arr) {
        result.push_back(attr.cast<IntegerAttr>().getInt());
      }
      return result;
    };

    auto startIndices = extractI64Array(
        op->getAttr("start_indices").cast<ArrayAttr>());
    auto limitIndices = extractI64Array(
        op->getAttr("limit_indices").cast<ArrayAttr>());
    auto strides = extractI64Array(
        op->getAttr("strides").cast<ArrayAttr>());

    int64_t rank = static_cast<int64_t>(startIndices.size());

    // Convert to DenseIntElementsAttr for stablehlo.slice.
    auto startAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({rank}, rewriter.getI64Type()), startIndices);
    auto limitAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({rank}, rewriter.getI64Type()), limitIndices);
    auto stridesDenseAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({rank}, rewriter.getI64Type()), strides);

    rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(
        op, adaptor.getInput(), startAttr, limitAttr, stridesDenseAttr);
    return success();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Populate conversion patterns
//===----------------------------------------------------------------------===//

void jules::populateJulesToStableHLOPatterns(RewritePatternSet &patterns,
                                              TypeConverter &typeConverter) {
  patterns.add<
    AddOpLowering,
    SubOpLowering,
    MulOpLowering,
    DivOpLowering,
    PowOpLowering,
    NegOpLowering,
    MatMulOpLowering,
    ReluOpLowering,
    SigmoidOpLowering,
    TanhOpLowering,
    MeanOpLowering,
    SumOpLowering,
    ZerosOpLowering,
    OnesOpLowering,
    CastOpLowering,
    TransposeOpLowering,
    ReshapeOpLowering,
    ConcatOpLowering,
    SelectOpLowering,
    CmpOpLowering,
    ConstantOpLowering,
    RandomOpLowering,
    SliceOpLowering
  >(typeConverter, patterns.getContext());
}

//===----------------------------------------------------------------------===//
// The MLIR Pass
//===----------------------------------------------------------------------===//

namespace {

struct JulesToStableHLOLoweringPass
    : public PassWrapper<JulesToStableHLOLoweringPass,
                         OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<stablehlo::StableHLODialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // Create the type converter.
    auto typeConverter = getJulesToStableHLOTypeConverter();

    // Create the conversion target.
    ConversionTarget target(*ctx);
    target.addLegalDialect<stablehlo::StableHLODialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addIllegalDialect<JulesDialect>();

    // Add conversion patterns.
    RewritePatternSet patterns(ctx);
    populateJulesToStableHLOPatterns(patterns, typeConverter);

    // Add func conversion pattern.
    populateFuncOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                typeConverter);

    // Apply full conversion.
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> jules::createJulesToStableHLOLoweringPass() {
  return std::make_unique<JulesToStableHLOLoweringPass>();
}

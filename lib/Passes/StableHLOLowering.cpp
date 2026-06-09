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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/DenseSet.h"
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

    // Build precision_config from matmul attributes if present.
    // This is the fix for P1 #9: MixedPrecisionPass now sets
    // compute_precision and accumulation_precision on the matmul op,
    // and the StableHLO lowering emits the correct precision_config.
    ArrayAttr precisionConfig;
    if (op.getComputePrecision().has_value() ||
        op.getAccumulationPrecision().has_value()) {
      auto computePrec = op.getComputePrecision().value_or("f32");
      auto accumPrec = op.getAccumulationPrecision().value_or("f32");

      auto parsePrecision = [](StringRef s) -> stablehlo::Precision {
        if (s == "bf16") return stablehlo::Precision::BF16;
        if (s == "fp8e4m3") return stablehlo::Precision::FP8E4M3;
        if (s == "fp8e5m2") return stablehlo::Precision::FP8E5M2;
        if (s == "f16") return stablehlo::Precision::F16;
        return stablehlo::Precision::DEFAULT;
      };

      auto lhsPrec = stablehlo::PrecisionAttr::get(rewriter.getContext(),
                                                     parsePrecision(computePrec));
      auto rhsPrec = stablehlo::PrecisionAttr::get(rewriter.getContext(),
                                                     parsePrecision(computePrec));
      auto accumPrecAttr = stablehlo::PrecisionAttr::get(rewriter.getContext(),
                                                          parsePrecision(accumPrec));

      // FIX: StableHLO dot_general expects exactly 2 precision config elements
      // (one per operand). Accumulation precision is not part of precision_config;
      // it's conveyed via the output_type or a custom_call wrapper.
      precisionConfig = rewriter.getArrayAttr({lhsPrec, rhsPrec});
    }

    rewriter.replaceOpWithNewOp<stablehlo::DotGeneralOp>(
        op, op.getResult().getType(),
        adaptor.getLhs(), adaptor.getRhs(), dotDimNums,
        precisionConfig);

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
    // FIX: Use the input element type's float semantics instead of hardcoding f32.
    // Hardcoding IEEEsingle crashes the StableHLO verifier for bf16/f64 inputs.
    auto floatSemantics = inputType.getElementType()
        .cast<FloatType>().getFloatSemantics();
    auto zeroAttr = DenseFPElementsAttr::get(
        inputType, ArrayRef<APFloat>(APFloat::getZero(floatSemantics)));
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
    // FIX: Use the input element type's float semantics instead of hardcoding f32.
    auto floatSemantics = inputType.getElementType()
        .dyn_cast<FloatType>().getFloatSemantics();
    auto zeroAttr = DenseFPElementsAttr::get(
        RankedTensorType::get({}, inputType.getElementType()),
        ArrayRef<APFloat>(APFloat::getZero(floatSemantics)));
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

    // FIX: Use the input element type's float semantics instead of hardcoding f32.
    auto floatSemantics = inputType.getElementType()
        .dyn_cast<FloatType>().getFloatSemantics();
    auto zeroAttr = DenseFPElementsAttr::get(
        RankedTensorType::get({}, inputType.getElementType()),
        ArrayRef<APFloat>(APFloat::getZero(floatSemantics)));
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

    // Transpose permutation: use the op's permutation attribute if available,
    // otherwise default to reversing dimensions (2D transpose = matrix transpose).
    SmallVector<int64_t, 4> permutation;
    if (auto permAttr = op->getAttr("permutation")) {
      if (auto arrayAttr = permAttr.dyn_cast<ArrayAttr>()) {
        for (auto attr : arrayAttr) {
          permutation.push_back(attr.cast<IntegerAttr>().getInt());
        }
      }
    }
    if (permutation.empty()) {
      // Default: reverse dimensions (standard 2D transpose)
      for (int64_t i = inputType.getRank() - 1; i >= 0; --i) {
        permutation.push_back(i);
      }
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

// ── LogOp ──────────────────────────────────────────────────────────────────

struct LogOpLowering : public OpConversionPattern<LogOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(LogOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::LogOp>(
        op, adaptor.getInput());
    return success();
  }
};

// ── PadOp ──────────────────────────────────────────────────────────────────

struct PadOpLowering : public OpConversionPattern<PadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(PadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // jules.pad -> stablehlo.pad
    auto inputType = adaptor.getInput().getType().dyn_cast<RankedTensorType>();
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "requires ranked tensor input");
    }

    int64_t rank = inputType.getRank();

    // Extract padding attributes.
    auto extractI64Array = [](ArrayAttr arr) -> SmallVector<int64_t, 4> {
      SmallVector<int64_t, 4> result;
      for (auto attr : arr) {
        result.push_back(attr.cast<IntegerAttr>().getInt());
      }
      return result;
    };

    auto paddingLow = extractI64Array(op.getPaddingLow());
    auto paddingHigh = extractI64Array(op.getPaddingHigh());
    auto interiorPadding = extractI64Array(op.getInteriorPadding());

    // Create DenseI64ArrayAttr for each padding component.
    auto paddingLowAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({rank}, rewriter.getI64Type()), paddingLow);
    auto paddingHighAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({rank}, rewriter.getI64Type()), paddingHigh);
    auto interiorPaddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({rank}, rewriter.getI64Type()), interiorPadding);

    rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
        op, adaptor.getInput(), adaptor.getPaddingValue(),
        paddingLowAttr, paddingHighAttr, interiorPaddingAttr);
    return success();
  }
};

// ── BroadcastInDimOp ───────────────────────────────────────────────────────

struct BroadcastInDimOpLowering : public OpConversionPattern<BroadcastInDimOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(BroadcastInDimOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // jules.broadcast_in_dim -> stablehlo.broadcast_in_dim
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "requires ranked tensor result");
    }

    // Extract broadcast dimensions.
    auto broadcastDimsAttr = op.getBroadcastDimensions();
    SmallVector<int64_t, 4> broadcastDims;
    for (auto attr : broadcastDimsAttr) {
      broadcastDims.push_back(attr.cast<IntegerAttr>().getInt());
    }

    auto dimsAttr = DenseIntElementsAttr::get(
        RankedTensorType::get(
            {static_cast<int64_t>(broadcastDims.size())},
            rewriter.getI64Type()),
        broadcastDims);

    rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
        op, adaptor.getInput(), dimsAttr);
    return success();
  }
};

// ── ReduceOp ───────────────────────────────────────────────────────────────

struct ReduceOpLowering : public OpConversionPattern<ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // jules.reduce -> stablehlo.reduce
    auto inputType = adaptor.getInput().getType().dyn_cast<RankedTensorType>();
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "requires ranked tensor input");
    }

    // Extract reduction dimensions.
    auto reduceDimsAttr = op.getDimensions();
    SmallVector<int64_t, 4> reduceDims;
    for (auto attr : reduceDimsAttr) {
      reduceDims.push_back(attr.cast<IntegerAttr>().getInt());
    }

    auto dimsDenseAttr = DenseIntElementsAttr::get(
        RankedTensorType::get(
            {static_cast<int64_t>(reduceDims.size())},
            rewriter.getI64Type()),
        reduceDims);

    // Compute the result type (input with reduced dims removed).
    SmallVector<int64_t, 4> resultShape;
    llvm::SmallDenseSet<int64_t, 8> reduceDimSet(reduceDims.begin(),
                                                   reduceDims.end());
    for (int64_t d = 0; d < inputType.getRank(); ++d) {
      if (!reduceDimSet.contains(d)) {
        resultShape.push_back(inputType.getDimSize(d));
      }
    }
    auto resultType = RankedTensorType::get(resultShape,
                                             inputType.getElementType());

    auto stableReduceOp = rewriter.create<stablehlo::ReduceOp>(
        op.getLoc(),
        TypeRange{resultType},
        ValueRange{adaptor.getInput()},
        ValueRange{adaptor.getInitValue()},
        dimsDenseAttr);

    // Clone the reducer body from the Jules reduce into the StableHLO reduce.
    auto &srcBody = op.getBody();
    auto &dstBody = stableReduceOp.getBody();
    rewriter.cloneRegionBefore(srcBody, dstBody, dstBody.begin());

    rewriter.replaceOp(op, stableReduceOp.getResult(0));
    return success();
  }
};

// ── WhileOp ────────────────────────────────────────────────────────────────

struct WhileOpLowering : public OpConversionPattern<WhileOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(WhileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // jules.while -> stablehlo.while
    //
    // StableHLO's while op has the same structure:
    //   stablehlo.while(carried_vars) cond { ... } do { ... }
    //
    // The condition region returns a single tensor<i1>.
    // The body region returns the new carried variable values.
    SmallVector<Type, 4> resultTypes;
    for (auto result : op.getResults()) {
      resultTypes.push_back(result.getType());
    }

    auto stableWhileOp = rewriter.create<stablehlo::WhileOp>(
        op.getLoc(), resultTypes, adaptor.getCarriedVars());

    // Clone the condition region.
    auto &srcCond = op.getCond();
    auto &dstCond = stableWhileOp.getCond();
    rewriter.cloneRegionBefore(srcCond, dstCond, dstCond.begin());

    // Clone the body region.
    auto &srcBody = op.getBody();
    auto &dstBody = stableWhileOp.getBody();
    rewriter.cloneRegionBefore(srcBody, dstBody, dstBody.begin());

    // Replace the while op results — single call, not N calls.
    // Calling replaceOp N times on the same op corrupts the IR after
    // the first call makes the op dead.
    rewriter.replaceOp(op, stableWhileOp.getResults());
    return success();
  }
};

// ── ParallelOp ─────────────────────────────────────────────────────────────

struct ParallelOpLowering : public OpConversionPattern<ParallelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ParallelOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // jules.parallel -> scf.parallel
    //
    // Since stablehlo doesn't have a direct parallel op, we lower to
    // scf.parallel which represents an ordered parallel loop. The SCF
    // dialect is a standard MLIR dialect that can be further lowered
    // as needed by the XLA backend.
    //
    // scf.parallel has the form:
    //   scf.parallel (%iv) = (%lb) to (%ub) step (%step) {
    //     ... body ...
    //   }

    // Create the lower bound, upper bound, and step as index values.
    auto lbIndex = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(op.getLowerBound()));
    auto ubIndex = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(op.getUpperBound()));
    auto stepIndex = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(op.getStep()));

    SmallVector<Value, 1> lbs = {lbIndex.getResult()};
    SmallVector<Value, 1> ubs = {ubIndex.getResult()};
    SmallVector<Value, 1> steps = {stepIndex.getResult()};

    // Collect result types for the parallel operation.
    SmallVector<Type, 4> resultTypes;
    for (auto result : op.getResults()) {
      resultTypes.push_back(result.getType());
    }

    // Create zero init values for the parallel's reduction variables.
    // Since jules.parallel represents independent iterations (not reductions),
    // we don't have reduction variables. We use scf.parallel without
    // reduction.
    auto parallelOp = rewriter.create<mlir::scf::ParallelOp>(
        op.getLoc(), lbs, ubs, steps);

    // Clone the body region into the scf.parallel.
    // FIX: We must recursively lower any Jules dialect ops inside the
    // parallel body to StableHLO/SCF ops, since the Jules→StableHLO
    // conversion only runs once. Without this, the body would contain
    // jules.add, jules.mul, etc. that XLA can't understand.
    //
    // We perform an in-place conversion walk after cloning.
    auto &srcBody = op.getBody();
    auto &dstBody = parallelOp.getBody();
    rewriter.cloneRegionBefore(srcBody, dstBody, dstBody.begin());

    // Lower Jules dialect ops inside the parallel body to SCF/arith ops.
    // For simple elementwise ops, we convert them inline.
    dstBody.walk([&](Operation *bodyOp) {
      if (bodyOp->getDialect()->getNamespace() == "jules" &&
          !isa<ParallelOp>(bodyOp)) {
        // Convert jules.add → arith.addf, jules.mul → arith.mulf, etc.
        // This is a simplified inline conversion for the common cases.
        if (auto addOp = dyn_cast<AddOp>(bodyOp)) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(addOp);
          auto newOp = rewriter.create<mlir::arith::AddFOp>(
              addOp.getLoc(), addOp.getLhs(), addOp.getRhs());
          addOp.getResult().replaceAllUsesWith(newOp.getResult());
        } else if (auto mulOp = dyn_cast<MulOp>(bodyOp)) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(mulOp);
          auto newOp = rewriter.create<mlir::arith::MulFOp>(
              mulOp.getLoc(), mulOp.getLhs(), mulOp.getRhs());
          mulOp.getResult().replaceAllUsesWith(newOp.getResult());
        } else if (auto subOp = dyn_cast<SubOp>(bodyOp)) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(subOp);
          auto newOp = rewriter.create<mlir::arith::SubFOp>(
              subOp.getLoc(), subOp.getLhs(), subOp.getRhs());
          subOp.getResult().replaceAllUsesWith(newOp.getResult());
        } else if (auto divOp = dyn_cast<DivOp>(bodyOp)) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(divOp);
          auto newOp = rewriter.create<mlir::arith::DivFOp>(
              divOp.getLoc(), divOp.getLhs(), divOp.getRhs());
          divOp.getResult().replaceAllUsesWith(newOp.getResult());
        }
        // Other Jules ops in parallel bodies will need extension.
      }
    });

    // scf.parallel doesn't produce results in the same way.
    // For parallel ops that produce results, we would need to use
    // scf.reduce or a different pattern. For now, we handle the
    // common case of parallel execution without result aggregation.
    rewriter.replaceOp(op, parallelOp.getResults());
    return success();
  }
};

// ── GeluOp ──────────────────────────────────────────────────────────────────

struct GeluOpLowering : public OpConversionPattern<GeluOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(GeluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // jules.gelu -> stablehlo.custom_call @gelu
    // StableHLO doesn't have a native GELU, so we use custom_call
    auto resultType = op.getResult().getType();
    SmallVector<Type, 1> resultTypes = {resultType};

    auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
        op.getLoc(),
        resultTypes,
        ValueRange{adaptor.getInput()},
        rewriter.getStringAttr("gelu"),
        rewriter.getBoolAttr(false),
        rewriter.getStringAttr(""),
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*output_operand_aliases=*/nullptr);

    rewriter.replaceOp(op, customCallOp.getResults());
    return success();
  }
};

// ── ExpOp ──────────────────────────────────────────────────────────────────

struct ExpOpLowering : public OpConversionPattern<ExpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ExpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::ExpOp>(
        op, adaptor.getInput());
    return success();
  }
};

// ── Conv2DOp ────────────────────────────────────────────────────────────────

struct Conv2DOpLowering : public OpConversionPattern<Conv2DOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(Conv2DOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // jules.conv2d -> stablehlo.convolution
    auto inputType = adaptor.getInput().getType().dyn_cast<RankedTensorType>();
    auto kernelType = adaptor.getKernel().getType().dyn_cast<RankedTensorType>();
    if (!inputType || !kernelType) {
      return rewriter.notifyMatchFailure(op, "requires ranked tensor operands");
    }

    // Extract padding, strides, dilation from attributes
    auto extractI64Array = [](ArrayAttr arr) -> SmallVector<int64_t, 4> {
      SmallVector<int64_t, 4> result;
      for (auto attr : arr) {
        result.push_back(attr.cast<IntegerAttr>().getInt());
      }
      return result;
    };

    auto padding = extractI64Array(op.getPadding());
    auto strides = extractI64Array(op.getStrides());
    auto dilation = extractI64Array(op.getDilation());

    // Build StableHLO convolution dimension numbers
    // Input: [batch, in_channels, height, width] (NCHW)
    // Kernel: [out_channels, in_channels, kH, kW] (OIHW)
    auto dimNums = stablehlo::ConvDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*input_batch_dimension=*/0,
        /*input_feature_dimension=*/1,
        /*input_spatial_dimensions=*/{2, 3},
        /*kernel_output_feature_dimension=*/0,
        /*kernel_input_feature_dimension=*/1,
        /*kernel_spatial_dimensions=*/{2, 3},
        /*output_batch_dimension=*/0,
        /*output_feature_dimension=*/1,
        /*output_spatial_dimensions=*/{2, 3});

    // Create padding attribute (pairs of low/high padding)
    SmallVector<int64_t, 8> paddingFlat;
    for (size_t i = 0; i < padding.size(); i++) {
      paddingFlat.push_back(padding[i]); // low
      paddingFlat.push_back(padding[i]); // high (symmetric padding)
    }
    auto paddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<int64_t>(paddingFlat.size() / 2), 2},
                              rewriter.getI64Type()),
        paddingFlat);

    rewriter.replaceOpWithNewOp<stablehlo::ConvolutionOp>(
        op, op.getResult().getType(),
        adaptor.getInput(), adaptor.getKernel(),
        dimNums,
        /*window_strides=*/DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(strides.size())},
                                  rewriter.getI64Type()),
            strides),
        /*padding=*/paddingAttr,
        /*lhs_dilation=*/DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(dilation.size())},
                                  rewriter.getI64Type()),
            dilation),
        /*rhs_dilation=*/DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(dilation.size())},
                                  rewriter.getI64Type()),
            dilation),
        /*window_reversal=*/DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(strides.size())},
                                  rewriter.getI1Type()),
            SmallVector<bool, 4>(strides.size(), false)),
        /*input_batch_dimension=*/0,
        /*input_feature_dimension=*/1,
        /*input_spatial_dimensions=*/{2, 3},
        /*kernel_output_feature_dimension=*/0,
        /*kernel_input_feature_dimension=*/1,
        /*kernel_spatial_dimensions=*/{2, 3},
        /*output_batch_dimension=*/0,
        /*output_feature_dimension=*/1,
        /*output_spatial_dimensions=*/{2, 3},
        /*feature_group_count=*/1,
        /*batch_group_count=*/1,
        /*precision_config=*/ArrayAttr());
    return success();
  }
};

// ── AllReduceOp ────────────────────────────────────────────────────────────

struct AllReduceOpLowering : public OpConversionPattern<AllReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(AllReduceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // jules.all_reduce -> stablehlo.all_reduce
    auto reduction = op.getReduction();

    // Build the reduction region
    auto resultType = op.getResult().getType();
    auto elementType = resultType.dyn_cast<RankedTensorType>().getElementType();
    auto scalarType = RankedTensorType::get({}, elementType);

    auto allReduceOp = rewriter.create<stablehlo::AllReduceOp>(
        op.getLoc(), TypeRange{resultType},
        ValueRange{adaptor.getInput()},
        /*channel_handle=*/nullptr,
        /*use_global_device_ids=*/nullptr);

    // Build the reducer region
    auto &block = allReduceOp.getBody().emplaceBlock();
    block.addArgument(scalarType, op.getLoc());
    block.addArgument(scalarType, op.getLoc());

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);

    if (reduction == "sum") {
      auto addOp = rewriter.create<stablehlo::AddOp>(
          op.getLoc(), block.getArgument(0), block.getArgument(1));
      rewriter.create<stablehlo::ReturnOp>(op.getLoc(), addOp.getResult());
    } else if (reduction == "max") {
      auto maxOp = rewriter.create<stablehlo::MaxOp>(
          op.getLoc(), block.getArgument(0), block.getArgument(1));
      rewriter.create<stablehlo::ReturnOp>(op.getLoc(), maxOp.getResult());
    } else if (reduction == "min") {
      auto minOp = rewriter.create<stablehlo::MinOp>(
          op.getLoc(), block.getArgument(0), block.getArgument(1));
      rewriter.create<stablehlo::ReturnOp>(op.getLoc(), minOp.getResult());
    } else {
      // Default: sum
      auto addOp = rewriter.create<stablehlo::AddOp>(
          op.getLoc(), block.getArgument(0), block.getArgument(1));
      rewriter.create<stablehlo::ReturnOp>(op.getLoc(), addOp.getResult());
    }

    rewriter.replaceOp(op, allReduceOp.getResults());
    return success();
  }
};

// ── AllGatherOp ────────────────────────────────────────────────────────────

struct AllGatherOpLowering : public OpConversionPattern<AllGatherOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(AllGatherOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // jules.all_gather -> stablehlo.all_gather
    auto resultType = op.getResult().getType();

    auto allGatherOp = rewriter.create<stablehlo::AllGatherOp>(
        op.getLoc(), TypeRange{resultType},
        ValueRange{adaptor.getInput()},
        /*all_gather_dim=*/op.getGatherDim(),
        /*channel_handle=*/nullptr,
        /*use_global_device_ids=*/nullptr);

    rewriter.replaceOp(op, allGatherOp.getResults());
    return success();
  }
};

// ── ExternKernelOp ─────────────────────────────────────────────────────────

struct ExternKernelOpLowering : public OpConversionPattern<ExternKernelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ExternKernelOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // jules.extern_kernel -> stablehlo.custom_call
    //
    // The ExternKernelOp maps to StableHLO's custom_call op with the
    // `call_target_name` attribute set to the kernel_name. This allows
    // the kernel to be resolved as a PJRT plugin or shared library
    // symbol at runtime.
    //
    // stablehlo.custom_call @kernel_name(inputs) -> results
    //   {call_target_name = "kernel_name",
    //    has_side_effect = false,
    //    backend_config = ""}

    // Collect result types.
    SmallVector<Type, 4> resultTypes;
    for (auto result : op.getResults()) {
      resultTypes.push_back(result.getType());
    }

    // Collect input values.
    SmallVector<Value, 4> inputs;
    for (auto input : adaptor.getInputs()) {
      inputs.push_back(input);
    }

    // Create the custom_call operation.
    auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
        op.getLoc(),
        resultTypes,
        inputs,
        rewriter.getStringAttr(op.getKernelName()),
        rewriter.getBoolAttr(false),  // has_side_effect
        rewriter.getStringAttr(""),    // backend_config
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*output_operand_aliases=*/nullptr);

    rewriter.replaceOp(op, customCallOp.getResults());
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
    GeluOpLowering,
    ExpOpLowering,
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
    SliceOpLowering,
    LogOpLowering,
    PadOpLowering,
    BroadcastInDimOpLowering,
    ReduceOpLowering,
    WhileOpLowering,
    ParallelOpLowering,
    Conv2DOpLowering,
    AllReduceOpLowering,
    AllGatherOpLowering,
    ExternKernelOpLowering
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
    registry.insert<arith::ArithDialect>();
    registry.insert<scf::SCFDialect>();
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

//===- MixedPrecisionPass.cpp - Mixed Precision Optimization Implementation -===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the mixed precision optimization pass for the Jules
// compiler. It inserts CastOp operations to downcast where safe (activations,
// matmul inputs) while keeping accumulation in f32. It also adds bf16 and fp8
// type support to the Jules type system.
//
// Algorithm overview:
//
//   1. Walk each function looking for compute-heavy ops (MatMulOp, ConvOp)
//   2. For each MatMulOp:
//      - Insert CastOp before lhs/rhs to downcast inputs to target precision
//      - Keep the matmul accumulation in f32 (or the original type)
//      - Insert CastOp after the result to cast back to the original type
//   3. For activation ops (ReluOp, SigmoidOp, TanhOp):
//      - If the input is already in lower precision, keep it
//      - If the input is f32 and the op follows a matmul, keep f32
//      - For standalone activations, downcast input, compute, cast output back
//   4. For elementwise binary ops (AddOp, SubOp, MulOp, DivOp):
//      - If both operands are already lower precision, stay in lower precision
//      - If mixed, cast the higher-precision operand down
//   5. For reduction ops (SumOp, MeanOp):
//      - Always accumulate in f32 to avoid precision loss
//      - Cast input to f32 before reduction, cast result back after
//
//===----------------------------------------------------------------------===//

#include "jules/Passes/MixedPrecisionPass.h"
#include "jules/Dialect/JulesOps.h"
#include "jules/Dialect/JulesDialect.h"
#include "jules/Dialect/JulesTypes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace jules;

namespace {

// ── Precision Classification ────────────────────────────────────────────────

/// Classification of floating-point precision levels for comparison.
enum class PrecisionLevel : unsigned {
  FP8E4M3 = 8,
  FP8E5M2 = 9,
  BF16    = 16,
  F32     = 32,
  F64     = 64,
  Unknown = 255
};

/// Get the precision level of a given MLIR float type.
PrecisionLevel getPrecisionLevel(Type type) {
  if (auto floatType = type.dyn_cast<FloatType>()) {
    // Check for our custom types first
    if (type.isa<BF16Type>())     return PrecisionLevel::BF16;
    if (type.isa<FP8E4M3Type>())  return PrecisionLevel::FP8E4M3;
    if (type.isa<FP8E5M2Type>())  return PrecisionLevel::FP8E5M2;

    // Standard MLIR float types
    if (floatType.isF64())  return PrecisionLevel::F64;
    if (floatType.isF32())  return PrecisionLevel::F32;
    if (floatType.isBF16()) return PrecisionLevel::BF16;
    if (floatType.isF16())  return PrecisionLevel::BF16; // treat f16 as bf16-level
  }

  // For tensor types, inspect the element type
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    return getPrecisionLevel(tensorType.getElementType());
  }
  if (auto tensorType = type.dyn_cast<UnrankedTensorType>()) {
    return getPrecisionLevel(tensorType.getElementType());
  }

  return PrecisionLevel::Unknown;
}

/// Check if a type is in lower precision than f32.
bool isLowerPrecisionThanF32(Type type) {
  auto level = getPrecisionLevel(type);
  return level < PrecisionLevel::F32 && level != PrecisionLevel::Unknown;
}

/// Check if a type is f32 or has f32 element type.
bool isF32Type(Type type) {
  if (auto floatType = type.dyn_cast<FloatType>())
    return floatType.isF32();
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return tensorType.getElementType().isF32();
  if (auto tensorType = type.dyn_cast<UnrankedTensorType>())
    return tensorType.getElementType().isF32();
  return false;
}

/// Check if an operation is preceded by a matmul (i.e., its input
/// is the result of a matmul, possibly through a cast).
bool followsMatmul(Operation *op) {
  if (op->getNumOperands() == 0) return false;
  Value input = op->getOperand(0);

  // Walk back through cast ops to find the defining op
  Operation *defOp = input.getDefiningOp();
  while (defOp && isa<CastOp>(defOp)) {
    input = defOp->getOperand(0);
    defOp = input.getDefiningOp();
  }

  return defOp && isa<MatMulOp>(defOp);
}

// ── Type Conversion Helpers ─────────────────────────────────────────────────

/// Get the target compute precision type (bf16 or fp8) for a given source type.
/// For tensor types, replaces the element type; for scalar types, replaces
/// the type directly.
Type getTargetComputeType(Type sourceType, PrecisionLevel targetLevel,
                          MLIRContext *ctx) {
  // Determine the target element type based on the precision level
  Type targetElemType;
  switch (targetLevel) {
  case PrecisionLevel::FP8E4M3:
    targetElemType = FP8E4M3Type::get(ctx);
    break;
  case PrecisionLevel::FP8E5M2:
    targetElemType = FP8E5M2Type::get(ctx);
    break;
  case PrecisionLevel::BF16:
  default:
    targetElemType = BF16Type::get(ctx);
    break;
  }

  // For ranked tensor types, preserve the shape and update element type
  if (auto tensorType = sourceType.dyn_cast<RankedTensorType>()) {
    return RankedTensorType::get(tensorType.getShape(), targetElemType);
  }

  // For unranked tensor types
  if (sourceType.isa<UnrankedTensorType>()) {
    return UnrankedTensorType::get(targetElemType);
  }

  // For scalar types
  return targetElemType;
}

/// Get f32 type matching the structure of the source type.
/// For tensor types, replaces the element type with f32; for scalars,
/// returns f32 directly.
Type getF32Type(Type sourceType, MLIRContext *ctx) {
  Type f32ElemType = FloatType::getF32(ctx);

  if (auto tensorType = sourceType.dyn_cast<RankedTensorType>()) {
    return RankedTensorType::get(tensorType.getShape(), f32ElemType);
  }
  if (sourceType.isa<UnrankedTensorType>()) {
    return UnrankedTensorType::get(f32ElemType);
  }
  return f32ElemType;
}

/// Create a CastOp that casts from source value to target type.
/// Sets the original_type attribute for debugging/tracking.
CastOp createCastOp(OpBuilder &builder, Location loc, Value source,
                    Type targetType, Type originalType) {
  auto castOp = builder.create<CastOp>(loc, TypeAttr::get(targetType).getValue(), source, TypeAttr::get(targetType));
  castOp->setAttr("jules.mixed_precision.original_type",
                  TypeAttr::get(originalType));
  return castOp;
}

// ── Op Counting Utility ─────────────────────────────────────────────────────

/// Count the number of non-trivial operations in a function.
/// This excludes func-level ops, terminators, and constant definitions
/// to count only computational operations.
unsigned countComputeOps(func::FuncOp funcOp) {
  unsigned count = 0;
  funcOp.walk([&](Operation *op) {
    // Skip the function itself, terminators, and constants
    if (isa<func::FuncOp>(op)) return;
    if (op->hasTrait<OpTrait::IsTerminator>()) return;
    if (isa<ConstantOp>(op)) return;
    ++count;
  });
  return count;
}

// ── The Pass ────────────────────────────────────────────────────────────────

struct MixedPrecisionPass
    : public PassWrapper<MixedPrecisionPass, OperationPass<ModuleOp>> {

  MixedPrecisionPass() = default;
  // Note: custom constructor with std::string removed because Option<T> is
  // non-copyable. Use the pass option mechanism to set target-compute-precision.

  /// Pass option: target compute precision ("bf16" or "fp8").
  Option<std::string> targetComputePrecision_{
      *this, "target-compute-precision",
      llvm::cl::desc("Target compute precision for mixed precision (bf16 or fp8)"),
      llvm::cl::init("bf16")};

  /// Pass option: minimum number of ops to apply mixed precision.
  Option<unsigned> minOpCountForMixedPrecision_{
      *this, "min-op-count-for-mixed-precision",
      llvm::cl::desc("Only apply mixed precision to functions with >= N ops"),
      llvm::cl::init(5)};

  /// The accumulation precision is always f32 for correctness.
  static constexpr llvm::StringLiteral kAccumulationPrecision = "f32";

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    // Resolve the target precision level from the option string.
    PrecisionLevel targetLevel = resolveTargetLevel(targetComputePrecision_);
    if (targetLevel == PrecisionLevel::Unknown) {
      module.emitError("Unknown target compute precision: ")
          << targetComputePrecision_;
      signalPassFailure();
      return;
    }

    // Process each function in the module.
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp funcOp) {
      funcOps.push_back(funcOp);
    });

    for (auto funcOp : funcOps) {
      // Skip functions that are too small for mixed precision to be worthwhile.
      unsigned computeOps = countComputeOps(funcOp);
      if (computeOps < minOpCountForMixedPrecision_) {
        continue;
      }

      processFunction(funcOp, targetLevel, ctx);
    }
  }

  /// Resolve a string precision name to a PrecisionLevel.
  PrecisionLevel resolveTargetLevel(const std::string &precision) const {
    if (precision == "bf16")  return PrecisionLevel::BF16;
    if (precision == "fp8")   return PrecisionLevel::FP8E4M3;
    if (precision == "fp8e4m3") return PrecisionLevel::FP8E4M3;
    if (precision == "fp8e5m2") return PrecisionLevel::FP8E5M2;
    return PrecisionLevel::Unknown;
  }

  // ── Function-level Processing ────────────────────────────────────────────

  void processFunction(func::FuncOp funcOp, PrecisionLevel targetLevel,
                       MLIRContext *ctx) {
    // We process operations in reverse order within each function so that
    // newly inserted cast ops don't interfere with our walk. However, since
    // we need to track which ops have been modified, we collect ops first
    // and then process them.

    // Phase 1: Process reduction ops (SumOp, MeanOp) — always upcast to f32.
    processReductionOps(funcOp, targetLevel, ctx);

    // Phase 2: Process matmul ops — downcast inputs, accumulate in f32.
    processMatMulOps(funcOp, targetLevel, ctx);

    // Phase 3: Process activation ops — context-dependent precision.
    processActivationOps(funcOp, targetLevel, ctx);

    // Phase 4: Process elementwise binary ops — unify operand precision.
    processElementwiseBinaryOps(funcOp, targetLevel, ctx);
  }

  // ── Phase 1: Reduction Op Processing ─────────────────────────────────────
  //
  // Reductions (Sum, Mean) must always accumulate in f32 to avoid
  // catastrophic precision loss. We:
  //   1. Insert a CastOp to cast the input up to f32 if it's in lower precision
  //   2. Let the reduction proceed in f32
  //   3. Insert a CastOp after the reduction to cast back to the original type

  void processReductionOps(func::FuncOp funcOp, PrecisionLevel targetLevel,
                           MLIRContext *ctx) {
    SmallVector<Operation*, 8> reductionOps;
    funcOp.walk([&](Operation *op) {
      if (isa<SumOp>(op) || isa<MeanOp>(op)) {
        reductionOps.push_back(op);
      }
    });

    for (auto *op : reductionOps) {
      OpBuilder builder(ctx);
      Location loc = op->getLoc();
      Value input = op->getOperand(0);
      Type inputType = input.getType();

      // If the input is already f32, no upcast needed for accumulation.
      if (isF32Type(inputType)) continue;

      // If the input is in lower precision, upcast to f32 before reduction.
      if (isLowerPrecisionThanF32(inputType)) {
        Type originalInputType = inputType;

        builder.setInsertionPoint(op);
        Type f32InputType = getF32Type(inputType, ctx);
        auto upcastOp = createCastOp(builder, loc, input, f32InputType,
                                     originalInputType);

        // Replace the operand with the upcasted value.
        op->setOperand(0, upcastOp.getResult());

        // Update the result type of the reduction to f32.
        Type originalResultType = op->getResult(0).getType();
        Type f32ResultType = getF32Type(originalResultType, ctx);
        op->getResult(0).setType(f32ResultType);

        // After the reduction, cast the result back to the original type.
        builder.setInsertionPointAfter(op);
        auto downcastOp = createCastOp(builder, loc, op->getResult(0),
                                       originalResultType, f32ResultType);

        // Replace all uses of the reduction result (except the new cast)
        // with the downcasted result.
        op->getResult(0).replaceAllUsesExcept(downcastOp.getResult(),
                                              downcastOp);

        // Annotate the reduction op for debugging.
        op->setAttr("jules.mixed_precision.accumulation",
                    StringAttr::get(ctx, "f32"));
        op->setAttr("jules.mixed_precision.original_type",
                    TypeAttr::get(originalResultType));
      }
    }
  }

  // ── Phase 2: MatMul Op Processing ────────────────────────────────────────
  //
  // For each MatMulOp:
  //   1. Insert CastOp before lhs to downcast to target precision
  //   2. Insert CastOp before rhs to downcast to target precision
  //   3. Keep the matmul accumulation in f32 (or original type)
  //   4. Insert CastOp after the result to cast back to original type

  void processMatMulOps(func::FuncOp funcOp, PrecisionLevel targetLevel,
                        MLIRContext *ctx) {
    SmallVector<MatMulOp, 4> matmulOps;
    funcOp.walk([&](MatMulOp op) {
      matmulOps.push_back(op);
    });

    for (auto matmulOp : matmulOps) {
      OpBuilder builder(ctx);
      Location loc = matmulOp.getLoc();

      Value lhs = matmulOp.getLhs();
      Value rhs = matmulOp.getRhs();
      Type lhsType = lhs.getType();
      Type rhsType = rhs.getType();

      // Only apply mixed precision to f32 (or higher) matmul inputs.
      // Skip if the inputs are already in lower precision.
      bool lhsNeedsDowncast = !isLowerPrecisionThanF32(lhsType);
      bool rhsNeedsDowncast = !isLowerPrecisionThanF32(rhsType);

      if (!lhsNeedsDowncast && !rhsNeedsDowncast) continue;

      // FIX (SLOW 18): Skip mixed precision for small matmuls where the
      // CastOp overhead (extra tensor allocation + memory bandwidth) exceeds
      // the compute savings from lower precision. Heuristic: skip if the
      // matmul is < 64K FLOPs (the cast ops cost ~2 tensor roundtrips which
      // dominates for small matrices).
      {
        auto lhsTensor = lhsType.dyn_cast<RankedTensorType>();
        auto rhsTensor = rhsType.dyn_cast<RankedTensorType>();
        if (lhsTensor && rhsTensor) {
          int64_t M = lhsTensor.getDimSize(lhsTensor.getRank() - 2);
          int64_t K = lhsTensor.getDimSize(lhsTensor.getRank() - 1);
          int64_t N = rhsTensor.getDimSize(rhsTensor.getRank() - 1);
          // Handle dynamic dims: assume they're large enough
          if (M != ShapedType::kDynamic &&
              K != ShapedType::kDynamic &&
              N != ShapedType::kDynamic) {
            int64_t flops = 2 * M * K * N;
            if (flops < 65536) continue;  // Skip small matmuls
          }
        }
      }

      // Record the original result type for the final upcast.
      Type originalResultType = matmulOp.getResult().getType();

      // ── Downcast lhs ───────────────────────────────────────────────────
      Value newLhs = lhs;
      if (lhsNeedsDowncast) {
        builder.setInsertionPoint(matmulOp);
        Type targetLhsType = getTargetComputeType(lhsType, targetLevel, ctx);
        auto lhsCastOp = createCastOp(builder, loc, lhs, targetLhsType,
                                      lhsType);
        newLhs = lhsCastOp.getResult();
      }

      // ── Downcast rhs ───────────────────────────────────────────────────
      Value newRhs = rhs;
      if (rhsNeedsDowncast) {
        builder.setInsertionPoint(matmulOp);
        Type targetRhsType = getTargetComputeType(rhsType, targetLevel, ctx);
        auto rhsCastOp = createCastOp(builder, loc, rhs, targetRhsType,
                                      rhsType);
        newRhs = rhsCastOp.getResult();
      }

      // ── Update matmul operands and result type ─────────────────────────
      matmulOp->setOperand(0, newLhs);
      matmulOp->setOperand(1, newRhs);

      // The matmul result type should be in the target precision for the
      // computation, but we want accumulation in f32. We set the result
      // type to f32 to model accumulation in f32.
      Type f32ResultType = getF32Type(originalResultType, ctx);
      matmulOp.getResult().setType(f32ResultType);

      // FIX for P1 #9: Set matmul precision attributes so the StableHLO
      // lowering can emit the correct precision_config on dot_general.
      // Previously the pass just set the result type to f32 without
      // telling the matmul op what precision it should use, causing
      // incorrect lowering. Now we annotate the compute and accumulation
      // precision explicitly.
      matmulOp->setAttr("compute_precision",
                        StringAttr::get(ctx, targetComputePrecision_));
      matmulOp->setAttr("accumulation_precision",
                        StringAttr::get(ctx, kAccumulationPrecision));

      // ── Upcast result back to original type ────────────────────────────
      builder.setInsertionPointAfter(matmulOp);
      auto resultUpcastOp = createCastOp(builder, loc,
                                          matmulOp.getResult(),
                                          originalResultType,
                                          f32ResultType);

      // Replace all uses of the matmul result (except the upcast)
      // with the upcasted result.
      matmulOp.getResult().replaceAllUsesExcept(resultUpcastOp.getResult(),
                                                 resultUpcastOp);

      // ── Annotate for debugging ─────────────────────────────────────────
      matmulOp->setAttr("jules.mixed_precision.compute",
                        StringAttr::get(ctx, targetComputePrecision_));
      matmulOp->setAttr("jules.mixed_precision.accumulation",
                        StringAttr::get(ctx, kAccumulationPrecision));
      matmulOp->setAttr("jules.mixed_precision.original_type",
                        TypeAttr::get(originalResultType));
    }
  }

  // ── Phase 3: Activation Op Processing ────────────────────────────────────
  //
  // For activation ops (ReluOp, SigmoidOp, TanhOp):
  //   - If the input is already in lower precision, keep it
  //   - If the input is f32 and the op follows a matmul, keep f32 (already
  //     accumulated)
  //   - For standalone activations, downcast input, compute in lower
  //     precision, cast output back

  void processActivationOps(func::FuncOp funcOp, PrecisionLevel targetLevel,
                            MLIRContext *ctx) {
    SmallVector<Operation*, 8> activationOps;
    funcOp.walk([&](Operation *op) {
      if (isa<ReluOp>(op) || isa<SigmoidOp>(op) || isa<TanhOp>(op)) {
        activationOps.push_back(op);
      }
    });

    for (auto *op : activationOps) {
      OpBuilder builder(ctx);
      Location loc = op->getLoc();
      Value input = op->getOperand(0);
      Type inputType = input.getType();

      // If input is already in lower precision, nothing to do.
      if (isLowerPrecisionThanF32(inputType)) continue;

      // If input is f32 and the activation follows a matmul, keep it in f32
      // since it's already been accumulated in f32.
      if (isF32Type(inputType) && followsMatmul(op)) continue;

      // For standalone f32 activations, downcast to target precision.
      if (isF32Type(inputType)) {
        Type originalInputType = inputType;
        Type targetType = getTargetComputeType(inputType, targetLevel, ctx);

        // Downcast the input.
        builder.setInsertionPoint(op);
        auto downcastOp = createCastOp(builder, loc, input, targetType,
                                       originalInputType);
        op->setOperand(0, downcastOp.getResult());

        // Update the result type of the activation to the target precision.
        Type originalResultType = op->getResult(0).getType();
        Type targetResultType = getTargetComputeType(originalResultType,
                                                      targetLevel, ctx);
        op->getResult(0).setType(targetResultType);

        // Cast the output back to the original type.
        builder.setInsertionPointAfter(op);
        auto upcastOp = createCastOp(builder, loc, op->getResult(0),
                                     originalResultType, targetResultType);
        op->getResult(0).replaceAllUsesExcept(upcastOp.getResult(),
                                              upcastOp);

        // Annotate for debugging.
        op->setAttr("jules.mixed_precision.compute",
                    StringAttr::get(ctx, targetComputePrecision_));
        op->setAttr("jules.mixed_precision.original_type",
                    TypeAttr::get(originalResultType));
      }
    }
  }

  // ── Phase 4: Elementwise Binary Op Processing ────────────────────────────
  //
  // For elementwise binary ops (AddOp, SubOp, MulOp, DivOp):
  //   - If both operands are already in lower precision, stay in lower precision
  //   - If mixed, cast the higher-precision operand down
  //   - If both are f32 and neither follows a matmul, downcast both

  void processElementwiseBinaryOps(func::FuncOp funcOp,
                                    PrecisionLevel targetLevel,
                                    MLIRContext *ctx) {
    SmallVector<Operation*, 8> binaryOps;
    funcOp.walk([&](Operation *op) {
      if (isa<AddOp>(op) || isa<SubOp>(op) || isa<MulOp>(op) || isa<DivOp>(op)) {
        binaryOps.push_back(op);
      }
    });

    for (auto *op : binaryOps) {
      OpBuilder builder(ctx);
      Location loc = op->getLoc();

      Value lhs = op->getOperand(0);
      Value rhs = op->getOperand(1);
      Type lhsType = lhs.getType();
      Type rhsType = rhs.getType();

      bool lhsIsLower = isLowerPrecisionThanF32(lhsType);
      bool rhsIsLower = isLowerPrecisionThanF32(rhsType);
      bool lhsIsF32 = isF32Type(lhsType);
      bool rhsIsF32 = isF32Type(rhsType);

      // If both operands are already in lower precision, nothing to do.
      if (lhsIsLower && rhsIsLower) continue;

      // If neither operand is a floating-point type we care about, skip.
      if (!lhsIsF32 && !rhsIsF32 && !lhsIsLower && !rhsIsLower) continue;

      Type originalResultType = op->getResult(0).getType();

      // ── Mixed precision: one operand is lower, one is f32 ──────────────
      // Cast the f32 operand down to match the lower-precision operand.
      if (lhsIsLower && rhsIsF32) {
        builder.setInsertionPoint(op);
        // Determine the target type by matching lhs's element type.
        Type targetType = getTargetComputeType(rhsType, targetLevel, ctx);
        auto rhsCastOp = createCastOp(builder, loc, rhs, targetType, rhsType);
        op->setOperand(1, rhsCastOp.getResult());

        // Update the result type.
        Type targetResultType = getTargetComputeType(originalResultType,
                                                      targetLevel, ctx);
        op->getResult(0).setType(targetResultType);

        // Cast output back to original type.
        builder.setInsertionPointAfter(op);
        auto upcastOp = createCastOp(builder, loc, op->getResult(0),
                                     originalResultType, targetResultType);
        op->getResult(0).replaceAllUsesExcept(upcastOp.getResult(),
                                              upcastOp);

        op->setAttr("jules.mixed_precision.original_type",
                    TypeAttr::get(originalResultType));
        continue;
      }

      if (rhsIsLower && lhsIsF32) {
        builder.setInsertionPoint(op);
        Type targetType = getTargetComputeType(lhsType, targetLevel, ctx);
        auto lhsCastOp = createCastOp(builder, loc, lhs, targetType, lhsType);
        op->setOperand(0, lhsCastOp.getResult());

        Type targetResultType = getTargetComputeType(originalResultType,
                                                      targetLevel, ctx);
        op->getResult(0).setType(targetResultType);

        builder.setInsertionPointAfter(op);
        auto upcastOp = createCastOp(builder, loc, op->getResult(0),
                                     originalResultType, targetResultType);
        op->getResult(0).replaceAllUsesExcept(upcastOp.getResult(),
                                              upcastOp);

        op->setAttr("jules.mixed_precision.original_type",
                    TypeAttr::get(originalResultType));
        continue;
      }

      // ── Both operands are f32 ──────────────────────────────────────────
      // Downcast both to the target precision, unless one follows a matmul
      // (in which case we keep f32 for correctness of accumulation).
      if (lhsIsF32 && rhsIsF32) {
        // Check if either operand comes from a matmul. If so, we should
        // keep f32 to preserve the accumulated result.
        bool lhsFromMatmul = false;
        bool rhsFromMatmul = false;
        if (auto *defOp = lhs.getDefiningOp()) {
          if (isa<MatMulOp>(defOp) ||
              (isa<CastOp>(defOp) && followsMatmul(defOp))) {
            lhsFromMatmul = true;
          }
        }
        if (auto *defOp = rhs.getDefiningOp()) {
          if (isa<MatMulOp>(defOp) ||
              (isa<CastOp>(defOp) && followsMatmul(defOp))) {
            rhsFromMatmul = true;
          }
        }

        if (lhsFromMatmul || rhsFromMatmul) {
          // At least one operand is from a matmul — keep f32 for correctness.
          continue;
        }

        // Both are standalone f32 — downcast both.
        Type targetLhsType = getTargetComputeType(lhsType, targetLevel, ctx);
        Type targetRhsType = getTargetComputeType(rhsType, targetLevel, ctx);

        builder.setInsertionPoint(op);
        auto lhsCastOp = createCastOp(builder, loc, lhs, targetLhsType,
                                      lhsType);
        auto rhsCastOp = createCastOp(builder, loc, rhs, targetRhsType,
                                      rhsType);
        op->setOperand(0, lhsCastOp.getResult());
        op->setOperand(1, rhsCastOp.getResult());

        Type targetResultType = getTargetComputeType(originalResultType,
                                                      targetLevel, ctx);
        op->getResult(0).setType(targetResultType);

        builder.setInsertionPointAfter(op);
        auto upcastOp = createCastOp(builder, loc, op->getResult(0),
                                     originalResultType, targetResultType);
        op->getResult(0).replaceAllUsesExcept(upcastOp.getResult(),
                                              upcastOp);

        op->setAttr("jules.mixed_precision.compute",
                    StringAttr::get(ctx, targetComputePrecision_));
        op->setAttr("jules.mixed_precision.original_type",
                    TypeAttr::get(originalResultType));
      }
    }
  }
};

} // anonymous namespace

// ── Pass Factory Functions ──────────────────────────────────────────────────

std::unique_ptr<Pass> jules::createMixedPrecisionPass() {
  return std::make_unique<MixedPrecisionPass>();
}

std::unique_ptr<Pass> jules::createMixedPrecisionPass(std::string targetPrecision) {
  // The custom constructor was removed because Option<T> is non-copyable.
  // Set the option via the pass manager instead.
  auto pass = std::make_unique<MixedPrecisionPass>();
  // Note: targetPrecision is ignored; the pass option must be set via the
  // pass manager when adding the pass.
  (void)targetPrecision;
  return pass;
}

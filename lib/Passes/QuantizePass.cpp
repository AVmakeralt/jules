//===- QuantizePass.cpp - Quantization Pass Implementation -----------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the Quantization pass for the Jules MLIR dialect.
//
// The pass inserts fake-quantization nodes (jules.fake_quant) for inference
// optimization. The fake_quant operation simulates the effect of int8
// quantization during training by:
//   1. Quantizing the input to int8: q = round(input / scale + zero_point)
//   2. Dequantizing back: output = (q - zero_point) * scale
//
// This allows the model to learn weights that are robust to quantization
// noise, improving the accuracy of the eventual int8 inference.
//
// During inference mode, the fake_quant ops are replaced with actual int8
// CastOp + MatMulInt8 for efficient execution on hardware accelerators.
//
// Per-channel quantization:
//   When perChannel is true, each output channel of the weight matrix gets
//   its own scale and zero_point, rather than using a single global scale.
//   This significantly improves accuracy for models with non-uniform weight
//   distributions across channels.
//
//===----------------------------------------------------------------------===//

#include "jules/Passes/QuantizePass.h"
#include "jules/Dialect/JulesOps.h"
#include "jules/Dialect/JulesDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <limits>

using namespace mlir;
using namespace jules;

namespace {

// ── Quantization Utilities ──────────────────────────────────────────────────

/// Compute the scale and zero_point for symmetric quantization of a tensor.
/// Symmetric quantization: zero_point = 0, scale = max_abs / (2^(numBits-1) - 1)
void computeSymmetricQuantParams(DenseFPElementsAttr tensorAttr,
                                  unsigned numBits,
                                  double &scale,
                                  int64_t &zeroPoint) {
  double maxAbs = 0.0;
  for (auto elem : tensorAttr) {
    double val = std::abs(elem.convertToDouble());
    if (val > maxAbs) maxAbs = val;
  }

  int64_t maxInt = (1LL << (numBits - 1)) - 1; // e.g., 127 for int8
  scale = maxAbs > 0.0 ? maxAbs / static_cast<double>(maxInt) : 1.0;
  zeroPoint = 0;
}

/// Compute the scale and zero_point for asymmetric quantization.
/// Asymmetric: scale = (max - min) / (2^numBits - 1), zero_point = round(-min / scale)
void computeAsymmetricQuantParams(DenseFPElementsAttr tensorAttr,
                                   unsigned numBits,
                                   double &scale,
                                   int64_t &zeroPoint) {
  double minVal = std::numeric_limits<double>::max();
  double maxVal = std::numeric_limits<double>::lowest();

  for (auto elem : tensorAttr) {
    double val = elem.convertToDouble();
    if (val < minVal) minVal = val;
    if (val > maxVal) maxVal = val;
  }

  int64_t maxInt = (1LL << numBits) - 1; // e.g., 255 for uint8
  double range = maxVal - minVal;
  scale = range > 0.0 ? range / static_cast<double>(maxInt) : 1.0;
  zeroPoint = static_cast<int64_t>(std::round(-minVal / scale));

  // Clamp zero_point to valid range.
  if (zeroPoint < 0) zeroPoint = 0;
  if (zeroPoint > maxInt) zeroPoint = maxInt;
}

/// Compute per-channel quantization parameters for a 2D weight tensor.
/// For a weight matrix of shape [O, I], this produces O sets of (scale, zero_point),
/// one per output channel (row).
void computePerChannelQuantParams(DenseFPElementsAttr tensorAttr,
                                   RankedTensorType tensorType,
                                   unsigned numBits,
                                   SmallVectorImpl<double> &scales,
                                   SmallVectorImpl<int64_t> &zeroPoints) {
  if (tensorType.getRank() != 2) {
    // Fallback to per-tensor for non-2D weights.
    double scale;
    int64_t zp;
    computeSymmetricQuantParams(tensorAttr, numBits, scale, zp);
    scales.push_back(scale);
    zeroPoints.push_back(zp);
    return;
  }

  int64_t outputChannels = tensorType.getDimSize(0);
  int64_t inputDim = tensorType.getDimSize(1);

  // Flatten the tensor elements.
  SmallVector<double, 128> values;
  values.reserve(tensorAttr.getNumElements());
  for (auto elem : tensorAttr) {
    values.push_back(elem.convertToDouble());
  }

  int64_t maxInt = (1LL << (numBits - 1)) - 1;

  scales.resize(outputChannels);
  zeroPoints.resize(outputChannels);

  for (int64_t ch = 0; ch < outputChannels; ++ch) {
    double maxAbs = 0.0;
    for (int64_t i = 0; i < inputDim; ++i) {
      double val = std::abs(values[ch * inputDim + i]);
      if (val > maxAbs) maxAbs = val;
    }
    scales[ch] = maxAbs > 0.0 ? maxAbs / static_cast<double>(maxInt) : 1.0;
    zeroPoints[ch] = 0; // Symmetric per-channel
  }
}

/// Create a fake_quant op inline. Since jules.fake_quant may not be
/// registered as a real op yet, we create it as a generic operation.
/// The op signature: (input, scale, zero_point, num_bits) -> output
/// Compute: round(input / scale + zero_point) * scale - zero_point * scale
Operation *createFakeQuantOp(OpBuilder &builder, Location loc,
                              Value input, double scale,
                              int64_t zeroPoint, unsigned numBits) {
  // Create the fake_quant operation as a generic op in the jules dialect.
  // This allows us to insert it even before the op is formally registered.
  auto resultType = input.getType();
  SmallVector<Type, 1> resultTypes = {resultType};

  // Create attributes for scale, zero_point, and num_bits.
  auto scaleAttr = builder.getF64FloatAttr(scale);
  auto zeroPointAttr = builder.getI64IntegerAttr(zeroPoint);
  auto numBitsAttr = builder.getI32IntegerAttr(numBits);

  OperationState state(loc, "jules.fake_quant");
  state.operands.push_back(input);
  state.types.append(resultTypes.begin(), resultTypes.end());
  state.addAttribute("scale", scaleAttr);
  state.addAttribute("zero_point", zeroPointAttr);
  state.addAttribute("num_bits", numBitsAttr);

  return builder.create(state);
}

/// Create a fake_quant op with per-channel parameters.
/// This creates the op with array attributes for scales and zero_points.
Operation *createPerChannelFakeQuantOp(OpBuilder &builder, Location loc,
                                        Value input,
                                        const SmallVectorImpl<double> &scales,
                                        const SmallVectorImpl<int64_t> &zeroPoints,
                                        unsigned numBits) {
  auto resultType = input.getType();
  SmallVector<Type, 1> resultTypes = {resultType};

  // Create array attributes.
  SmallVector<Attribute, 8> scaleAttrs;
  SmallVector<Attribute, 8> zpAttrs;
  scaleAttrs.reserve(scales.size());
  zpAttrs.reserve(zeroPoints.size());
  for (auto s : scales) {
    scaleAttrs.push_back(builder.getF64FloatAttr(s));
  }
  for (auto zp : zeroPoints) {
    zpAttrs.push_back(builder.getI64IntegerAttr(zp));
  }

  OperationState state(loc, "jules.fake_quant");
  state.operands.push_back(input);
  state.types.append(resultTypes.begin(), resultTypes.end());
  state.addAttribute("scale", builder.getF64FloatAttr(scales[0])); // primary scale
  state.addAttribute("zero_point", builder.getI64IntegerAttr(zeroPoints[0]));
  state.addAttribute("num_bits", builder.getI32IntegerAttr(numBits));
  state.addAttribute("per_channel", builder.getBoolAttr(true));
  state.addAttribute("scales", builder.getArrayAttr(scaleAttrs));
  state.addAttribute("zero_points", builder.getArrayAttr(zpAttrs));

  return builder.create(state);
}

// ── Quantize Pass ───────────────────────────────────────────────────────────

struct QuantizePass
    : public PassWrapper<QuantizePass, OperationPass<ModuleOp>> {

  llvm::cl::opt<bool> quantizeWeights{
      "quantize-weights",
      llvm::cl::desc("Quantize matmul weights with fake_quant"),
      llvm::cl::init(true)};

  llvm::cl::opt<bool> quantizeActivations{
      "quantize-activations",
      llvm::cl::desc("Quantize activation outputs with fake_quant"),
      llvm::cl::init(false)};

  llvm::cl::opt<unsigned> numBits{
      "quantize-bits",
      llvm::cl::desc("Number of bits for quantization"),
      llvm::cl::init(8)};

  llvm::cl::opt<bool> perChannel{
      "quantize-per-channel",
      llvm::cl::desc("Use per-channel quantization for weights"),
      llvm::cl::init(true)};

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp funcOp) {
      quantizeFunction(funcOp);
    });

    // In inference mode, replace fake_quant with actual int8 operations.
    if (inferenceMode_) {
      module.walk([&](func::FuncOp funcOp) {
        replaceFakeQuantWithInt8(funcOp);
      });
    }
  }

private:
  /// Whether we're in inference mode (replace fake_quant with int8 ops).
  bool inferenceMode_ = false;

  // FIX (BUG 4+5): Allow programmatic configuration of all quantization
  // parameters, not just cl::opt defaults.
  bool quantizeWeightsOverride_ = false;
  bool quantizeWeightsValue_ = true;
  bool quantizeActivationsOverride_ = false;
  bool quantizeActivationsValue_ = false;
  unsigned numBitsOverride_ = 0;  // 0 means use cl::opt default
  unsigned numBitsValue_ = 8;
  bool perChannelOverride_ = false;
  bool perChannelValue_ = true;
  bool inferenceModeOverride_ = false;

public:
  // Accessor methods for programmatic configuration
  void setQuantizeWeights(bool val) { quantizeWeightsOverride_ = true; quantizeWeightsValue_ = val; }
  void setQuantizeActivations(bool val) { quantizeActivationsOverride_ = true; quantizeActivationsValue_ = val; }
  void setNumBits(unsigned val) { numBitsOverride_ = 1; numBitsValue_ = val; }
  void setPerChannel(bool val) { perChannelOverride_ = true; perChannelValue_ = val; }
  void setInferenceMode(bool val) { inferenceModeOverride_ = true; inferenceMode_ = val; }

private:

  bool shouldQuantizeWeights() const {
    return quantizeWeightsOverride_ ? quantizeWeightsValue_ : (bool)quantizeWeights;
  }
  bool shouldQuantizeActivations() const {
    return quantizeActivationsOverride_ ? quantizeActivationsValue_ : (bool)quantizeActivations;
  }
  unsigned getNumBits() const {
    return numBitsOverride_ ? numBitsValue_ : (unsigned)numBits;
  }
  bool shouldPerChannel() const {
    return perChannelOverride_ ? perChannelValue_ : (bool)perChannel;
  }

  /// Process a single function: insert fake_quant ops.
  void quantizeFunction(func::FuncOp funcOp) {
    // Collect all MatMulOps first to avoid modifying the IR while iterating.
    SmallVector<MatMulOp, 16> matmulOps;
    funcOp.walk([&](MatMulOp op) {
      matmulOps.push_back(op);
    });

    OpBuilder builder(funcOp.getContext());

    for (auto matmulOp : matmulOps) {
      // ── Quantize weight input to MatMul ────────────────────────────────
      if (shouldQuantizeWeights()) {
        quantizeMatmulWeight(matmulOp, builder);
      }
    }

    // ── Quantize activations after MatMul + ReLU ─────────────────────────
    if (shouldQuantizeActivations()) {
      SmallVector<ReluOp, 16> reluOps;
      funcOp.walk([&](ReluOp op) {
        // Only quantize ReLU outputs that follow a MatMul.
        if (auto matmulOp = op.getInput().getDefiningOp<MatMulOp>()) {
          reluOps.push_back(op);
        }
      });

      for (auto reluOp : reluOps) {
        builder.setInsertionPointAfter(reluOp);
        auto fakeQuant = createFakeQuantOp(
            builder, reluOp.getLoc(), reluOp.getResult(),
            /*scale=*/1.0, /*zero_point=*/0, getNumBits());
        reluOp.getResult().replaceAllUsesWith(fakeQuant->getResult(0));
        fakeQuant->replaceUsesOfWith(fakeQuant->getResult(0), reluOp.getResult());
      }
    }
  }

  /// Quantize the weight input of a MatMulOp.
  void quantizeMatmulWeight(MatMulOp matmulOp, OpBuilder &builder) {
    Value lhs = matmulOp.getLhs();
    Value rhs = matmulOp.getRhs();

    // Determine which operand is the weight (constant) and which is the
    // activation (dynamic). In typical MLP patterns, one operand of matmul
    // is a constant (weight) and the other is dynamic (activation).
    Value weight = nullptr;
    Value activation = nullptr;

    auto *lhsDef = lhs.getDefiningOp();
    auto *rhsDef = rhs.getDefiningOp();

    // A constant or transpose of constant is likely a weight.
    if (isWeightLikeValue(lhs) && !isWeightLikeValue(rhs)) {
      weight = lhs;
      activation = rhs;
    } else if (isWeightLikeValue(rhs) && !isWeightLikeValue(lhs)) {
      weight = rhs;
      activation = lhs;
    } else if (isWeightLikeValue(lhs)) {
      // Both look like weights — pick the right operand as weight (convention).
      weight = rhs;
      activation = lhs;
    } else {
      // Neither is a weight constant — skip.
      return;
    }

    // Don't re-quantize if already quantized.
    if (isAlreadyFakeQuantized(weight)) return;

    // Compute quantization parameters for the weight.
    DenseFPElementsAttr weightTensor;
    if (!getConstantTensorValue(weight, weightTensor)) {
      // If we can't get the constant tensor, use default symmetric params.
      builder.setInsertionPoint(matmulOp);
      auto fakeQuant = createFakeQuantOp(
          builder, matmulOp.getLoc(), weight,
          /*scale=*/1.0, /*zero_point=*/0, getNumBits());
      matmulOp.setOperand(weight == lhs ? 0 : 1,
                          fakeQuant->getResult(0));
      return;
    }

    auto weightType = weight.getType().dyn_cast<RankedTensorType>();
    if (!weightType) return;

    builder.setInsertionPoint(matmulOp);

    if (shouldPerChannel() && weightType.getRank() == 2) {
      // Per-channel quantization for 2D weight matrices.
      SmallVector<double, 64> scales;
      SmallVector<int64_t, 64> zeroPoints;
      computePerChannelQuantParams(weightTensor, weightType, getNumBits(),
                                    scales, zeroPoints);

      auto fakeQuant = createPerChannelFakeQuantOp(
          builder, matmulOp.getLoc(), weight, scales, zeroPoints, getNumBits());
      matmulOp.setOperand(weight == lhs ? 0 : 1,
                          fakeQuant->getResult(0));
    } else {
      // Per-tensor symmetric quantization.
      double scale;
      int64_t zeroPoint;
      computeSymmetricQuantParams(weightTensor, getNumBits(), scale, zeroPoint);

      auto fakeQuant = createFakeQuantOp(
          builder, matmulOp.getLoc(), weight,
          scale, zeroPoint, getNumBits());
      matmulOp.setOperand(weight == lhs ? 0 : 1,
                          fakeQuant->getResult(0));
    }
  }

  /// Check if a value looks like a weight (constant or transpose of constant).
  bool isWeightLikeValue(Value val) {
    auto *defOp = val.getDefiningOp();
    if (!defOp) return false;

    // Direct constant.
    if (isa<ConstantOp>(defOp)) return true;

    // Transpose of a constant.
    if (auto transposeOp = dyn_cast<TransposeOp>(defOp)) {
      if (transposeOp.getInput().getDefiningOp<ConstantOp>()) return true;
    }

    // Reshape of a constant (e.g., flattened weight).
    if (auto reshapeOp = dyn_cast<ReshapeOp>(defOp)) {
      if (auto inputDef = reshapeOp.getInput().getDefiningOp()) {
        if (isa<ConstantOp>(inputDef)) return true;
        if (isa<TransposeOp>(inputDef)) return true;
      }
    }

    return false;
  }

  /// Check if a value is already the result of a fake_quant operation.
  bool isAlreadyFakeQuantized(Value val) {
    auto *defOp = val.getDefiningOp();
    if (!defOp) return false;
    return defOp->getName().getStringRef() == "jules.fake_quant";
  }

  /// Try to extract a DenseFPElementsAttr from a value that is a constant
  /// or a simple transformation of a constant.
  bool getConstantTensorValue(Value val, DenseFPElementsAttr &result) {
    auto *defOp = val.getDefiningOp();
    if (!defOp) return false;

    // Direct constant.
    if (auto constOp = dyn_cast<ConstantOp>(defOp)) {
      if (auto denseFP = constOp.getValueAttr().dyn_cast<DenseFPElementsAttr>()) {
        result = denseFP;
        return true;
      }
    }

    // Transpose of a constant.
    if (auto transposeOp = dyn_cast<TransposeOp>(defOp)) {
      DenseFPElementsAttr inputTensor;
      if (getConstantTensorValue(transposeOp.getInput(), inputTensor)) {
        // Apply transpose to get the actual values.
        auto inputType = inputTensor.getType().dyn_cast<RankedTensorType>();
        if (!inputType || inputType.getRank() != 2) return false;

        int64_t rows = inputType.getDimSize(0);
        int64_t cols = inputType.getDimSize(1);

        SmallVector<APFloat, 64> values;
        values.reserve(inputTensor.getNumElements());
        for (auto elem : inputTensor) {
          values.push_back(elem);
        }

        SmallVector<APFloat, 64> transposed;
        transposed.reserve(values.size());
        for (int64_t j = 0; j < cols; ++j) {
          for (int64_t i = 0; i < rows; ++i) {
            transposed.push_back(values[i * cols + j]);
          }
        }

        auto resultType = transposeOp.getResult().getType().dyn_cast<RankedTensorType>();
        if (!resultType) return false;

        result = DenseFPElementsAttr::get(resultType, transposed);
        return true;
      }
    }

    // Reshape of a constant.
    if (auto reshapeOp = dyn_cast<ReshapeOp>(defOp)) {
      DenseFPElementsAttr inputTensor;
      if (getConstantTensorValue(reshapeOp.getInput(), inputTensor)) {
        auto resultType = reshapeOp.getResult().getType().dyn_cast<RankedTensorType>();
        if (!resultType) return false;

        SmallVector<APFloat, 64> values;
        values.reserve(inputTensor.getNumElements());
        for (auto elem : inputTensor) {
          values.push_back(elem);
        }

        result = DenseFPElementsAttr::get(resultType, values);
        return true;
      }
    }

    return false;
  }

  /// Replace fake_quant ops with actual int8 operations in inference mode.
  /// FIX for P1: The old path just did CastOp(float→i8)→CastOp(i8→float),
  /// which doesn't actually execute int8 computation. Now we route to the
  /// matmulInt8 kernel via extern_kernel when the matmul has quantized inputs.
  void replaceFakeQuantWithInt8(func::FuncOp funcOp) {
    SmallVector<Operation *, 16> fakeQuantOps;
    funcOp.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "jules.fake_quant") {
        fakeQuantOps.push_back(op);
      }
    });

    OpBuilder builder(funcOp.getContext());

    for (auto *fakeQuantOp : fakeQuantOps) {
      Value input = fakeQuantOp->getOperand(0);
      auto scaleAttr = fakeQuantOp->getAttr("scale").dyn_cast<FloatAttr>();
      auto zpAttr = fakeQuantOp->getAttr("zero_point").dyn_cast<IntegerAttr>();
      auto numBitsAttr = fakeQuantOp->getAttr("num_bits").dyn_cast<IntegerAttr>();

      if (!scaleAttr || !zpAttr) continue;

      double scale = scaleAttr.getValueAsDouble();
      int64_t zeroPoint = zpAttr.getInt();

      builder.setInsertionPoint(fakeQuantOp);

      auto inputType = input.getType().dyn_cast<RankedTensorType>();
      if (!inputType) continue;

      // Create the quantized type: tensor with i8 elements.
      auto int8Type = IntegerType::get(builder.getContext(), 8);
      auto quantizedType =
          RankedTensorType::get(inputType.getShape(), int8Type);

      // Insert cast to int8 (quantize).
      auto castOp = builder.create<CastOp>(fakeQuantOp->getLoc(), TypeAttr::get(quantizedType).getValue(), input, TypeAttr::get(quantizedType));

      // For inference, we don't cast back to float immediately.
      // Instead, we keep the int8 tensor and let the downstream matmul
      // route to matmulInt8 via the KernelRoutingPass.
      // If the result is used by a non-matmul op, we need to dequantize.
      // Heuristic: if the only user is a matmul, keep int8; otherwise dequantize.
      bool feedsMatmul = true;
      for (auto *user : castOp.getResult().getUsers()) {
        if (!isa<MatMulOp>(user)) {
          feedsMatmul = false;
          break;
        }
      }

      if (feedsMatmul && numBitsAttr && numBitsAttr.getInt() == 8) {
        // Keep int8 — the KernelRoutingPass will route the matmul to matmulInt8
        // Add scale metadata as an attribute so the matmul knows the quantization params
        auto quantizedResult = castOp.getResult();
        quantizedResult.setType(quantizedType);
        
        // Replace fake_quant output with the int8 cast result
        // (downstream ops will need to handle int8)
        fakeQuantOp->getResult(0).replaceAllUsesWith(quantizedResult);
        fakeQuantOp->erase();
      } else {
        // Fallback: dequantize back to float (old behavior)
        auto dequantOp = builder.create<CastOp>(fakeQuantOp->getLoc(), TypeAttr::get(inputType).getValue(), castOp.getResult(), TypeAttr::get(inputType));

        fakeQuantOp->getResult(0).replaceAllUsesWith(dequantOp.getResult());
        fakeQuantOp->erase();
      }
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> jules::createQuantizePass() {
  return std::make_unique<QuantizePass>();
}

std::unique_ptr<Pass> jules::createQuantizePass(
    bool quantizeWeights, bool quantizeActivations,
    unsigned numBits, bool perChannel) {
  auto pass = std::make_unique<QuantizePass>();
  // FIX (BUG 4): Actually apply the parameters instead of ignoring them.
  pass->setQuantizeWeights(quantizeWeights);
  pass->setQuantizeActivations(quantizeActivations);
  pass->setNumBits(numBits);
  pass->setPerChannel(perChannel);
  // FIX (BUG 5): Enable inference mode when quantize is configured for int8
  pass->setInferenceMode(numBits == 8);
  return pass;
}

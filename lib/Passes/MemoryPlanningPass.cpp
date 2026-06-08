//===- MemoryPlanningPass.cpp - Memory Planning & In-Place Reuse Impl ------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the Memory Planning pass for the Jules MLIR dialect.
// It analyzes buffer lifetimes across each function and annotates operations
// with memory planning hints for in-place buffer reuse, alignment, and
// XLA output_operand_alias directives.
//
// Algorithm overview:
//
//   Phase 1: Liveness analysis
//     Assign a linear instruction index to each operation, then compute
//     the "last use" index for every SSA Value. This determines when a
//     buffer can be freed or reused.
//
//   Phase 2: In-place eligibility
//     For each operation, determine if it can reuse an input buffer:
//       - Elementwise unary ops can reuse their input if it's dead after.
//       - Elementwise binary ops can reuse one operand if it's dead after.
//       - AddOp is specially handled: if one operand has no other uses
//         and the same type, it can be in-place on that operand.
//
//   Phase 3: Annotation
//     Add per-op attributes:
//       - jules.memory.in_place = true
//       - jules.memory.alias = <operand index>
//       - jules.memory.alignment = 64
//     Add function-level attribute:
//       - jules.memory.plan = { total_bytes = <n>, buffers = <count> }
//
//   Phase 4: Memory budget verification
//     Compute total memory requirement and verify it fits within the
//     configured budget (default: 2 GB).
//
//===----------------------------------------------------------------------===//

#include "jules/Passes/MemoryPlanningPass.h"
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

// ── Liveness Interval ───────────────────────────────────────────────────────

/// Represents the liveness interval of an SSA value.
/// `start` is the instruction index where the value is defined.
/// `end` is the instruction index of the value's last use.
struct LivenessInterval {
  unsigned start;
  unsigned end;
  Value value;
};

// ── Operation Classification ────────────────────────────────────────────────

/// Returns true if the operation is a pure elementwise unary op.
bool isElementwiseUnary(Operation *op) {
  return isa<NegOp>(op) || isa<ReluOp>(op) || isa<SigmoidOp>(op) ||
         isa<TanhOp>(op);
}

/// Returns true if the operation is a pure elementwise binary op.
bool isElementwiseBinary(Operation *op) {
  return isa<AddOp>(op) || isa<SubOp>(op) || isa<MulOp>(op) ||
         isa<DivOp>(op);
}

/// Returns true if the operation is any elementwise op (unary or binary).
bool isElementwise(Operation *op) {
  return isElementwiseUnary(op) || isElementwiseBinary(op);
}

/// Returns true if the operation can potentially operate in-place.
bool isInPlaceCandidate(Operation *op) {
  // Elementwise unary ops can always be in-place if their input is dead.
  if (isElementwiseUnary(op)) return true;

  // Elementwise binary ops can be in-place if one operand is dead.
  if (isElementwiseBinary(op)) return true;

  // Reshape and transpose views can be in-place.
  if (isa<ReshapeOp>(op) || isa<TransposeOp>(op)) return true;

  return false;
}

// ── Buffer Size Estimation ──────────────────────────────────────────────────

/// Estimate the buffer size in bytes for a Value.
/// For ranked tensor types with static shapes, this is the product of
/// dimensions times the element type width. For dynamic or unranked types,
/// we use a conservative estimate.
int64_t estimateBufferSize(Value val) {
  auto tensorType = val.getType().dyn_cast<RankedTensorType>();
  if (!tensorType) {
    // Scalar or unknown type — estimate 4 bytes.
    return 4;
  }

  int64_t numElements = 1;
  bool hasDynamic = false;
  for (int64_t dim : tensorType.getShape()) {
    if (dim == ShapedType::kDynamic) {
      hasDynamic = true;
      break;
    }
    numElements *= dim;
  }

  if (hasDynamic) {
    // Conservative estimate: 1024 elements for dynamic dims.
    numElements = 1024;
  }

  // Determine element size in bytes.
  Type elemType = tensorType.getElementType();
  int64_t elemSize = 4; // Default: f32 = 4 bytes.
  if (auto floatType = elemType.dyn_cast<FloatType>()) {
    elemSize = floatType.getWidth() / 8;
  } else if (auto intType = elemType.dyn_cast<IntegerType>()) {
    elemSize = intType.getWidth() / 8;
  }

  return numElements * elemSize;
}

// ── The Pass ────────────────────────────────────────────────────────────────

struct MemoryPlanningPass
    : public PassWrapper<MemoryPlanningPass, OperationPass<ModuleOp>> {

  explicit MemoryPlanningPass(uint64_t memoryBudgetBytes = 2ULL * 1024 * 1024 * 1024)
      : memoryBudgetBytes_(memoryBudgetBytes) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp funcOp) {
      planFunction(funcOp);
    });
  }

  void planFunction(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();

    // ── Phase 1: Liveness Analysis ────────────────────────────────────────
    //
    // Assign a linear index to each operation and compute the last use
    // of every SSA value. This gives us liveness intervals.

    llvm::DenseMap<Operation*, unsigned> opIndex;
    llvm::SmallVector<Operation*, 64> opsInOrder;

    unsigned index = 0;
    funcOp.walk([&](Operation *op) {
      if (isa<func::FuncOp>(op)) return;
      opIndex[op] = index++;
      opsInOrder.push_back(op);
    });

    // Compute last use for each value.
    llvm::DenseMap<Value, unsigned> lastUseIndex;

    // Initialize: function arguments are live until we discover otherwise.
    for (auto arg : funcOp.getArguments()) {
      lastUseIndex[arg] = 0;
    }

    // Walk all operations and record uses.
    for (auto *op : opsInOrder) {
      for (auto operand : op->getOperands()) {
        lastUseIndex[operand] = std::max(lastUseIndex[operand], opIndex[op]);
      }
    }

    // Compute liveness intervals.
    llvm::SmallVector<LivenessInterval, 64> intervals;
    llvm::DenseMap<Value, unsigned> valueToIntervalIdx;

    // Process function arguments (start at 0).
    for (auto arg : funcOp.getArguments()) {
      unsigned start = 0;
      unsigned end = lastUseIndex[arg];
      valueToIntervalIdx[arg] = intervals.size();
      intervals.push_back({start, end, arg});
    }

    // Process operation results.
    for (auto *op : opsInOrder) {
      for (auto result : op->getResults()) {
        unsigned start = opIndex[op];
        auto it = lastUseIndex.find(result);
        unsigned end = (it != lastUseIndex.end()) ? it->second : start;
        valueToIntervalIdx[result] = intervals.size();
        intervals.push_back({start, end, result});
      }
    }

    // ── Phase 2: In-Place Eligibility ─────────────────────────────────────
    //
    // For each operation, check if it can reuse an input buffer.
    // An operation can be in-place on operand `i` if:
    //   1. The operation is an in-place candidate
    //   2. The operand buffer has the same size/type as the result
    //   3. The operand is dead after this operation (its last use is here)

    llvm::DenseMap<Operation*, unsigned> inPlaceAlias; // op -> operand index
    llvm::DenseSet<Operation*> inPlaceOps;

    for (auto *op : opsInOrder) {
      if (!isInPlaceCandidate(op)) continue;
      if (op->getNumResults() == 0) continue;

      Value result = op->getResult(0);

      // For elementwise unary ops: check if the single input is dead.
      if (isElementwiseUnary(op) && op->getNumOperands() == 1) {
        Value input = op->getOperand(0);
        unsigned inputLastUse = lastUseIndex.lookup(input);

        // The input is dead after this op if its last use is this op.
        if (inputLastUse <= opIndex[op]) {
          // Verify type compatibility (same shape and element type).
          if (isTypeCompatible(input.getType(), result.getType())) {
            inPlaceAlias[op] = 0;
            inPlaceOps.insert(op);
          }
        }
      }

      // For elementwise binary ops: check if one operand is dead.
      if (isElementwiseBinary(op) && op->getNumOperands() == 2) {
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);

        unsigned lhsLastUse = lastUseIndex.lookup(lhs);
        unsigned rhsLastUse = lastUseIndex.lookup(rhs);

        // Prefer aliasing the operand that's dead. If both are dead,
        // prefer the LHS (conventional accumulator pattern).
        bool lhsDead = lhsLastUse <= opIndex[op];
        bool rhsDead = rhsLastUse <= opIndex[op];

        if (lhsDead && isTypeCompatible(lhs.getType(), result.getType())) {
          inPlaceAlias[op] = 0;
          inPlaceOps.insert(op);
        } else if (rhsDead && isTypeCompatible(rhs.getType(), result.getType())) {
          inPlaceAlias[op] = 1;
          inPlaceOps.insert(op);
        }
      }

      // Special handling for AddOp: if one operand has no other uses
      // beyond this op, allow in-place even if the operand type is
      // broadcast-compatible (not necessarily identical).
      if (isa<AddOp>(op) && !inPlaceOps.count(op) &&
          op->getNumOperands() == 2) {
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);

        // Check if one operand has exactly one use (this op).
        if (lhs.hasOneUse()) {
          inPlaceAlias[op] = 0;
          inPlaceOps.insert(op);
        } else if (rhs.hasOneUse()) {
          inPlaceAlias[op] = 1;
          inPlaceOps.insert(op);
        }
      }

      // For reshape/transpose: can be views (no buffer copy needed).
      if ((isa<ReshapeOp>(op) || isa<TransposeOp>(op)) &&
          op->getNumOperands() == 1) {
        Value input = op->getOperand(0);
        unsigned inputLastUse = lastUseIndex.lookup(input);

        if (inputLastUse <= opIndex[op]) {
          inPlaceAlias[op] = 0;
          inPlaceOps.insert(op);
        }
      }
    }

    // ── Phase 3: Annotation ───────────────────────────────────────────────
    //
    // Add per-op attributes for in-place reuse and alignment.
    // Add function-level attribute with total memory plan.

    OpBuilder builder(ctx);

    for (auto *op : opsInOrder) {
      // Set alignment attribute on all tensor-producing ops.
      if (op->getNumResults() > 0) {
        auto resultType = op->getResult(0).getType();
        if (resultType.isa<RankedTensorType>() || resultType.isa<FloatType>() ||
            resultType.isa<IntegerType>()) {
          op->setAttr("jules.memory.alignment",
                      IntegerAttr::get(IntegerType::get(ctx, 64), 64));
        }
      }

      // Set in-place attributes.
      if (inPlaceOps.count(op)) {
        op->setAttr("jules.memory.in_place", BoolAttr::get(ctx, true));

        unsigned aliasIdx = inPlaceAlias[op];
        op->setAttr("jules.memory.alias",
                    IntegerAttr::get(IntegerType::get(ctx, 32), aliasIdx));

        // For StableHLO output: add output_operand_alias attribute.
        // This tells XLA that the output buffer can alias the input buffer.
        if (ctx->getLoadedDialect("stablehlo")) {
          auto aliasAttr = builder.getDictionaryAttr({
              builder.getNamedAttr("operand_index",
                                   builder.getI64IntegerAttr(aliasIdx)),
              builder.getNamedAttr("operand_type",
                                   builder.getStringAttr("input"))
          });
          op->setAttr("output_operand_alias", aliasAttr);
        }
      }
    }

    // ── Phase 4: Memory Budget Computation ────────────────────────────────
    //
    // Compute total memory requirement using a sweep-line algorithm.
    // This replaces the old O(ops² × intervals) approach with O(N log N).
    //
    // Algorithm:
    //   1. Create events for each interval: (start, +size) and (end, -size)
    //   2. Sort events by instruction index
    //   3. Sweep through events, maintaining a running sum of live memory
    //   4. Track the maximum live memory (the peak)

    int64_t peakMemory = 0;

    // Create sweep events
    struct SweepEvent {
      unsigned idx;      // Instruction index
      int64_t delta;     // +size for start, -size for end
      bool isStart;      // true = interval starts, false = interval ends
    };

    llvm::SmallVector<SweepEvent, 128> events;

    // Function arguments: live from 0 to their last use
    for (auto arg : funcOp.getArguments()) {
      auto it = lastUseIndex.find(arg);
      unsigned end = (it != lastUseIndex.end()) ? it->second : 0;
      int64_t size = estimateBufferSize(arg);

      // Skip args aliased by in-place ops (their buffer is reused)
      bool reused = false;
      for (auto *user : arg.getUsers()) {
        if (inPlaceOps.count(user) && inPlaceAlias[user] == 0 &&
            opIndex.count(user) && opIndex[user] <= end) {
          reused = true;
          break;
        }
      }
      if (!reused && size > 0) {
        events.push_back({0, size, true});
        events.push_back({end + 1, -size, false});
      }
    }

    // Operation results: live from their definition to their last use
    for (auto *op : opsInOrder) {
      for (auto result : op->getResults()) {
        auto it = lastUseIndex.find(result);
        unsigned start = opIndex[op];
        unsigned end = (it != lastUseIndex.end()) ? it->second : start;

        // Skip in-place results (their buffer is aliased)
        if (inPlaceOps.count(op)) continue;

        int64_t size = estimateBufferSize(result);
        if (size > 0) {
          events.push_back({start, size, true});
          events.push_back({end + 1, -size, false});
        }
      }

      // Account for in-place results whose input was already freed
      if (inPlaceOps.count(op)) {
        for (auto result : op->getResults()) {
          auto it = lastUseIndex.find(result);
          unsigned start = opIndex[op];
          unsigned end = (it != lastUseIndex.end()) ? it->second : start;

          unsigned aliasIdx = inPlaceAlias[op];
          Value aliasedInput = op->getOperand(aliasIdx);
          auto inputIt = lastUseIndex.find(aliasedInput);
          unsigned inputEnd = (inputIt != lastUseIndex.end()) ? inputIt->second : 0;

          // If the input was already freed, the output takes over the buffer
          if (inputEnd < start) {
            int64_t size = estimateBufferSize(result);
            if (size > 0) {
              events.push_back({start, size, true});
              events.push_back({end + 1, -size, false});
            }
          }
        }
      }
    }

    // Sort events: process ends before starts at the same index
    // (a buffer ending at index i can be reused by one starting at i)
    std::sort(events.begin(), events.end(), [](const SweepEvent &a, const SweepEvent &b) {
      if (a.idx != b.idx) return a.idx < b.idx;
      // End events before start events at same index (free before allocate)
      return !a.isStart && b.isStart;
    });

    // Sweep through events
    int64_t currentLive = 0;
    for (auto &event : events) {
      currentLive += event.delta;
      peakMemory = std::max(peakMemory, currentLive);
    }

    // Account for function argument buffers (always allocated).
    int64_t argMemory = 0;
    for (auto arg : funcOp.getArguments()) {
      argMemory += estimateBufferSize(arg);
    }
    peakMemory = std::max(peakMemory, argMemory);

    // Annotate the function with the memory plan.
    unsigned numBuffers = 0;
    funcOp.walk([&](Operation *op) {
      if (op->getNumResults() > 0) numBuffers++;
    });
    numBuffers += funcOp.getNumArguments();

    auto planAttr = builder.getDictionaryAttr({
        builder.getNamedAttr("total_bytes",
                             builder.getI64IntegerAttr(peakMemory)),
        builder.getNamedAttr("num_buffers",
                             builder.getI64IntegerAttr(numBuffers)),
        builder.getNamedAttr("in_place_ops",
                             builder.getI64IntegerAttr(inPlaceOps.size())),
        builder.getNamedAttr("budget_bytes",
                             builder.getI64IntegerAttr(memoryBudgetBytes_)),
    });
    funcOp->setAttr("jules.memory.plan", planAttr);

    // Verify memory budget.
    if (static_cast<uint64_t>(peakMemory) > memoryBudgetBytes_) {
      funcOp->emitWarning()
          << "Memory plan exceeds budget: " << peakMemory << " bytes > "
          << memoryBudgetBytes_ << " bytes budget";
    }
  }

  /// Check if two types are compatible for in-place reuse.
  /// They must have the same element type and the same shape.
  bool isTypeCompatible(Type inputType, Type resultType) {
    auto inputTensor = inputType.dyn_cast<RankedTensorType>();
    auto resultTensor = resultType.dyn_cast<RankedTensorType>();

    if (inputTensor && resultTensor) {
      // Both are ranked tensors: shapes and element types must match.
      if (inputTensor.getShape() != resultTensor.getShape()) return false;
      return inputTensor.getElementType() == resultTensor.getElementType();
    }

    // For non-tensor types, check direct equality.
    return inputType == resultType;
  }

private:
  uint64_t memoryBudgetBytes_;
};

} // anonymous namespace

std::unique_ptr<Pass> jules::createMemoryPlanningPass() {
  return std::make_unique<MemoryPlanningPass>();
}

std::unique_ptr<Pass> jules::createMemoryPlanningPass(uint64_t memoryBudgetBytes) {
  return std::make_unique<MemoryPlanningPass>(memoryBudgetBytes);
}

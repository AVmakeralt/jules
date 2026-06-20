//===- ProducerConsumerFusionPass.cpp - Producer-Consumer Fusion Impl ------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the Producer-Consumer Fusion pass for the Jules
// MLIR dialect. It identifies fusible chains of operations and wraps
// them into fusion clusters.
//
// Algorithm overview:
//
//   Phase 1: Build def-use chains
//     Walk each function, collect all pure operations, and record
//     producer-consumer relationships.
//
//   Phase 2: Identify fusible groups using union-find
//     Two operations are in the same group if:
//       - Both are elementwise and the producer has exactly one user
//       - One is matmul and the other is a follow-on activation
//       - An elementwise chain feeds into a reduction
//
//   Phase 3: Create fusion regions
//     For each group, create a fusion wrapper (stablehlo.fusion or
//     jules.fusion), move all ops into the fusion region, and replace
//     original results with the fusion outputs.
//
//===----------------------------------------------------------------------===//

#include "jules/Passes/ProducerConsumerFusionPass.h"
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
#include "llvm/ADT/EquivalenceClasses.h"

using namespace mlir;
using namespace jules;

namespace {

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

/// Returns true if the operation is a reduction op.
bool isReduction(Operation *op) {
  return isa<SumOp>(op) || isa<MeanOp>(op);
}

/// Returns true if the operation is an activation function that can
/// be fused with a matmul.
bool isFusibleActivation(Operation *op) {
  return isa<ReluOp>(op) || isa<SigmoidOp>(op) || isa<TanhOp>(op);
}

// ── Union-Find Wrapper ──────────────────────────────────────────────────────

/// A wrapper around LLVM's EquivalenceClasses that provides a simpler
/// interface for our fusion grouping. Each Operation* is a member, and
/// we union two operations when they should be in the same fusion group.
class FusionGroupMap {
public:
  /// Union two operations into the same fusion group.
  void unionOps(Operation *a, Operation *b) {
    classes_.unionSets(a, b);
  }

  /// Check if two operations are in the same group.
  bool sameGroup(Operation *a, Operation *b) {
    return classes_.findLeader(a) == classes_.findLeader(b);
  }

  /// Iterate over all groups. Returns a vector of groups, where each
  /// group is a vector of Operation*.
  llvm::SmallVector<llvm::SmallVector<Operation*, 8>, 8>
  getGroups() const {
    llvm::SmallVector<llvm::SmallVector<Operation*, 8>, 8> result;

    // Collect all members by leader.
    llvm::DenseMap<Operation*, llvm::SmallVector<Operation*, 8>> leaderToMembers;
    for (auto it = classes_.begin(); it != classes_.end(); ++it) {
      if (!it->isLeader()) continue;
      Operation *leader = it->getData();
      // Collect all members of this leader's class
      for (auto mi = classes_.member_begin(it); mi != classes_.member_end(); ++mi) {
        leaderToMembers[leader].push_back(*mi);
      }
    }

    for (auto &[leader, members] : leaderToMembers) {
      if (members.size() >= 2) {
        result.push_back(std::move(members));
      }
    }

    return result;
  }

  /// Insert an operation into the group map (as a singleton).
  void insert(Operation *op) {
    classes_.insert(op);
  }

private:
  llvm::EquivalenceClasses<Operation*> classes_;
};

// ── Def-Use Chain Builder ───────────────────────────────────────────────────

/// Represents a producer-consumer edge in the def-use graph.
struct DefUseEdge {
  Operation *producer;
  Operation *consumer;
  unsigned producerResultIdx; // Which result of the producer is consumed.
  unsigned consumerOperandIdx; // Which operand of the consumer uses it.
};

/// Build the def-use edges for all operations in a function.
llvm::SmallVector<DefUseEdge, 32>
buildDefUseChains(func::FuncOp funcOp) {
  llvm::SmallVector<DefUseEdge, 32> edges;

  funcOp.walk([&](Operation *op) {
    // Skip the function itself and terminators.
    if (isa<func::FuncOp>(op) || op->hasTrait<OpTrait::IsTerminator>())
      return;

    for (unsigned operandIdx = 0; operandIdx < op->getNumOperands();
         ++operandIdx) {
      Value operand = op->getOperand(operandIdx);
      Operation *defOp = operand.getDefiningOp();
      if (!defOp) continue; // Block argument — no producer op.

      // Determine which result of the producer this operand comes from.
      unsigned resultIdx = 0;
      for (unsigned i = 0; i < defOp->getNumResults(); ++i) {
        if (defOp->getResult(i) == operand) {
          resultIdx = i;
          break;
        }
      }

      edges.push_back({defOp, op, resultIdx, operandIdx});
    }
  });

  return edges;
}

// ── Topological Sort Within a Group ─────────────────────────────────────────

/// Sort operations within a fusion group in topological order so that
/// producers come before consumers. Uses a simple Kahn's algorithm.
llvm::SmallVector<Operation*, 8>
topologicalSort(const llvm::SmallVector<Operation*, 8> &group) {
  llvm::DenseSet<Operation*> groupSet;
  for (auto *op : group) groupSet.insert(op);

  // Build in-degree map within the group.
  llvm::DenseMap<Operation*, unsigned> inDegree;
  llvm::DenseMap<Operation*, llvm::SmallVector<Operation*, 4>> successors;

  for (auto *op : group) {
    inDegree[op] = 0;
    successors[op] = {};
  }

  for (auto *op : group) {
    for (auto operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (defOp && groupSet.count(defOp)) {
        inDegree[op]++;
        successors[defOp].push_back(op);
      }
    }
  }

  // Kahn's algorithm.
  llvm::SmallVector<Operation*, 8> sorted;
  llvm::SmallVector<Operation*, 8> worklist;

  for (auto *op : group) {
    if (inDegree[op] == 0) worklist.push_back(op);
  }

  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    sorted.push_back(current);

    for (auto *succ : successors[current]) {
      inDegree[succ]--;
      if (inDegree[succ] == 0) worklist.push_back(succ);
    }
  }

  // If we couldn't sort all ops (cycle), fall back to original order.
  if (sorted.size() != group.size()) {
    return group;
  }

  return sorted;
}

// ── The Pass ────────────────────────────────────────────────────────────────

struct ProducerConsumerFusionPass
    : public PassWrapper<ProducerConsumerFusionPass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    // Check if the StableHLO dialect is loaded.
    bool hasStableHLO = ctx->getLoadedDialect("stablehlo") != nullptr;

    module.walk([&](func::FuncOp funcOp) {
      fuseInFunction(funcOp, hasStableHLO);
    });
  }

  void fuseInFunction(func::FuncOp funcOp, bool hasStableHLO) {
    MLIRContext *ctx = funcOp.getContext();

    // ── Phase 1: Build def-use chains ─────────────────────────────────────
    auto edges = buildDefUseChains(funcOp);

    // ── Phase 2: Identify fusible groups using union-find ─────────────────
    FusionGroupMap groupMap;

    // First, insert all fusible ops as singletons.
    funcOp.walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->hasTrait<OpTrait::IsTerminator>())
        return;
      if (isElementwise(op) || isa<MatMulOp>(op) || isReduction(op)) {
        groupMap.insert(op);
      }
    });

    // ── 2a: Fuse elementwise chains ───────────────────────────────────────
    //
    // A chain of elementwise ops is fusible when each intermediate op has
    // exactly one user. We walk the edges and union producer-consumer pairs
    // that satisfy this condition.

    for (auto &edge : edges) {
      if (!isElementwise(edge.producer) || !isElementwise(edge.consumer))
        continue;

      // The producer must have exactly one user for chain fusion.
      // This ensures we don't duplicate computation in the fused kernel.
      if (!edge.producer->getResult(edge.producerResultIdx).hasOneUse())
        continue;

      groupMap.unionOps(edge.producer, edge.consumer);
    }

    // ── 2b: Fuse matmul + activation patterns ─────────────────────────────
    //
    // MatMulOp followed by ReluOp/SigmoidOp/TanhOp is a common pattern
    // in neural networks. Fuse them when the matmul result has only one user.

    for (auto &edge : edges) {
      if (!isa<MatMulOp>(edge.producer)) continue;
      if (!isFusibleActivation(edge.consumer)) continue;

      // Only fuse if the matmul result is used solely by this activation.
      if (!edge.producer->getResult(edge.producerResultIdx).hasOneUse())
        continue;

      groupMap.unionOps(edge.producer, edge.consumer);
    }

    // ── 2c: Fuse elementwise + reduction patterns ─────────────────────────
    //
    // E.g. add → mul → sum: the reduction can consume the elementwise
    // chain without materializing intermediates. Fuse when the elementwise
    // op has exactly one user (the reduction) or is already in a group
    // that feeds the reduction.

    for (auto &edge : edges) {
      if (!isElementwise(edge.producer)) continue;
      if (!isReduction(edge.consumer)) continue;

      // The elementwise result must have exactly one use.
      if (!edge.producer->getResult(edge.producerResultIdx).hasOneUse())
        continue;

      groupMap.unionOps(edge.producer, edge.consumer);
    }

    // Also fuse elementwise chains that feed into reduction through
    // other elementwise ops already in the same group.
    for (auto &edge : edges) {
      if (!isReduction(edge.consumer)) continue;

      // Check if any operand of the reduction is an elementwise op
      // that's already grouped with other elementwise ops.
      if (!isElementwise(edge.producer)) continue;

      // The producer must be single-use so we don't duplicate work.
      if (!edge.producer->getResult(edge.producerResultIdx).hasOneUse())
        continue;

      groupMap.unionOps(edge.producer, edge.consumer);
    }

    // ── Phase 3: Create fusion regions ────────────────────────────────────
    auto groups = groupMap.getGroups();

    for (auto &group : groups) {
      // Sort the group topologically.
      auto sortedOps = topologicalSort(group);

      // Collect all external inputs (operands whose definitions are
      // outside the fusion group) and all external outputs (results
      // used outside the fusion group).
      llvm::SetVector<Value> externalInputs;
      llvm::DenseSet<Operation*> groupSet;
      for (auto *op : sortedOps) groupSet.insert(op);

      for (auto *op : sortedOps) {
        for (auto operand : op->getOperands()) {
          Operation *defOp = operand.getDefiningOp();
          if (!defOp || !groupSet.count(defOp)) {
            externalInputs.insert(operand);
          }
        }
      }

      // Collect results that are used outside the group.
      llvm::SmallVector<Value, 4> externalOutputs;
      llvm::SmallVector<Type, 4> externalOutputTypes;
      for (auto *op : sortedOps) {
        for (auto result : op->getResults()) {
          bool hasExternalUse = false;
          for (auto *user : result.getUsers()) {
            if (!groupSet.count(user)) {
              hasExternalUse = true;
              break;
            }
          }
          // Also check if the result is a function return operand.
          if (!hasExternalUse) {
            for (auto &use : result.getUses()) {
              if (isa<func::ReturnOp>(use.getOwner())) {
                hasExternalUse = true;
                break;
              }
            }
          }
          if (hasExternalUse) {
            externalOutputs.push_back(result);
            externalOutputTypes.push_back(result.getType());
          }
        }
      }

      // Skip groups with no external outputs (dead code).
      if (externalOutputs.empty()) continue;

      // Determine the insertion point: just before the first op in the
      // group (they are topologically sorted).
      OpBuilder builder(sortedOps.front());

      if (hasStableHLO) {
        // ── StableHLO fusion path ──────────────────────────────────────
        // NOTE: StableHLO support requires the stablehlo dialect to be
        // available at compile time. When not available, we fall through
        // to the Jules fusion path.
        createJulesFusion(builder, sortedOps, externalInputs,
                          externalOutputs, externalOutputTypes, groupSet);
      } else {
        // ── Jules fusion fallback path ─────────────────────────────────
        createJulesFusion(builder, sortedOps, externalInputs,
                          externalOutputs, externalOutputTypes, groupSet);
      }
    }
  }

  /// Create a stablehlo.fusion op wrapping the fusion group.
  /// NOTE: Disabled because the stablehlo dialect is not available in this
  /// build. The function body is wrapped in #if 0 to avoid compile errors.
#if 0
  void createStableHLOFusion(OpBuilder &builder,
                              llvm::SmallVector<Operation*, 8> &sortedOps,
                              const llvm::SetVector<Value> &externalInputs,
                              const llvm::SmallVector<Value, 4> &externalOutputs,
                              const llvm::SmallVector<Type, 4> &externalOutputTypes,
                              const llvm::DenseSet<Operation*> &groupSet) {
    MLIRContext *ctx = builder.getContext();
    Location loc = sortedOps.front()->getLoc();

    // Create the stablehlo.fusion op with a region.
    auto fusionOp = builder.create<stablehlo::FusionOp>(
        loc, externalOutputTypes, ValueRange{externalInputs.getArrayRef()});

    // Build the fusion region.
    Region &region = fusionOp.getBody();
    Block *block = builder.createBlock(&region);

    // Add block arguments corresponding to external inputs.
    for (auto input : externalInputs) {
      block->addArgument(input.getType(), input.getLoc());
    }

    // Map external input Values to block arguments.
    llvm::DenseMap<Value, Value> externalInputToBlockArg;
    for (size_t i = 0; i < externalInputs.size(); ++i) {
      externalInputToBlockArg[externalInputs[i]] = block->getArgument(i);
    }

    // Move ops into the fusion block, remapping external inputs.
    builder.setInsertionPointToStart(block);

    llvm::DenseMap<Value, Value> valueMap;

    // Initialize value map with external input remappings.
    for (size_t i = 0; i < externalInputs.size(); ++i) {
      valueMap[externalInputs[i]] = block->getArgument(i);
    }

    // Clone each operation into the fusion block with remapped operands.
    llvm::SmallVector<Value, 4> fusionResults;
    llvm::DenseSet<Value> externalOutputSet;
    for (auto &v : externalOutputs) externalOutputSet.insert(v);

    for (auto *op : sortedOps) {
      // Remap operands.
      llvm::SmallVector<Value, 4> remappedOperands;
      for (auto operand : op->getOperands()) {
        auto it = valueMap.find(operand);
        if (it != valueMap.end()) {
          remappedOperands.push_back(it->second);
        } else {
          remappedOperands.push_back(operand);
        }
      }

      // Clone the operation.
      auto *clonedOp = builder.clone(*op);

      // Replace operands with remapped values.
      for (unsigned i = 0; i < clonedOp->getNumOperands(); ++i) {
        auto it = valueMap.find(op->getOperand(i));
        if (it != valueMap.end()) {
          clonedOp->setOperand(i, it->second);
        }
      }

      // Record the mapping from original results to cloned results.
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        valueMap[op->getResult(i)] = clonedOp->getResult(i);

        // If this is an external output, record it.
        if (externalOutputSet.count(op->getResult(i))) {
          fusionResults.push_back(clonedOp->getResult(i));
        }
      }
    }

    // Create the terminator: stablehlo.return with the external outputs.
    builder.create<stablehlo::ReturnOp>(loc, fusionResults);

    // Replace external uses of the original outputs with the fusion results.
    for (size_t i = 0; i < externalOutputs.size(); ++i) {
      externalOutputs[i].replaceAllUsesWith(fusionOp.getResult(i));
    }

    // Erase the original operations (in reverse order to avoid dangling refs).
    for (auto it = sortedOps.rbegin(); it != sortedOps.rend(); ++it) {
      (*it)->erase();
    }
  }
#endif // 0 (createStableHLOFusion disabled)

  /// Create a jules.fusion wrapper when StableHLO isn't available.
  /// This uses a jules.fusion_group attribute to mark the cluster and
  /// wraps the group in a jules.fusion composite operation.
  void createJulesFusion(OpBuilder &builder,
                          llvm::SmallVector<Operation*, 8> &sortedOps,
                          const llvm::SetVector<Value> &externalInputs,
                          const llvm::SmallVector<Value, 4> &externalOutputs,
                          const llvm::SmallVector<Type, 4> &externalOutputTypes,
                          const llvm::DenseSet<Operation*> &groupSet) {
    MLIRContext *ctx = builder.getContext();
    Location loc = sortedOps.front()->getLoc();

    // Assign a unique fusion group ID.
    static unsigned fusionGroupCounter = 0;
    unsigned groupId = fusionGroupCounter++;
    auto groupIdAttr = IntegerAttr::get(IntegerType::get(ctx, 32), groupId);

    // Annotate each operation in the group with jules.fusion_group.
    for (auto *op : sortedOps) {
      op->setAttr("jules.fusion_group", groupIdAttr);
    }

    // Determine a fusion kind string for debugging/annotation.
    std::string fusionKind = "elementwise";
    for (auto *op : sortedOps) {
      if (isa<MatMulOp>(op)) { fusionKind = "matmul_activation"; break; }
      if (isReduction(op)) { fusionKind = "elementwise_reduction"; break; }
    }

    auto fusionKindAttr = StringAttr::get(ctx, fusionKind);
    for (auto *op : sortedOps) {
      op->setAttr("jules.fusion_kind", fusionKindAttr);
    }

    // Create a region-based fusion wrapper using a generic operation.
    // We use a jules.fusion op as a container. Since we may not have
    // a formal jules.fusion op defined in the dialect, we create an
    // MLIR generic operation with the right properties.
    auto fusionOp = builder.create(
        loc,
        builder.getStringAttr("jules.fusion"),
        ValueRange{externalInputs.getArrayRef()},
        externalOutputTypes,
        llvm::SmallVector<NamedAttribute, 4>{
            {builder.getStringAttr("fusion_group"), groupIdAttr},
            {builder.getStringAttr("fusion_kind"), fusionKindAttr},
        });

    // Build the fusion region.
    Region &region = fusionOp->getRegion(0);
    Block *block = builder.createBlock(&region);

    // Add block arguments for external inputs.
    for (auto input : externalInputs) {
      block->addArgument(input.getType(), input.getLoc());
    }

    // Map external inputs to block arguments.
    llvm::DenseMap<Value, Value> valueMap;
    for (size_t i = 0; i < externalInputs.size(); ++i) {
      valueMap[externalInputs[i]] = block->getArgument(i);
    }

    builder.setInsertionPointToStart(block);

    // Clone each operation into the fusion block.
    llvm::SmallVector<Value, 4> fusionResults;
    llvm::DenseSet<Value> externalOutputSet;
    for (auto &v : externalOutputs) externalOutputSet.insert(v);

    for (auto *op : sortedOps) {
      auto *clonedOp = builder.clone(*op);

      // Remap operands.
      for (unsigned i = 0; i < clonedOp->getNumOperands(); ++i) {
        auto it = valueMap.find(op->getOperand(i));
        if (it != valueMap.end()) {
          clonedOp->setOperand(i, it->second);
        }
      }

      // Record result mappings.
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        valueMap[op->getResult(i)] = clonedOp->getResult(i);

        if (externalOutputSet.count(op->getResult(i))) {
          fusionResults.push_back(clonedOp->getResult(i));
        }
      }

      // Preserve the fusion group attributes on the cloned op.
      clonedOp->setAttr("jules.fusion_group", groupIdAttr);
      clonedOp->setAttr("jules.fusion_kind", fusionKindAttr);
    }

    // Create a jules.return terminator with the fusion results.
    builder.create(
        loc, builder.getStringAttr("jules.return"),
        fusionResults, TypeRange{},
        llvm::SmallVector<NamedAttribute, 0>{});

    // Replace external uses with fusion outputs.
    for (size_t i = 0; i < externalOutputs.size(); ++i) {
      externalOutputs[i].replaceAllUsesWith(fusionOp->getResult(i));
    }

    // Erase original operations in reverse order.
    for (auto it = sortedOps.rbegin(); it != sortedOps.rend(); ++it) {
      (*it)->erase();
    }
  }

private:
  bool hasStableHLO_ = false;
};

} // anonymous namespace

std::unique_ptr<Pass> jules::createProducerConsumerFusionPass() {
  return std::make_unique<ProducerConsumerFusionPass>();
}

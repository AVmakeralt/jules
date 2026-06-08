//===- SymbolDCEPass.cpp - Symbol Dead Code Elimination Implementation -----===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the SymbolDCE pass. It performs two levels of
// dead code elimination:
//
//   Module-level: Remove functions that are never called
//   Function-level: Remove operations whose results have no users
//
//===----------------------------------------------------------------------===//

#include "jules/Passes/SymbolDCEPass.h"
#include "jules/Dialect/JulesOps.h"
#include "jules/Dialect/JulesDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace jules;

namespace {

struct SymbolDCEPass : public PassWrapper<SymbolDCEPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // ── Phase 1: Module-level DCE ─────────────────────────────────────────
    //
    // Find all functions that are reachable from the entry point and
    // remove the rest.

    // Determine the entry point.
    StringRef entryPoint = "main";

    // Collect all function names.
    llvm::SmallSetVector<StringRef, 32> reachableFunctions;

    // The entry point is always reachable.
    reachableFunctions.insert(entryPoint);

    // BFS/DFS to find all transitively called functions.
    llvm::SmallVector<StringRef, 8> worklist;
    worklist.push_back(entryPoint);

    while (!worklist.empty()) {
      StringRef fnName = worklist.pop_back_val();

      // Find the function definition.
      auto funcOp = module.lookupSymbol<func::FuncOp>(fnName);
      if (!funcOp) continue;

      // Walk the function body and find all call operations.
      funcOp.walk([&](func::CallOp callOp) {
        StringRef callee = callOp.getCallee();
        if (!reachableFunctions.count(callee)) {
          reachableFunctions.insert(callee);
          worklist.push_back(callee);
        }
      });
    }

    // Remove all unreachable functions.
    llvm::SmallVector<func::FuncOp, 8> deadFunctions;
    module.walk([&](func::FuncOp funcOp) {
      if (!reachableFunctions.count(funcOp.getName())) {
        deadFunctions.push_back(funcOp);
      }
    });

    for (auto funcOp : deadFunctions) {
      funcOp.erase();
    }

    // ── Phase 2: Function-level DCE ───────────────────────────────────────
    //
    // Within each surviving function, remove operations whose results
    // have no users. This is the standard dead code elimination.

    module.walk([&](func::FuncOp funcOp) {
      eliminateDeadCode(funcOp);
    });
  }

  /// Eliminate dead code within a function.
  /// Iteratively removes operations with no users until no more can be removed.
  void eliminateDeadCode(func::FuncOp funcOp) {
    bool changed = true;
    while (changed) {
      changed = false;

      // Collect operations with no result users.
      llvm::SmallVector<Operation*, 32> deadOps;
      funcOp.walk([&](Operation *op) {
        // Don't remove terminators, function ops, or ops with side effects.
        if (op->hasTrait<OpTrait::IsTerminator>()) return;
        if (isa<func::FuncOp>(op)) return;
        if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) return;

        // Check if all results are unused.
        bool allUnused = true;
        for (auto result : op->getResults()) {
          if (!result.use_empty()) {
            allUnused = false;
            break;
          }
        }

        if (allUnused && op->getNumResults() > 0) {
          deadOps.push_back(op);
        }
      });

      // Remove dead operations.
      for (auto *op : deadOps) {
        op->erase();
        changed = true;
      }
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> jules::createSymbolDCEPass() {
  return std::make_unique<SymbolDCEPass>();
}

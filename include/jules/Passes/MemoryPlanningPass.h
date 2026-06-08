//===- MemoryPlanningPass.h - Memory Planning & In-Place Reuse Pass --------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Memory Planning pass for the Jules MLIR dialect.
// The pass analyzes buffer lifetimes across each function and annotates
// operations with memory planning hints to enable in-place buffer reuse,
// cache-line alignment, and XLA output_operand_alias directives.
//
// Analysis phases:
//
//   1. Liveness analysis: for each SSA Value, compute its last use point
//      to determine when its backing buffer can be reclaimed or reused.
//
//   2. In-place eligibility: determine which operations can safely reuse
//      an input buffer for their output:
//        - Elementwise unary ops (neg, relu, sigmoid, tanh) can always
//          reuse their input buffer if it has no other live users.
//        - Elementwise binary ops (add, sub, mul, div) can reuse one
//          operand's buffer if that operand is dead after the op.
//
//   3. Memory budget: compute total planned memory and verify it fits
//      within limits, annotating the function with jules.memory.plan.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_MEMORY_PLANNING_PASS_H
#define JULES_PASSES_MEMORY_PLANNING_PASS_H

#include <memory>
#include <cstdint>

namespace mlir {
class Pass;
}

namespace jules {

/// Create the Memory Planning pass.
/// This pass analyzes buffer lifetimes and annotates operations with
/// in-place reuse hints, alignment requirements, and alias information.
std::unique_ptr<mlir::Pass> createMemoryPlanningPass();

/// Create the Memory Planning pass with a specific memory budget limit
/// in bytes. Operations whose planned memory exceeds this limit will
/// trigger a warning.
std::unique_ptr<mlir::Pass> createMemoryPlanningPass(uint64_t memoryBudgetBytes);

} // namespace jules

#endif // JULES_PASSES_MEMORY_PLANNING_PASS_H

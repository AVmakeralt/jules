//===- SCCPPass.h - Sparse Conditional Constant Propagation Pass ------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Sparse Conditional Constant Propagation (SCCP) pass
// for the Jules MLIR dialect. SCCP is a global (interprocedural) optimization
// that discovers constant values and propagates them throughout the program.
//
// Unlike simple constant folding, SCCP can discover constants that arise from
// conditional branches where only one branch is taken. It uses a worklist
// algorithm to propagate lattice values (top/unknown, constant, bottom/overdefined)
// through the program's dataflow graph.
//
// In the context of the two-tier AOT/JIT system, SCCP is critical for
// whole-program graph collapsing: it discovers that computations like
// get_mask() = [1.0, 2.0, 3.0, 4.0] * 2.0 are constant, enabling the
// compiler to replace the entire function with a hardcoded constant.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_SCCP_PASS_H
#define JULES_PASSES_SCCP_PASS_H

#include <memory>

namespace mlir {
class Pass;
}

namespace jules {

/// Create the Sparse Conditional Constant Propagation pass.
/// This pass discovers and propagates constant values throughout the
/// MLIR module, enabling whole-program graph collapsing.
std::unique_ptr<mlir::Pass> createSCCPPass();

} // namespace jules

#endif // JULES_PASSES_SCCP_PASS_H

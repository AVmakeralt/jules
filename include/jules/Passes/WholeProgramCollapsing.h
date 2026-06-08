//===- WholeProgramCollapsing.h - Whole-Program Graph Collapsing Pass -------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the whole-program graph collapsing pass that performs
// aggressive interprocedural constant folding, partial evaluation, and
// dead code elimination across the entire module.
//
// Unlike the per-function GraphCollapsingPass, this pass operates at the
// module level and can collapse computations across function boundaries.
// If a function's output is derived entirely from constants, the entire
// function is dissolved and replaced with a hardcoded constant.
//
// Example:
//
//   get_mask : () -> [4]f32
//   get_mask () = [1.0, 2.0, 3.0, 4.0] * 2.0
//
//   main : () -> [4]f32
//   main () =
//     let M = get_mask()
//     in X * M
//
// After whole-program collapsing:
//   - get_mask is dissolved
//   - M is replaced with [2.0, 4.0, 6.0, 8.0] (computed at compile time)
//   - The multiply uses a hardcoded dense attribute constant
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_WHOLE_PROGRAM_COLLAPSING_H
#define JULES_PASSES_WHOLE_PROGRAM_COLLAPSING_H

#include <memory>

namespace mlir {
class Pass;
}

namespace jules {

/// Create the whole-program graph collapsing pass.
/// This pass operates at the module level and performs:
///   - Interprocedural constant propagation
///   - Partial evaluation of pure functions
///   - Inline-and-fold for constant arguments
///   - Dead function elimination after collapsing
std::unique_ptr<mlir::Pass> createWholeProgramCollapsingPass();

} // namespace jules

#endif // JULES_PASSES_WHOLE_PROGRAM_COLLAPSING_H

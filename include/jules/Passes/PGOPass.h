//===- PGOPass.h - PGO-Informed Optimization Pass --------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PGO-informed optimization pass. This pass uses
// runtime profiling data to:
//
//   - Replace dynamic shape dimensions with concrete sizes from profiles
//   - Insert shape specialization assumptions
//   - Guide memory layout decisions based on access patterns
//   - Select optimal kernel configurations based on observed shapes
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PASSES_PGOPASS_H
#define JULES_PASSES_PGOPASS_H

#include "jules/Profiler.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace jules {

/// Create the PGO-informed optimization pass.
/// \p profiler provides the runtime profile data.
std::unique_ptr<mlir::Pass> createPGOPass(const Profiler &profiler);

} // namespace jules

#endif // JULES_PASSES_PGOPASS_H

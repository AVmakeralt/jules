//===- BailoutHandler.cpp - Deoptimization / Bailout Implementation ---------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the bailout handler for graceful deoptimization
// when Tier 2 shape guards fail.
//
//===----------------------------------------------------------------------===//

#include "jules/BailoutHandler.h"
#include "jules/Diagnostics.h"
#include <algorithm>
#include <chrono>
#include <vector>

namespace jules {

BailoutHandler::BailoutHandler(DiagnosticsEngine &diag) : diag_(diag) {}

ShapeGuard BailoutHandler::createGuard(
    const ShapeSignature &specializedShapes) const {
  ShapeGuard guard;
  guard.expectedShapes = specializedShapes;
  return guard;
}

ExecutableHandle* BailoutHandler::executeWithGuard(
    const std::string &functionName,
    const ShapeSignature &actualShapes,
    const ShapeSignature &specializedShapes,
    ExecutableHandle *tier2Exec,
    ExecutableHandle *tier1Fallback,
    DispatchTable &dispatchTable) {

  // Create and check the shape guard.
  auto guard = createGuard(specializedShapes);

  if (guard.check(actualShapes)) {
    // Shapes match — execute Tier 2 (fast path).
    return tier2Exec;
  }

  // ── BAILOUT: Shapes don't match Tier 2 specialization ───────────────────
  //
  // This is the critical deoptimization path. We fall back to Tier 1
  // without crashing or freezing. The new shapes are recorded for
  // potential future Tier 2 compilation.

  BailoutEvent event;
  event.functionName = functionName;
  event.reason = BailoutReason::ShapeMismatch;
  event.expectedShapes = specializedShapes;
  event.actualShapes = actualShapes;

  auto now = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()).count();
  event.timestamp = static_cast<uint64_t>(ms);

  recordBailout(event);

  // Record the bailout in the dispatch table.
  dispatchTable.recordBailout(functionName, actualShapes);

  // Notify the cache policy about the new shape observation.
  if (newShapeCallback_) {
    // Use a dummy trace ID; in production, we'd correlate with the profiler.
    newShapeCallback_(functionName, actualShapes, 0);
  }

  diag_.info(SourceLocation{},
             "Bailout: '" + functionName + "' shape mismatch, falling back to Tier 1");

  return tier1Fallback;
}

void BailoutHandler::recordBailout(const BailoutEvent &event) {
  totalBailouts_++;
  functionBailouts_[event.functionName]++;

  // Add to the recent events ring buffer.
  recentBailouts_.push_back(event);
  if (recentBailouts_.size() > kMaxRecentBailouts) {
    recentBailouts_.erase(recentBailouts_.begin());
  }
}

uint64_t BailoutHandler::bailoutsForFunction(
    const std::string &functionName) const {
  auto it = functionBailouts_.find(functionName);
  return it != functionBailouts_.end() ? it->second : 0;
}

} // namespace jules

//===- BailoutHandler.h - Deoptimization / Bailout Mechanism ----------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the bailout handler for the two-tier AOT/JIT system.
//
// When the user passes a batch size of 64 instead of 32 (the shape that
// Tier 2 was compiled for), the system must gracefully fall back to the
// generic Tier 1 AOT binary without crashing or freezing.
//
// The bailout mechanism:
//
//   1. Before executing Tier 2, check if input shapes match the
//      specialized shapes that Tier 2 was compiled for.
//   2. If shapes match → execute Tier 2 (fast path).
//   3. If shapes don't match → bail out to Tier 1 (safe path).
//   4. Record the new shape observation for potential Tier 2 compilation.
//   5. If the new shape becomes stable, queue a new Tier 2 compilation.
//
// This is similar to how V8's TurboFan handles deoptimization: the JIT
// code includes shape guards, and if they fail, execution falls back to
// the interpreter (Tier 1 in our case).
//
//===----------------------------------------------------------------------===//

#ifndef JULES_BAILOUT_HANDLER_H
#define JULES_BAILOUT_HANDLER_H

#include "jules/DispatchTable.h"
#include "jules/Profiler.h"
#include "jules/Diagnostics.h"
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace jules {

// ── Shape Guard ─────────────────────────────────────────────────────────────

/// A shape guard: checks that runtime input shapes match the specialized
/// shapes that a Tier 2 executable was compiled for.
struct ShapeGuard {
  /// The expected shapes for each input.
  ShapeSignature expectedShapes;

  /// Check if the given actual shapes match the expected shapes.
  bool check(const ShapeSignature &actual) const {
    return expectedShapes == actual;
  }
};

// ── Bailout Reason ──────────────────────────────────────────────────────────

/// The reason a bailout occurred.
enum class BailoutReason {
  ShapeMismatch,     // Input shapes don't match Tier 2 specialization
  CompilationFailed, // Tier 2 compilation failed, fall back to Tier 1
  Tier2NotReady,     // Tier 2 not yet compiled for this shape
  ExplicitRequest,   // User explicitly requested bailout
};

// ── Bailout Event ───────────────────────────────────────────────────────────

/// A recorded bailout event.
struct BailoutEvent {
  /// The function that bailed out.
  std::string functionName;

  /// The reason for the bailout.
  BailoutReason reason;

  /// The shapes that were expected (Tier 2 specialization).
  ShapeSignature expectedShapes;

  /// The actual shapes that caused the bailout.
  ShapeSignature actualShapes;

  /// The trace ID.
  uint64_t traceId = 0;

  /// Timestamp (epoch milliseconds).
  uint64_t timestamp = 0;
};

// ── Bailout Handler ─────────────────────────────────────────────────────────

/// The bailout handler: manages shape guards and fallback execution.
///
/// When a Tier 2 executable is dispatched, the bailout handler wraps
/// it with a shape guard. If the guard fails, the handler:
///   1. Records the bailout event
///   2. Falls back to Tier 1 execution
///   3. Feeds the new shape observation to the profiler
///   4. Optionally triggers a new Tier 2 compilation for the new shape
///
class BailoutHandler {
public:
  explicit BailoutHandler(DiagnosticsEngine &diag);

  // ── Shape guard creation ─────────────────────────────────────────────────

  /// Create a shape guard for a Tier 2 executable.
  ShapeGuard createGuard(const ShapeSignature &specializedShapes) const;

  // ── Execution with guard ─────────────────────────────────────────────────

  /// Execute a function with shape guard protection.
  /// If the shapes match the Tier 2 specialization, execute Tier 2.
  /// Otherwise, bail out to Tier 1 and record the event.
  ///
  /// Returns the executable that should be used (Tier 2 or Tier 1 fallback).
  ExecutableHandle* executeWithGuard(
      const std::string &functionName,
      const ShapeSignature &actualShapes,
      const ShapeSignature &specializedShapes,
      ExecutableHandle *tier2Exec,
      ExecutableHandle *tier1Fallback,
      DispatchTable &dispatchTable);

  // ── Event recording ──────────────────────────────────────────────────────

  /// Record a bailout event.
  void recordBailout(const BailoutEvent &event);

  /// Get recent bailout events.
  const std::vector<BailoutEvent> &getRecentBailouts() const {
    return recentBailouts_;
  }

  /// Get the total number of bailouts.
  uint64_t totalBailouts() const { return totalBailouts_; }

  /// Get the number of bailouts for a specific function.
  uint64_t bailoutsForFunction(const std::string &functionName) const;

  // ── Shape observation feeding ────────────────────────────────────────────

  /// Set a callback to be invoked when a new shape is observed during bailout.
  /// This allows the cache policy to trigger a new Tier 2 compilation.
  using NewShapeCallback = std::function<void(
      const std::string &functionName,
      const ShapeSignature &newShapes,
      uint64_t traceId)>;
  void setNewShapeCallback(NewShapeCallback cb) { newShapeCallback_ = std::move(cb); }

private:
  DiagnosticsEngine    &diag_;

  /// Recent bailout events (ring buffer, last 64 events).
  static constexpr size_t kMaxRecentBailouts = 64;
  std::vector<BailoutEvent> recentBailouts_;

  /// Total bailouts across all functions.
  uint64_t totalBailouts_ = 0;

  /// Per-function bailout counts.
  std::unordered_map<std::string, uint64_t> functionBailouts_;

  /// Callback for new shape observations.
  NewShapeCallback newShapeCallback_;
};

} // namespace jules

#endif // JULES_BAILOUT_HANDLER_H

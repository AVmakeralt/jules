//===- Profiler.h - Runtime Profiler for JIT PGO ---------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the runtime profiler for Profile-Guided Optimization
// in the Jules JIT compiler.
//
// The profiler collects telemetry during trace execution:
//   - Actual tensor shapes flowing through each operation
//   - Execution frequency of each trace (hot path detection)
//   - Memory layout and access patterns
//   - Kernel execution timing
//
// This data drives the PGO recompilation loop:
//   1. First execution: compile with dynamic shapes (fast JIT, no specialization)
//   2. Warmup: profiler collects shape/frequency data
//   3. Hot path detected: background thread recompiles with static shapes
//   4. XLA generates hyper-optimized kernels for the specific shapes
//   5. Old executable is swapped out atomically
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PROFILER_H
#define JULES_PROFILER_H

#include "jules/Tracing.h"
#include "jules/AST.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace jules {

// ── Shape Profile ───────────────────────────────────────────────────────────

/// A concrete shape observed at runtime for a specific trace value.
struct ShapeProfile {
  std::vector<int64_t> shape;
  ScalarType::ScalarKind elementType;
  uint64_t observationCount = 0;

  /// Check if this shape is static (no dynamic dimensions).
  bool isStatic() const;

  /// Check if this shape matches a given TensorType.
  bool matches(const TensorType &type) const;
};

// ── Trace Profile ───────────────────────────────────────────────────────────

/// Profile data for a single compiled trace.
struct TraceProfile {
  /// Unique ID for this trace.
  uint64_t traceId = 0;

  /// How many times this trace has been executed.
  std::atomic<uint64_t> executionCount{0};

  /// Per-value shape observations.
  /// Maps TraceValueId -> observed shapes.
  std::unordered_map<TraceValueId, ShapeProfile> valueShapes;

  /// Total execution time across all invocations (nanoseconds).
  std::atomic<uint64_t> totalExecutionTimeNs{0};

  /// Whether this trace has been recompiled with PGO data.
  std::atomic<bool> pgoRecompiled{false};

  /// Whether the shapes in this trace are stable enough for PGO.
  bool isShapeStable() const;

  /// Threshold for considering a trace "hot" (eligible for PGO recompilation).
  static constexpr uint64_t kHotTraceThreshold = 10;

  /// Minimum number of shape observations before declaring stability.
  static constexpr uint64_t kMinObservations = 5;

  /// Is this trace hot enough for PGO recompilation?
  bool isHot() const {
    return executionCount.load() >= kHotTraceThreshold;
  }
};

// ── Kernel Timing ───────────────────────────────────────────────────────────

/// Timing data for a single kernel execution.
struct KernelTiming {
  std::string opName;         // e.g., "matmul", "add"
  std::vector<int64_t> inputShapes;
  uint64_t executionTimeNs;
  uint64_t memoryBytes;
};

// ── Profiler ────────────────────────────────────────────────────────────────

/// The runtime profiler: collects and analyzes PGO telemetry.
class Profiler {
public:
  Profiler();

  // ── Trace lifecycle ──────────────────────────────────────────────────────

  /// Start profiling a new trace execution.
  void beginTraceExecution(uint64_t traceId);

  /// End profiling for the current trace execution.
  void endTraceExecution(uint64_t traceId);

  // ── Shape observation ────────────────────────────────────────────────────

  /// Observe the concrete shape of a trace value at runtime.
  void observeShape(uint64_t traceId, TraceValueId valueId,
                    const std::vector<int64_t> &shape,
                    ScalarType::ScalarKind elementType);

  /// Get the observed shape profile for a value in a trace.
  std::optional<ShapeProfile> getShapeProfile(uint64_t traceId,
                                               TraceValueId valueId) const;

  // ── Kernel timing ────────────────────────────────────────────────────────

  /// Record a kernel execution timing.
  void recordKernelTiming(const KernelTiming &timing);

  /// Get the latest timing for a specific operation.
  std::optional<KernelTiming> getLatestTiming(const std::string &opName) const;

  // ── PGO decisions ────────────────────────────────────────────────────────

  /// Should this trace be recompiled with PGO data?
  bool shouldPGORecompile(uint64_t traceId) const;

  /// Get the profile for a trace.
  TraceProfile *getTraceProfile(uint64_t traceId);

  /// Get the static shapes for a trace (for PGO recompilation).
  /// Returns nullopt if shapes aren't stable yet.
  std::optional<std::unordered_map<TraceValueId, std::vector<int64_t>>>
  getStaticShapes(uint64_t traceId) const;

  // ── Statistics ───────────────────────────────────────────────────────────

  /// Total number of trace executions.
  uint64_t totalExecutions() const;

  /// Number of PGO recompilations.
  uint64_t pgoRecompilationCount() const { return pgoRecompilationCount_; }

  /// Reset all profile data.
  void reset();

private:
  mutable std::mutex mutex_;

  /// Profile data for each trace.
  std::unordered_map<uint64_t, std::unique_ptr<TraceProfile>> profiles_;

  /// Kernel timing data.
  std::unordered_map<std::string, KernelTiming> latestTimings_;

  /// Number of PGO recompilations performed.
  uint64_t pgoRecompilationCount_ = 0;

  /// Next trace ID.
  std::atomic<uint64_t> nextTraceId_{1};
};

} // namespace jules

#endif // JULES_PROFILER_H

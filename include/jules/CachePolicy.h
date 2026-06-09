//===- CachePolicy.h - Hot Path Detection & Cache Policy --------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the cache policy that determines when a function path
// is "hot" enough to justify the overhead of compiling a Tier 2 specialized
// binary. The policy uses execution frequency, shape stability, and
// cost-benefit analysis to make compilation decisions.
//
// Key design:
//   - Count-based triggering: compile after N executions with stable shapes
//   - Shape stability: only compile when shapes are consistent across runs
//   - Cost model: estimate compilation cost vs. expected speedup
//   - Eviction policy: remove Tier 2 entries that are no longer hot
//
//===----------------------------------------------------------------------===//

#ifndef JULES_CACHE_POLICY_H
#define JULES_CACHE_POLICY_H

#include "jules/DispatchTable.h"
#include "jules/Profiler.h"
#include "jules/Diagnostics.h"
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace jules {

// ── Cache Policy Configuration ──────────────────────────────────────────────

struct CachePolicyConfig {
  /// Number of executions before considering Tier 2 compilation.
  uint64_t warmupThreshold = 10;

  /// Minimum number of consistent shape observations before compiling.
  uint64_t shapeStabilityThreshold = 5;

  /// Maximum number of Tier 2 entries per function (prevents bloat).
  size_t maxTier2PerFunction = 8;

  /// Maximum total Tier 2 entries across all functions.
  size_t maxTotalTier2Entries = 64;

  /// Minimum dispatch count for a Tier 2 entry to be considered "hot".
  /// Entries below this threshold may be evicted.
  uint64_t coldEvictionThreshold = 5;

  /// How often (in dispatches) to run the eviction check.
  uint64_t evictionCheckInterval = 100;

  /// Estimated compilation time cost (milliseconds).
  double estimatedCompileCostMs = 500.0;

  /// Estimated minimum speedup factor from Tier 2.
  double minSpeedupFactor = 2.0;

  /// Enable verbose logging of cache policy decisions.
  bool verbose = false;
};

// ── Compilation Decision ────────────────────────────────────────────────────

/// The result of a cache policy evaluation.
struct CompilationDecision {
  /// Should this function+shape be compiled to Tier 2?
  bool shouldCompile = false;

  /// The priority of this compilation (higher = sooner).
  int priority = 0;

  /// The reason for the decision (for logging/diagnostics).
  std::string reason;

  /// The concrete shapes to compile for.
  ShapeSignature targetShapes;

  /// Estimated speedup factor.
  double estimatedSpeedup = 0.0;
};

// ── Cache Policy ────────────────────────────────────────────────────────────

/// The cache policy engine: evaluates whether functions should be compiled
/// to Tier 2 based on profiling data and cost-benefit analysis.
class CachePolicy {
public:
  explicit CachePolicy(DiagnosticsEngine &diag, CachePolicyConfig config = {});

  // ── Decision making ──────────────────────────────────────────────────────

  /// Evaluate whether a function should be compiled to Tier 2.
  /// This is called after each execution to check if the function
  /// meets the criteria for Tier 2 compilation.
  CompilationDecision evaluate(const std::string &functionName,
                                uint64_t traceId,
                                const Profiler &profiler);

  /// Check if a function has already been compiled to Tier 2
  /// for a given shape signature.
  bool isAlreadyCompiled(const std::string &functionName,
                          const ShapeSignature &shapes,
                          const DispatchTable &dispatchTable) const;

  // ── Eviction ─────────────────────────────────────────────────────────────

  /// Check if eviction should be performed (based on check interval).
  bool shouldEvict(uint64_t totalDispatchCount) const;

  /// Get the number of Tier 2 entries that should be evicted.
  /// Returns a list of (function_name, shape_signature) pairs to evict.
  std::vector<std::pair<std::string, ShapeSignature>>
  getEvictionCandidates(const DispatchTable &dispatchTable) const;

  // ── Statistics ───────────────────────────────────────────────────────────

  /// Total number of compilation decisions made.
  uint64_t totalDecisions() const { return totalDecisions_; }

  /// Number of positive compilation decisions.
  uint64_t positiveDecisions() const { return positiveDecisions_; }

  /// Number of negative compilation decisions.
  uint64_t negativeDecisions() const { return negativeDecisions_; }

  // ── Configuration ────────────────────────────────────────────────────────

  const CachePolicyConfig &getConfig() const { return config_; }

private:
  DiagnosticsEngine    &diag_;
  CachePolicyConfig     config_;

  /// Track which function+shape combos have already been decided.
  std::unordered_map<std::string, std::vector<ShapeSignature>> compiledEntries_;

  /// Statistics.
  uint64_t              totalDecisions_ = 0;
  uint64_t              positiveDecisions_ = 0;
  uint64_t              negativeDecisions_ = 0;
};

} // namespace jules

#endif // JULES_CACHE_POLICY_H

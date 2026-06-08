//===- CachePolicy.cpp - Hot Path Detection & Cache Policy Implementation ---===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the cache policy engine that determines when functions
// should be promoted from Tier 1 (AOT) to Tier 2 (specialized JIT).
//
//===----------------------------------------------------------------------===//

#include "jules/CachePolicy.h"
#include "jules/Diagnostics.h"
#include <algorithm>
#include <vector>

namespace jules {

CachePolicy::CachePolicy(DiagnosticsEngine &diag, CachePolicyConfig config)
    : diag_(diag), config_(std::move(config)) {}

CompilationDecision CachePolicy::evaluate(const std::string &functionName,
                                           uint64_t traceId,
                                           const Profiler &profiler) {
  CompilationDecision decision;
  totalDecisions_++;

  // Step 1: Check if we already have a Tier 2 entry for this function.
  auto &existingEntries = compiledEntries_[functionName];

  // Step 2: Get the profile data for this trace.
  auto *profile = profiler.getTraceProfile(traceId);
  if (!profile) {
    decision.shouldCompile = false;
    decision.reason = "no profile data available";
    negativeDecisions_++;
    return decision;
  }

  // Step 3: Check execution frequency (is it "hot"?).
  uint64_t execCount = profile->executionCount.load();
  if (execCount < config_.warmupThreshold) {
    decision.shouldCompile = false;
    decision.reason = "not hot enough (" + std::to_string(execCount) +
                      "/" + std::to_string(config_.warmupThreshold) + " executions)";
    negativeDecisions_++;
    return decision;
  }

  // Step 4: Check shape stability.
  if (!profile->isShapeStable()) {
    decision.shouldCompile = false;
    decision.reason = "shapes not stable yet";
    negativeDecisions_++;
    return decision;
  }

  // Step 5: Check if we've already compiled for these shapes.
  auto staticShapes = profiler.getStaticShapes(traceId);
  if (!staticShapes) {
    decision.shouldCompile = false;
    decision.reason = "could not resolve static shapes";
    negativeDecisions_++;
    return decision;
  }

  // Build the target shape signature from the profiler data.
  ShapeSignature targetSig;
  for (const auto &[valueId, shape] : *staticShapes) {
    targetSig.inputShapes.push_back(shape);
  }

  // Check if already compiled.
  for (const auto &existing : existingEntries) {
    if (existing == targetSig) {
      decision.shouldCompile = false;
      decision.reason = "already compiled for these shapes";
      negativeDecisions_++;
      return decision;
    }
  }

  // Step 6: Check if we've exceeded the maximum Tier 2 entries.
  if (existingEntries.size() >= config_.maxTier2PerFunction) {
    decision.shouldCompile = false;
    decision.reason = "max Tier 2 entries reached for this function";
    negativeDecisions_++;
    return decision;
  }

  // Step 7: Cost-benefit analysis.
  // Estimate speedup based on the ratio of dynamic to static dimensions.
  // More dynamic dimensions → more potential speedup from specialization.
  double dynamicRatio = 0.0;
  size_t totalDims = 0;
  size_t dynamicDims = 0;
  for (const auto &[valueId, shape] : *staticShapes) {
    totalDims += shape.size();
    for (int64_t dim : shape) {
      if (dim < 0) dynamicDims++;
    }
  }
  if (totalDims > 0) {
    dynamicRatio = static_cast<double>(dynamicDims) / totalDims;
  }

  // Estimated speedup: more dynamic dimensions = higher speedup from specialization.
  decision.estimatedSpeedup = 1.0 + dynamicRatio * 3.0; // 1x to 4x

  if (decision.estimatedSpeedup < config_.minSpeedupFactor) {
    decision.shouldCompile = false;
    decision.reason = "estimated speedup too low (" +
                      std::to_string(decision.estimatedSpeedup) + "x < " +
                      std::to_string(config_.minSpeedupFactor) + "x)";
    negativeDecisions_++;
    return decision;
  }

  // Step 8: Approve compilation.
  decision.shouldCompile = true;
  decision.priority = static_cast<int>(execCount); // Higher exec count = higher priority
  decision.targetShapes = targetSig;
  decision.reason = "hot path with stable shapes (" +
                    std::to_string(execCount) + " executions, " +
                    std::to_string(decision.estimatedSpeedup) + "x estimated speedup)";

  existingEntries.push_back(targetSig);
  positiveDecisions_++;

  if (config_.verbose || true) { // Always log positive decisions.
    diag_.info(SourceLocation{},
               "Cache policy: compile '" + functionName + "' to Tier 2 — " +
               decision.reason);
  }

  return decision;
}

bool CachePolicy::isAlreadyCompiled(const std::string &functionName,
                                      const ShapeSignature &shapes,
                                      const DispatchTable &dispatchTable) const {
  return dispatchTable.hasTier2(functionName, shapes);
}

bool CachePolicy::shouldEvict(uint64_t totalDispatchCount) const {
  return totalDispatchCount > 0 &&
         totalDispatchCount % config_.evictionCheckInterval == 0;
}

std::vector<std::pair<std::string, ShapeSignature>>
CachePolicy::getEvictionCandidates(const DispatchTable &dispatchTable) const {
  std::vector<std::pair<std::string, ShapeSignature>> candidates;

  for (const auto &fnName : dispatchTable.getRegisteredFunctions()) {
    for (const auto &sig : dispatchTable.getTier2Signatures(fnName)) {
      uint64_t dispatches = dispatchTable.getDispatchCount(fnName);
      uint64_t bailouts = dispatchTable.getBailoutCount(fnName);

      // Evict if dispatches are low or bailouts are high.
      if (dispatches < config_.coldEvictionThreshold ||
          (bailouts > 0 && bailouts > dispatches / 2)) {
        candidates.emplace_back(fnName, sig);
      }
    }
  }

  return candidates;
}

} // namespace jules

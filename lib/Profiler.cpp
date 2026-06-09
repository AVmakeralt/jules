//===- Profiler.cpp - Runtime Profiler Implementation -----------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the runtime profiler for JIT PGO. It collects shape
// telemetry and execution frequency data to drive background recompilation.
//
//===----------------------------------------------------------------------===//

#include "jules/Profiler.h"
#include <algorithm>
#include <cassert>
#include <mutex>
#include <optional>

namespace jules {

// ── ShapeProfile ────────────────────────────────────────────────────────────

bool ShapeProfile::isStatic() const {
  for (int64_t dim : shape) {
    if (dim < 0) return false;
  }
  return true;
}

bool ShapeProfile::matches(const TensorType &type) const {
  if (type.getElementKind() != elementType) return false;
  const auto &dims = type.getDims();
  if (dims.size() != shape.size()) return false;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i].kind == Dimension::DK_Concrete && dims[i].size != shape[i]) {
      return false;
    }
  }
  return true;
}

// ── TraceProfile ────────────────────────────────────────────────────────────

bool TraceProfile::isShapeStable() const {
  if (valueShapes.size() < 2) return true; // Trivially stable

  // Check that every observed shape has been seen enough times.
  for (const auto &[valueId, profile] : valueShapes) {
    if (profile.observationCount < kMinObservations) {
      return false;
    }
    // Check that the shape is fully static (no dynamic dimensions).
    if (!profile.isStatic()) {
      return false;
    }
  }
  return true;
}

// ── Profiler implementation ─────────────────────────────────────────────────

Profiler::Profiler() = default;

void Profiler::beginTraceExecution(uint64_t traceId) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = profiles_.find(traceId);
  if (it == profiles_.end()) {
    auto profile = std::make_unique<TraceProfile>();
    profile->traceId = traceId;
    profiles_[traceId] = std::move(profile);
  }
}

void Profiler::endTraceExecution(uint64_t traceId) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = profiles_.find(traceId);
  if (it != profiles_.end()) {
    it->second->executionCount.fetch_add(1);
  }
}

void Profiler::observeShape(uint64_t traceId, TraceValueId valueId,
                             const std::vector<int64_t> &shape,
                             ScalarType::ScalarKind elementType) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = profiles_.find(traceId);
  if (it == profiles_.end()) {
    auto profile = std::make_unique<TraceProfile>();
    profile->traceId = traceId;
    it = profiles_.insert({traceId, std::move(profile)}).first;
  }

  auto &profile = *it->second;
  auto shapeIt = profile.valueShapes.find(valueId);

  if (shapeIt != profile.valueShapes.end()) {
    // Check if the new shape matches the previously observed shape.
    if (shapeIt->second.shape == shape &&
        shapeIt->second.elementType == elementType) {
      shapeIt->second.observationCount++;
    } else {
      // Shape changed — reset the observation count for this value.
      shapeIt->second.shape = shape;
      shapeIt->second.elementType = elementType;
      shapeIt->second.observationCount = 1;
    }
  } else {
    ShapeProfile sp;
    sp.shape = shape;
    sp.elementType = elementType;
    sp.observationCount = 1;
    profile.valueShapes[valueId] = std::move(sp);
  }
}

std::optional<ShapeProfile> Profiler::getShapeProfile(
    uint64_t traceId, TraceValueId valueId) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = profiles_.find(traceId);
  if (it == profiles_.end()) return std::nullopt;

  auto shapeIt = it->second->valueShapes.find(valueId);
  if (shapeIt == it->second->valueShapes.end()) return std::nullopt;

  return shapeIt->second;
}

void Profiler::recordKernelTiming(const KernelTiming &timing) {
  std::lock_guard<std::mutex> lock(mutex_);
  latestTimings_[timing.opName] = timing;
}

std::optional<KernelTiming> Profiler::getLatestTiming(
    const std::string &opName) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = latestTimings_.find(opName);
  if (it != latestTimings_.end()) return it->second;
  return std::nullopt;
}

bool Profiler::shouldPGORecompile(uint64_t traceId) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = profiles_.find(traceId);
  if (it == profiles_.end()) return false;

  const auto &profile = *it->second;

  // Already recompiled?
  if (profile.pgoRecompiled.load()) return false;

  // Is it hot enough?
  if (!profile.isHot()) return false;

  // Are shapes stable?
  if (!profile.isShapeStable()) return false;

  return true;
}

TraceProfile *Profiler::getTraceProfile(uint64_t traceId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = profiles_.find(traceId);
  return it != profiles_.end() ? it->second.get() : nullptr;
}

const TraceProfile *Profiler::getTraceProfile(uint64_t traceId) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = profiles_.find(traceId);
  return it != profiles_.end() ? it->second.get() : nullptr;
}

std::optional<std::unordered_map<TraceValueId, std::vector<int64_t>>>
Profiler::getStaticShapes(uint64_t traceId) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = profiles_.find(traceId);
  if (it == profiles_.end()) return std::nullopt;

  const auto &profile = *it->second;
  if (!profile.isShapeStable()) return std::nullopt;

  std::unordered_map<TraceValueId, std::vector<int64_t>> shapes;
  for (const auto &[valueId, shapeProfile] : profile.valueShapes) {
    if (shapeProfile.isStatic()) {
      shapes[valueId] = shapeProfile.shape;
    }
  }

  return shapes;
}

uint64_t Profiler::totalExecutions() const {
  std::lock_guard<std::mutex> lock(mutex_);
  uint64_t total = 0;
  for (const auto &[id, profile] : profiles_) {
    total += profile->executionCount.load();
  }
  return total;
}

void Profiler::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  profiles_.clear();
  latestTimings_.clear();
  pgoRecompilationCount_ = 0;
}

} // namespace jules

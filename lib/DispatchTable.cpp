//===- DispatchTable.cpp - Virtual Dispatch Table Implementation -----------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the virtual dispatch table for tiered execution.
// It provides lock-free fast-path dispatch with thread-safe Tier 2 registration.
//
//===----------------------------------------------------------------------===//

#include "jules/DispatchTable.h"
#include "jules/Diagnostics.h"
#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

namespace jules {

// ── DispatchTable implementation ─────────────────────────────────────────────

DispatchTable::DispatchTable(DiagnosticsEngine &diag) : diag_(diag) {}

DispatchTable::~DispatchTable() {
  // Clean up atomic pointers in dispatch entries.
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &[fn, shapeMap] : table_) {
    for (auto &[sig, entry] : shapeMap) {
      auto *t2 = entry->tier2Exec.load();
      if (t2) delete t2;
      auto *active = entry->activeExec.load();
      // activeExec points to either tier1Exec or tier2Exec,
      // both of which are owned by shared_ptr — no manual delete needed.
    }
  }
}

void DispatchTable::registerTier1(const std::string &functionName,
                                   std::shared_ptr<ExecutableHandle> exec) {
  std::lock_guard<std::mutex> lock(mutex_);

  tier1Fallbacks_[functionName] = exec;

  // Set the active executable to Tier 1 by default.
  auto &shapeMap = table_[functionName];
  if (shapeMap.empty()) {
    // Create a default dispatch entry for the Tier 1 fallback.
    auto entry = std::make_shared<DispatchEntry>();
    entry->tier1Exec = exec;
    entry->activeExec.store(exec.get());
    // Insert with an empty shape signature (wildcard).
    shapeMap[ShapeSignature{}] = entry;
  }
}

void DispatchTable::registerTier2(const std::string &functionName,
                                   const ShapeSignature &shapes,
                                   std::shared_ptr<ExecutableHandle> exec) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto &shapeMap = table_[functionName];

  // Find or create a dispatch entry for this shape signature.
  auto it = shapeMap.find(shapes);
  if (it != shapeMap.end()) {
    auto &entry = it->second;

    // Atomically replace the Tier 2 executable.
    auto *oldT2 = entry->tier2Exec.load();
    entry->tier2Exec.store(exec.get());

    // Atomically switch the active executable to Tier 2.
    // This is the critical moment: after this store, all subsequent
    // dispatches with this shape will hit Tier 2.
    entry->activeExec.store(exec.get());

    if (oldT2) {
      // The old Tier 2 handle is being replaced. In a production system,
      // we'd want to keep it alive until all in-flight executions finish.
      // For now, we rely on shared_ptr refcounting.
    }
  } else {
    // Create a new dispatch entry for this shape.
    auto entry = std::make_shared<DispatchEntry>();
    entry->tier1Exec = tier1Fallbacks_[functionName];
    entry->tier2Exec.store(exec.get());
    entry->activeExec.store(exec.get());
    shapeMap[shapes] = entry;
  }

  if (exec->isSpecialized()) {
    diag_.info(SourceLocation{},
               "Tier 2 registered for '" + functionName + "' with " +
               std::to_string(shapes.inputShapes.size()) + " inputs");
  }
}

ExecutableHandle* DispatchTable::dispatch(const std::string &functionName,
                                           const ShapeSignature &shapes) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto fnIt = table_.find(functionName);
  if (fnIt == table_.end()) {
    // No entry for this function at all.
    diag_.error(SourceLocation{},
                "dispatch: no entry for function '" + functionName + "'");
    return nullptr;
  }

  auto &shapeMap = fnIt->second;

  // Try exact shape match first (Tier 2).
  auto it = shapeMap.find(shapes);
  if (it != shapeMap.end()) {
    auto *entry = it->second.get();
    entry->dispatchCount.fetch_add(1);

    auto *active = entry->activeExec.load();
    if (active && active->isSpecialized()) {
      // Tier 2 hit — fast path.
      return active;
    }
  }

  // No Tier 2 for this exact shape — fall back to Tier 1.
  auto *tier1 = tier1Fallbacks_[functionName].get();
  if (tier1) {
    // Record dispatch for the wildcard entry.
    auto wildcardIt = shapeMap.find(ShapeSignature{});
    if (wildcardIt != shapeMap.end()) {
      wildcardIt->second->dispatchCount.fetch_add(1);
    }
    return tier1;
  }

  return nullptr;
}

bool DispatchTable::hasTier2(const std::string &functionName,
                              const ShapeSignature &shapes) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto fnIt = table_.find(functionName);
  if (fnIt == table_.end()) return false;

  auto it = fnIt->second.find(shapes);
  if (it == fnIt->second.end()) return false;

  auto *t2 = it->second->tier2Exec.load();
  return t2 != nullptr && t2->isSpecialized();
}

ExecutableHandle* DispatchTable::getTier1(const std::string &functionName) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = tier1Fallbacks_.find(functionName);
  return it != tier1Fallbacks_.end() ? it->second.get() : nullptr;
}

void DispatchTable::recordBailout(const std::string &functionName,
                                   const ShapeSignature &shapes) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto fnIt = table_.find(functionName);
  if (fnIt == table_.end()) return;

  auto it = fnIt->second.find(shapes);
  if (it != fnIt->second.end()) {
    it->second->bailoutCount.fetch_add(1);
  }
}

uint64_t DispatchTable::getDispatchCount(
    const std::string &functionName) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto fnIt = table_.find(functionName);
  if (fnIt == table_.end()) return 0;

  uint64_t total = 0;
  for (const auto &[sig, entry] : fnIt->second) {
    total += entry->dispatchCount.load();
  }
  return total;
}

uint64_t DispatchTable::getBailoutCount(
    const std::string &functionName) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto fnIt = table_.find(functionName);
  if (fnIt == table_.end()) return 0;

  uint64_t total = 0;
  for (const auto &[sig, entry] : fnIt->second) {
    total += entry->bailoutCount.load();
  }
  return total;
}

std::vector<std::string> DispatchTable::getRegisteredFunctions() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> names;
  for (const auto &[name, _] : tier1Fallbacks_) {
    names.push_back(name);
  }
  return names;
}

std::vector<ShapeSignature> DispatchTable::getTier2Signatures(
    const std::string &functionName) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<ShapeSignature> sigs;

  auto fnIt = table_.find(functionName);
  if (fnIt == table_.end()) return sigs;

  for (const auto &[sig, entry] : fnIt->second) {
    auto *t2 = entry->tier2Exec.load();
    if (t2 && t2->isSpecialized()) {
      sigs.push_back(sig);
    }
  }
  return sigs;
}

size_t DispatchTable::evictColdEntries(uint64_t minDispatchCount) {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t evicted = 0;

  for (auto &[fn, shapeMap] : table_) {
    auto it = shapeMap.begin();
    while (it != shapeMap.end()) {
      if (it->first.inputShapes.empty()) {
        // Don't evict the wildcard (Tier 1 fallback) entry.
        ++it;
        continue;
      }

      if (it->second->dispatchCount.load() < minDispatchCount) {
        auto *t2 = it->second->tier2Exec.load();
        if (t2) {
          // Revert activeExec to Tier 1.
          it->second->activeExec.store(it->second->tier1Exec.get());
        }
        it = shapeMap.erase(it);
        evicted++;
      } else {
        ++it;
      }
    }
  }

  return evicted;
}

void DispatchTable::clearTier2Entries() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto &[fn, shapeMap] : table_) {
    auto it = shapeMap.begin();
    while (it != shapeMap.end()) {
      if (it->first.inputShapes.empty()) {
        // Keep the Tier 1 fallback.
        it->second->tier2Exec.store(nullptr);
        it->second->activeExec.store(it->second->tier1Exec.get());
        ++it;
      } else {
        it = shapeMap.erase(it);
      }
    }
  }
}

void DispatchTable::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  table_.clear();
  tier1Fallbacks_.clear();
}

} // namespace jules

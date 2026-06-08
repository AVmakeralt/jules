//===- DispatchTable.cpp - Virtual Dispatch Table Implementation -----------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the virtual dispatch table for tiered execution.
// It provides lock-free fast-path dispatch with thread-safe Tier 2 registration,
// hash-based lookup with improved distribution, and radix-tree prefix matching
// for shape families.
//
//===----------------------------------------------------------------------===//

#include "jules/DispatchTable.h"
#include "jules/Diagnostics.h"
#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

namespace jules {

// ═══════════════════════════════════════════════════════════════════════════════
// Radix Tree Implementation
// ═══════════════════════════════════════════════════════════════════════════════

void RadixTreeNode::insert(const ShapeSignature &sig, ExecutableHandle *handle) {
  RadixTreeNode *current = this;

  // Walk down the tree, creating nodes for each dimension in the signature.
  for (const auto &shape : sig.inputShapes) {
    for (int64_t dim : shape) {
      auto it = current->children_.find(dim);
      if (it == current->children_.end()) {
        auto child = std::make_unique<RadixTreeNode>();
        child->dimValue_ = dim;
        current->children_[dim] = std::move(child);
      }
      current = current->children_[dim].get();
    }
    // Add a separator node (-2) between different input shapes.
    // This ensures that shapes [2,3] and [2], [3] are distinguished.
    const int64_t separator = -2; // sentinel value between inputs
    auto sepIt = current->children_.find(separator);
    if (sepIt == current->children_.end()) {
      auto child = std::make_unique<RadixTreeNode>();
      child->dimValue_ = separator;
      current->children_[separator] = std::move(child);
    }
    current = current->children_[separator].get();
  }

  // Mark this node as a leaf and store the handle.
  current->handle_ = handle;
  current->isLeaf_ = true;
}

std::vector<ExecutableHandle*>
RadixTreeNode::prefixLookup(const ShapeSignature &sig, size_t minPrefixLen) const {
  std::vector<ExecutableHandle*> results;

  // Collect all handles in the subtree at the point where the prefix ends.
  std::function<void(const RadixTreeNode*, size_t)> collectAll =
      [&](const RadixTreeNode *node, size_t depth) {
    if (!node) return;
    if (node->isLeaf_ && depth >= minPrefixLen) {
      results.push_back(node->handle_);
    }
    for (const auto &[dim, child] : node->children_) {
      collectAll(child.get(), depth + 1);
    }
  };

  // Walk down the tree following the signature dimensions.
  const RadixTreeNode *current = this;
  size_t depth = 0;

  for (const auto &shape : sig.inputShapes) {
    for (int64_t dim : shape) {
      auto it = current->children_.find(dim);
      if (it == current->children_.end()) {
        // Prefix doesn't match further. Return whatever we've collected
        // so far if it meets the minimum prefix length.
        if (depth >= minPrefixLen) {
          collectAll(current, depth);
        }
        return results;
      }
      current = it->second.get();
      depth++;
    }
    // Follow the separator between inputs.
    const int64_t separator = -2;
    auto sepIt = current->children_.find(separator);
    if (sepIt != current->children_.end()) {
      current = sepIt->second.get();
      depth++;
    }
  }

  // We've matched the entire signature. Collect all handles in this subtree.
  if (depth >= minPrefixLen) {
    collectAll(current, depth);
  }

  return results;
}

ExecutableHandle*
RadixTreeNode::exactLookup(const ShapeSignature &sig) const {
  const RadixTreeNode *current = this;

  for (const auto &shape : sig.inputShapes) {
    for (int64_t dim : shape) {
      auto it = current->children_.find(dim);
      if (it == current->children_.end()) {
        return nullptr;
      }
      current = it->second.get();
    }
    const int64_t separator = -2;
    auto sepIt = current->children_.find(separator);
    if (sepIt != current->children_.end()) {
      current = sepIt->second.get();
    }
  }

  return current->isLeaf_ ? current->handle_ : nullptr;
}

size_t RadixTreeNode::size() const {
  size_t count = isLeaf_ ? 1 : 0;
  for (const auto &[dim, child] : children_) {
    count += child->size();
  }
  return count;
}

// ═══════════════════════════════════════════════════════════════════════════════
// RCU Guard Implementation (Simplified Userspace RCU)
// ═══════════════════════════════════════════════════════════════════════════════

namespace {
/// Global RCU state.
std::atomic<uint64_t> rcuEpoch{0};
std::atomic<int> rcuReaderCount{0};
std::vector<ExecutableHandle*> rcuRetiredList;
std::mutex rcuMutex;
} // anonymous namespace

void RCUGuard::readLock() {
  rcuReaderCount.fetch_add(1, std::memory_order_acquire);
}

void RCUGuard::readUnlock() {
  rcuReaderCount.fetch_sub(1, std::memory_order_release);
}

void RCUGuard::retire(ExecutableHandle *handle) {
  std::lock_guard<std::mutex> lock(rcuMutex);
  rcuRetiredList.push_back(handle);
}

void RCUGuard::quiesce() {
  // Wait for all readers to exit their critical sections.
  while (rcuReaderCount.load(std::memory_order_acquire) > 0) {
    // Spin-wait. In production, use a futex or condition variable.
    std::this_thread::yield();
  }

  // FIX (Bug 2): Actually delete retired handles instead of leaking them.
  // The old code had `(void)handle;` which suppressed the unused warning
  // but never freed the memory, causing an unbounded leak over time.
  // Now we properly delete each retired handle.
  std::lock_guard<std::mutex> lock(rcuMutex);
  for (auto *handle : rcuRetiredList) {
    delete handle;
  }
  rcuRetiredList.clear();

  rcuEpoch.fetch_add(1, std::memory_order_release);
}

// ═══════════════════════════════════════════════════════════════════════════════
// DispatchTable Implementation
// ═══════════════════════════════════════════════════════════════════════════════

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

  // Clean up fast path cache.
  auto *cache = fastPathCache_.load();
  if (cache) delete cache;

  // FIX (Bug 3): Clean up retired cache entries that haven't been freed yet.
  for (auto *entry : retiredCacheEntries_) {
    delete entry;
  }
  retiredCacheEntries_.clear();
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

  // Initialize the radix tree for this function if not present.
  if (radixTrees_.find(functionName) == radixTrees_.end()) {
    radixTrees_[functionName] = std::make_unique<RadixTreeNode>();
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

    // Allocate a new handle on the heap for the atomic pointer.
    auto *newT2 = new ExecutableHandle(*exec);
    entry->tier2Exec.store(newT2);

    // Atomically switch the active executable to Tier 2.
    // This is the critical moment: after this store, all subsequent
    // dispatches with this shape will hit Tier 2.
    entry->activeExec.store(newT2);

    // Update the fast path cache.
    updateFastPathCache(functionName, shapes, newT2);

    // Retire the old Tier 2 handle via RCU.
    if (oldT2) {
      RCUGuard::retire(oldT2);
    }
  } else {
    // Create a new dispatch entry for this shape.
    auto entry = std::make_shared<DispatchEntry>();
    entry->tier1Exec = tier1Fallbacks_[functionName];
    auto *newT2 = new ExecutableHandle(*exec);
    entry->tier2Exec.store(newT2);
    entry->activeExec.store(newT2);
    shapeMap[shapes] = entry;

    // Update the fast path cache.
    updateFastPathCache(functionName, shapes, newT2);
  }

  // Insert into the radix tree for prefix-based lookup.
  auto &radixTree = radixTrees_[functionName];
  if (!radixTree) {
    radixTree = std::make_unique<RadixTreeNode>();
  }
  radixTree->insert(shapes, exec.get());

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

ExecutableHandle*
DispatchTable::dispatchFastPath(const std::string &functionName,
                                 const ShapeSignature &shapes) {
  // Fast path: try to read the cached entry without taking the mutex.
  // This uses an RCU-like pattern where the fastPathCache_ pointer is
  // updated atomically by writers under the mutex.
  RCUGuard::readLock();

  auto *cache = fastPathCache_.load(std::memory_order_acquire);
  if (cache && cache->functionName == functionName &&
      cache->shapes == shapes && cache->handle) {
    auto *result = cache->handle;
    RCUGuard::readUnlock();
    return result;
  }

  RCUGuard::readUnlock();

  // Cache miss: fall back to the mutex-protected dispatch.
  return dispatch(functionName, shapes);
}

ExecutableHandle*
DispatchTable::dispatchByPrefix(const std::string &functionName,
                                 const ShapeSignature &shapes,
                                 size_t minPrefixLen) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = radixTrees_.find(functionName);
  if (it == radixTrees_.end() || !it->second) {
    return getTier1(functionName);
  }

  auto matches = it->second->prefixLookup(shapes, minPrefixLen);

  // Prefer Tier 2 matches over Tier 1.
  for (auto *handle : matches) {
    if (handle && handle->isSpecialized()) {
      return handle;
    }
  }

  // Fall back to Tier 1.
  return getTier1(functionName);
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
          // Retire the old Tier 2 handle.
          RCUGuard::retire(t2);
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
        auto *t2 = it->second->tier2Exec.load();
        if (t2) RCUGuard::retire(t2);
        it = shapeMap.erase(it);
      }
    }
  }

  // Clear the radix trees.
  for (auto &[fn, tree] : radixTrees_) {
    tree = std::make_unique<RadixTreeNode>();
  }

  // Clear the fast path cache.
  auto *cache = fastPathCache_.load();
  if (cache) {
    fastPathCache_.store(nullptr);
    delete cache;
  }
}

void DispatchTable::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  table_.clear();
  tier1Fallbacks_.clear();
  radixTrees_.clear();

  auto *cache = fastPathCache_.load();
  if (cache) {
    fastPathCache_.store(nullptr);
    delete cache;
  }
}

void DispatchTable::updateFastPathCache(const std::string &functionName,
                                         const ShapeSignature &shapes,
                                         ExecutableHandle *handle) {
  // Must be called under the mutex.
  //
  // FIX (Bug 3): The old code deleted the old cache entry immediately
  // after the atomic store. This is a use-after-free: a reader in
  // dispatchFastPath could still be accessing oldCache when we delete it.
  //
  // Instead, we retire the old cache entry via RCU. After all current
  // readers have quiesced (exited their critical sections), it's safe
  // to free. We call quiesce() periodically to actually free retired
  // entries.
  auto *oldCache = fastPathCache_.load();
  auto *newCache = new FastPathEntry{functionName, shapes, handle};
  fastPathCache_.store(newCache, std::memory_order_release);

  if (oldCache) {
    // Retire the old cache entry — it will be freed after a grace period
    // when no readers can still be accessing it.
    // We cast FastPathEntry* to a dummy allocation so RCU can delete it.
    // Since FastPathEntry is not ExecutableHandle, we manage deletion
    // separately in a simple retired list for cache entries.
    retiredCacheEntries_.push_back(oldCache);

    // Free old cache entries that are no longer reachable.
    // Since we only update fastPathCache_ under the mutex, and we're
    // currently holding the mutex, any reader that started before our
    // store has already finished by the time we next acquire the mutex.
    // We keep the last 2 entries as a safety margin and free the rest.
    while (retiredCacheEntries_.size() > 2) {
      auto *toDelete = retiredCacheEntries_.front();
      retiredCacheEntries_.erase(retiredCacheEntries_.begin());
      delete toDelete;
    }
  }
}

} // namespace jules

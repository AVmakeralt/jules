//===- DispatchTable.h - Virtual Dispatch Table for Tiered Execution --------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Virtual Dispatch Table, the core routing mechanism
// for the two-tier AOT/JIT hybrid execution engine.
//
// When a function is called at runtime, the dispatch table checks the
// incoming tensor shapes and routes execution to either:
//
//   Tier 1 (AOT): A generic, dynamic-shape binary that runs instantly
//   Tier 2 (JIT): A specialized, static-shape XLA binary that runs faster
//
// The dispatch table uses:
//   - A hash-based lookup (ShapeSignatureHash) for O(1) exact shape matching
//   - A radix-tree based prefix lookup for shape families (same leading dims)
//   - A lock-free fast path (dispatchFastPath) using std::atomic + RCU-like
//     pattern for the common case where Tier 2 entries exist
//
//===----------------------------------------------------------------------===//

#ifndef JULES_DISPATCH_TABLE_H
#define JULES_DISPATCH_TABLE_H

#include "jules/AST.h"
#include "jules/Diagnostics.h"
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace jules {

// ── Shape Signature ─────────────────────────────────────────────────────────

/// A concrete shape signature for dispatch table lookup.
/// This represents the actual runtime shapes of all function inputs.
struct ShapeSignature {
  /// The shapes of each input tensor (empty for scalar inputs).
  std::vector<std::vector<int64_t>> inputShapes;

  /// Equality comparison for hash map lookup.
  bool operator==(const ShapeSignature &other) const {
    return inputShapes == other.inputShapes;
  }
  bool operator!=(const ShapeSignature &other) const {
    return !(*this == other);
  }

  /// Hash function for use in unordered_map.
  struct Hash {
    size_t operator()(const ShapeSignature &sig) const {
      size_t h = 0;
      for (const auto &shape : sig.inputShapes) {
        for (int64_t dim : shape) {
          h ^= std::hash<int64_t>{}(dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        h ^= std::hash<size_t>{}(shape.size()) + 0x9e3779b9 + (h << 6) + (h >> 2);
      }
      return h;
    }
  };
};

// ── Shape Signature Hash (Improved) ─────────────────────────────────────────

/// An improved hash function for shape signatures that provides better
/// distribution and avalanche properties than the basic XOR-based hash.
/// Uses FNV-1a mixing for each dimension value.
struct ShapeSignatureHash {
  /// FNV-1a offset basis and prime for 64-bit.
  static constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
  static constexpr uint64_t FNV_PRIME  = 1099511628211ULL;

  size_t operator()(const ShapeSignature &sig) const {
    uint64_t h = FNV_OFFSET;

    // Hash the number of inputs first (helps distinguish different arities).
    h ^= static_cast<uint64_t>(sig.inputShapes.size());
    h *= FNV_PRIME;

    for (const auto &shape : sig.inputShapes) {
      // Hash the rank of each input.
      h ^= static_cast<uint64_t>(shape.size());
      h *= FNV_PRIME;

      for (int64_t dim : shape) {
        // Mix each dimension value using FNV-1a.
        uint64_t dimBits = static_cast<uint64_t>(dim);
        h ^= dimBits;
        h *= FNV_PRIME;
        // Extra mixing for better avalanche.
        h ^= (dimBits >> 32);
        h *= FNV_PRIME;
      }
    }

    // Final avalanche mixing.
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;

    return static_cast<size_t>(h);
  }
};

// ── Executable Handle ───────────────────────────────────────────────────────

/// A handle to a compiled executable, tagged with its tier.
struct ExecutableHandle {
  /// Unique ID for this executable.
  uint64_t id = 0;

  /// Which tier this executable belongs to.
  enum Tier : uint8_t {
    Tier1_AOT = 1,   // Generic, dynamic-shape AOT binary
    Tier2_JIT = 2,   // Specialized, static-shape JIT binary
  };
  Tier tier = Tier1_AOT;

  /// The serialized MLIR/StableHLO module (or compiled binary blob).
  std::string serializedModule;

  /// The concrete shapes this executable was compiled for (empty for Tier 1).
  ShapeSignature specializedShapes;

  /// Whether this handle is valid.
  bool isValid() const { return id != 0; }

  /// Is this a Tier 2 (specialized) executable?
  bool isSpecialized() const { return tier == Tier2_JIT; }

  /// FIX (Perf 3): Cached PJRT executable. The old code re-compiled from
  /// the serialized MLIR string on every execute() call, which is devastating
  /// for hot loops. Now the compiled executable is cached here after the
  /// first compilation and reused on subsequent calls.
  std::shared_ptr<PJRTExecutable> cachedExecutable;
};

// ── Dispatch Entry ──────────────────────────────────────────────────────────

/// An entry in the dispatch table: maps a shape signature to an executable.
struct DispatchEntry {
  /// The Tier 1 (AOT) executable — always present as the fallback.
  std::shared_ptr<ExecutableHandle> tier1Exec;

  /// The Tier 2 (JIT) executable — may be null if not yet compiled.
  std::atomic<ExecutableHandle*> tier2Exec{nullptr};

  /// The current active executable pointer (points to either tier1 or tier2).
  /// This is what the dispatch fast-path reads. Updated atomically when
  /// a Tier 2 compilation completes.
  std::atomic<ExecutableHandle*> activeExec{nullptr};

  /// Number of times this entry has been dispatched.
  std::atomic<uint64_t> dispatchCount{0};

  /// Number of times execution fell back from Tier 2 to Tier 1 (bailout).
  std::atomic<uint64_t> bailoutCount{0};
};

// ── Radix Tree Node ─────────────────────────────────────────────────────────

/// A node in the radix tree used for prefix-based shape family lookup.
/// The radix tree groups shape signatures by their leading dimensions,
/// enabling efficient lookup of all functions with the same batch or
/// leading dimension sizes.
class RadixTreeNode {
public:
  RadixTreeNode() = default;

  /// Insert a shape signature into the radix tree.
  /// \param sig     The shape signature to insert
  /// \param depth   The current dimension depth (input index, dim index)
  /// \param handle  The executable handle associated with this signature
  void insert(const ShapeSignature &sig, ExecutableHandle *handle);

  /// Look up all executables matching a prefix of the shape signature.
  /// Returns all handles whose signatures share the same leading dimensions.
  std::vector<ExecutableHandle*> prefixLookup(const ShapeSignature &sig,
                                               size_t minPrefixLen = 1) const;

  /// Look up an exact match.
  ExecutableHandle* exactLookup(const ShapeSignature &sig) const;

  /// Get the number of entries in this subtree.
  size_t size() const;

private:
  /// The dimension value at this level of the radix tree.
  /// -1 represents the root (no dimension value).
  int64_t dimValue_ = -1;

  /// Children indexed by dimension value.
  /// Using unordered_map for O(1) child lookup.
  std::unordered_map<int64_t, std::unique_ptr<RadixTreeNode>> children_;

  /// The executable handle at this node (leaf nodes only).
  ExecutableHandle *handle_ = nullptr;

  /// Whether this node is a leaf (has an associated handle).
  bool isLeaf_ = false;
};

// ── RCU Guard for Lock-Free Dispatch ─────────────────────────────────────────

/// A simple RCU (Read-Copy-Update) guard for lock-free dispatch table reads.
/// Writers publish new dispatch entries by atomically updating a pointer;
/// readers access the table without locks. Old entries are retired and
/// freed after a grace period.
///
/// This is a simplified userspace RCU implementation. In production,
/// one would use liburcu or a similar library.
class RCUGuard {
public:
  /// Enter an RCU read-side critical section.
  static void readLock();

  /// Exit an RCU read-side critical section.
  static void readUnlock();

  /// Retire an old dispatch entry (will be freed after a grace period).
  static void retire(ExecutableHandle *handle);

  /// Process retired entries (call periodically from a background thread).
  static void quiesce();
};

// ── Dispatch Table ──────────────────────────────────────────────────────────

/// The virtual dispatch table: routes function calls to the correct tier.
///
/// The table is keyed by (function_name) -> per-shape dispatch entries.
/// For each function, multiple shape signatures may have Tier 2 entries.
/// The lookup order is:
///
///   1. Check if there's a Tier 2 entry for the exact input shapes
///   2. If yes, execute Tier 2 (specialized, fast)
///   3. If no, execute Tier 1 (generic, always works)
///   4. Record the shape observation for potential Tier 2 compilation
///
/// The table supports three dispatch modes:
///   - dispatch():           Standard mutex-protected dispatch
///   - dispatchFastPath():   Lock-free dispatch using RCU-like pattern
///   - dispatchByPrefix():   Radix-tree prefix lookup for shape families
class DispatchTable {
public:
  explicit DispatchTable(DiagnosticsEngine &diag);
  ~DispatchTable();

  // ── Registration ─────────────────────────────────────────────────────────

  /// Register a Tier 1 (AOT) executable for a function.
  void registerTier1(const std::string &functionName,
                     std::shared_ptr<ExecutableHandle> exec);

  /// Register a Tier 2 (JIT) executable for a function and shape signature.
  /// This atomically updates the active executable pointer so that
  /// subsequent dispatches immediately use the Tier 2 binary.
  void registerTier2(const std::string &functionName,
                     const ShapeSignature &shapes,
                     std::shared_ptr<ExecutableHandle> exec);

  // ── Dispatch ─────────────────────────────────────────────────────────────

  /// Standard dispatch: look up the best executable for a function call.
  /// Returns the Tier 2 executable if one exists for these exact shapes,
  /// otherwise returns the Tier 1 fallback.
  ExecutableHandle* dispatch(const std::string &functionName,
                              const ShapeSignature &shapes);

  /// Lock-free fast-path dispatch using RCU-like pattern.
  /// This avoids taking the mutex in the common case (Tier 2 hit).
  /// Falls back to the mutex-protected path on miss.
  ExecutableHandle* dispatchFastPath(const std::string &functionName,
                                      const ShapeSignature &shapes);

  /// Prefix-based dispatch using the radix tree.
  /// Returns the best executable for a shape family (functions with
  /// the same leading dimensions). Useful for batched workloads.
  ExecutableHandle* dispatchByPrefix(const std::string &functionName,
                                      const ShapeSignature &shapes,
                                      size_t minPrefixLen = 1);

  /// Check if a Tier 2 executable exists for a function + shape combo.
  bool hasTier2(const std::string &functionName,
                const ShapeSignature &shapes) const;

  /// Get the Tier 1 (fallback) executable for a function.
  ExecutableHandle* getTier1(const std::string &functionName) const;

  // ── Bailout ──────────────────────────────────────────────────────────────

  /// Record a bailout event (Tier 2 failed, fell back to Tier 1).
  void recordBailout(const std::string &functionName,
                     const ShapeSignature &shapes);

  // ── Statistics ───────────────────────────────────────────────────────────

  /// Get the number of dispatches for a function.
  uint64_t getDispatchCount(const std::string &functionName) const;

  /// Get the number of bailouts for a function.
  uint64_t getBailoutCount(const std::string &functionName) const;

  /// Get all registered function names.
  std::vector<std::string> getRegisteredFunctions() const;

  /// Get all shape signatures that have Tier 2 entries for a function.
  std::vector<ShapeSignature> getTier2Signatures(
      const std::string &functionName) const;

  // ── Maintenance ──────────────────────────────────────────────────────────

  /// Evict Tier 2 entries that haven't been used recently.
  /// Returns the number of entries evicted.
  size_t evictColdEntries(uint64_t minDispatchCount);

  /// Clear all Tier 2 entries (e.g., before re-profiling).
  void clearTier2Entries();

  /// Clear everything.
  void clear();

private:
  DiagnosticsEngine &diag_;

  /// Per-function dispatch entries, keyed by shape signature.
  /// function_name -> (shape_signature -> dispatch entry)
  /// Uses the improved ShapeSignatureHash for better distribution.
  using ShapeMap = std::unordered_map<ShapeSignature,
                                       std::shared_ptr<DispatchEntry>,
                                       ShapeSignatureHash>;
  using FunctionMap = std::unordered_map<std::string, ShapeMap>;

  mutable std::mutex mutex_;
  FunctionMap table_;

  /// Tier 1 fallback executables per function.
  std::unordered_map<std::string, std::shared_ptr<ExecutableHandle>> tier1Fallbacks_;

  /// Radix trees for prefix-based shape family lookup.
  /// function_name -> radix tree root
  std::unordered_map<std::string, std::unique_ptr<RadixTreeNode>> radixTrees_;

  /// Next executable ID.
  std::atomic<uint64_t> nextExecId_{1};

  /// Atomic pointer to a read-only snapshot of the table, for RCU-like
  /// lock-free dispatch. Updated by writers under the mutex; read by
  /// dispatchFastPath without the mutex.
  struct FastPathEntry {
    std::string functionName;
    ShapeSignature shapes;
    ExecutableHandle *handle;
  };
  std::atomic<FastPathEntry*> fastPathCache_{nullptr};

  /// Build/update the fast path cache entry for a function+shape combo.
  void updateFastPathCache(const std::string &functionName,
                           const ShapeSignature &shapes,
                           ExecutableHandle *handle);

  /// FIX (Bug 3): Retired fast path cache entries pending deletion.
  /// We can't delete them immediately because a reader in dispatchFastPath
  /// might still be accessing them. We keep a small buffer and free old
  /// entries once they're guaranteed unreachable.
  std::vector<FastPathEntry*> retiredCacheEntries_;
};

} // namespace jules

#endif // JULES_DISPATCH_TABLE_H

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
// The dispatch table is a lock-free hash map keyed by (function_name, shape).
// When a Tier 2 binary is compiled in the background, it is atomically
// inserted into the table, and subsequent calls with matching shapes bypass
// the AOT tier entirely.
//
// If the shapes don't match any Tier 2 entry, execution falls back to
// the AOT tier (bailout mechanism).
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

  /// Look up the best executable for a function call with the given shapes.
  /// Returns the Tier 2 executable if one exists for these exact shapes,
  /// otherwise returns the Tier 1 fallback.
  ExecutableHandle* dispatch(const std::string &functionName,
                              const ShapeSignature &shapes);

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
  using ShapeMap = std::unordered_map<ShapeSignature,
                                       std::shared_ptr<DispatchEntry>,
                                       ShapeSignature::Hash>;
  using FunctionMap = std::unordered_map<std::string, ShapeMap>;

  mutable std::mutex mutex_;
  FunctionMap table_;

  /// Tier 1 fallback executables per function.
  std::unordered_map<std::string, std::shared_ptr<ExecutableHandle>> tier1Fallbacks_;

  /// Next executable ID.
  std::atomic<uint64_t> nextExecId_{1};
};

} // namespace jules

#endif // JULES_DISPATCH_TABLE_H

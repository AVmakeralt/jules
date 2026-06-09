//===- JIT.h - Tier 2 JIT Compiler with PGO --------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Tier 2 JIT compiler that handles:
//
//   1. Specialized compilation: takes PGO-profiled static shapes and
//      generates hyper-optimized XLA binaries
//   2. Background recompilation: triggered by the JITQueue when a hot
//      path with stable shapes is detected
//   3. Atomic executable swap: replaces Tier 1 with Tier 2 in the
//      dispatch table without blocking the main execution loop
//
// The JIT is the "Tier 2 Optimizer Engine" in the two-tier hybrid system.
// It produces specialized, static-shape XLA binaries that are:
//   - Hyper-fused (kernel fusion for the specific shape)
//   - SIMD-vectorized (native AVX-512/Neon instructions)
//   - Cache-optimized (polyhedral tiling for the specific dimensions)
//   - Dead-code-free (SymbolDCE strips unused paths)
//
//===----------------------------------------------------------------------===//

#ifndef JULES_JIT_H
#define JULES_JIT_H

#include "jules/Tracing.h"
#include "jules/Profiler.h"
#include "jules/Diagnostics.h"
#include "jules/DispatchTable.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace mlir {
class MLIRContext;
class ModuleOp;
class PassManager;
} // namespace mlir

namespace jules {

// ── Executable ──────────────────────────────────────────────────────────────

/// A compiled executable ready to run on a device.
class Executable {
public:
  virtual ~Executable() = default;

  /// Execute this executable with the given input values.
  virtual std::vector<void *> execute(const std::vector<void *> &inputs) = 0;

  uint64_t getId() const { return id_; }
  bool isSpecialized() const { return specialized_; }
  uint64_t getTraceId() const { return traceId_; }

protected:
  uint64_t id_ = 0;
  uint64_t traceId_ = 0;
  bool     specialized_ = false;
};

// ── JIT Configuration ───────────────────────────────────────────────────────

struct JITConfig {
  int       optLevel = 2;            // Aggressive optimization for Tier 2
  bool      enablePGO = true;
  uint64_t  pgoWarmupThreshold = 10;
  bool      enableGraphCollapsing = true;
  bool      enableAutodiffPruning = true;
  bool      enableSCCP = true;
  bool      enableSymbolDCE = true;
  bool      enableSIMDLayout = true;
  bool      enablePolyhedral = true;
  std::string targetDevice = "cpu";
  uint64_t  maxPGORecompiles = 3;
  bool      verbose = false;
};

// ── JIT Compiler (Tier 2) ──────────────────────────────────────────────────

/// The Tier 2 JIT compiler: manages specialized compilation and PGO lifecycle.
class JITCompiler {
public:
  explicit JITCompiler(DiagnosticsEngine &diag, JITConfig config = {});
  ~JITCompiler();

  // ── Trace compilation ────────────────────────────────────────────────────

  /// Compile an active trace into an executable.
  std::shared_ptr<Executable> compileTrace(ActiveTrace &trace,
                                            bool useStaticShapes = false);

  /// Compile a trace with dynamic shapes (Tier 1 fallback).
  std::shared_ptr<Executable> compileSpeculative(ActiveTrace &trace);

  /// Compile a trace with static shapes from PGO data (Tier 2).
  std::shared_ptr<Executable> compileSpecialized(ActiveTrace &trace,
                                                   uint64_t traceId);

  // ── Executable management ────────────────────────────────────────────────

  std::shared_ptr<Executable> lookupExecutable(uint64_t traceId) const;
  void registerExecutable(uint64_t traceId, std::shared_ptr<Executable> exec);
  void swapExecutable(uint64_t traceId, std::shared_ptr<Executable> newExec);

  // ── PGO ──────────────────────────────────────────────────────────────────

  Profiler &getProfiler() { return profiler_; }
  bool shouldPGORecompile(uint64_t traceId) const;
  void launchPGORecompile(uint64_t traceId, ActiveTrace &trace);
  void waitForPGORecompiles();

  // ── Accessors ────────────────────────────────────────────────────────────

  const JITConfig &getConfig() const { return config_; }
  DiagnosticsEngine &getDiag() { return diag_; }

private:
  DiagnosticsEngine                                    &diag_;
  JITConfig                                             config_;
  Profiler                                              profiler_;

  mutable std::mutex                                    execMutex_;
  std::unordered_map<uint64_t, std::shared_ptr<Executable>> executables_;

  std::vector<std::thread>                              pgoThreads_;
  std::mutex                                            pgoMutex_;
  std::condition_variable                               pgoCV_;
  std::atomic<bool>                                     shuttingDown_{false};
  std::atomic<uint32_t>                                 activePGORecompiles_{0};

  /// Compile a trace through the full Tier 2 MLIR pipeline.
  std::shared_ptr<Executable> compileThroughMLIR(ActiveTrace &trace,
                                                   bool useStaticShapes,
                                                   uint64_t traceId);

  /// FIX (Perf 5): Cached MLIRContext for reuse across JIT compilations.
  /// Creating a new MLIRContext and loading dialects is expensive (hundreds
  /// of milliseconds). The old code created a new one on every compile
  /// call. Now we cache it and reuse it across compilations.
  std::shared_ptr<mlir::MLIRContext> cachedContext_;
  std::mutex contextMutex_;
};

} // namespace jules

#endif // JULES_JIT_H

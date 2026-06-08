//===- JIT.h - JIT Compiler with PGO --------------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the JIT compiler that orchestrates:
//
//   1. Speculative compilation: first execution uses dynamic shapes
//   2. Warmup profiling: collect shape/frequency telemetry
//   3. PGO recompilation: background thread compiles specialized version
//   4. Atomic executable swap: replace dynamic with static executable
//
// The JIT integrates with:
//   - GlobalTracer: receives traces for compilation
//   - Profiler: provides PGO telemetry
//   - PJRT: executes compiled programs on devices
//   - MLIR: compiles traces through the Jules/StableHLO pipeline
//
//===----------------------------------------------------------------------===//

#ifndef JULES_JIT_H
#define JULES_JIT_H

#include "jules/Tracing.h"
#include "jules/Profiler.h"
#include "jules/Diagnostics.h"
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
  /// Returns the output values.
  virtual std::vector<void *> execute(const std::vector<void *> &inputs) = 0;

  /// Get the unique ID of this executable.
  uint64_t getId() const { return id_; }

  /// Is this executable compiled with static (specialized) shapes?
  bool isSpecialized() const { return specialized_; }

  /// Get the trace ID this executable was compiled from.
  uint64_t getTraceId() const { return traceId_; }

protected:
  uint64_t id_ = 0;
  uint64_t traceId_ = 0;
  bool     specialized_ = false;
};

// ── JIT Configuration ───────────────────────────────────────────────────────

struct JITConfig {
  /// Optimization level (0 = none, 1 = basic, 2 = aggressive).
  int optLevel = 1;

  /// Whether to enable PGO recompilation.
  bool enablePGO = true;

  /// Number of executions before PGO recompilation kicks in.
  uint64_t pgoWarmupThreshold = 10;

  /// Whether to enable graph collapsing passes.
  bool enableGraphCollapsing = true;

  /// Whether to enable autodiff pruning.
  bool enableAutodiffPruning = true;

  /// Target device: "cpu", "cuda", "rocm", "tpu".
  std::string targetDevice = "cpu";

  /// Maximum number of PGO recompilations per trace.
  uint64_t maxPGORecompiles = 3;

  /// Whether to enable verbose JIT logging.
  bool verbose = false;
};

// ── JIT Compiler ────────────────────────────────────────────────────────────

/// The JIT compiler: manages the compilation, caching, and PGO lifecycle.
class JITCompiler {
public:
  explicit JITCompiler(DiagnosticsEngine &diag, JITConfig config = {});
  ~JITCompiler();

  // ── Trace compilation ────────────────────────────────────────────────────

  /// Compile an active trace into an executable.
  /// If useStaticShapes is true, the compiler will use PGO-profiled shapes
  /// to generate specialized code.
  std::shared_ptr<Executable> compileTrace(ActiveTrace &trace,
                                            bool useStaticShapes = false);

  /// Compile a trace with dynamic shapes (speculative compilation).
  std::shared_ptr<Executable> compileSpeculative(ActiveTrace &trace);

  /// Compile a trace with static shapes from PGO data.
  std::shared_ptr<Executable> compileSpecialized(ActiveTrace &trace,
                                                   uint64_t traceId);

  // ── Executable management ────────────────────────────────────────────────

  /// Look up a cached executable for a trace.
  std::shared_ptr<Executable> lookupExecutable(uint64_t traceId) const;

  /// Register an executable for a trace.
  void registerExecutable(uint64_t traceId,
                           std::shared_ptr<Executable> exec);

  /// Atomically swap an old executable with a new (PGO-optimized) one.
  void swapExecutable(uint64_t traceId,
                       std::shared_ptr<Executable> newExec);

  // ── PGO ──────────────────────────────────────────────────────────────────

  /// Get the profiler.
  Profiler &getProfiler() { return profiler_; }

  /// Check if a trace should be PGO-recompiled.
  bool shouldPGORecompile(uint64_t traceId) const;

  /// Launch background PGO recompilation for a trace.
  void launchPGORecompile(uint64_t traceId, ActiveTrace &trace);

  /// Wait for all background PGO recompilations to finish.
  void waitForPGORecompiles();

  // ── Accessors ────────────────────────────────────────────────────────────

  const JITConfig &getConfig() const { return config_; }
  DiagnosticsEngine &getDiag() { return diag_; }

private:
  DiagnosticsEngine                                    &diag_;
  JITConfig                                             config_;
  Profiler                                              profiler_;

  /// Cached executables: trace ID -> executable.
  mutable std::mutex                                    execMutex_;
  std::unordered_map<uint64_t, std::shared_ptr<Executable>> executables_;

  /// Background PGO recompilation threads.
  std::vector<std::thread>                              pgoThreads_;
  std::mutex                                            pgoMutex_;
  std::condition_variable                               pgoCV_;
  std::atomic<bool>                                     shuttingDown_{false};

  /// Number of active PGO recompilations.
  std::atomic<uint32_t>                                 activePGORecompiles_{0};

  /// Compile a trace through the MLIR pipeline.
  std::shared_ptr<Executable> compileThroughMLIR(ActiveTrace &trace,
                                                   bool useStaticShapes,
                                                   uint64_t traceId);
};

} // namespace jules

#endif // JULES_JIT_H

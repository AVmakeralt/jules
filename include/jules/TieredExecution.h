//===- TieredExecution.h - Two-Tier AOT/JIT Hybrid Execution Engine ---------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TieredExecution engine — the top-level orchestrator
// for the two-tier AOT/JIT hybrid compilation and execution system.
//
// Architecture:
//
//   ┌─────────────────────────────────────────────────────────────┐
//   │                     TieredExecution                         │
//   │                                                              │
//   │  ┌────────────┐   ┌─────────────┐   ┌──────────────────┐  │
//   │  │ AOTCompiler │   │  JITCompiler │   │  DispatchTable   │  │
//   │  │  (Tier 1)   │   │  (Tier 2)   │   │  (routing)       │  │
//   │  └─────┬──────┘   └──────┬──────┘   └────────┬─────────┘  │
//   │        │                  │                    │             │
//   │        │    ┌─────────────┴──────────┐        │             │
//   │        │    │                        │        │             │
//   │        ▼    ▼                        ▼        ▼             │
//   │  ┌────────────────┐          ┌──────────────┐              │
//   │  │  JITQueue       │          │  BailoutHandler │            │
//   │  │  (background)   │          │  (deopt)        │            │
//   │  └────────────────┘          └──────────────┘              │
//   │                                                              │
//   │  ┌────────────┐   ┌─────────────┐   ┌──────────────────┐  │
//   │  │  Profiler   │   │ CachePolicy │   │    PJRTClient    │  │
//   │  │  (PGO)      │   │ (decisions) │   │    (execution)   │  │
//   │  └────────────┘   └─────────────┘   └──────────────────┘  │
//   └─────────────────────────────────────────────────────────────┘
//
// The lifecycle of a function call through the tiered execution engine:
//
//   1. Function called with input tensors
//   2. DispatchTable routes to Tier 2 if available, else Tier 1
//   3. BailoutHandler checks shape guards (if Tier 2)
//   4. Executable runs via PJRT on the target device
//   5. Profiler records shape/frequency telemetry
//   6. CachePolicy evaluates if Tier 2 compilation is warranted
//   7. If yes, JITQueue compiles Tier 2 in background
//   8. When done, DispatchTable is updated atomically
//   9. Next call with matching shapes hits Tier 2
//
//===----------------------------------------------------------------------===//

#ifndef JULES_TIERED_EXECUTION_H
#define JULES_TIERED_EXECUTION_H

#include "jules/AOTCompiler.h"
#include "jules/DispatchTable.h"
#include "jules/JITQueue.h"
#include "jules/CachePolicy.h"
#include "jules/BailoutHandler.h"
#include "jules/Profiler.h"
#include "jules/PJRT.h"
#include "jules/Tracing.h"
#include "jules/Diagnostics.h"
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace jules {

// ── Tiered Execution Configuration ──────────────────────────────────────────

struct TieredExecutionConfig {
  /// AOT compiler configuration.
  AOTCompilerConfig aotConfig;

  /// JIT queue configuration.
  JITQueueConfig jitQueueConfig;

  /// Cache policy configuration.
  CachePolicyConfig cachePolicyConfig;

  /// PJRT platform: "cpu", "cuda", "rocm", "tpu".
  std::string pjrtPlatform = "cpu";

  /// Whether to enable the Tier 2 JIT (if false, only Tier 1 AOT runs).
  bool enableTier2 = true;

  /// Whether to enable PGO profiling.
  bool enablePGO = true;

  /// Whether to enable verbose logging.
  bool verbose = false;
};

// ── Execution Result ────────────────────────────────────────────────────────

/// The result of a tiered execution call.
struct ExecutionResult {
  /// Did the execution succeed?
  bool success = false;

  /// Which tier was used (1 or 2).
  int tierUsed = 1;

  /// Did a bailout occur?
  bool bailedOut = false;

  /// Execution time (milliseconds).
  double executionTimeMs = 0.0;

  /// Output buffers (owned by the device, transferred to host).
  std::vector<std::shared_ptr<DeviceBuffer>> outputs;

  /// Error message (if execution failed).
  std::string errorMessage;
};

// ── Tiered Execution Engine ─────────────────────────────────────────────────

/// The top-level orchestrator for the two-tier AOT/JIT hybrid system.
class TieredExecution {
public:
  explicit TieredExecution(DiagnosticsEngine &diag,
                            TieredExecutionConfig config = {});
  ~TieredExecution();

  // ── Setup ────────────────────────────────────────────────────────────────

  /// Initialize the execution engine.
  /// This creates the PJRT client, starts the JIT queue, etc.
  bool initialize();

  /// Shut down the execution engine.
  /// Waits for background compilations and cleans up resources.
  void shutdown();

  /// Is the engine initialized and ready?
  bool isReady() const { return ready_.load(); }

  // ── Program Loading ──────────────────────────────────────────────────────

  /// Load a Jules source program.
  /// This compiles all functions to Tier 1 (AOT) and registers them
  /// in the dispatch table. Returns true on success.
  bool loadProgram(const std::string &source,
                   const std::string &sourceName = "<input>");

  /// Load a single function from source.
  bool loadFunction(const std::string &source,
                    const std::string &functionName);

  // ── Execution ────────────────────────────────────────────────────────────

  /// Execute a function by name with the given input buffers.
  /// This is the main entry point for the tiered execution system.
  ExecutionResult execute(const std::string &functionName,
                          const std::vector<std::shared_ptr<DeviceBuffer>> &inputs);

  /// Execute a function with shape hints.
  /// The shape hints are used for dispatch table lookup and profiling.
  ExecutionResult executeWithShapes(
      const std::string &functionName,
      const ShapeSignature &inputShapes,
      const std::vector<std::shared_ptr<DeviceBuffer>> &inputs);

  // ── Profiling & PGO ─────────────────────────────────────────────────────

  /// Manually trigger a PGO evaluation for a function.
  /// This checks if the function should be compiled to Tier 2.
  void evaluatePGO(const std::string &functionName, uint64_t traceId);

  /// Wait for all background Tier 2 compilations to finish.
  void waitForCompilations();

  // ── Statistics ───────────────────────────────────────────────────────────

  /// Total number of executions.
  uint64_t totalExecutions() const { return totalExecutions_.load(); }

  /// Number of Tier 1 executions.
  uint64_t tier1Executions() const { return tier1Executions_.load(); }

  /// Number of Tier 2 executions.
  uint64_t tier2Executions() const { return tier2Executions_.load(); }

  /// Number of bailouts.
  uint64_t totalBailouts() const { return bailoutHandler_.totalBailouts(); }

  /// Average Tier 2 compilation time.
  double averageCompileTimeMs() const {
    return jitQueue_.averageCompileTimeMs();
  }

  // ── Component Access ─────────────────────────────────────────────────────

  DispatchTable   &getDispatchTable()   { return dispatchTable_; }
  JITQueue        &getJITQueue()        { return jitQueue_; }
  CachePolicy     &getCachePolicy()     { return cachePolicy_; }
  BailoutHandler  &getBailoutHandler()  { return bailoutHandler_; }
  Profiler        &getProfiler()        { return profiler_; }
  AOTCompiler     &getAOTCompiler()     { return aotCompiler_; }
  PJRTClient      *getPJRTClient()      { return pjrtClient_.get(); }

private:
  DiagnosticsEngine                        &diag_;
  TieredExecutionConfig                      config_;

  /// Core components.
  DispatchTable                             dispatchTable_;
  JITQueue                                  jitQueue_;
  CachePolicy                               cachePolicy_;
  BailoutHandler                            bailoutHandler_;
  Profiler                                  profiler_;
  AOTCompiler                               aotCompiler_;

  /// PJRT client for device execution.
  std::unique_ptr<PJRTClient>               pjrtClient_;

  /// State.
  std::atomic<bool>                         ready_{false};
  std::atomic<bool>                         shuttingDown_{false};

  /// Execution statistics.
  std::atomic<uint64_t>                     totalExecutions_{0};
  std::atomic<uint64_t>                     tier1Executions_{0};
  std::atomic<uint64_t>                     tier2Executions_{0};

  /// Trace ID counter.
  std::atomic<uint64_t>                     nextTraceId_{1};

  /// Function trace ID mapping.
  std::unordered_map<std::string, uint64_t> functionTraceIds_;
  std::mutex                                traceIdMutex_;

  /// Handle a Tier 2 compilation completion callback.
  void handleTier2Completion(const CompilationResult &result);
};

} // namespace jules

#endif // JULES_TIERED_EXECUTION_H

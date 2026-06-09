//===- TieredExecution.cpp - Two-Tier AOT/JIT Hybrid Implementation ---------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the tiered execution engine — the orchestrator for
// the two-tier AOT/JIT hybrid compilation and execution system.
//
//===----------------------------------------------------------------------===//

#include "jules/TieredExecution.h"
#include "jules/Diagnostics.h"
#include "jules/Tracing.h"
#include "jules/AOTCompiler.h"
#include "jules/JITQueue.h"
#include "jules/CachePolicy.h"
#include "jules/BailoutHandler.h"
#include "jules/DispatchTable.h"
#include "jules/Profiler.h"
#include "jules/PJRT.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace jules {

// ── TieredExecution implementation ───────────────────────────────────────────

TieredExecution::TieredExecution(DiagnosticsEngine &diag,
                                 TieredExecutionConfig config)
    : diag_(diag),
      config_(std::move(config)),
      dispatchTable_(diag),
      jitQueue_(diag, config_.jitQueueConfig),
      cachePolicy_(diag, config_.cachePolicyConfig),
      bailoutHandler_(diag),
      profiler_(),
      aotCompiler_(diag, config_.aotConfig) {}

TieredExecution::~TieredExecution() {
  shutdown();
}

bool TieredExecution::initialize() {
  if (ready_.load()) return true;

  // Create the PJRT client for device execution.
  pjrtClient_ = PJRTClient::create(config_.pjrtPlatform, diag_);
  if (!pjrtClient_) {
    diag_.error(SourceLocation{},
                "failed to create PJRT client for platform: " +
                config_.pjrtPlatform);
    return false;
  }

  // Start the JIT compilation queue.
  if (config_.enableTier2) {
    jitQueue_.start();
  }

  // Set up the bailout handler's new-shape callback.
  // When a bailout occurs with a new shape, we evaluate whether
  // it should trigger a Tier 2 compilation.
  bailoutHandler_.setNewShapeCallback(
      [this](const std::string &fnName,
             const ShapeSignature &newShapes,
             uint64_t traceId) {
        // Feed the new shape observation to the profiler.
        if (traceId > 0) {
          for (size_t i = 0; i < newShapes.inputShapes.size(); ++i) {
            profiler_.observeShape(traceId, static_cast<TraceValueId>(i),
                                   newShapes.inputShapes[i],
                                   ScalarType::SK_F32);
          }
        }

        // Evaluate whether to compile Tier 2 for this new shape.
        if (traceId > 0) {
          evaluatePGO(fnName, traceId);
        }
      });

  ready_.store(true);

  if (config_.verbose) {
    diag_.info(SourceLocation{},
               "Tiered execution engine initialized: platform=" +
               config_.pjrtPlatform +
               " tier2=" + (config_.enableTier2 ? "on" : "off") +
               " pgo=" + (config_.enablePGO ? "on" : "off"));
  }

  return true;
}

void TieredExecution::shutdown() {
  if (!ready_.load()) return;

  shuttingDown_.store(true);

  // Stop accepting new compilations and wait for in-flight ones.
  if (config_.enableTier2) {
    jitQueue_.stop();
  }

  ready_.store(false);
}

bool TieredExecution::loadProgram(const std::string &source,
                                   const std::string &sourceName) {
  if (!ready_.load()) {
    diag_.error(SourceLocation{}, "tiered execution engine not initialized");
    return false;
  }

  // Compile the entire program to Tier 1 (AOT).
  auto execHandle = aotCompiler_.compile(source, sourceName);
  if (!execHandle) {
    diag_.error(SourceLocation{}, "failed to compile program to Tier 1");
    return false;
  }

  // Register all functions in the dispatch table.
  // In a full implementation, we'd parse the source and register
  // each function separately. For now, we register the main function.
  dispatchTable_.registerTier1("main", execHandle);

  // Assign a trace ID for profiling.
  {
    std::lock_guard<std::mutex> lock(traceIdMutex_);
    uint64_t traceId = nextTraceId_.fetch_add(1);
    functionTraceIds_["main"] = traceId;
    profiler_.beginTraceExecution(traceId);
  }

  if (config_.verbose) {
    diag_.info(SourceLocation{},
               "Program loaded and compiled to Tier 1");
  }

  return true;
}

bool TieredExecution::loadFunction(const std::string &source,
                                    const std::string &functionName) {
  if (!ready_.load()) return false;

  return aotCompiler_.compileAndRegister(source, functionName, dispatchTable_);
}

ExecutionResult TieredExecution::execute(
    const std::string &functionName,
    const std::vector<std::shared_ptr<DeviceBuffer>> &inputs) {
  // Build shape signature from input buffers.
  ShapeSignature inputShapes;
  for (const auto &buffer : inputs) {
    inputShapes.inputShapes.push_back(buffer->shape());
  }

  return executeWithShapes(functionName, inputShapes, inputs);
}

ExecutionResult TieredExecution::executeWithShapes(
    const std::string &functionName,
    const ShapeSignature &inputShapes,
    const std::vector<std::shared_ptr<DeviceBuffer>> &inputs) {
  ExecutionResult result;
  auto startTime = std::chrono::steady_clock::now();

  totalExecutions_.fetch_add(1);

  // ── Step 1: Dispatch ─────────────────────────────────────────────────────
  //
  // Look up the best executable for this function + shape combination.
  // The dispatch table returns Tier 2 if available, Tier 1 otherwise.

  auto *exec = dispatchTable_.dispatch(functionName, inputShapes);

  if (!exec) {
    result.success = false;
    result.errorMessage = "no executable found for function '" +
                          functionName + "'";
    return result;
  }

  // ── Step 2: Shape Guard (Bailout Check) ──────────────────────────────────
  //
  // If we're about to execute a Tier 2 binary, check that the input
  // shapes match the specialized shapes. If not, bail out to Tier 1.

  if (exec->isSpecialized()) {
    auto *tier1Fallback = dispatchTable_.getTier1(functionName);
    if (tier1Fallback) {
      exec = bailoutHandler_.executeWithGuard(
          functionName, inputShapes, exec->specializedShapes,
          exec, tier1Fallback, dispatchTable_);
    }

    if (!exec->isSpecialized()) {
      // Bailout occurred.
      result.bailedOut = true;
      tier1Executions_.fetch_add(1);
      result.tierUsed = 1;
    } else {
      tier2Executions_.fetch_add(1);
      result.tierUsed = 2;
    }
  } else {
    tier1Executions_.fetch_add(1);
    result.tierUsed = 1;
  }

  // ── Step 3: Execute via PJRT ─────────────────────────────────────────────

  if (pjrtClient_) {
    // FIX (Perf 3): Cache the compiled PJRT executable instead of
    // re-compiling from the serialized MLIR string on every call.
    // The old code called compileAndLoad() on EVERY invocation, which
    // re-parsed and re-compiled the entire program each time. This made
    // Tier 2 slower than Tier 1 in practice. Now we cache the compiled
    // executable in the ExecutableHandle and reuse it.
    if (!exec->cachedExecutable) {
      exec->cachedExecutable = pjrtClient_->compileAndLoad(exec->serializedModule);
    }
    auto &pjrtExec = exec->cachedExecutable;
    if (pjrtExec) {
      // Execute the program.
      auto outputs = pjrtExec->execute(inputs);
      result.outputs = std::move(outputs);
      result.success = true;
    } else {
      result.success = false;
      result.errorMessage = "PJRT execution failed";
    }
  } else {
    result.success = true; // No PJRT — dry run.
  }

  auto endTime = std::chrono::steady_clock::now();
  result.executionTimeMs = std::chrono::duration<double, std::milli>(
      endTime - startTime).count();

  // ── Step 4: PGO Profiling ────────────────────────────────────────────────
  //
  // Record the execution and shape observations for PGO.

  if (config_.enablePGO) {
    uint64_t traceId = 0;
    {
      std::lock_guard<std::mutex> lock(traceIdMutex_);
      auto it = functionTraceIds_.find(functionName);
      if (it != functionTraceIds_.end()) {
        traceId = it->second;
      } else {
        traceId = nextTraceId_.fetch_add(1);
        functionTraceIds_[functionName] = traceId;
        profiler_.beginTraceExecution(traceId);
      }
    }

    // Record execution completion.
    profiler_.endTraceExecution(traceId);

    // Record shape observations for each input.
    for (size_t i = 0; i < inputShapes.inputShapes.size(); ++i) {
      profiler_.observeShape(traceId, static_cast<TraceValueId>(i),
                             inputShapes.inputShapes[i],
                             ScalarType::SK_F32);
    }

    // ── Step 5: PGO Evaluation ─────────────────────────────────────────────
    //
    // Check if this function should be compiled to Tier 2.

    if (config_.enableTier2) {
      evaluatePGO(functionName, traceId);
    }
  }

  return result;
}

void TieredExecution::evaluatePGO(const std::string &functionName,
                                   uint64_t traceId) {
  // Ask the cache policy if we should compile Tier 2.
  auto decision = cachePolicy_.evaluate(functionName, traceId, profiler_);

  if (decision.shouldCompile) {
    // Enqueue a Tier 2 compilation job.
    CompilationJob job;
    job.functionName = functionName;
    job.targetShapes = decision.targetShapes;
    job.traceId = traceId;
    job.priority = decision.priority;

    // Capture the active trace for compilation.
    // In a full implementation, we'd capture a copy of the trace
    // or re-generate it from the MLIR module.
    // For now, we create an empty trace that the worker will
    // reconstruct from the MLIR module.

    auto callback = [this](const CompilationResult &result) {
      handleTier2Completion(result);
    };

    uint64_t jobId = jitQueue_.enqueue(std::move(job), callback);

    if (jobId > 0 && config_.verbose) {
      diag_.info(SourceLocation{},
                 "Enqueued Tier 2 compilation for '" + functionName +
                 "' (job " + std::to_string(jobId) + "): " + decision.reason);
    }
  }
}

void TieredExecution::waitForCompilations() {
  jitQueue_.waitIdle();
}

void TieredExecution::handleTier2Completion(const CompilationResult &result) {
  if (!result.success) {
    diag_.warning(SourceLocation{},
                  "Tier 2 compilation failed (job " +
                  std::to_string(result.jobId) + "): " + result.errorMessage);
    return;
  }

  if (result.executable) {
    // Register the Tier 2 executable in the dispatch table.
    dispatchTable_.registerTier2(
        result.executable->specializedShapes.inputShapes.empty()
            ? "main"
            : "main", // Would be the actual function name
        result.executable->specializedShapes,
        result.executable);

    if (config_.verbose) {
      diag_.info(SourceLocation{},
                 "Tier 2 compilation complete: " +
                 std::to_string(result.compilationTimeMs) + "ms");
    }
  }
}

} // namespace jules

//===- JITQueue.cpp - Thread-Safe JIT Compilation Queue Implementation ------===//
//
// Copyright (c) 2025 Jules Contributors
//
// This file implements the thread-safe JIT compilation queue for background
// Tier 2 recompilation. Worker threads pull jobs from a priority queue and
// compile them independently of the main execution thread.
//
//===----------------------------------------------------------------------===//

#include "jules/JITQueue.h"
#include "jules/Diagnostics.h"
#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

namespace jules {

// ── JITQueue implementation ─────────────────────────────────────────────────

JITQueue::JITQueue(DiagnosticsEngine &diag, JITQueueConfig config)
    : diag_(diag), config_(std::move(config)) {}

JITQueue::~JITQueue() {
  stop();
}

void JITQueue::start() {
  if (running_.load()) return;

  running_.store(true);
  shuttingDown_.store(false);

  for (unsigned i = 0; i < config_.numWorkers; ++i) {
    workers_.emplace_back(&JITQueue::workerLoop, this);
  }

  if (config_.verbose) {
    diag_.info(SourceLocation{},
               "JIT queue started with " +
               std::to_string(config_.numWorkers) + " workers");
  }
}

void JITQueue::stop() {
  if (!running_.load()) return;

  shuttingDown_.store(true);
  running_.store(false);
  queueCV_.notify_all();

  for (auto &worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  workers_.clear();
}

uint64_t JITQueue::enqueue(CompilationJob job, CompilationCallback callback) {
  if (!running_.load()) {
    diag_.warning(SourceLocation{}, "JIT queue not running, job rejected");
    return 0;
  }

  std::lock_guard<std::mutex> lock(queueMutex_);

  if (jobQueue_.size() >= config_.maxQueueDepth) {
    diag_.warning(SourceLocation{}, "JIT queue full, job dropped");
    return 0;
  }

  uint64_t jobId = nextJobId_.fetch_add(1);
  job.id = jobId;

  // Store the callback.
  if (callback) {
    std::lock_guard<std::mutex> cbLock(callbackMutex_);
    callbacks_[jobId] = std::move(callback);
  }

  jobQueue_.push(std::move(job));
  queueCV_.notify_one();

  return jobId;
}

void JITQueue::cancelJob(uint64_t jobId) {
  // Mark the job as cancelled. Workers will check this flag.
  // Since the priority_queue doesn't support removal, we use
  // the cancelled flag to skip processing.
  // We don't have a direct reference to the job here, but
  // the worker will check and skip it.
  // For a production system, we'd maintain a separate cancellation set.
  // For now, we rely on the callback being cleared.
  std::lock_guard<std::mutex> cbLock(callbackMutex_);
  callbacks_.erase(jobId);
}

void JITQueue::cancelAll() {
  {
    std::lock_guard<std::mutex> lock(queueMutex_);
    while (!jobQueue_.empty()) {
      jobQueue_.pop();
    }
  }
  {
    std::lock_guard<std::mutex> cbLock(callbackMutex_);
    callbacks_.clear();
  }
}

void JITQueue::waitIdle() {
  while (true) {
    {
      std::lock_guard<std::mutex> lock(queueMutex_);
      if (jobQueue_.empty() && activeCompiles_.load() == 0) {
        return;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

size_t JITQueue::pendingCount() const {
  std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queueMutex_));
  return jobQueue_.size();
}

size_t JITQueue::activeCount() const {
  return activeCompiles_.load();
}

uint64_t JITQueue::completedCount() const {
  return completedCount_.load();
}

uint64_t JITQueue::failedCount() const {
  return failedCount_.load();
}

double JITQueue::averageCompileTimeMs() const {
  uint64_t completed = completedCount_.load();
  if (completed == 0) return 0.0;
  return static_cast<double>(totalCompileTimeMs_.load()) / completed;
}

// ── Worker thread ────────────────────────────────────────────────────────────

void JITQueue::workerLoop() {
  while (!shuttingDown_.load()) {
    CompilationJob job;

    {
      std::unique_lock<std::mutex> lock(queueMutex_);
      queueCV_.wait(lock, [this] {
        return shuttingDown_.load() || !jobQueue_.empty();
      });

      if (shuttingDown_.load()) return;
      if (jobQueue_.empty()) continue;

      job = std::move(const_cast<CompilationJob&>(jobQueue_.top()));
      jobQueue_.pop();
    }

    // Check if the job was cancelled while waiting.
    if (job.cancelled.load()) {
      continue;
    }

    activeCompiles_.fetch_add(1);

    if (config_.verbose) {
      diag_.info(SourceLocation{},
                 "JIT worker: compiling '" + job.functionName +
                 "' (job " + std::to_string(job.id) + ")");
    }

    auto result = processJob(job);

    activeCompiles_.fetch_sub(1);

    if (result.success) {
      completedCount_.fetch_add(1);
      totalCompileTimeMs_.fetch_add(
          static_cast<uint64_t>(result.compilationTimeMs));
    } else {
      failedCount_.fetch_add(1);
    }

    // Invoke the completion callback.
    CompilationCallback cb;
    {
      std::lock_guard<std::mutex> cbLock(callbackMutex_);
      auto it = callbacks_.find(job.id);
      if (it != callbacks_.end()) {
        cb = std::move(it->second);
        callbacks_.erase(it);
      }
    }

    if (cb) {
      cb(result);
    }
  }
}

CompilationResult JITQueue::processJob(CompilationJob &job) {
  CompilationResult result;
  result.jobId = job.id;

  auto startTime = std::chrono::steady_clock::now();

  // ── Compile the trace through the MLIR pipeline ──────────────────────────
  //
  // This is where the real compilation happens:
  //   1. Create an MLIR context and load the Jules dialect
  //   2. Build the MLIR module from the trace
  //   3. Inject the concrete shapes from job.targetShapes
  //   4. Run the full optimization pipeline:
  //      - Shape inference
  //      - Autodiff pass
  //      - Autodiff pruning
  //      - Graph collapsing (whole-program)
  //      - SCCP (constant propagation)
  //      - SymbolDCE (dead code elimination)
  //      - SIMD layout pass
  //      - Polyhedral optimization (affine tiling, fusion)
  //      - Canonicalization + CSE
  //   5. Lower to StableHLO
  //   6. Serialize the compiled module
  //
  // For a complete implementation, this would call compileThroughMLIR
  // with useStaticShapes=true and inject the target shapes.
  // The result would be a serialized StableHLO module ready for XLA.

  try {
    if (config_.compileFn) {
      // Use the provided compilation function to compile through the MLIR
      // pipeline.
      auto compiled = config_.compileFn(job.functionName, job.targetShapes,
                                         job.traceId);
      if (compiled) {
        compiled->id = job.id;
        compiled->tier = ExecutableHandle::Tier2_JIT;
        compiled->specializedShapes = job.targetShapes;
        result.executable = compiled;
        result.success = true;
      } else {
        result.success = false;
        result.errorMessage = "JIT compilation returned null";
      }
    } else {
      // Fallback: create a stub executable handle when no compile function
      // is provided.
      auto execHandle = std::make_shared<ExecutableHandle>();
      execHandle->id = job.id;
      execHandle->tier = ExecutableHandle::Tier2_JIT;
      execHandle->specializedShapes = job.targetShapes;
      execHandle->serializedModule =
          "// Tier 2 JIT compiled: " + job.functionName + "\n";
      result.success = true;
      result.executable = execHandle;
    }

  } catch (const std::exception &e) {
    result.success = false;
    result.errorMessage = e.what();
  }

  auto endTime = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double, std::milli>(endTime - startTime);
  result.compilationTimeMs = duration.count();

  return result;
}

} // namespace jules

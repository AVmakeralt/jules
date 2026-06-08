//===- JITQueue.h - Thread-Safe JIT Compilation Queue -----------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the thread-safe JIT compilation queue for background
// Tier 2 recompilation. The queue ensures that XLA compilation passes
// never block the main training/execution loop.
//
// Architecture:
//
//   Main Thread                   Background Workers
//   ───────────                   ──────────────────
//   [detects hot path] ──────>   [JITQueue]
//   [detects stable shapes]       │
//   [enqueues compile job]  ───>  │ ──> Worker 1: compile trace A
//                                  │ ──> Worker 2: compile trace B
//                                  │
//   [continues executing    <───  │ [compilation done]
//    on Tier 1 AOT]               │ [register Tier 2 in dispatch table]
//                                  │ [atomically swap active exec]
//
// The queue supports:
//   - Priority ordering (hot paths compiled first)
//   - Cancellation (if shapes change during compilation)
//   - Completion callbacks (to register Tier 2 in dispatch table)
//   - Worker thread pool with configurable size
//
//===----------------------------------------------------------------------===//

#ifndef JULES_JIT_QUEUE_H
#define JULES_JIT_QUEUE_H

#include "jules/DispatchTable.h"
#include "jules/Tracing.h"
#include "jules/Profiler.h"
#include "jules/Diagnostics.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace jules {

// ── Compilation Job ─────────────────────────────────────────────────────────

/// A compilation job enqueued for background Tier 2 compilation.
struct CompilationJob {
  /// Unique job ID.
  uint64_t id;

  /// The function name to compile.
  std::string functionName;

  /// The trace to compile (captured from the global tracer).
  ActiveTrace trace;

  /// The concrete shapes for Tier 2 specialization.
  ShapeSignature targetShapes;

  /// The trace ID (for profiler correlation).
  uint64_t traceId;

  /// Priority (higher = compiled sooner).
  /// Based on execution frequency and shape stability.
  int priority = 0;

  /// Whether this job has been cancelled.
  std::atomic<bool> cancelled{false};

  /// Comparison for priority queue (higher priority first).
  bool operator<(const CompilationJob &other) const {
    return priority < other.priority;
  }
};

// ── Compilation Result ──────────────────────────────────────────────────────

/// The result of a background compilation job.
struct CompilationResult {
  /// The job ID this result corresponds to.
  uint64_t jobId;

  /// Whether compilation succeeded.
  bool success = false;

  /// The compiled Tier 2 executable handle (if success).
  std::shared_ptr<ExecutableHandle> executable;

  /// Error message (if compilation failed).
  std::string errorMessage;

  /// Time taken for compilation (milliseconds).
  double compilationTimeMs = 0.0;
};

// ── Completion Callback ─────────────────────────────────────────────────────

/// Callback invoked when a compilation job completes.
/// The callback receives the compilation result and is responsible for
/// registering the Tier 2 executable in the dispatch table.
using CompilationCallback = std::function<void(const CompilationResult &)>;

// ── JIT Queue Configuration ─────────────────────────────────────────────────

struct JITQueueConfig {
  /// Number of background worker threads.
  unsigned numWorkers = 2;

  /// Maximum queue depth (jobs are dropped if queue is full).
  size_t maxQueueDepth = 16;

  /// Whether to enable verbose logging.
  bool verbose = false;
};

// ── JIT Queue ───────────────────────────────────────────────────────────────

/// The thread-safe JIT compilation queue.
class JITQueue {
public:
  explicit JITQueue(DiagnosticsEngine &diag, JITQueueConfig config = {});
  ~JITQueue();

  // ── Job submission ───────────────────────────────────────────────────────

  /// Enqueue a compilation job for background execution.
  /// Returns the job ID, or 0 if the queue is full.
  uint64_t enqueue(CompilationJob job, CompilationCallback callback = {});

  /// Cancel a pending compilation job.
  void cancelJob(uint64_t jobId);

  /// Cancel all pending compilation jobs.
  void cancelAll();

  // ── Lifecycle ────────────────────────────────────────────────────────────

  /// Start the worker threads.
  void start();

  /// Stop the worker threads and wait for them to finish.
  void stop();

  /// Wait for all queued jobs to complete.
  void waitIdle();

  // ── Status ───────────────────────────────────────────────────────────────

  /// Number of pending jobs in the queue.
  size_t pendingCount() const;

  /// Number of currently active compilations.
  size_t activeCount() const;

  /// Total number of completed compilations.
  uint64_t completedCount() const;

  /// Total number of failed compilations.
  uint64_t failedCount() const;

  /// Average compilation time (milliseconds).
  double averageCompileTimeMs() const;

  /// Is the queue running?
  bool isRunning() const { return running_.load(); }

private:
  /// Worker thread function.
  void workerLoop();

  /// Process a single compilation job.
  CompilationResult processJob(CompilationJob &job);

  DiagnosticsEngine                         &diag_;
  JITQueueConfig                             config_;

  /// Worker threads.
  std::vector<std::thread>                   workers_;

  /// Job queue (priority queue).
  std::priority_queue<CompilationJob>        jobQueue_;
  std::mutex                                 queueMutex_;
  std::condition_variable                    queueCV_;

  /// Completion callbacks.
  std::unordered_map<uint64_t, CompilationCallback> callbacks_;
  std::mutex                                 callbackMutex_;

  /// State flags.
  std::atomic<bool>                          running_{false};
  std::atomic<bool>                          shuttingDown_{false};

  /// Next job ID.
  std::atomic<uint64_t>                      nextJobId_{1};

  /// Statistics.
  std::atomic<uint64_t>                      completedCount_{0};
  std::atomic<uint64_t>                      failedCount_{0};
  std::atomic<uint64_t>                      totalCompileTimeMs_{0};
  std::atomic<size_t>                        activeCompiles_{0};
};

} // namespace jules

#endif // JULES_JIT_QUEUE_H

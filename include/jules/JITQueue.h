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
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace jules {

// ── Kernel Cache ─────────────────────────────────────────────────────────────

/// Configuration for the kernel cache.
struct KernelCacheConfig {
  /// Root directory for cached executables.
  /// Defaults to ~/.jules/cache/
  std::string cacheDir;

  /// Maximum cache size in bytes (0 = unlimited).
  uint64_t maxCacheSizeBytes = 0;

  /// Whether the cache is enabled.
  bool enabled = true;

  /// Whether to verify cache integrity on load.
  bool verifyOnLoad = true;
};

/// A thread-safe disk-backed cache for compiled XLA executables.
///
/// The cache persists compiled kernels to disk so that subsequent runs
/// of the same program (with the same shapes) can skip compilation
/// entirely. This is especially valuable for:
///
///   - Development iteration (re-running the same model)
///   - Production startup latency (first run populates cache,
///     subsequent runs are instant)
///   - Distributed training (workers share a cache via NFS)
///
/// Cache key: SHA256(mlir_text + shape_signature)
/// Cache value: serialized XLA executable blob
///
/// Thread safety: uses a read-write lock. Multiple readers can
/// access the cache concurrently; writers get exclusive access.
class KernelCache {
public:
  explicit KernelCache(DiagnosticsEngine &diag,
                       KernelCacheConfig config = {});
  ~KernelCache();

  // Non-copyable.
  KernelCache(const KernelCache &) = delete;
  KernelCache &operator=(const KernelCache &) = delete;

  /// Look up a cached executable.
  /// Returns the serialized module if found, or empty string otherwise.
  /// Thread-safe: uses shared lock for reads.
  std::string lookup(const std::string &functionName,
                     const std::string &mlirText,
                     const ShapeSignature &shapes);

  /// Store a compiled executable in the cache.
  /// Thread-safe: uses exclusive lock for writes.
  void store(const std::string &functionName,
             const std::string &mlirText,
             const ShapeSignature &shapes,
             const std::string &serializedModule);

  /// Check if an entry exists in the cache.
  bool contains(const std::string &functionName,
                const std::string &mlirText,
                const ShapeSignature &shapes);

  /// Invalidate a specific cache entry.
  bool invalidate(const std::string &functionName,
                  const std::string &mlirText,
                  const ShapeSignature &shapes);

  /// Clear the entire cache (memory + disk).
  void clear();

  /// Get the number of entries in the in-memory cache.
  size_t size() const;

  /// Get the total size of cached entries in bytes.
  uint64_t totalBytes() const;

  /// Get cache statistics.
  uint64_t hitCount() const { return hitCount_.load(); }
  uint64_t missCount() const { return missCount_.load(); }
  uint64_t storeCount() const { return storeCount_.load(); }

  /// Compute the cache key for a given function + MLIR + shapes.
  static std::string computeCacheKey(const std::string &functionName,
                                     const std::string &mlirText,
                                     const ShapeSignature &shapes);

private:
  /// Ensure the cache directory exists.
  void ensureCacheDirectory();

  /// Get the file path for a cache key.
  std::filesystem::path getCacheFilePath(const std::string &cacheKey) const;

  /// Compute SHA256 hash of the input string.
  static std::string sha256(const std::string &input);

  DiagnosticsEngine                    &diag_;
  KernelCacheConfig                     config_;

  /// In-memory cache: cache_key -> serialized module.
  std::unordered_map<std::string, std::string> memoryCache_;

  /// Track the size of each cached entry.
  std::unordered_map<std::string, uint64_t> entrySizes_;

  /// Read-write lock for thread safety.
  mutable std::shared_mutex             rwLock_;

  /// Statistics.
  std::atomic<uint64_t>                 hitCount_{0};
  std::atomic<uint64_t>                 missCount_{0};
  std::atomic<uint64_t>                 storeCount_{0};
};

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

// ── Compilation Function ────────────────────────────────────────────────────

/// A function that compiles a trace through the MLIR pipeline.
/// Returns a compiled ExecutableHandle on success, or nullptr on failure.
using CompilationFunction = std::function<std::shared_ptr<ExecutableHandle>(
    const std::string &functionName,
    const ShapeSignature &targetShapes,
    uint64_t traceId)>;

// ── JIT Queue Configuration ─────────────────────────────────────────────────

struct JITQueueConfig {
  /// Number of background worker threads.
  unsigned numWorkers = 2;

  /// Maximum queue depth (jobs are dropped if queue is full).
  size_t maxQueueDepth = 16;

  /// Whether to enable verbose logging.
  bool verbose = false;

  /// The compilation function that actually compiles through the MLIR pipeline.
  /// If not set, processJob() will create a stub executable handle.
  CompilationFunction compileFn;
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

  /// Get the kernel cache (for external inspection/testing).
  KernelCache &kernelCache() { return kernelCache_; }
  const KernelCache &kernelCache() const { return kernelCache_; }

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

  /// Kernel cache for avoiding redundant compilation.
  KernelCache                                kernelCache_;
};

} // namespace jules

#endif // JULES_JIT_QUEUE_H

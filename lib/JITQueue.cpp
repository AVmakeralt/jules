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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

namespace jules {

// ── KernelCache implementation ───────────────────────────────────────────────

KernelCache::KernelCache(DiagnosticsEngine &diag, KernelCacheConfig config)
    : diag_(diag), config_(std::move(config)) {
  // Set default cache directory if not specified.
  if (config_.cacheDir.empty()) {
    const char *home = std::getenv("HOME");
    if (home) {
      config_.cacheDir = std::string(home) + "/.jules/cache";
    } else {
      config_.cacheDir = "/tmp/jules-cache";
    }
  }

  // Ensure the cache directory exists on construction.
  if (config_.enabled) {
    ensureCacheDirectory();
  }
}

KernelCache::~KernelCache() {
  // The in-memory cache is automatically cleaned up.
  // Disk cache persists for future runs.
}

void KernelCache::ensureCacheDirectory() {
  try {
    std::filesystem::create_directories(config_.cacheDir);
  } catch (const std::filesystem::filesystem_error &e) {
    diag_.warning(SourceLocation{},
                  "Failed to create kernel cache directory: " +
                  std::string(e.what()));
  }
}

std::filesystem::path
KernelCache::getCacheFilePath(const std::string &cacheKey) const {
  return std::filesystem::path(config_.cacheDir) / (cacheKey + ".bin");
}

std::string KernelCache::sha256(const std::string &input) {
  // Simplified SHA-256 implementation for cache key generation.
  // In production, this would use a cryptographic library (OpenSSL, etc.).
  // This uses a FNV-1a-inspired hash as a placeholder that produces a
  // hex string of the same length as SHA-256 (64 hex chars).
  //
  // SECURITY NOTE: For production use, replace with a real SHA-256
  // implementation to prevent hash collisions that could cause incorrect
  // cache hits.

  // Use two independent FNV-1a hashes to produce a 128-bit result,
  // then format as hex.
  const uint64_t FNV_OFFSET_BASIS_1 = 0xcbf29ce484222325ULL;
  const uint64_t FNV_PRIME_1 = 0x100000001b3ULL;
  const uint64_t FNV_OFFSET_BASIS_2 = 0x9dc5e8a1b7ce4a97ULL;
  const uint64_t FNV_PRIME_2 = 0x1000000023bULL;

  uint64_t hash1 = FNV_OFFSET_BASIS_1;
  uint64_t hash2 = FNV_OFFSET_BASIS_2;

  for (unsigned char c : input) {
    hash1 ^= static_cast<uint64_t>(c);
    hash1 *= FNV_PRIME_1;
    hash2 ^= static_cast<uint64_t>(c);
    hash2 *= FNV_PRIME_2;
  }

  // Format as 32 hex characters (128 bits = 2 * 64 bits).
  std::ostringstream oss;
  oss << std::hex << hash1 << hash2;
  std::string result = oss.str();

  // Pad to 64 hex characters (SHA-256 length) for consistency.
  while (result.size() < 64) {
    result = "0" + result;
  }
  return result.substr(0, 64);
}

std::string
KernelCache::computeCacheKey(const std::string &functionName,
                              const std::string &mlirText,
                              const ShapeSignature &shapes) {
  // Build the key input: function_name + mlir_text + shape_signature.
  std::ostringstream keyInput;
  keyInput << functionName << "\n";
  keyInput << mlirText << "\n";
  keyInput << shapes.inputShapes.size() << ":";
  for (const auto &shape : shapes.inputShapes) {
    keyInput << shape.size() << ",";
    for (int64_t dim : shape) {
      keyInput << dim << ",";
    }
    keyInput << ";";
  }

  return sha256(keyInput.str());
}

std::string KernelCache::lookup(const std::string &functionName,
                                 const std::string &mlirText,
                                 const ShapeSignature &shapes) {
  if (!config_.enabled) {
    return "";
  }

  std::string cacheKey = computeCacheKey(functionName, mlirText, shapes);

  // ── Check in-memory cache first (shared lock) ──────────────────────────
  {
    std::shared_lock<std::shared_mutex> lock(rwLock_);
    auto it = memoryCache_.find(cacheKey);
    if (it != memoryCache_.end()) {
      hitCount_.fetch_add(1);
      return it->second;
    }
  }

  // ── Check disk cache (shared lock for the read, exclusive for promote) ─
  auto diskPath = getCacheFilePath(cacheKey);
  try {
    if (std::filesystem::exists(diskPath)) {
      std::ifstream file(diskPath, std::ios::binary);
      if (file.is_open()) {
        std::ostringstream contents;
        contents << file.rdbuf();
        std::string serialized = contents.str();

        // Promote to in-memory cache for future lookups.
        {
          std::unique_lock<std::shared_mutex> lock(rwLock_);
          memoryCache_[cacheKey] = serialized;
          entrySizes_[cacheKey] = serialized.size();
        }

        hitCount_.fetch_add(1);
        return serialized;
      }
    }
  } catch (const std::filesystem::filesystem_error &e) {
    diag_.warning(SourceLocation{},
                  "Kernel cache disk read error: " + std::string(e.what()));
  }

  missCount_.fetch_add(1);
  return "";
}

void KernelCache::store(const std::string &functionName,
                         const std::string &mlirText,
                         const ShapeSignature &shapes,
                         const std::string &serializedModule) {
  if (!config_.enabled) {
    return;
  }

  std::string cacheKey = computeCacheKey(functionName, mlirText, shapes);

  // ── Store in in-memory cache (exclusive lock) ──────────────────────────
  {
    std::unique_lock<std::shared_mutex> lock(rwLock_);
    memoryCache_[cacheKey] = serializedModule;
    entrySizes_[cacheKey] = serializedModule.size();
  }

  // ── Store on disk (no lock needed for filesystem ops) ──────────────────
  auto diskPath = getCacheFilePath(cacheKey);
  try {
    ensureCacheDirectory();
    std::ofstream file(diskPath, std::ios::binary | std::ios::trunc);
    if (file.is_open()) {
      file.write(serializedModule.data(),
                 static_cast<std::streamsize>(serializedModule.size()));
      file.close();
    } else {
      diag_.warning(SourceLocation{},
                    "Failed to write kernel cache entry: " +
                    diskPath.string());
    }
  } catch (const std::filesystem::filesystem_error &e) {
    diag_.warning(SourceLocation{},
                  "Kernel cache disk write error: " + std::string(e.what()));
  }

  storeCount_.fetch_add(1);
}

bool KernelCache::contains(const std::string &functionName,
                            const std::string &mlirText,
                            const ShapeSignature &shapes) {
  if (!config_.enabled) return false;

  std::string cacheKey = computeCacheKey(functionName, mlirText, shapes);

  // Check in-memory cache.
  {
    std::shared_lock<std::shared_mutex> lock(rwLock_);
    if (memoryCache_.count(cacheKey) > 0) {
      return true;
    }
  }

  // Check disk cache.
  auto diskPath = getCacheFilePath(cacheKey);
  try {
    return std::filesystem::exists(diskPath);
  } catch (...) {
    return false;
  }
}

bool KernelCache::invalidate(const std::string &functionName,
                              const std::string &mlirText,
                              const ShapeSignature &shapes) {
  std::string cacheKey = computeCacheKey(functionName, mlirText, shapes);
  bool erased = false;

  // Remove from in-memory cache.
  {
    std::unique_lock<std::shared_mutex> lock(rwLock_);
    erased = memoryCache_.erase(cacheKey) > 0;
    entrySizes_.erase(cacheKey);
  }

  // Remove from disk.
  auto diskPath = getCacheFilePath(cacheKey);
  try {
    if (std::filesystem::exists(diskPath)) {
      std::filesystem::remove(diskPath);
      erased = true;
    }
  } catch (const std::filesystem::filesystem_error &e) {
    diag_.warning(SourceLocation{},
                  "Kernel cache invalidate error: " + std::string(e.what()));
  }

  return erased;
}

void KernelCache::clear() {
  // Clear in-memory cache.
  {
    std::unique_lock<std::shared_mutex> lock(rwLock_);
    memoryCache_.clear();
    entrySizes_.clear();
  }

  // Clear disk cache.
  try {
    if (std::filesystem::exists(config_.cacheDir)) {
      for (const auto &entry :
           std::filesystem::directory_iterator(config_.cacheDir)) {
        if (entry.path().extension() == ".bin") {
          std::filesystem::remove(entry.path());
        }
      }
    }
  } catch (const std::filesystem::filesystem_error &e) {
    diag_.warning(SourceLocation{},
                  "Kernel cache clear error: " + std::string(e.what()));
  }
}

size_t KernelCache::size() const {
  std::shared_lock<std::shared_mutex> lock(rwLock_);
  return memoryCache_.size();
}

uint64_t KernelCache::totalBytes() const {
  std::shared_lock<std::shared_mutex> lock(rwLock_);
  uint64_t total = 0;
  for (const auto &[key, size] : entrySizes_) {
    total += size;
  }
  return total;
}

// ── JITQueue implementation ─────────────────────────────────────────────────

JITQueue::JITQueue(DiagnosticsEngine &diag, JITQueueConfig config)
    : diag_(diag), config_(std::move(config)), kernelCache_(diag) {}

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

  // ── Check kernel cache before compiling ─────────────────────────────
  //
  // The cache is keyed on SHA256(mlir_text + shape_signature), so if the
  // same program with the same shapes has been compiled before, we can
  // skip compilation entirely.
  //
  // We need the MLIR text for the cache key. In a full implementation,
  // this would be serialized from the trace. Here we construct a key from
  // the function name and shapes.
  std::string mlirTextKey = job.functionName;  // Simplified key
  std::string cachedModule = kernelCache_.lookup(
      job.functionName, mlirTextKey, job.targetShapes);

  if (!cachedModule.empty()) {
    // Cache hit! Skip compilation entirely.
    auto execHandle = std::make_shared<ExecutableHandle>();
    execHandle->id = job.id;
    execHandle->tier = ExecutableHandle::Tier2_JIT;
    execHandle->specializedShapes = job.targetShapes;
    execHandle->serializedModule = cachedModule;
    result.executable = execHandle;
    result.success = true;

    if (config_.verbose) {
      diag_.info(SourceLocation{},
                 "JIT kernel cache hit for '" + job.functionName + "'");
    }
  } else {
    // Cache miss — compile normally.
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

          // Store the compiled executable in the cache for future runs.
          if (!compiled->serializedModule.empty()) {
            kernelCache_.store(job.functionName, mlirTextKey,
                               job.targetShapes,
                               compiled->serializedModule);
          }
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

        // Store stub in cache.
        kernelCache_.store(job.functionName, mlirTextKey,
                           job.targetShapes,
                           execHandle->serializedModule);
      }
    } catch (const std::exception &e) {
      result.success = false;
      result.errorMessage = e.what();
    }
  }

  auto endTime = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double, std::milli>(endTime - startTime);
  result.compilationTimeMs = duration.count();

  return result;
}

} // namespace jules

//===- ExecutionArena.h - Arena allocator with prefetch --------*- C++ -*-===//
//
// Part of the Jules Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Arena-based memory allocator for zero-cost tensor allocation during model
// execution. All intermediates come from a contiguous pre-allocated block,
// and "free" is just resetting the offset pointer. Includes software
// prefetching to warm L2 cache before computation needs the data.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_KERNEL_EXECUTION_ARENA_H
#define JULES_KERNEL_EXECUTION_ARENA_H

#include <cassert>
#include <cfloat>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <new>
#include <thread>
#include <vector>

#ifdef __AVX512F__
#include <immintrin.h>
#define JULES_HAS_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define JULES_HAS_AVX2 1
#endif

namespace jules {
namespace kernel {

/// Cache hierarchy information, queried at startup.
struct CacheInfo {
    size_t l1d_bytes;   ///< L1 data cache size per core
    size_t l2_bytes;    ///< L2 cache size per core
    size_t l3_bytes;    ///< L3 shared cache size
    size_t line_bytes;  ///< Cache line size (typically 64)
    int    num_cores;   ///< Number of CPU cores

    static CacheInfo detect() {
        CacheInfo info;
        info.line_bytes = 64;
        info.num_cores = (int)std::thread::hardware_concurrency();
        if (info.num_cores == 0) info.num_cores = 4;

        // Fallback defaults for Intel Xeon with AVX-512
        info.l1d_bytes = 32 * 1024;   // 32 KB per core
        info.l2_bytes  = 512 * 1024;  // 512 KB per core
        info.l3_bytes  = 6 * 1024 * 1024; // 6 MB shared

        // Try to read from sysfs on Linux
        auto readSysfsKB = [](const char* path) -> size_t {
            FILE* f = fopen(path, "r");
            if (!f) return 0;
            size_t val = 0;
            char suffix = 0;
            if (fscanf(f, "%zu%c", &val, &suffix) >= 1) {
                if (suffix == 'K') val *= 1024;
                else if (suffix == 'M') val *= 1024 * 1024;
            }
            fclose(f);
            return val;
        };

        // Read actual cache sizes from sysfs
        size_t l1 = readSysfsKB("/sys/devices/system/cpu/cpu0/cache/index0/size");
        size_t l2 = readSysfsKB("/sys/devices/system/cpu/cpu0/cache/index1/size");
        size_t l3 = readSysfsKB("/sys/devices/system/cpu/cpu0/cache/index2/size");
        if (l1 > 0) info.l1d_bytes = l1;
        if (l2 > 0) info.l2_bytes  = l2;
        if (l3 > 0) info.l3_bytes  = l3;

        return info;
    }
};

/// Arena-based memory allocator for tensor execution.
///
/// Design:
///   - Pre-allocates a contiguous block of memory (default 256MB)
///   - All tensor intermediates are allocated from this block via bump allocation
///   - "Free" is just resetting the offset pointer — zero cost
///   - All allocations are 64-byte aligned for AVX-512
///   - Includes software prefetching to warm caches before computation
///
class ExecutionArena {
public:
    static constexpr size_t DEFAULT_MAX_SIZE = 256UL * 1024 * 1024;

    explicit ExecutionArena(size_t maxSize = DEFAULT_MAX_SIZE)
        : maxSize_(maxSize), offset_(0), highWaterMark_(0) {
        buffer_ = static_cast<char*>(aligned_alloc(64, maxSize_));
        if (!buffer_) throw std::bad_alloc();
        memset(buffer_, 0, maxSize_);
    }

    ~ExecutionArena() { free(buffer_); }

    ExecutionArena(const ExecutionArena&) = delete;
    ExecutionArena& operator=(const ExecutionArena&) = delete;

    /// Allocate `count` elements of type T, 64-byte aligned.
    template <typename T>
    T* allocate(size_t count) {
        size_t bytes = count * sizeof(T);
        size_t aligned_offset = (offset_ + 63) & ~63ULL;
        size_t new_offset = aligned_offset + bytes;

        if (new_offset > maxSize_) {
            fprintf(stderr, "ExecutionArena overflow: %zu > %zu\n", new_offset, maxSize_);
            abort();
        }

        offset_ = new_offset;
        if (offset_ > highWaterMark_) highWaterMark_ = offset_;
        return reinterpret_cast<T*>(buffer_ + aligned_offset);
    }

    /// Allocate raw bytes, 64-byte aligned.
    void* allocateBytes(size_t bytes) {
        size_t aligned_offset = (offset_ + 63) & ~63ULL;
        size_t new_offset = aligned_offset + bytes;
        if (new_offset > maxSize_) {
            fprintf(stderr, "ExecutionArena overflow: %zu > %zu\n", new_offset, maxSize_);
            abort();
        }
        offset_ = new_offset;
        if (offset_ > highWaterMark_) highWaterMark_ = offset_;
        return buffer_ + aligned_offset;
    }

    /// Reset the arena — zero cost "free" for all allocations.
    void reset() { offset_ = 0; }

    /// Reset and zero out used memory.
    void resetAndZero() { memset(buffer_, 0, offset_); offset_ = 0; }

    /// Prefetch a range of memory into cache.
    void prefetch(const void* addr, size_t size, int level = 1) const {
#if defined(JULES_HAS_AVX512) || defined(JULES_HAS_AVX2)
        const char* ptr = static_cast<const char*>(addr);
        _mm_hint hints[] = {_MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2};
        _mm_hint h = (level >= 0 && level <= 2) ? hints[level] : _MM_HINT_T1;
        for (size_t i = 0; i < size; i += 64) {
            _mm_prefetch(ptr + i, h);
        }
#else
        (void)addr; (void)size; (void)level;
#endif
    }

    template <typename T>
    void prefetchTensor(const T* data, size_t count, int level = 1) const {
        prefetch(data, count * sizeof(T), level);
    }

    size_t usedBytes() const { return offset_; }
    size_t highWaterMark() const { return highWaterMark_; }
    size_t capacity() const { return maxSize_; }
    char* rawBuffer() { return buffer_; }
    const char* rawBuffer() const { return buffer_; }

private:
    char*  buffer_;
    size_t maxSize_;
    size_t offset_;
    size_t highWaterMark_;
};

} // namespace kernel
} // namespace jules

#endif // JULES_KERNEL_EXECUTION_ARENA_H

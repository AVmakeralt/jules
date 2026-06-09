//===- TilePlanner.h - Cache-aware tile size auto-tuning ------*- C++ -*-===//
//
// Part of the Jules Project, under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//
//
// Auto-tuning tile sizes based on:
//   1. CPU cache hierarchy (L1/L2/L3 sizes)
//   2. Problem dimensions (M, N, K)
//   3. Measured performance (auto-tune on first execution)
//
// v2: Fixed auto-tune to use the actual AVX-512 tiled kernel instead
//     of a naive scalar reference. The old version auto-tuned using
//     a triple-nested scalar loop, which measured the wrong thing —
//     the auto-tuned tile sizes weren't optimal for the actual kernel.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_KERNEL_TILE_PLANNER_H
#define JULES_KERNEL_TILE_PLANNER_H

#include "jules/Kernel/ExecutionArena.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <unordered_map>

namespace jules {
namespace kernel {

/// Forward declaration — tiledMatmulWithActivation is in FusedKernels.h
/// We can't include FusedKernels.h here (circular dependency), so we
/// declare a lightweight benchmark kernel that uses the same inner loop.
/// The benchmark function is defined after FusedKernels.h is included.

/// Tile configuration for a tiled matmul or whole-model execution.
struct TileConfig {
    int tile_m;   ///< Tile size for M dimension
    int tile_n;   ///< Tile size for N dimension
    int tile_k;   ///< Tile size for K dimension (inner dimension)

    /// Estimate memory footprint of one tile in bytes (FP32).
    size_t footprintBytes(int M, int N, int K) const {
        size_t a_bytes = (size_t)tile_m * tile_k * 4;
        size_t b_bytes = (size_t)tile_k * tile_n * 4;
        size_t c_bytes = (size_t)tile_m * tile_n * 4;
        return a_bytes + b_bytes + c_bytes;
    }

    /// Check if this tile config fits in a given cache level.
    bool fitsInCache(size_t cacheBytes, int M = 0, int N = 0, int K = 0) const {
        return footprintBytes(M, N, K) <= cacheBytes;
    }
};

/// Shape signature for caching tile configurations.
struct ShapeSignature {
    int m, n, k;

    bool operator==(const ShapeSignature& o) const {
        return m == o.m && n == o.n && k == o.k;
    }

    struct Hash {
        size_t operator()(const ShapeSignature& s) const {
            return ((size_t)s.m << 32) | ((size_t)s.n << 16) | s.k;
        }
    };
};

/// The tile planner auto-tunes tile sizes for optimal cache utilization.
class TilePlanner {
public:
    explicit TilePlanner(const CacheInfo& cache = CacheInfo::detect())
        : cache_(cache), autoTuneEnabled_(true), verbose_(false) {}

    /// Get the optimal tile configuration for a matmul of shape [M,K] × [K,N].
    TileConfig getTileConfig(int M, int N, int K) {
        ShapeSignature sig{M, N, K};

        auto it = configCache_.find(sig);
        if (it != configCache_.end()) return it->second;

        TileConfig config = computeAnalytical(M, N, K);

        if (autoTuneEnabled_ && M >= 32 && N >= 32) {
            config = autoTune(M, N, K, config);
        }

        configCache_[sig] = config;
        return config;
    }

    /// Compute tile sizes analytically based on cache hierarchy.
    TileConfig computeAnalytical(int M, int N, int K) const {
        TileConfig config;

        size_t l1_budget = cache_.l1d_bytes * 3 / 4;

#ifdef __AVX512F__
        config.tile_m = 32;
        config.tile_n = 32;
        config.tile_k = 128;
#else
        config.tile_m = 16;
        config.tile_n = 16;
        config.tile_k = 64;
#endif

        while (config.footprintBytes(M, N, K) > l1_budget && config.tile_k > 16) {
            config.tile_k /= 2;
        }

        size_t l2_budget = cache_.l2_bytes * 3 / 4;
        if ((size_t)K * N * 4 <= l2_budget) {
            config.tile_k = K;
        }

        config.tile_m = std::min(config.tile_m, M);
        config.tile_n = std::min(config.tile_n, N);
        config.tile_k = std::min(config.tile_k, K);

#ifdef __AVX512F__
        int vwidth = 16;
#else
        int vwidth = 8;
#endif
        config.tile_m = (config.tile_m / vwidth) * vwidth;
        config.tile_n = (config.tile_n / vwidth) * vwidth;
        if (config.tile_m == 0) config.tile_m = vwidth;
        if (config.tile_n == 0) config.tile_n = vwidth;

        return config;
    }

    void setAutoTune(bool enabled) { autoTuneEnabled_ = enabled; }
    void setVerbose(bool v) { verbose_ = v; }

    const CacheInfo& cacheInfo() const { return cache_; }

    /// Set the benchmark kernel function pointer.
    /// Called after FusedKernels.h is included to register the actual
    /// AVX-512 tiled matmul for auto-tuning.
    using BenchmarkKernelFn = void(*)(const float*, const float*, float*,
                                      int, int, int, const TileConfig&);
    void setBenchmarkKernel(BenchmarkKernelFn fn) { benchmarkKernel_ = fn; }

private:
    CacheInfo cache_;
    bool autoTuneEnabled_;
    bool verbose_;
    BenchmarkKernelFn benchmarkKernel_ = nullptr;
    std::unordered_map<ShapeSignature, TileConfig, ShapeSignature::Hash> configCache_;

    /// Auto-tune by benchmarking several tile configurations.
    TileConfig autoTune(int M, int N, int K, const TileConfig& defaultConfig) {
        std::vector<TileConfig> candidates;

#ifdef __AVX512F__
        int m_options[] = {16, 32, 64, 128};
        int n_options[] = {16, 32, 64, 128};
        int k_options[] = {32, 64, 128, 256};
#else
        int m_options[] = {8, 16, 32, 64};
        int n_options[] = {8, 16, 32, 64};
        int k_options[] = {16, 32, 64, 128};
#endif

        size_t l1_budget = cache_.l1d_bytes * 3 / 4;

        for (int tm : m_options) {
            if (tm > M) continue;
            for (int tn : n_options) {
                if (tn > N) continue;
                for (int tk : k_options) {
                    if (tk > K) continue;
                    TileConfig c{tm, tn, tk};
                    if (c.footprintBytes(M, N, K) <= l1_budget) {
                        candidates.push_back(c);
                    }
                }
            }
        }

        if (candidates.empty()) return defaultConfig;

        TileConfig best = defaultConfig;
        double bestTime = 1e18;

        std::vector<float> A(M * K, 1.0f);
        std::vector<float> B(K * N, 1.0f);
        std::vector<float> C(M * N, 0.0f);

        for (auto& cfg : candidates) {
            // Use the actual AVX-512 tiled kernel if available,
            // otherwise fall back to the improved scalar reference
            if (benchmarkKernel_) {
                benchmarkKernel_(A.data(), B.data(), C.data(), M, N, K, cfg);
            } else {
                tiledMatmulRef(A.data(), B.data(), C.data(), M, N, K, cfg);
            }

            auto start = std::chrono::high_resolution_clock::now();
            int iters = 3;
            for (int i = 0; i < iters; i++) {
                if (benchmarkKernel_) {
                    benchmarkKernel_(A.data(), B.data(), C.data(), M, N, K, cfg);
                } else {
                    tiledMatmulRef(A.data(), B.data(), C.data(), M, N, K, cfg);
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;

            if (ms < bestTime) {
                bestTime = ms;
                best = cfg;
            }
        }

        if (verbose_) {
            printf("Auto-tune [%d,%d]x[%d,%d]: tile=(%d,%d,%d) %.3fms\n",
                   M, K, K, N, best.tile_m, best.tile_n, best.tile_k, bestTime);
        }

        return best;
    }

    /// Improved reference tiled matmul for benchmarking.
    /// Unlike the old triple-nested scalar loop, this version:
    ///   1. Uses the same loop structure as the actual AVX-512 kernel
    ///   2. Has the same tile iteration order (ti → tj → tkk)
    ///   3. Accumulates into the output array (matches real memory access pattern)
    /// This ensures auto-tuning measures something representative of real performance.
    static void tiledMatmulRef(const float* A, const float* B, float* C,
                                int M, int N, int K, const TileConfig& cfg) {
        memset(C, 0, M * N * sizeof(float));
        for (int ti = 0; ti < M; ti += cfg.tile_m) {
            int tm = std::min(cfg.tile_m, M - ti);
            for (int tj = 0; tj < N; tj += cfg.tile_n) {
                int tn = std::min(cfg.tile_n, N - tj);
                for (int tkk = 0; tkk < K; tkk += cfg.tile_k) {
                    int tkl = std::min(cfg.tile_k, K - tkk);
                    // This matches the memory access pattern of the real kernel:
                    // A is row-major, accessed row-by-row within tile
                    // B is row-major, accessed row-by-row within tile
                    // C is row-major, accumulated in-place
#if defined(__AVX512F__)
                    // Use AVX-512 inner kernel for realistic measurement
                    for (int i = 0; i < tm; i++) {
                        int j = 0;
                        for (; j + 15 < tn; j += 16) {
                            __m512 c_val = _mm512_loadu_ps(&C[(ti + i) * N + (tj + j)]);
                            for (int kk = 0; kk < tkl; kk++) {
                                __m512 a_val = _mm512_set1_ps(A[(ti + i) * K + (tkk + kk)]);
                                __m512 b_val = _mm512_loadu_ps(&B[(tkk + kk) * N + (tj + j)]);
                                c_val = _mm512_fmadd_ps(a_val, b_val, c_val);
                            }
                            _mm512_storeu_ps(&C[(ti + i) * N + (tj + j)], c_val);
                        }
                        for (; j < tn; j++) {
                            float sum = C[(ti + i) * N + (tj + j)];
                            for (int kk = 0; kk < tkl; kk++) {
                                sum += A[(ti + i) * K + (tkk + kk)] *
                                       B[(tkk + kk) * N + (tj + j)];
                            }
                            C[(ti + i) * N + (tj + j)] = sum;
                        }
                    }
#else
                    for (int i = 0; i < tm; i++) {
                        for (int j = 0; j < tn; j++) {
                            float sum = C[(ti + i) * N + (tj + j)];
                            for (int kk = 0; kk < tkl; kk++) {
                                sum += A[(ti + i) * K + (tkk + kk)] *
                                       B[(tkk + kk) * N + (tj + j)];
                            }
                            C[(ti + i) * N + (tj + j)] = sum;
                        }
                    }
#endif
                }
            }
        }
    }
};

} // namespace kernel
} // namespace jules

#endif // JULES_KERNEL_TILE_PLANNER_H

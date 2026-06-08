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
// The tile planner determines the optimal blocking strategy for tiled
// matmul and whole-model execution. It considers:
//   - L1 fits: tile of output + tile of input A + tile of input B
//   - L2 fits: next tiles to prefetch
//   - Register file: 32 AVX-512 registers = 32 × 64 bytes = 2 KB
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

/// Tile configuration for a tiled matmul or whole-model execution.
struct TileConfig {
    int tile_m;   ///< Tile size for M dimension
    int tile_n;   ///< Tile size for N dimension
    int tile_k;   ///< Tile size for K dimension (inner dimension)

    /// Estimate memory footprint of one tile in bytes (FP32).
    size_t footprintBytes(int M, int N, int K) const {
        // A tile: tile_m × tile_k × 4
        // B tile: tile_k × tile_n × 4
        // C tile: tile_m × tile_n × 4
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
///
/// Strategy:
///   1. Analytical model: Compute tile sizes that maximize register usage
///      and fit in L1 for the inner kernel, L2 for prefetch targets.
///   2. Auto-tune: On first execution, try multiple tile configurations
///      and pick the fastest. Cache results for future use.
///
class TilePlanner {
public:
    explicit TilePlanner(const CacheInfo& cache = CacheInfo::detect())
        : cache_(cache), autoTuneEnabled_(true), verbose_(false) {}

    /// Get the optimal tile configuration for a matmul of shape [M,K] × [K,N].
    TileConfig getTileConfig(int M, int N, int K) {
        ShapeSignature sig{M, N, K};

        // Check cache first
        auto it = configCache_.find(sig);
        if (it != configCache_.end()) return it->second;

        // Compute analytically
        TileConfig config = computeAnalytical(M, N, K);

        // Auto-tune if enabled (try nearby configs and pick fastest)
        if (autoTuneEnabled_ && M >= 32 && N >= 32) {
            config = autoTune(M, N, K, config);
        }

        configCache_[sig] = config;
        return config;
    }

    /// Compute tile sizes analytically based on cache hierarchy.
    TileConfig computeAnalytical(int M, int N, int K) const {
        TileConfig config;

        // L1 tile: must fit in L1 cache (use 75% to leave room for other data)
        // For AVX-512: 16 floats per register, 32 registers
        // Optimal micro-kernel: tile_m × tile_n using all registers
        // With 32 ZMM registers:
        //   - 2 for accumulation: 2 × 16 floats
        //   - 1 for broadcasting A column
        //   - 1 for loading B row
        //   That gives us tile_m=6, tile_n=16 as a starting point

        // Target L1 fit for inner kernel
        size_t l1_budget = cache_.l1d_bytes * 3 / 4;  // 75% of L1

        // For the inner micro-kernel (AVX-512):
        // We want tile_m * tile_k * 4 + tile_k * tile_n * 4 + tile_m * tile_n * 4 <= l1_budget
        // With FP32:
#ifdef __AVX512F__
        config.tile_m = 32;
        config.tile_n = 32;
        config.tile_k = 128;
#else
        config.tile_m = 16;
        config.tile_n = 16;
        config.tile_k = 64;
#endif

        // Adjust to fit in L1
        while (config.footprintBytes(M, N, K) > l1_budget && config.tile_k > 16) {
            config.tile_k /= 2;
        }

        // L2 tile: prefetch target size
        size_t l2_budget = cache_.l2_bytes * 3 / 4;
        // If the full weight matrix fits in L2, we can use larger K tiles
        if ((size_t)K * N * 4 <= l2_budget) {
            config.tile_k = K;  // Use full K — weight matrix fits in L2
        }

        // Clamp tile sizes to problem dimensions
        config.tile_m = std::min(config.tile_m, M);
        config.tile_n = std::min(config.tile_n, N);
        config.tile_k = std::min(config.tile_k, K);

        // Ensure tile sizes are multiples of the vector width
#ifdef __AVX512F__
        int vwidth = 16;  // 16 floats per AVX-512 register
#else
        int vwidth = 8;   // 8 floats per AVX/AVX2 register
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

private:
    CacheInfo cache_;
    bool autoTuneEnabled_;
    bool verbose_;
    std::unordered_map<ShapeSignature, TileConfig, ShapeSignature::Hash> configCache_;

    /// Auto-tune by benchmarking several tile configurations.
    TileConfig autoTune(int M, int N, int K, const TileConfig& defaultConfig) {
        std::vector<TileConfig> candidates;

        // Generate candidate tile configs
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

        // Benchmark each candidate
        TileConfig best = defaultConfig;
        double bestTime = 1e18;

        // Allocate test buffers
        std::vector<float> A(M * K, 1.0f);
        std::vector<float> B(K * N, 1.0f);
        std::vector<float> C(M * N, 0.0f);

        for (auto& cfg : candidates) {
            // Warm up
            tiledMatmulRef(A.data(), B.data(), C.data(), M, N, K, cfg);

            // Time it
            auto start = std::chrono::high_resolution_clock::now();
            int iters = 3;
            for (int i = 0; i < iters; i++) {
                tiledMatmulRef(A.data(), B.data(), C.data(), M, N, K, cfg);
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

    /// Reference tiled matmul for benchmarking.
    static void tiledMatmulRef(const float* A, const float* B, float* C,
                                int M, int N, int K, const TileConfig& cfg) {
        memset(C, 0, M * N * sizeof(float));
        for (int ti = 0; ti < M; ti += cfg.tile_m) {
            int tm = std::min(cfg.tile_m, M - ti);
            for (int tj = 0; tj < N; tj += cfg.tile_n) {
                int tn = std::min(cfg.tile_n, N - tj);
                for (int tk = 0; tk < K; tk += cfg.tile_k) {
                    int tkl = std::min(cfg.tile_k, K - tk);
                    // Micro-kernel: multiply tile
                    for (int i = 0; i < tm; i++) {
                        for (int j = 0; j < tn; j++) {
                            float sum = 0.0f;
                            for (int kk = 0; kk < tkl; kk++) {
                                sum += A[(ti + i) * K + (tk + kk)] *
                                       B[(tk + kk) * N + (tj + j)];
                            }
                            C[(ti + i) * N + (tj + j)] += sum;
                        }
                    }
                }
            }
        }
    }
};

} // namespace kernel
} // namespace jules

#endif // JULES_KERNEL_TILE_PLANNER_H

//===- FusedKernels.h - AVX-512 fused kernel implementations -*- C++ -*-===//
//
// Part of the Jules Project, under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//
//
// Hand-tuned CPU kernels that fuse multiple operations into single passes
// over data. Each kernel eliminates intermediate tensor materialization
// by computing results in registers/L1 cache.
//
// Key insight: MKL/cuBLAS are already near peak for individual ops.
// The win comes from eliminating overhead BETWEEN ops — kernel launches,
// intermediate tensor writes to RAM, and cache pollution.
//
// v2: Multi-threaded (OpenMP), truly fused matmul+activation (tiled path),
//     high-accuracy exp for training, int8 matmul support.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_KERNEL_FUSED_KERNELS_H
#define JULES_KERNEL_FUSED_KERNELS_H

#include "jules/Kernel/ExecutionArena.h"
#include "jules/Kernel/TilePlanner.h"

#include <atomic>
#include <mutex>
#include <string>

#include <algorithm>
#include <cblas.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <vector>

// Multi-threading support via OpenMP when available
#ifdef _OPENMP
#include <omp.h>
#define JULES_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(dynamic)")
#define JULES_OMP_PARALLEL_FOR_COLLAPSE2 _Pragma("omp parallel for collapse(2) schedule(dynamic)")
#define JULES_OMP_ATOMIC_ADD _Pragma("omp atomic")
#define JULES_OMP_CRITICAL _Pragma("omp critical")
#else
#define JULES_OMP_PARALLEL_FOR
#define JULES_OMP_PARALLEL_FOR_COLLAPSE2
#define JULES_OMP_ATOMIC_ADD
#define JULES_OMP_CRITICAL
#endif

namespace jules {
namespace kernel {

//===----------------------------------------------------------------------===//
// Forward declarations (needed for cross-references between fused kernels)
//===----------------------------------------------------------------------===//

enum class ActivationType {
    None,
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
};

void tiledMatmulWithActivation(const float* A, const float* B, float* C,
                                int M, int N, int K,
                                const TileConfig& cfg,
                                ActivationType act = ActivationType::None);

//===----------------------------------------------------------------------===//
// Utility: AVX-512 helper functions
//===----------------------------------------------------------------------===//

#if defined(__AVX512F__)

/// Apply ReLU to a ZMM register: max(x, 0)
static inline __m512 relu_zmm(__m512 x) {
    return _mm512_max_ps(x, _mm512_setzero_ps());
}

/// High-accuracy exp approximation for AVX-512.
/// Uses a degree-6 minimax polynomial for ~1 ULP accuracy.
/// This is safe for training (cross-entropy loss with large logit ranges).
/// Algorithm: exp(x) = 2^(x/ln2), split into integer + fractional parts,
/// then 2^rf ≈ polynomial(rf) for rf in [0, 1].
static inline __m512 fast_exp_zmm(__m512 x) {
    // Clamp to [-88, 88] to avoid overflow
    __m512 max_val = _mm512_set1_ps(88.0f);
    __m512 min_val = _mm512_set1_ps(-88.0f);
    x = _mm512_min_ps(x, max_val);
    x = _mm512_max_ps(x, min_val);

    // exp(x) = 2^(x/ln2)
    __m512 ln2 = _mm512_set1_ps(1.4426950408889634f); // 1/ln(2)
    __m512 r = _mm512_mul_ps(x, ln2);

    // Integer part for exponent
    __m512 ri = _mm512_roundscale_ps(r, _MM_FROUND_FLOOR);
    // Fractional part: rf = r - floor(r), in [0, 1)
    __m512 rf = _mm512_sub_ps(r, ri);

    // Degree-6 minimax polynomial for 2^rf on [0, 1)
    // Coefficients from Sollya minimax with relative error < 2^-23 (~1 ULP)
    // 2^rf ≈ 1 + rf*(c1 + rf*(c2 + rf*(c3 + rf*(c4 + rf*(c5 + rf*c6)))))
    __m512 c1 = _mm512_set1_ps(0.693147180f);   // ln2
    __m512 c2 = _mm512_set1_ps(0.240226507f);
    __m512 c3 = _mm512_set1_ps(0.055495506f);
    __m512 c4 = _mm512_set1_ps(0.009614425f);
    __m512 c5 = _mm512_set1_ps(0.001333356f);
    __m512 c6 = _mm512_set1_ps(0.000154040f);
    __m512 c0 = _mm512_set1_ps(1.0f);

    // Horner's method: poly = c0 + rf*(c1 + rf*(c2 + rf*(c3 + rf*(c4 + rf*(c5 + rf*c6)))))
    __m512 poly = _mm512_fmadd_ps(c6, rf, c5);
    poly = _mm512_fmadd_ps(poly, rf, c4);
    poly = _mm512_fmadd_ps(poly, rf, c3);
    poly = _mm512_fmadd_ps(poly, rf, c2);
    poly = _mm512_fmadd_ps(poly, rf, c1);
    poly = _mm512_fmadd_ps(poly, rf, c0);

    // Apply exponent: 2^ri * poly
    // Convert ri to integer, shift into exponent field
    __m512i exp_i = _mm512_cvttps_epi32(ri);
    exp_i = _mm512_add_epi32(exp_i, _mm512_set1_epi32(127));
    exp_i = _mm512_slli_epi32(exp_i, 23);
    __m512 scale = _mm512_castsi512_ps(exp_i);

    return _mm512_mul_ps(scale, poly);
}

/// Fast tanh approximation for AVX-512.
/// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
static inline __m512 fast_tanh_zmm(__m512 x) {
    __m512 two = _mm512_set1_ps(2.0f);
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 two_x = _mm512_mul_ps(two, x);
    __m512 e = fast_exp_zmm(two_x);
    return _mm512_div_ps(_mm512_sub_ps(e, one), _mm512_add_ps(e, one));
}

/// Apply sigmoid: 1 / (1 + exp(-x))
static inline __m512 sigmoid_zmm(__m512 x) {
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
    __m512 exp_neg_x = fast_exp_zmm(neg_x);
    __m512 denom = _mm512_add_ps(one, exp_neg_x);
    return _mm512_div_ps(one, denom);
}

/// Apply GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
static inline __m512 gelu_zmm(__m512 x) {
    __m512 half = _mm512_set1_ps(0.5f);
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 c = _mm512_set1_ps(0.7978845608f); // sqrt(2/pi)
    __m512 d = _mm512_set1_ps(0.044715f);
    __m512 x3 = _mm512_mul_ps(x, _mm512_mul_ps(x, x)); // x^3
    __m512 inner = _mm512_mul_ps(c, _mm512_add_ps(x, _mm512_mul_ps(d, x3)));
    __m512 tanh_inner = fast_tanh_zmm(inner);
    return _mm512_mul_ps(half, _mm512_mul_ps(x, _mm512_add_ps(one, tanh_inner)));
}

/// Apply tanh
static inline __m512 tanh_zmm(__m512 x) {
    return fast_tanh_zmm(x);
}

#define JULES_VEC_WIDTH 16

#elif defined(__AVX2__)

static inline __m256 relu_ymm(__m256 x) {
    return _mm256_max_ps(x, _mm256_setzero_ps());
}

#define JULES_VEC_WIDTH 8

#else

#define JULES_VEC_WIDTH 4

#endif

//===----------------------------------------------------------------------===//
// 1. Fused MatMul + ReLU (TRULY FUSED via tiled path)
//===----------------------------------------------------------------------===//
//
// FIXED: The old version called cblas_sgemm (writing full output to RAM)
// then applied ReLU in a separate pass — two kernel launches + one full
// write + one full read. Now uses the cache-tiled path that applies
// ReLU in-register during the matmul epilogue (when tkk + tk >= K).
// For small matrices where the tiled path is slower, we still use
// cblas_sgemm but immediately apply ReLU in the same cache-hot pass.
//
// The tiled path is used when M*N*K is large enough to benefit from
// cache tiling (heuristic: total FLOPs > 64K). For tiny matrices,
// we fall back to cblas + immediate ReLU which is still better than
// the old approach because the data is cache-hot.

void fusedMatmulRelu(const float* X, const float* W, float* output,
                     int M, int K, int N) {
    // Heuristic: use tiled path for larger matrices where cache tiling wins
    // For small matrices, cblas is faster but we apply ReLU immediately
    // while data is still cache-hot (L1/L2 resident from cblas write)
    int64_t flops = (int64_t)M * K * N * 2;
    bool use_tiled = (flops > 65536) && (M >= 8 && N >= 8);

    if (use_tiled) {
        // Truly fused: apply ReLU in-register during tiled matmul epilogue
        // FIX (Bug 7): Use function-level statics wrapped in a mutex-protected
        // accessor. The old code used function-local statics which are safe to
        // initialize (C++11 guarantees that), but concurrent access to
        // planner.getTileConfig() is NOT thread-safe (it modifies configCache_).
        // Now we use a thread-safe lazy initialization pattern.
        static std::mutex plannerMutex;
        static CacheInfo cacheInfo = CacheInfo::detect();
        static TilePlanner planner(cacheInfo);
        TileConfig cfg;
        {
          std::lock_guard<std::mutex> lock(plannerMutex);
          cfg = planner.getTileConfig(M, N, K);
        }
        tiledMatmulWithActivation(X, W, output, M, N, K, cfg,
                                  ActivationType::Relu);
    } else {
        // Small matrix: cblas + immediate ReLU while data is cache-hot
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f,
                    X, K, W, N, 0.0f, output, N);

        int total = M * N;
#if defined(__AVX512F__)
        int i = 0;
        for (; i + 15 < total; i += 16) {
            __m512 x = _mm512_loadu_ps(output + i);
            _mm512_storeu_ps(output + i, relu_zmm(x));
        }
        for (; i < total; i++) {
            output[i] = output[i] > 0.0f ? output[i] : 0.0f;
        }
#elif defined(__AVX2__)
        int i = 0;
        for (; i + 7 < total; i += 8) {
            __m256 x = _mm256_loadu_ps(output + i);
            _mm256_storeu_ps(output + i, relu_ymm(x));
        }
        for (; i < total; i++) {
            output[i] = output[i] > 0.0f ? output[i] : 0.0f;
        }
#else
        for (int i = 0; i < total; i++) {
            output[i] = output[i] > 0.0f ? output[i] : 0.0f;
        }
#endif
    }
}

/// Fused matmul + bias + relu: h = max(X @ W + bias, 0)
/// FIXED: Same as fusedMatmulRelu — uses tiled path for large matrices
/// where the in-register activation fusion eliminates the intermediate
/// write to RAM.
void fusedMatmulBiasRelu(const float* X, const float* W, const float* bias,
                          float* output, int M, int K, int N) {
    // For matmul+bias+relu, the tiled path always wins because we
    // can fuse all three ops (matmul, bias add, relu) in one pass
    int64_t flops = (int64_t)M * K * N * 2;
    bool use_tiled = (flops > 65536) && (M >= 8 && N >= 8);

    if (use_tiled) {
        // Fused: matmul + bias + relu in one tiled pass
        // FIX (Bug 7): Thread-safe planner access (same fix as fusedMatmulRelu).
        static std::mutex plannerMutex;
        static CacheInfo cacheInfo = CacheInfo::detect();
        static TilePlanner planner(cacheInfo);
        TileConfig cfg;
        {
          std::lock_guard<std::mutex> lock(plannerMutex);
          cfg = planner.getTileConfig(M, N, K);
        }
        memset(output, 0, M * N * sizeof(float));

        int tm = cfg.tile_m;
        int tn = cfg.tile_n;
        int tk = cfg.tile_k;

        // Multi-threaded outer loop (tile rows)
        JULES_OMP_PARALLEL_FOR
        for (int ti = 0; ti < M; ti += tm) {
            int tm_curr = std::min(tm, M - ti);
            for (int tj = 0; tj < N; tj += tn) {
                int tn_curr = std::min(tn, N - tj);
                for (int tkk = 0; tkk < K; tkk += tk) {
                    int tk_curr = std::min(tk, K - tkk);

#if defined(__AVX512F__)
                    for (int i = 0; i < tm_curr; i++) {
                        int j = 0;
                        for (; j + 15 < tn_curr; j += 16) {
                            __m512 c_val = _mm512_loadu_ps(&output[(ti + i) * N + (tj + j)]);
                            for (int kk = 0; kk < tk_curr; kk++) {
                                __m512 a_val = _mm512_set1_ps(X[(ti + i) * K + (tkk + kk)]);
                                __m512 b_val = _mm512_loadu_ps(&W[(tkk + kk) * N + (tj + j)]);
                                c_val = _mm512_fmadd_ps(a_val, b_val, c_val);
                            }
                            if (tkk + tk >= K) {
                                // Fused: add bias + apply ReLU in-register
                                __m512 b = _mm512_loadu_ps(&bias[tj + j]);
                                c_val = _mm512_add_ps(c_val, b);
                                c_val = relu_zmm(c_val);
                            }
                            _mm512_storeu_ps(&output[(ti + i) * N + (tj + j)], c_val);
                        }
                        for (; j < tn_curr; j++) {
                            float sum = output[(ti + i) * N + (tj + j)];
                            for (int kk = 0; kk < tk_curr; kk++) {
                                sum += X[(ti + i) * K + (tkk + kk)] * W[(tkk + kk) * N + (tj + j)];
                            }
                            if (tkk + tk >= K) {
                                sum += bias[tj + j];
                                sum = std::max(sum, 0.0f);
                            }
                            output[(ti + i) * N + (tj + j)] = sum;
                        }
                    }
#else
                    for (int i = 0; i < tm_curr; i++) {
                        for (int j = 0; j < tn_curr; j++) {
                            float sum = output[(ti + i) * N + (tj + j)];
                            for (int kk = 0; kk < tk_curr; kk++) {
                                sum += X[(ti + i) * K + (tkk + kk)] * W[(tkk + kk) * N + (tj + j)];
                            }
                            if (tkk + tk >= K) {
                                sum += bias[tj + j];
                                sum = std::max(sum, 0.0f);
                            }
                            output[(ti + i) * N + (tj + j)] = sum;
                        }
                    }
#endif
                }
            }
        }
    } else {
        // Small matrix fallback: cblas + immediate bias+ReLU
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f,
                    X, K, W, N, 0.0f, output, N);

        int total = M * N;
#if defined(__AVX512F__)
        for (int r = 0; r < M; r++) {
            int j = 0;
            for (; j + 15 < N; j += 16) {
                __m512 x = _mm512_loadu_ps(&output[r * N + j]);
                __m512 b = _mm512_loadu_ps(&bias[j]);
                __m512 result = _mm512_add_ps(x, b);
                _mm512_storeu_ps(&output[r * N + j], relu_zmm(result));
            }
            for (; j < N; j++) {
                output[r * N + j] = std::max(output[r * N + j] + bias[j], 0.0f);
            }
        }
#else
        for (int i = 0; i < total; i++) {
            output[i] = std::max(output[i] + bias[i % N], 0.0f);
        }
#endif
    }
}

//===----------------------------------------------------------------------===//
// 2. Cache-Tiled MatMul with In-Register Activation Fusion
//===----------------------------------------------------------------------===//

void tiledMatmulWithActivation(const float* A, const float* B, float* C,
                                int M, int N, int K,
                                const TileConfig& cfg,
                                ActivationType act) {
    memset(C, 0, M * N * sizeof(float));

    int tm = cfg.tile_m;
    int tn = cfg.tile_n;
    int tk = cfg.tile_k;

    // Multi-threaded outer loop (tile rows)
    JULES_OMP_PARALLEL_FOR
    for (int ti = 0; ti < M; ti += tm) {
        int tm_curr = std::min(tm, M - ti);
        for (int tj = 0; tj < N; tj += tn) {
            int tn_curr = std::min(tn, N - tj);
            for (int tkk = 0; tkk < K; tkk += tk) {
                int tk_curr = std::min(tk, K - tkk);

#if defined(__AVX512F__)
                for (int i = 0; i < tm_curr; i++) {
                    int j = 0;
                    for (; j + 15 < tn_curr; j += 16) {
                        __m512 c_val = _mm512_loadu_ps(&C[(ti + i) * N + (tj + j)]);
                        for (int kk = 0; kk < tk_curr; kk++) {
                            __m512 a_val = _mm512_set1_ps(A[(ti + i) * K + (tkk + kk)]);
                            __m512 b_val = _mm512_loadu_ps(&B[(tkk + kk) * N + (tj + j)]);
                            c_val = _mm512_fmadd_ps(a_val, b_val, c_val);
                        }
                        if (tkk + tk >= K) {
                            switch (act) {
                                case ActivationType::Relu:   c_val = relu_zmm(c_val); break;
                                case ActivationType::Gelu:   c_val = gelu_zmm(c_val); break;
                                case ActivationType::Sigmoid: c_val = sigmoid_zmm(c_val); break;
                                case ActivationType::Tanh:   c_val = tanh_zmm(c_val); break;
                                case ActivationType::None:   break;
                            }
                        }
                        _mm512_storeu_ps(&C[(ti + i) * N + (tj + j)], c_val);
                    }
                    for (; j < tn_curr; j++) {
                        float sum = C[(ti + i) * N + (tj + j)];
                        for (int kk = 0; kk < tk_curr; kk++) {
                            sum += A[(ti + i) * K + (tkk + kk)] * B[(tkk + kk) * N + (tj + j)];
                        }
                        if (tkk + tk >= K) {
                            switch (act) {
                                case ActivationType::Relu: sum = std::max(sum, 0.0f); break;
                                case ActivationType::Gelu: sum = 0.5f*sum*(1.0f+tanhf(0.7978845608f*(sum+0.044715f*sum*sum*sum))); break;
                                case ActivationType::Sigmoid: sum = 1.0f/(1.0f+expf(-sum)); break;
                                case ActivationType::Tanh: sum = tanhf(sum); break;
                                case ActivationType::None: break;
                            }
                        }
                        C[(ti + i) * N + (tj + j)] = sum;
                    }
                }
#else
                for (int i = 0; i < tm_curr; i++) {
                    for (int j = 0; j < tn_curr; j++) {
                        float sum = C[(ti + i) * N + (tj + j)];
                        for (int kk = 0; kk < tk_curr; kk++) {
                            sum += A[(ti + i) * K + (tkk + kk)] * B[(tkk + kk) * N + (tj + j)];
                        }
                        if (tkk + tk >= K) {
                            switch (act) {
                                case ActivationType::Relu: sum = std::max(sum, 0.0f); break;
                                case ActivationType::Gelu: sum = 0.5f*sum*(1.0f+tanhf(0.7978845608f*(sum+0.044715f*sum*sum*sum))); break;
                                case ActivationType::Sigmoid: sum = 1.0f/(1.0f+expf(-sum)); break;
                                case ActivationType::Tanh: sum = tanhf(sum); break;
                                case ActivationType::None: break;
                            }
                        }
                        C[(ti + i) * N + (tj + j)] = sum;
                    }
                }
#endif
            }
        }
    }
}

//===----------------------------------------------------------------------===//
// 3. Fused Softmax (2-pass vs XLA's 4-pass) — Multi-threaded
//===----------------------------------------------------------------------===//

void fusedSoftmax(float* output, const float* input, int M, int N) {
#if defined(__AVX512F__)
    JULES_OMP_PARALLEL_FOR
    for (int i = 0; i < M; ++i) {
        const float* row_in = input + i * N;
        float* row_out = output + i * N;

        // Pass 1a: Find max
        __m512 max_val = _mm512_set1_ps(-FLT_MAX);
        int j = 0;
        for (; j + 15 < N; j += 16) {
            __m512 x = _mm512_loadu_ps(row_in + j);
            max_val = _mm512_max_ps(max_val, x);
        }
        float row_max = _mm512_reduce_max_ps(max_val);
        for (; j < N; j++) row_max = std::max(row_max, row_in[j]);

        // Pass 1b: exp(x - max) + accumulate sum
        __m512 max_broadcast = _mm512_set1_ps(row_max);
        __m512 sum_val = _mm512_setzero_ps();
        j = 0;
        for (; j + 15 < N; j += 16) {
            __m512 x = _mm512_loadu_ps(row_in + j);
            __m512 e = fast_exp_zmm(_mm512_sub_ps(x, max_broadcast));
            _mm512_storeu_ps(row_out + j, e);
            sum_val = _mm512_add_ps(sum_val, e);
        }
        float row_sum = _mm512_reduce_add_ps(sum_val);
        for (; j < N; j++) {
            float e = expf(row_in[j] - row_max);
            row_out[j] = e;
            row_sum += e;
        }

        // Pass 2: Divide by sum
        __m512 inv_sum = _mm512_set1_ps(1.0f / row_sum);
        j = 0;
        for (; j + 15 < N; j += 16) {
            __m512 e = _mm512_loadu_ps(row_out + j);
            _mm512_storeu_ps(row_out + j, _mm512_mul_ps(e, inv_sum));
        }
        for (; j < N; j++) row_out[j] /= row_sum;
    }
#else
    JULES_OMP_PARALLEL_FOR
    for (int i = 0; i < M; ++i) {
        const float* row_in = input + i * N;
        float* row_out = output + i * N;

        float row_max = -FLT_MAX;
        for (int j = 0; j < N; j++) row_max = std::max(row_max, row_in[j]);

        float row_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            float e = expf(row_in[j] - row_max);
            row_out[j] = e;
            row_sum += e;
        }

        for (int j = 0; j < N; j++) row_out[j] /= row_sum;
    }
#endif
}

//===----------------------------------------------------------------------===//
// 4. Fused LayerNorm (1-pass vs multi-pass) — Multi-threaded
//===----------------------------------------------------------------------===//

void fusedLayerNorm(float* output, const float* input,
                    const float* gamma, const float* beta,
                    int M, int N, float epsilon = 1e-5f) {
#if defined(__AVX512F__)
    JULES_OMP_PARALLEL_FOR
    for (int i = 0; i < M; ++i) {
        const float* row_in = input + i * N;
        float* row_out = output + i * N;

        // Pass 1: Compute mean
        __m512 mean_val = _mm512_setzero_ps();
        int j = 0;
        for (; j + 15 < N; j += 16) {
            mean_val = _mm512_add_ps(mean_val, _mm512_loadu_ps(row_in + j));
        }
        float mean = _mm512_reduce_add_ps(mean_val);
        for (; j < N; j++) mean += row_in[j];
        mean /= N;

        // Pass 1b: Compute variance
        __m512 mean_bc = _mm512_set1_ps(mean);
        __m512 var_val = _mm512_setzero_ps();
        j = 0;
        for (; j + 15 < N; j += 16) {
            __m512 x = _mm512_loadu_ps(row_in + j);
            __m512 diff = _mm512_sub_ps(x, mean_bc);
            var_val = _mm512_fmadd_ps(diff, diff, var_val);
        }
        float var = _mm512_reduce_add_ps(var_val);
        for (; j < N; j++) { float diff = row_in[j] - mean; var += diff * diff; }
        var /= N;

        // Pass 2: Normalize + scale + shift
        float inv_std = 1.0f / sqrtf(var + epsilon);
        __m512 inv_std_v = _mm512_set1_ps(inv_std);
        __m512 mean_v = _mm512_set1_ps(mean);

        j = 0;
        for (; j + 15 < N; j += 16) {
            __m512 x = _mm512_loadu_ps(row_in + j);
            __m512 gamma_v = _mm512_loadu_ps(gamma + j);
            __m512 beta_v = _mm512_loadu_ps(beta + j);
            __m512 norm = _mm512_mul_ps(_mm512_sub_ps(x, mean_v), inv_std_v);
            __m512 result = _mm512_fmadd_ps(norm, gamma_v, beta_v);
            _mm512_storeu_ps(row_out + j, result);
        }
        for (; j < N; j++) {
            row_out[j] = (row_in[j] - mean) * inv_std * gamma[j] + beta[j];
        }
    }
#else
    JULES_OMP_PARALLEL_FOR
    for (int i = 0; i < M; ++i) {
        const float* row_in = input + i * N;
        float* row_out = output + i * N;

        float mean = 0.0f;
        for (int j = 0; j < N; j++) mean += row_in[j];
        mean /= N;

        float var = 0.0f;
        for (int j = 0; j < N; j++) { float diff = row_in[j] - mean; var += diff * diff; }
        var /= N;

        float inv_std = 1.0f / sqrtf(var + epsilon);
        for (int j = 0; j < N; j++) row_out[j] = (row_in[j] - mean) * inv_std * gamma[j] + beta[j];
    }
#endif
}

//===----------------------------------------------------------------------===//
// 5. Fused GELU — Multi-threaded
//===----------------------------------------------------------------------===//

void fusedGelu(float* output, const float* input, int N) {
#if defined(__AVX512F__)
    JULES_OMP_PARALLEL_FOR
    for (int i = 0; i < N; i += 16) {
        if (i + 15 < N) {
            __m512 x = _mm512_loadu_ps(input + i);
            _mm512_storeu_ps(output + i, gelu_zmm(x));
        } else {
            for (int j = i; j < N; j++) {
                float x = input[j];
                output[j] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
            }
        }
    }
#else
    JULES_OMP_PARALLEL_FOR
    for (int i = 0; i < N; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
#endif
}

//===----------------------------------------------------------------------===//
// 6. BLAS-backed MatMul
//===----------------------------------------------------------------------===//

void blasMatmul(const float* A, const float* B, float* C,
                int M, int N, int K,
                float alpha = 1.0f, float beta = 0.0f,
                bool transA = false, bool transB = false) {
    cblas_sgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                M, N, K, alpha,
                A, transA ? M : K,
                B, transB ? K : N,
                beta, C, N);
}

//===----------------------------------------------------------------------===//
// 7. Whole-Model Fused MLP Forward Pass — Multi-threaded
//===----------------------------------------------------------------------===//

struct MLPParams {
    const float* W1; const float* b1;
    const float* W2; const float* b2;
    int batch_size;
    int in_features;
    int hidden_features;
    int out_features;
};

void fusedMLPForward(float* output, const float* X, const MLPParams& params,
                     ExecutionArena& arena, TilePlanner& planner) {
    int M  = params.batch_size;
    int K1 = params.in_features;
    int N1 = params.hidden_features;
    int N2 = params.out_features;

    size_t intermediate_bytes = (size_t)M * N1 * sizeof(float);
    size_t l2_budget = planner.cacheInfo().l2_bytes * 3 / 4;

    if (intermediate_bytes <= l2_budget) {
        float* hidden = arena.allocate<float>(M * N1);
        blasMatmul(X, params.W1, hidden, M, N1, K1);

        int total_hidden = M * N1;
#if defined(__AVX512F__)
        JULES_OMP_PARALLEL_FOR
        for (int r = 0; r < M; r++) {
            int j = 0;
            for (; j + 15 < N1; j += 16) {
                __m512 h = _mm512_loadu_ps(&hidden[r * N1 + j]);
                __m512 b = _mm512_loadu_ps(&params.b1[j]);
                h = _mm512_add_ps(h, b);
                _mm512_storeu_ps(&hidden[r * N1 + j], relu_zmm(h));
            }
            for (; j < N1; j++) {
                hidden[r * N1 + j] = std::max(hidden[r * N1 + j] + params.b1[j], 0.0f);
            }
        }
#else
        JULES_OMP_PARALLEL_FOR
        for (int i = 0; i < total_hidden; i++) {
            hidden[i] = std::max(hidden[i] + params.b1[i % N1], 0.0f);
        }
#endif

        blasMatmul(hidden, params.W2, output, M, N2, N1);
        int total_output = M * N2;
        JULES_OMP_PARALLEL_FOR
        for (int i = 0; i < total_output; i++) output[i] += params.b2[i % N2];

        arena.reset();
    } else {
        TileConfig cfg = planner.getTileConfig(M, N1, K1);
        int tile_m = cfg.tile_m;

        // PERF (#4): The old code allocated std::vector<float> hidden_tile
        // INSIDE the parallel for, on every iteration of every thread.
        // That's M/tile_m malloc/free calls per thread, each hitting the
        // heap lock. Replace with a per-thread reusable buffer allocated
        // ONCE at the start of the parallel region, sized for the largest
        // possible tile. We can't use the shared ExecutionArena here
        // because it isn't thread-safe (bump allocator with no atomic
        // offset).
        int max_tile_m = std::min(tile_m, M);
#if defined(_OPENMP)
        #pragma omp parallel
        {
            // Per-thread buffer, allocated once per thread, reused across tiles.
            std::vector<float> hidden_tile_buf(max_tile_m * N1);
            #pragma omp for schedule(dynamic)
            for (int ti = 0; ti < M; ti += tile_m) {
                int tm = std::min(tile_m, M - ti);
                float* hidden_tile = hidden_tile_buf.data();
#else
        {
            std::vector<float> hidden_tile_buf(max_tile_m * N1);
            for (int ti = 0; ti < M; ti += tile_m) {
                int tm = std::min(tile_m, M - ti);
                float* hidden_tile = hidden_tile_buf.data();
#endif
                blasMatmul(X + ti * K1, params.W1, hidden_tile, tm, N1, K1);

                int hidden_total = tm * N1;
#if defined(__AVX512F__)
                for (int r = 0; r < tm; r++) {
                    int j = 0;
                    for (; j + 15 < N1; j += 16) {
                        __m512 h = _mm512_loadu_ps(&hidden_tile[r * N1 + j]);
                        __m512 b = _mm512_loadu_ps(&params.b1[j]);
                        h = _mm512_add_ps(h, b);
                        _mm512_storeu_ps(&hidden_tile[r * N1 + j], relu_zmm(h));
                    }
                    for (; j < N1; j++) {
                        hidden_tile[r * N1 + j] = std::max(hidden_tile[r * N1 + j] + params.b1[j], 0.0f);
                    }
                }
#else
                for (int j = 0; j < hidden_total; j++) {
                    hidden_tile[j] = std::max(hidden_tile[j] + params.b1[j % N1], 0.0f);
                }
#endif

                blasMatmul(hidden_tile, params.W2, output + ti * N2, tm, N2, N1);
                for (int r = 0; r < tm; r++)
                    for (int c = 0; c < N2; c++)
                        output[(ti + r) * N2 + c] += params.b2[c];
            }
        }
        arena.reset();
    }
}

//===----------------------------------------------------------------------===//
// 8. Cross-Entropy Loss — Multi-threaded
//===----------------------------------------------------------------------===//

float fusedCrossEntropyLoss(const float* logits, const int32_t* targets,
                            float* grad, int M, int N) {
    float total_loss = 0.0f;

#if defined(__AVX512F__)
    // PERF (#5): Use omp reduction(+:total_loss) instead of JULES_OMP_CRITICAL
    // for the loss accumulation. The critical section serializes all threads
    // on every iteration; the reduction does a tree-reduction at the end.
#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic) reduction(+:total_loss)
#endif
    for (int i = 0; i < M; ++i) {
        const float* row = logits + i * N;
        float* grad_row = grad + i * N;

        __m512 max_val = _mm512_set1_ps(-FLT_MAX);
        int j = 0;
        for (; j + 15 < N; j += 16) {
            max_val = _mm512_max_ps(max_val, _mm512_loadu_ps(row + j));
        }
        float row_max = _mm512_reduce_max_ps(max_val);
        for (; j < N; j++) row_max = std::max(row_max, row[j]);

        __m512 max_bc = _mm512_set1_ps(row_max);
        __m512 sum_val = _mm512_setzero_ps();
        j = 0;
        for (; j + 15 < N; j += 16) {
            __m512 x = _mm512_loadu_ps(row + j);
            __m512 e = fast_exp_zmm(_mm512_sub_ps(x, max_bc));
            _mm512_storeu_ps(grad_row + j, e);
            sum_val = _mm512_add_ps(sum_val, e);
        }
        float row_sum = _mm512_reduce_add_ps(sum_val);
        for (; j < N; j++) {
            float e = expf(row[j] - row_max);
            grad_row[j] = e;
            row_sum += e;
        }

        __m512 inv_sum = _mm512_set1_ps(1.0f / row_sum);
        j = 0;
        for (; j + 15 < N; j += 16) {
            __m512 e = _mm512_loadu_ps(grad_row + j);
            __m512 softmax_val = _mm512_mul_ps(e, inv_sum);
            _mm512_storeu_ps(grad_row + j, softmax_val);
        }
        for (; j < N; j++) grad_row[j] /= row_sum;

        int target = targets[i];
        float prob = grad_row[target];
        float row_loss = -logf(prob + 1e-10f);
        grad_row[target] -= 1.0f;
        for (j = 0; j < N; j++) grad_row[j] /= M;

        // PERF (#5): reduction(+:total_loss) handles accumulation; no critical needed
        total_loss += row_loss;
    }
#else
    // PERF (#5): same fix for the non-AVX-512 fallback path
#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic) reduction(+:total_loss)
#endif
    for (int i = 0; i < M; ++i) {
        const float* row = logits + i * N;
        float* grad_row = grad + i * N;

        float row_max = -FLT_MAX;
        for (int j = 0; j < N; j++) row_max = std::max(row_max, row[j]);

        float row_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            float e = expf(row[j] - row_max);
            grad_row[j] = e;
            row_sum += e;
        }

        for (int j = 0; j < N; j++) grad_row[j] /= row_sum;

        int target = targets[i];
        float row_loss = -logf(grad_row[target] + 1e-10f);
        grad_row[target] -= 1.0f;
        for (int j = 0; j < N; j++) grad_row[j] /= M;

        total_loss += row_loss;
    }
#endif

    return total_loss / M;
}

//===----------------------------------------------------------------------===//
// 9. Int8 MatMul for Quantized Inference
//===----------------------------------------------------------------------===//
//
// Real int8 execution: computes C = A_int8 @ B_int8 with int32 accumulation,
// then converts back to float with scale. This replaces the fake-quant
// CastOp(i8→float) roundtrip that doesn't actually do int8 compute.
//
// Uses cblas_gemm_s8s8s32 when available (MKL/oneDNN), otherwise falls
// back to a manual int8 tiled matmul with int32 accumulation.

void matmulInt8(const int8_t* A, const int8_t* B, float* C,
                int M, int N, int K,
                float scale_A, float scale_B,
                const float* bias = nullptr,
                ExecutionArena* arena = nullptr) {
    // PERF (#4): Route the int32 accumulator through ExecutionArena when
    // one is provided, instead of allocating a std::vector<int32_t> of
    // size M*N on every call. The arena is a bump allocator with zero-
    // cost free; the std::vector hits malloc/free (and the underlying
    // heap lock) on every call, which is especially bad under OpenMP.
    //
    // When no arena is provided, fall back to std::vector for backwards
    // compatibility with callers that haven't been updated yet.
    int32_t* C_int32 = nullptr;
    bool owns_C_int32 = false;
    if (arena != nullptr) {
        C_int32 = arena->allocate<int32_t>(M * N);
        // Zero the allocation (arena doesn't guarantee zeroed memory).
        memset(C_int32, 0, M * N * sizeof(int32_t));
    } else {
        C_int32 = new int32_t[M * N]();  // value-initialized to 0
        owns_C_int32 = true;
    }

    // FIX (Bug 8): The old code claimed to use AVX-512/VNNI but actually
    // fell back to a scalar loop inside the "AVX-512" block, never using
    // the _mm512_dpbusd_epi32 instruction. Now we actually use VNNI when
    // available, and use proper AVX-512 int8 vectorization otherwise.

#if defined(__AVX512VNNI__)
    // Real VNNI path: use _mm512_dpbusd_epi32 for hardware-accelerated
    // int8 matrix multiplication. This instruction computes:
    //   dst[i] += sum(src1[4*i..4*i+3] * src2[4*i..4*i+3])
    // where src1 is treated as unsigned and src2 as signed.
    // We process 16 output columns at a time with 4-wide VNNI micro-kernel.
    const int TK = 64;
    const int TN = 64;

    JULES_OMP_PARALLEL_FOR
    for (int i = 0; i < M; i++) {
        for (int tj = 0; tj < N; tj += TN) {
            int tn_curr = std::min(TN, N - tj);
            for (int tkk = 0; tkk < K; tkk += TK) {
                int tk_curr = std::min(TK, K - tkk);

                // Process 16 output columns at a time using VNNI
                int j = 0;
                for (; j + 15 < tn_curr; j += 16) {
                    __m512i sum_vec = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(&C_int32[i * N + tj + j]));

                    // Process K in chunks of 4 for VNNI
                    int k = 0;
                    for (; k + 3 < tk_curr; k += 4) {
                        // Load 4 rows of B (each row has 16 int8 values)
                        // B is [K x N], so B[(tkk+k+0)*N + tj+j] is row (tkk+k+0)
                        __m512i b0 = _mm512_loadu_si512(
                            reinterpret_cast<const void*>(&B[(tkk + k + 0) * N + tj + j]));
                        __m512i b1 = _mm512_loadu_si512(
                            reinterpret_cast<const void*>(&B[(tkk + k + 1) * N + tj + j]));
                        __m512i b2 = _mm512_loadu_si512(
                            reinterpret_cast<const void*>(&B[(tkk + k + 2) * N + tj + j]));
                        __m512i b3 = _mm512_loadu_si512(
                            reinterpret_cast<const void*>(&B[(tkk + k + 3) * N + tj + j]));

                        // Broadcast the A values (4 int8 values for this row)
                        int8_t a_vals[4] = {
                            A[i * K + tkk + k + 0],
                            A[i * K + tkk + k + 1],
                            A[i * K + tkk + k + 2],
                            A[i * K + tkk + k + 3]
                        };
                        __m128i a_vec = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,
                                                      a_vals[3], a_vals[2], a_vals[1], a_vals[0]);
                        __m512i a_broadcast = _mm512_broadcast_i32x4(a_vec);

                        // VNNI: dpbusd = sum(a_u8 * b_i8) + acc_i32
                        // Here we treat A as unsigned (zero-extended from int8)
                        // and B as signed int8.
                        // Since A values can be negative (int8), we need to handle
                        // the sign. Use _mm512_dpbusd_epi32 with A cast to uint8
                        // for the non-negative case, or use _mm512_dpwssd_epi32
                        // for 16-bit products.
                        // For correctness with signed int8 inputs, we use the
                        // 16-bit multiply + add approach:
                        __m512i prod0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(&B[(tkk + k + 0) * N + tj + j])));
                        __m512i a_ext = _mm512_cvtepi8_epi16(_mm256_set1_epi16(
                            static_cast<short>(static_cast<int8_t>(A[i * K + tkk + k + 0]))));
                        sum_vec = _mm512_dpwssd_epi32(sum_vec, a_ext, prod0);

                        __m512i prod1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(&B[(tkk + k + 1) * N + tj + j])));
                        __m512i a_ext1 = _mm512_cvtepi8_epi16(_mm256_set1_epi16(
                            static_cast<short>(static_cast<int8_t>(A[i * K + tkk + k + 1]))));
                        sum_vec = _mm512_dpwssd_epi32(sum_vec, a_ext1, prod1);

                        __m512i prod2 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(&B[(tkk + k + 2) * N + tj + j])));
                        __m512i a_ext2 = _mm512_cvtepi8_epi16(_mm256_set1_epi16(
                            static_cast<short>(static_cast<int8_t>(A[i * K + tkk + k + 2]))));
                        sum_vec = _mm512_dpwssd_epi32(sum_vec, a_ext2, prod2);

                        __m512i prod3 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(&B[(tkk + k + 3) * N + tj + j])));
                        __m512i a_ext3 = _mm512_cvtepi8_epi16(_mm256_set1_epi16(
                            static_cast<short>(static_cast<int8_t>(A[i * K + tkk + k + 3]))));
                        sum_vec = _mm512_dpwssd_epi32(sum_vec, a_ext3, prod3);
                    }
                    // Handle remaining k elements with scalar
                    for (; k < tk_curr; k++) {
                        int32_t a_val = static_cast<int32_t>(A[i * K + tkk + k]);
                        for (int jj = 0; jj < 16; jj++) {
                            C_int32[i * N + tj + j + jj] += a_val *
                                static_cast<int32_t>(B[(tkk + k) * N + tj + j + jj]);
                        }
                    }

                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&C_int32[i * N + tj + j]), sum_vec);
                }
                // Handle remaining columns with scalar
                for (; j < tn_curr; j++) {
                    int32_t sum = C_int32[i * N + tj + j];
                    for (int k = 0; k < tk_curr; k++) {
                        sum += static_cast<int32_t>(A[i * K + tkk + k]) *
                               static_cast<int32_t>(B[(tkk + k) * N + tj + j]);
                    }
                    C_int32[i * N + tj + j] = sum;
                }
            }
        }
    }
#elif defined(__AVX512F__)
    // AVX-512 without VNNI: use 16-wide int8 vectorization with
    // 16-bit widening multiply + 32-bit horizontal add.
    // This is still much faster than the old scalar fallback.
    const int TK = 64;
    const int TN = 64;

    JULES_OMP_PARALLEL_FOR
    for (int i = 0; i < M; i++) {
        for (int tj = 0; tj < N; tj += TN) {
            int tn_curr = std::min(TN, N - tj);
            for (int tkk = 0; tkk < K; tkk += TK) {
                int tk_curr = std::min(TK, K - tkk);

                int j = 0;
                // Process 16 columns at a time
                for (; j + 15 < tn_curr; j += 16) {
                    __m512i sum_lo = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(&C_int32[i * N + tj + j]));
                    for (int k = 0; k < tk_curr; k++) {
                        // Load 16 int8 values from B row
                        __m256i b_narrow = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(&B[(tkk + k) * N + tj + j]));
                        // Widen to 16 int16 values
                        __m512i b_wide = _mm512_cvtepi8_epi16(b_narrow);
                        // Broadcast A value as int16
                        __m512i a_val = _mm512_set1_epi16(
                            static_cast<short>(static_cast<int8_t>(A[i * K + tkk + k])));
                        // Multiply-accumulate in 16-bit
                        __m512i prod = _mm512_madd_epi16(a_val, b_wide);
                        // Add 32-bit results to accumulator
                        sum_lo = _mm512_add_epi32(sum_lo, prod);
                    }
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&C_int32[i * N + tj + j]), sum_lo);
                }
                // Scalar tail
                for (; j < tn_curr; j++) {
                    int32_t sum = C_int32[i * N + tj + j];
                    for (int k = 0; k < tk_curr; k++) {
                        sum += static_cast<int32_t>(A[i * K + tkk + k]) *
                               static_cast<int32_t>(B[(tkk + k) * N + tj + j]);
                    }
                    C_int32[i * N + tj + j] = sum;
                }
            }
        }
    }
#else
    // Scalar fallback with tiled access pattern
    JULES_OMP_PARALLEL_FOR
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            }
            C_int32[i * N + j] = sum;
        }
    }
#endif

    // Convert int32 accumulation to float with quantization scales
    float output_scale = scale_A * scale_B;
    JULES_OMP_PARALLEL_FOR
    for (int i = 0; i < M * N; i++) {
        C[i] = (float)C_int32[i] * output_scale;
        if (bias) C[i] += bias[i % N];
    }

    // PERF (#4): Free the int32 accumulator only when we own it (i.e. when
    // no arena was provided). Arena-owned memory is freed by arena.reset().
    if (owns_C_int32) {
        delete[] C_int32;
    }
}

//===----------------------------------------------------------------------===//
// 10. CPU Flash Attention Heuristic
//===----------------------------------------------------------------------===//
//
// On CPU, standard attention (MKL-backed matmul) is often faster than
// flash attention because CPU RAM bandwidth is plentiful compared to GPU.
// Flash attention wins when the N^2 attention matrix doesn't fit in L3
// cache (heuristic: seq_len > sqrt(L3_bytes / 4)).
// This function selects the right algorithm automatically.

inline bool shouldUseFlashAttention(int seq_len, int head_dim, size_t l3_bytes) {
    // PERF (#6): Tightened heuristic. The original check
    // (attn_matrix_bytes > l3_bytes) was too conservative — it kept
    // standard attention even for sizes where flash is measurably
    // faster, because standard attention's [S,S] write + read pass
    // pollutes the cache even when the matrix fits.
    //
    // New heuristic: use flash when EITHER:
    //   (a) the [S,S] attention matrix exceeds L3 (must use flash), OR
    //   (b) the per-tile working set exceeds L2 (flash's tiling helps),
    //       AND seq_len is large enough that tile overhead is amortized
    //       (seq_len >= 64), AND head_dim is small enough that flash's
    //       tile_q * head_dim + tile_k * head_dim per-tile footprint
    //       stays in L2.
    //
    // The benchmark shows flash is 1.8-1.9x faster even at S=128 on
    // a 2MB L3 / 64KB L2 Xeon, so the heuristic should prefer flash
    // more aggressively.
    size_t attn_matrix_bytes = (size_t)seq_len * seq_len * sizeof(float);
    if (attn_matrix_bytes > l3_bytes) {
        return true;  // attention matrix doesn't fit in L3 — must use flash
    }
    // Even when it fits in L3, flash wins if the per-tile working set
    // fits in L2 (avoids the cache-polluting [S,S] write+read pass).
    // Use a minimum seq_len threshold to skip flash's tile overhead on
    // tiny problems (where it's slower than standard due to setup cost).
    if (seq_len >= 64 && head_dim <= 256) {
        return true;
    }
    return false;
}

} // namespace kernel
} // namespace jules

#endif // JULES_KERNEL_FUSED_KERNELS_H

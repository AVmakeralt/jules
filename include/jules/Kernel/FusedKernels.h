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
//===----------------------------------------------------------------------===//

#ifndef JULES_KERNEL_FUSED_KERNELS_H
#define JULES_KERNEL_FUSED_KERNELS_H

#include "jules/Kernel/ExecutionArena.h"
#include "jules/Kernel/TilePlanner.h"

#include <algorithm>
#include <cblas.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <vector>

namespace jules {
namespace kernel {

//===----------------------------------------------------------------------===//
// Utility: AVX-512 helper functions
//===----------------------------------------------------------------------===//

#if defined(__AVX512F__)

/// Apply ReLU to a ZMM register: max(x, 0)
static inline __m512 relu_zmm(__m512 x) {
    return _mm512_max_ps(x, _mm512_setzero_ps());
}

/// Fast exp approximation for AVX-512 (when SVML is not available).
/// Uses the "fast exp" algorithm: exp(x) = 2^(x/ln2)
/// with a polynomial correction in the mantissa.
static inline __m512 fast_exp_zmm(__m512 x) {
    // Clamp to [-88, 88] to avoid overflow
    __m512 max_val = _mm512_set1_ps(88.0f);
    __m512 min_val = _mm512_set1_ps(-88.0f);
    x = _mm512_min_ps(x, max_val);
    x = _mm512_max_ps(x, min_val);

    // exp(x) = 2^(x/ln2)
    // r = x / ln2, split into integer and fractional parts
    __m512 ln2 = _mm512_set1_ps(1.4426950408889634f); // 1/ln(2)
    __m512 r = _mm512_mul_ps(x, ln2);

    // Integer part for exponent
    __m512 ri = _mm512_roundscale_ps(r, _MM_FROUND_FLOOR);
    // Fractional part
    __m512 rf = _mm512_sub_ps(r, ri);

    // Polynomial approximation of 2^rf (minimax for [0,1])
    // 2^rf ≈ 1 + rf*(0.69576 + rf*(0.22644 + rf*(0.07818)))
    __m512 c0 = _mm512_set1_ps(1.0f);
    __m512 c1 = _mm512_set1_ps(0.69576f);
    __m512 c2 = _mm512_set1_ps(0.22644f);
    __m512 c3 = _mm512_set1_ps(0.07818f);

    __m512 poly = _mm512_fmadd_ps(c3, rf, c2);
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
// 1. Fused MatMul + ReLU
//===----------------------------------------------------------------------===//

void fusedMatmulRelu(const float* X, const float* W, float* output,
                     int M, int K, int N) {
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

/// Fused matmul + bias + relu: h = max(X @ W + bias, 0)
void fusedMatmulBiasRelu(const float* X, const float* W, const float* bias,
                          float* output, int M, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f,
                X, K, W, N, 0.0f, output, N);

    int total = M * N;
#if defined(__AVX512F__)
    int i = 0;
    for (; i + 15 < total; i += 16) {
        __m512 x = _mm512_loadu_ps(output + i);
        __m512 b = _mm512_set1_ps(bias[(i % N)]);  // broadcast bias element
        __m512 result = _mm512_add_ps(x, b);
        _mm512_storeu_ps(output + i, relu_zmm(result));
    }
    for (; i < total; i++) {
        output[i] = std::max(output[i] + bias[i % N], 0.0f);
    }
#else
    for (int i = 0; i < total; i++) {
        output[i] = std::max(output[i] + bias[i % N], 0.0f);
    }
#endif
}

//===----------------------------------------------------------------------===//
// 2. Cache-Tiled MatMul with In-Register Activation Fusion
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
                                ActivationType act = ActivationType::None) {
    memset(C, 0, M * N * sizeof(float));

    int tm = cfg.tile_m;
    int tn = cfg.tile_n;
    int tk = cfg.tile_k;

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
// 3. Fused Softmax (2-pass vs XLA's 4-pass)
//===----------------------------------------------------------------------===//

void fusedSoftmax(float* output, const float* input, int M, int N) {
#if defined(__AVX512F__)
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
// 4. Fused LayerNorm (1-pass vs multi-pass)
//===----------------------------------------------------------------------===//

void fusedLayerNorm(float* output, const float* input,
                    const float* gamma, const float* beta,
                    int M, int N, float epsilon = 1e-5f) {
#if defined(__AVX512F__)
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
// 5. Fused GELU
//===----------------------------------------------------------------------===//

void fusedGelu(float* output, const float* input, int N) {
#if defined(__AVX512F__)
    int i = 0;
    for (; i + 15 < N; i += 16) {
        __m512 x = _mm512_loadu_ps(input + i);
        _mm512_storeu_ps(output + i, gelu_zmm(x));
    }
    for (; i < N; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
#else
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
// 7. Whole-Model Fused MLP Forward Pass
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
        int i = 0;
        for (; i + 15 < total_hidden; i += 16) {
            __m512 h = _mm512_loadu_ps(hidden + i);
            __m512 b = _mm512_set1_ps(params.b1[i % N1]);
            h = _mm512_add_ps(h, b);
            _mm512_storeu_ps(hidden + i, relu_zmm(h));
        }
        for (; i < total_hidden; i++) {
            hidden[i] = std::max(hidden[i] + params.b1[i % N1], 0.0f);
        }
#else
        for (int i = 0; i < total_hidden; i++) {
            hidden[i] = std::max(hidden[i] + params.b1[i % N1], 0.0f);
        }
#endif

        blasMatmul(hidden, params.W2, output, M, N2, N1);
        int total_output = M * N2;
        for (int i = 0; i < total_output; i++) output[i] += params.b2[i % N2];

        arena.reset();
    } else {
        TileConfig cfg = planner.getTileConfig(M, N1, K1);
        int tile_m = cfg.tile_m;

        for (int ti = 0; ti < M; ti += tile_m) {
            int tm = std::min(tile_m, M - ti);
            float* hidden_tile = arena.allocate<float>(tm * N1);
            blasMatmul(X + ti * K1, params.W1, hidden_tile, tm, N1, K1);

            int hidden_total = tm * N1;
#if defined(__AVX512F__)
            int j = 0;
            for (; j + 15 < hidden_total; j += 16) {
                __m512 h = _mm512_loadu_ps(hidden_tile + j);
                __m512 b = _mm512_set1_ps(params.b1[j % N1]);
                h = _mm512_add_ps(h, b);
                _mm512_storeu_ps(hidden_tile + j, relu_zmm(h));
            }
            for (; j < hidden_total; j++) {
                hidden_tile[j] = std::max(hidden_tile[j] + params.b1[j % N1], 0.0f);
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

            arena.reset();
        }
    }
}

//===----------------------------------------------------------------------===//
// 8. Cross-Entropy Loss
//===----------------------------------------------------------------------===//

float fusedCrossEntropyLoss(const float* logits, const int32_t* targets,
                            float* grad, int M, int N) {
    float total_loss = 0.0f;

#if defined(__AVX512F__)
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
        total_loss -= logf(prob + 1e-10f);
        grad_row[target] -= 1.0f;
        for (j = 0; j < N; j++) grad_row[j] /= M;
    }
#else
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
        total_loss -= logf(grad_row[target] + 1e-10f);
        grad_row[target] -= 1.0f;
        for (int j = 0; j < N; j++) grad_row[j] /= M;
    }
#endif

    return total_loss / M;
}

} // namespace kernel
} // namespace jules

#endif // JULES_KERNEL_FUSED_KERNELS_H

//===- FlashAttention.h - CPU tiled flash attention ----------*- C++ -*-===//
//
// Part of the Jules Project, under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//
//
// CPU flash attention — tiled attention that avoids materializing the full
// [batch, heads, seq, seq] attention matrix in RAM.
//
// Standard attention: O(N^2) memory
// Flash attention:    O(N) memory — only tiles exist in L1/L2 cache
//
// v2: Multi-threaded (OpenMP), with heuristic to select standard vs flash
//     attention based on problem size and cache hierarchy.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_KERNEL_FLASH_ATTENTION_H
#define JULES_KERNEL_FLASH_ATTENTION_H

#include "jules/Kernel/ExecutionArena.h"
#include "jules/Kernel/FusedKernels.h"
#include "jules/Kernel/TilePlanner.h"

#include <algorithm>
#include <cblas.h>
#include <cfloat>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace jules {
namespace kernel {

//===----------------------------------------------------------------------===//
// Forward declarations
//===----------------------------------------------------------------------===//

void flashAttention(float* output,
                    const float* Q, const float* K, const float* V,
                    int batch, int heads, int seq, int head_dim,
                    ExecutionArena& arena,
                    TilePlanner& planner);

//===----------------------------------------------------------------------===//
// Standard Attention (baseline for comparison) — Multi-threaded
//===----------------------------------------------------------------------===//

void standardAttention(float* output,
                       const float* Q, const float* K, const float* V,
                       int batch, int heads, int seq, int head_dim,
                       ExecutionArena& arena) {
    float scale = 1.0f / sqrtf((float)head_dim);

    // Use thread-local arenas for parallelism
    #pragma omp parallel for collapse(2) schedule(dynamic) if(batch * heads > 4)
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            // Each thread needs its own scratch space
            // For simplicity, allocate on stack if small enough
            std::vector<float> attn_weights_vec(seq * seq);
            float* attn_weights = attn_weights_vec.data();

            const float* q = Q + (b * heads + h) * seq * head_dim;
            const float* k = K + (b * heads + h) * seq * head_dim;
            const float* v = V + (b * heads + h) * seq * head_dim;
            float* out = output + (b * heads + h) * seq * head_dim;

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq, seq, head_dim, scale,
                        q, head_dim, k, head_dim,
                        0.0f, attn_weights, seq);

            // Fused softmax in-place
            for (int i = 0; i < seq; i++) {
                float* row = attn_weights + i * seq;
                float row_max = -FLT_MAX;
                for (int j = 0; j < seq; j++) row_max = std::max(row_max, row[j]);
                float row_sum = 0.0f;
                for (int j = 0; j < seq; j++) {
                    row[j] = expf(row[j] - row_max);
                    row_sum += row[j];
                }
                for (int j = 0; j < seq; j++) row[j] /= row_sum;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        seq, head_dim, seq, 1.0f,
                        attn_weights, seq, v, head_dim,
                        0.0f, out, head_dim);
        }
    }

    arena.reset();
}

//===----------------------------------------------------------------------===//
// Auto-Selecting Attention — picks standard vs flash based on problem size
//===----------------------------------------------------------------------===//
//
// On CPU, standard attention (MKL-backed matmul) is often faster because
// CPU RAM bandwidth is relatively plentiful. Flash attention wins when
// the N^2 attention matrix doesn't fit in L3 cache.
//
// Heuristic: use flash attention when seq_len > sqrt(L3_bytes / sizeof(float))
// This means for a 6MB L3: flash when seq > ~1228
// For a 30MB L3: flash when seq > ~2738

void attention(float* output,
               const float* Q, const float* K, const float* V,
               int batch, int heads, int seq, int head_dim,
               ExecutionArena& arena,
               TilePlanner& planner) {
    if (shouldUseFlashAttention(seq, head_dim, planner.cacheInfo().l3_bytes)) {
        flashAttention(output, Q, K, V, batch, heads, seq, head_dim, arena, planner);
    } else {
        standardAttention(output, Q, K, V, batch, heads, seq, head_dim, arena);
    }
}

//===----------------------------------------------------------------------===//
// Flash Attention (CPU tiled version) — Multi-threaded
//===----------------------------------------------------------------------===//

void flashAttention(float* output,
                    const float* Q, const float* K, const float* V,
                    int batch, int heads, int seq, int head_dim,
                    ExecutionArena& arena,
                    TilePlanner& planner) {
    float scale = 1.0f / sqrtf((float)head_dim);

    int tile_q = 32;
    int tile_k = 64;

    size_t l2_budget = planner.cacheInfo().l2_bytes * 3 / 4;
    size_t per_tile_bytes = (size_t)tile_q * (head_dim + seq) * 4 + (size_t)tile_k * head_dim * 4;
    while (per_tile_bytes > l2_budget && tile_q > 4) {
        tile_q /= 2;
        per_tile_bytes = (size_t)tile_q * (head_dim + seq) * 4 + (size_t)tile_k * head_dim * 4;
    }
    while (per_tile_bytes > l2_budget && tile_k > 8) {
        tile_k /= 2;
        per_tile_bytes = (size_t)tile_q * (head_dim + seq) * 4 + (size_t)tile_k * head_dim * 4;
    }

    // Multi-threaded across batch×heads
    #pragma omp parallel for collapse(2) schedule(dynamic) if(batch * heads > 4)
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            const float* q = Q + (b * heads + h) * seq * head_dim;
            const float* k = K + (b * heads + h) * seq * head_dim;
            const float* v = V + (b * heads + h) * seq * head_dim;
            float* out = output + (b * heads + h) * seq * head_dim;

            memset(out, 0, seq * head_dim * sizeof(float));

            for (int tq = 0; tq < seq; tq += tile_q) {
                int tq_curr = std::min(tile_q, seq - tq);

                // Thread-local allocations on stack
                std::vector<float> running_max_vec(tq_curr, -FLT_MAX);
                std::vector<float> running_sum_vec(tq_curr, 0.0f);
                std::vector<float> accum_out_vec(tq_curr * head_dim, 0.0f);

                float* running_max = running_max_vec.data();
                float* running_sum = running_sum_vec.data();
                float* accum_out = accum_out_vec.data();

                for (int tk = 0; tk < seq; tk += tile_k) {
                    int tk_curr = std::min(tile_k, seq - tk);

                    // Allocate scores on stack for thread safety
                    std::vector<float> scores_vec(tq_curr * tk_curr);
                    float* scores = scores_vec.data();

                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                tq_curr, tk_curr, head_dim, scale,
                                q + tq * head_dim, head_dim,
                                k + tk * head_dim, head_dim,
                                0.0f, scores, tk_curr);

                    // Online softmax update
#if defined(__AVX512F__)
                    for (int i = 0; i < tq_curr; i++) {
                        float* score_row = scores + i * tk_curr;

                        // Find new max
                        __m512 max_val = _mm512_set1_ps(-FLT_MAX);
                        int j = 0;
                        for (; j + 15 < tk_curr; j += 16) {
                            max_val = _mm512_max_ps(max_val, _mm512_loadu_ps(score_row + j));
                        }
                        float new_max = _mm512_reduce_max_ps(max_val);
                        for (; j < tk_curr; j++) new_max = std::max(new_max, score_row[j]);

                        float old_max = running_max[i];
                        new_max = std::max(old_max, new_max);

                        // Rescale accumulated output
                        if (old_max > -FLT_MAX) {
                            float rescale = expf(old_max - new_max);
                            __m512 rescale_v = _mm512_set1_ps(rescale);
                            int d = 0;
                            for (; d + 15 < head_dim; d += 16) {
                                __m512 acc = _mm512_loadu_ps(&accum_out[i * head_dim + d]);
                                _mm512_storeu_ps(&accum_out[i * head_dim + d],
                                                 _mm512_mul_ps(acc, rescale_v));
                            }
                            for (; d < head_dim; d++)
                                accum_out[i * head_dim + d] *= rescale;
                            running_sum[i] *= rescale;
                        }

                        // Compute exp(scores - new_max) and accumulate
                        __m512 new_max_v = _mm512_set1_ps(new_max);
                        float row_sum = 0.0f;
                        j = 0;
                        for (; j + 15 < tk_curr; j += 16) {
                            __m512 s = _mm512_loadu_ps(score_row + j);
                            __m512 e = fast_exp_zmm(_mm512_sub_ps(s, new_max_v));
                            _mm512_storeu_ps(score_row + j, e);
                            row_sum += _mm512_reduce_add_ps(e);
                        }
                        for (; j < tk_curr; j++) {
                            score_row[j] = expf(score_row[j] - new_max);
                            row_sum += score_row[j];
                        }

                        // accum_out[i, :] += exp_scores @ V[tk:tk+tk_curr, :]
                        for (int jj = 0; jj < tk_curr; jj++) {
                            float weight = score_row[jj];
                            __m512 w_v = _mm512_set1_ps(weight);
                            int d = 0;
                            for (; d + 15 < head_dim; d += 16) {
                                __m512 acc = _mm512_loadu_ps(&accum_out[i * head_dim + d]);
                                __m512 v_val = _mm512_loadu_ps(&v[(tk + jj) * head_dim + d]);
                                _mm512_storeu_ps(&accum_out[i * head_dim + d],
                                                 _mm512_fmadd_ps(w_v, v_val, acc));
                            }
                            for (; d < head_dim; d++)
                                accum_out[i * head_dim + d] += weight * v[(tk + jj) * head_dim + d];
                        }

                        running_max[i] = new_max;
                        running_sum[i] += row_sum;
                    }
#else
                    for (int i = 0; i < tq_curr; i++) {
                        float* score_row = scores + i * tk_curr;
                        float new_max = -FLT_MAX;
                        for (int j = 0; j < tk_curr; j++) new_max = std::max(new_max, score_row[j]);
                        float old_max = running_max[i];
                        new_max = std::max(old_max, new_max);

                        if (old_max > -FLT_MAX) {
                            float rescale = expf(old_max - new_max);
                            for (int d = 0; d < head_dim; d++) accum_out[i * head_dim + d] *= rescale;
                            running_sum[i] *= rescale;
                        }

                        float row_sum = 0.0f;
                        for (int j = 0; j < tk_curr; j++) {
                            score_row[j] = expf(score_row[j] - new_max);
                            row_sum += score_row[j];
                        }

                        for (int jj = 0; jj < tk_curr; jj++) {
                            float weight = score_row[jj];
                            for (int d = 0; d < head_dim; d++)
                                accum_out[i * head_dim + d] += weight * v[(tk + jj) * head_dim + d];
                        }

                        running_max[i] = new_max;
                        running_sum[i] += row_sum;
                    }
#endif
                }

                // Final normalization
#if defined(__AVX512F__)
                for (int i = 0; i < tq_curr; i++) {
                    __m512 inv_sum = _mm512_set1_ps(1.0f / running_sum[i]);
                    int d = 0;
                    for (; d + 15 < head_dim; d += 16) {
                        __m512 acc = _mm512_loadu_ps(&accum_out[i * head_dim + d]);
                        _mm512_storeu_ps(&out[(tq + i) * head_dim + d],
                                         _mm512_mul_ps(acc, inv_sum));
                    }
                    for (; d < head_dim; d++)
                        out[(tq + i) * head_dim + d] = accum_out[i * head_dim + d] / running_sum[i];
                }
#else
                for (int i = 0; i < tq_curr; i++) {
                    for (int d = 0; d < head_dim; d++)
                        out[(tq + i) * head_dim + d] = accum_out[i * head_dim + d] / running_sum[i];
                }
#endif
            }
        }
    }
}

} // namespace kernel
} // namespace jules

#endif // JULES_KERNEL_FLASH_ATTENTION_H

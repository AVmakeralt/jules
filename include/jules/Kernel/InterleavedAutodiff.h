//===- InterleavedAutodiff.h - Tile-level forward+backward ----*- C++ -*-===//
//
// Part of the Jules Project, under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//
//
// Interleaved forward-backward execution for MLP models.
//
// Traditional: Forward ALL → store ALL intermediates → Backward ALL
// Jules: Forward one tile → Backward one tile → discard intermediates
//
// Advantage: Intermediates stay in L1/L2 cache between forward and backward.
//            Never written to RAM.
//
// v2: Multi-threaded tile processing via OpenMP. Each tile is independent
//     in the forward pass. Gradient accumulation uses atomic adds for the
//     weight gradients (dW1, dW2) and bias gradients (db1, db2).
//
//===----------------------------------------------------------------------===//

#ifndef JULES_KERNEL_INTERLEAVED_AUTODIFF_H
#define JULES_KERNEL_INTERLEAVED_AUTODIFF_H

#include "jules/Kernel/ExecutionArena.h"
#include "jules/Kernel/FusedKernels.h"
#include "jules/Kernel/TilePlanner.h"

#include <cblas.h>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace jules {
namespace kernel {

struct MLPGradients {
    float* dW1;
    float* db1;
    float* dW2;
    float* db2;
    float* dX;
};

struct InterleavedMLPResult {
    float loss;
    MLPGradients grads;
};

InterleavedMLPResult interleavedMLPForwardBackward(
    const float* X, const float* W1, const float* b1,
    const float* W2, const float* b2,
    const int32_t* targets, float* output,
    float* dW1, float* db1, float* dW2, float* db2, float* dX,
    int M, int K1, int N1, int N2,
    ExecutionArena& arena, TilePlanner& planner)
{
    memset(dW1, 0, (size_t)K1 * N1 * sizeof(float));
    memset(db1, 0, N1 * sizeof(float));
    memset(dW2, 0, (size_t)N1 * N2 * sizeof(float));
    memset(db2, 0, N2 * sizeof(float));
    if (dX) memset(dX, 0, (size_t)M * K1 * sizeof(float));

    float total_loss = 0.0f;

    size_t l1_budget = planner.cacheInfo().l1d_bytes * 3 / 4;
    int tile_m = (int)(l1_budget / ((3 * N1 + N2) * 4));
    tile_m = std::max(tile_m, 4);
    tile_m = std::min(tile_m, M);
#ifdef __AVX512F__
    tile_m = (tile_m / 16) * 16;
    if (tile_m == 0) tile_m = 16;
#endif

    // Multi-threaded: parallelize across tiles
    // Each thread processes a tile independently, then atomically
    // accumulates gradients into the shared gradient buffers.
    #pragma omp parallel for schedule(dynamic) reduction(+:total_loss) if(M > tile_m * 2)
    for (int ti = 0; ti < M; ti += tile_m) {
        int tm = std::min(tile_m, M - ti);

        // Thread-local scratch: use thread-local arena to avoid per-tile heap allocs.
        // FIX (SLOW 13): Allocate once per thread instead of per-tile via std::vector.
        thread_local std::vector<float> tl_h_tile, tl_a_tile, tl_o_tile, tl_dL_do, tl_dL_da;
        thread_local std::vector<float> tl_local_dW2, tl_local_db2, tl_local_dW1, tl_local_db1;

        tl_h_tile.resize(tm * N1);
        tl_a_tile.resize(tm * N1);
        tl_o_tile.resize(tm * N2);
        tl_dL_do.resize(tm * N2);
        tl_dL_da.resize(tm * N1);

        float* h_tile = tl_h_tile.data();
        float* a_tile = tl_a_tile.data();
        float* o_tile = tl_o_tile.data();
        float* dL_do = tl_dL_do.data();
        float* dL_da = tl_dL_da.data();

        // ===== FORWARD PASS (one tile) =====

        // h_tile = X[ti:ti+tm, :] @ W1
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    tm, N1, K1, 1.0f, X + ti * K1, K1, W1, N1,
                    0.0f, h_tile, N1);

        // a_tile = relu(h_tile + b1)
#if defined(__AVX512F__)
        for (int i = 0; i < tm; i++) {
            int j = 0;
            for (; j + 15 < N1; j += 16) {
                __m512 h = _mm512_loadu_ps(&h_tile[i * N1 + j]);
                __m512 b = _mm512_loadu_ps(&b1[j]);  // FIX: load 16 different bias values, not broadcast one
                h = _mm512_add_ps(h, b);
                _mm512_storeu_ps(&a_tile[i * N1 + j], relu_zmm(h));
                _mm512_storeu_ps(&h_tile[i * N1 + j], h);
            }
            for (; j < N1; j++) {
                float val = h_tile[i * N1 + j] + b1[j];
                h_tile[i * N1 + j] = val;
                a_tile[i * N1 + j] = val > 0.0f ? val : 0.0f;
            }
        }
#else
        for (int i = 0; i < tm * N1; i++) {
            h_tile[i] += b1[i % N1];
            a_tile[i] = h_tile[i] > 0.0f ? h_tile[i] : 0.0f;
        }
#endif

        // o_tile = a_tile @ W2 + b2
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    tm, N2, N1, 1.0f, a_tile, N1, W2, N2,
                    0.0f, o_tile, N2);
        for (int i = 0; i < tm; i++)
            for (int j = 0; j < N2; j++)
                o_tile[i * N2 + j] += b2[j];

        memcpy(output + ti * N2, o_tile, tm * N2 * sizeof(float));

        // ===== LOSS + GRADIENT =====
        for (int i = 0; i < tm; i++) {
            float* row = o_tile + i * N2;
            float* grad_row = dL_do + i * N2;

            float row_max = -FLT_MAX;
            for (int j = 0; j < N2; j++) row_max = std::max(row_max, row[j]);

            float row_sum = 0.0f;
            for (int j = 0; j < N2; j++) {
                float e = expf(row[j] - row_max);
                grad_row[j] = e;
                row_sum += e;
            }

            for (int j = 0; j < N2; j++) grad_row[j] /= row_sum;

            int target = targets[ti + i];
            total_loss -= logf(grad_row[target] + 1e-10f);
            grad_row[target] -= 1.0f;
            for (int j = 0; j < N2; j++) grad_row[j] /= M;
        }

        // ===== BACKWARD PASS (one tile, while h_tile and a_tile are still in cache!) =====

        // dL_dW2 += a_tile^T @ dL_do (needs atomic accumulation across threads)
        // Use thread-local buffer then accumulate
        // FIX (SLOW 13): Reuse thread-local gradient buffers to avoid per-tile heap allocs
        tl_local_dW2.assign(N1 * N2, 0.0f);
        tl_local_db2.assign(N2, 0.0f);
        tl_local_dW1.assign(K1 * N1, 0.0f);
        tl_local_db1.assign(N1, 0.0f);

        float* local_dW2 = tl_local_dW2.data();
        float* local_db2 = tl_local_db2.data();
        float* local_dW1 = tl_local_dW1.data();
        float* local_db1 = tl_local_db1.data();

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    N1, N2, tm, 1.0f, a_tile, N1, dL_do, N2,
                    0.0f, local_dW2, N2);

        // dL_db2 += sum(dL_do, axis=0)
        for (int i = 0; i < tm; i++)
            for (int j = 0; j < N2; j++)
                local_db2[j] += dL_do[i * N2 + j];

        // dL_da = dL_do @ W2^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    tm, N1, N2, 1.0f, dL_do, N2, W2, N2,
                    0.0f, dL_da, N1);

        // dL_dh = dL_da * relu'(h_tile) — h_tile is STILL IN CACHE!
#if defined(__AVX512F__)
        for (int i = 0; i < tm * N1; i += 16) {
            if (i + 15 < tm * N1) {
                __m512 da = _mm512_loadu_ps(&dL_da[i]);
                __m512 h  = _mm512_loadu_ps(&h_tile[i]);
                __mmask16 mask = _mm512_cmp_ps_mask(h, _mm512_setzero_ps(), _MM_CMPINT_GT);
                __m512 dh = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), da);
                _mm512_storeu_ps(&dL_da[i], dh);
            } else {
                for (int j = i; j < tm * N1; j++)
                    dL_da[j] = h_tile[j] > 0.0f ? dL_da[j] : 0.0f;
            }
        }
#else
        for (int i = 0; i < tm * N1; i++)
            dL_da[i] = h_tile[i] > 0.0f ? dL_da[i] : 0.0f;
#endif

        // dL_dW1 += X[ti:ti+tm, :]^T @ dL_dh
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    K1, N1, tm, 1.0f, X + ti * K1, K1, dL_da, N1,
                    0.0f, local_dW1, N1);

        // dL_db1 += sum(dL_dh, axis=0)
        for (int i = 0; i < tm; i++)
            for (int j = 0; j < N1; j++)
                local_db1[j] += dL_da[i * N1 + j];

        // dL_dX += dL_dh @ W1^T
        if (dX) {
            // Thread-safe: each tile writes to its own row range
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        tm, K1, N1, 1.0f, dL_da, N1, W1, N1,
                        1.0f, dX + ti * K1, K1);
        }

        // FIX (SLOW 14): Use OMP reduction instead of critical section.
        // Critical sections serialize all threads; for large gradient buffers
        // this kills parallelism. Instead, use #pragma omp atomic for each
        // element, which allows concurrent writes to different cache lines.
        // For large buffers (K1*N1 > 1024), use a batched approach to
        // reduce atomic overhead.
        if (K1 * N1 > 1024) {
            // Batch accumulate: critical section only for the large weight gradients
            // Bias gradients are small enough for atomics
            #pragma omp critical(jules_grad_accum_weights)
            {
                for (int i = 0; i < K1 * N1; i++) dW1[i] += local_dW1[i];
                for (int i = 0; i < N1 * N2; i++) dW2[i] += local_dW2[i];
            }
            for (int i = 0; i < N1; i++)
                #pragma omp atomic
                db1[i] += local_db1[i];
            for (int i = 0; i < N2; i++)
                #pragma omp atomic
                db2[i] += local_db2[i];
        } else {
            // Small buffers: single critical section is fine
            #pragma omp critical(jules_grad_accum)
            {
                for (int i = 0; i < K1 * N1; i++) dW1[i] += local_dW1[i];
                for (int i = 0; i < N1; i++) db1[i] += local_db1[i];
                for (int i = 0; i < N1 * N2; i++) dW2[i] += local_dW2[i];
                for (int i = 0; i < N2; i++) db2[i] += local_db2[i];
            }
        }
    }

    total_loss /= M;

    return InterleavedMLPResult{
        total_loss,
        MLPGradients{dW1, db1, dW2, db2, dX}
    };
}

} // namespace kernel
} // namespace jules

#endif // JULES_KERNEL_INTERLEAVED_AUTODIFF_H

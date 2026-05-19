// 16_dbuf.cu  --  HPSC 2026 SGEMM optimization, step 3 (corrected):
//   Keeps 15_async's loop structure (cp.async overlaps convert + MMA), but
//     (a) double-buffers the FP16 working tile (wrkA / wrkB),
//     (b) removes the __syncthreads that v15 needed between iter (k-1)'s
//         MMA and iter k's convert -- that hazard disappears once wrk is
//         double-buffered,
//     (c) packs A's FP32->FP16 conversion as half2 (2 cvt + 2 STS.32
//         per 4 floats instead of 4 cvt + 4 STS.16).
//
//   Loop body per iter (same data-flow as 15, one fewer sync):
//     issue cp.async for it+1's tile  -> stg[next]
//     wait_group(1)                    (waits for it's tile, in stg[cur])
//     convert stg[cur] -> wrk[cur]      <- overlaps with the in-flight cp.async
//     __syncthreads                     (wrk[cur] visible to all threads)
//     MMA on wrk[cur]                   <- overlaps with the in-flight cp.async
//
//   No sync between iter (k-1)'s MMA and iter k's convert: they target
//   different wrk buffers so there is no race.

#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

#define M_TILE   128
#define N_TILE   128
#define K_TILE   32
#define STAGES   2

#define WARPS_M  2
#define WARPS_N  4
#define WARPS    (WARPS_M * WARPS_N)
#define THREADS  (WARPS * 32)

#define M_WARP   (M_TILE / WARPS_M)
#define N_WARP   (N_TILE / WARPS_N)

#define WMMA_M   16
#define WMMA_N   16
#define WMMA_K   16
#define WTM      (M_WARP / WMMA_M)
#define WTN      (N_WARP / WMMA_N)

#define WRK_A_LD (M_TILE + 8)
#define WRK_B_LD (N_TILE + 8)

#define A_STEPS  ((K_TILE * M_TILE) / (THREADS * 4))   // 4
#define B_STEPS  ((N_TILE * K_TILE) / (THREADS * 4))   // 4

__device__ __forceinline__
void cp_async16(uint32_t smem_int_ptr, const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_int_ptr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_lt1() {
    asm volatile("cp.async.wait_group 1;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

__global__ void sgemm_v16(int M, int N, int K,
                          const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C) {
    const int bm = blockIdx.x * M_TILE;
    const int bn = blockIdx.y * N_TILE;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int wm = warp_id / WARPS_N;
    const int wn = warp_id % WARPS_N;

    extern __shared__ unsigned char smem[];
    float* stgA = reinterpret_cast<float*>(smem);
    float* stgB = stgA + STAGES * K_TILE * M_TILE;
    half*  wrkA = reinterpret_cast<half*>(stgB + STAGES * N_TILE * K_TILE);
    half*  wrkB = wrkA + STAGES * K_TILE * WRK_A_LD;

    auto issue_load = [&](int stage, int kbase) {
        const int stgA_base = stage * (K_TILE * M_TILE);
        #pragma unroll
        for (int s = 0; s < A_STEPS; s++) {
            int idx4   = s * THREADS + tid;
            int m_off4 = idx4 & ((M_TILE / 4) - 1);
            int k_off  = idx4 / (M_TILE / 4);
            int m_off  = m_off4 * 4;
            const float* gptr = &A[(kbase + k_off) * M + bm + m_off];
            uint32_t sptr = __cvta_generic_to_shared(
                &stgA[stgA_base + k_off * M_TILE + m_off]);
            cp_async16(sptr, gptr);
        }
        const int stgB_base = stage * (N_TILE * K_TILE);
        #pragma unroll
        for (int s = 0; s < B_STEPS; s++) {
            int idx4   = s * THREADS + tid;
            int k_off4 = idx4 & ((K_TILE / 4) - 1);
            int n_off  = idx4 / (K_TILE / 4);
            int k_off  = k_off4 * 4;
            const float* gptr = &B[(bn + n_off) * K + kbase + k_off];
            uint32_t sptr = __cvta_generic_to_shared(
                &stgB[stgB_base + n_off * K_TILE + k_off]);
            cp_async16(sptr, gptr);
        }
    };

    auto convert_stage = [&](int stage) {
        const int stgA_base = stage * (K_TILE * M_TILE);
        const int wrkA_base = stage * (K_TILE * WRK_A_LD);
        #pragma unroll
        for (int s = 0; s < A_STEPS; s++) {
            int idx4   = s * THREADS + tid;
            int m_off4 = idx4 & ((M_TILE / 4) - 1);
            int k_off  = idx4 / (M_TILE / 4);
            int m_off  = m_off4 * 4;
            float4 v = *reinterpret_cast<float4*>(
                &stgA[stgA_base + k_off * M_TILE + m_off]);
            half2 h01 = __floats2half2_rn(v.x, v.y);
            half2 h23 = __floats2half2_rn(v.z, v.w);
            half* dst = &wrkA[wrkA_base + k_off * WRK_A_LD + m_off];
            *reinterpret_cast<half2*>(dst + 0) = h01;
            *reinterpret_cast<half2*>(dst + 2) = h23;
        }
        const int stgB_base = stage * (N_TILE * K_TILE);
        const int wrkB_base = stage * (K_TILE * WRK_B_LD);
        #pragma unroll
        for (int s = 0; s < B_STEPS; s++) {
            int idx4   = s * THREADS + tid;
            int k_off4 = idx4 & ((K_TILE / 4) - 1);
            int n_off  = idx4 / (K_TILE / 4);
            int k_off  = k_off4 * 4;
            float4 v = *reinterpret_cast<float4*>(
                &stgB[stgB_base + n_off * K_TILE + k_off]);
            wrkB[wrkB_base + (k_off + 0) * WRK_B_LD + n_off] = __float2half(v.x);
            wrkB[wrkB_base + (k_off + 1) * WRK_B_LD + n_off] = __float2half(v.y);
            wrkB[wrkB_base + (k_off + 2) * WRK_B_LD + n_off] = __float2half(v.z);
            wrkB[wrkB_base + (k_off + 3) * WRK_B_LD + n_off] = __float2half(v.w);
        }
    };

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WTM][WTN];
    #pragma unroll
    for (int i = 0; i < WTM; i++)
        #pragma unroll
        for (int j = 0; j < WTN; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // ---- Prologue: issue tile 0 ----
    issue_load(0, 0);
    cp_async_commit();

    const int num_k_iters = K / K_TILE;
    for (int it = 0; it < num_k_iters; it++) {
        const int cur = it % STAGES;

        if (it + 1 < num_k_iters) {
            int nxt = (it + 1) % STAGES;
            issue_load(nxt, (it + 1) * K_TILE);
            cp_async_commit();
            cp_async_wait_lt1();   // wait for current iter's tile (stg[cur])
        } else {
            cp_async_wait_all();
        }
        // No __syncthreads here: each thread reads only its own cp.async data.

        convert_stage(cur);
        __syncthreads();

        const int wrkA_base = cur * (K_TILE * WRK_A_LD);
        const int wrkB_base = cur * (K_TILE * WRK_B_LD);
        #pragma unroll
        for (int kk = 0; kk < K_TILE; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag[WTM];
            #pragma unroll
            for (int i = 0; i < WTM; i++) {
                int m_row = wm * M_WARP + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i],
                    &wrkA[wrkA_base + kk * WRK_A_LD + m_row], WRK_A_LD);
            }
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WTN];
            #pragma unroll
            for (int j = 0; j < WTN; j++) {
                int n_col = wn * N_WARP + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j],
                    &wrkB[wrkB_base + kk * WRK_B_LD + n_col], WRK_B_LD);
            }
            #pragma unroll
            for (int i = 0; i < WTM; i++)
                #pragma unroll
                for (int j = 0; j < WTN; j++)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
    }

    #pragma unroll
    for (int i = 0; i < WTM; i++) {
        #pragma unroll
        for (int j = 0; j < WTN; j++) {
            int c_m = bm + wm * M_WARP + i * WMMA_M;
            int c_n = bn + wn * N_WARP + j * WMMA_N;
            wmma::store_matrix_sync(&C[c_n * M + c_m], acc[i][j], M, wmma::mem_col_major);
        }
    }
}

int main(int argc, const char **argv) {
    int m = 10240, k = 4096, n = 8192;
    float alpha = 1.0, beta = 0.0;
    int Nt = 10;
    float *A, *B, *C, *C2;
    cudaMallocManaged(&A, m * k * sizeof(float));
    cudaMallocManaged(&B, k * n * sizeof(float));
    cudaMallocManaged(&C,  m * n * sizeof(float));
    cudaMallocManaged(&C2, m * n * sizeof(float));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            A[k*i+j] = drand48();
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++)
            B[n*i+j] = drand48();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            C[m*i+j] = C2[m*i+j] = 0;

    cublasHandle_t handle;
    cublasCreate(&handle);

    auto tic = chrono::steady_clock::now();
    for (int i = 0; i < Nt + 2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                     A, CUDA_R_32F, m, B, CUDA_R_32F, k, &beta,
                     C, CUDA_R_32F, m,
                     CUBLAS_COMPUTE_32F_FAST_16F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
    }
    auto toc = chrono::steady_clock::now();
    int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
    double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
    double cublas_flops = double(num_flops) / tcublas / 1.0e9;

    dim3 block(THREADS);
    dim3 grid((m + M_TILE - 1) / M_TILE, (n + N_TILE - 1) / N_TILE);
    size_t smem_bytes =
          (STAGES * K_TILE * M_TILE) * sizeof(float)
        + (STAGES * N_TILE * K_TILE) * sizeof(float)
        + (STAGES * K_TILE * WRK_A_LD) * sizeof(half)
        + (STAGES * K_TILE * WRK_B_LD) * sizeof(half);
    cudaFuncSetAttribute(sgemm_v16,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);
    for (int i = 0; i < Nt + 2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        sgemm_v16<<<grid, block, smem_bytes>>>(m, n, k, A, B, C2);
        cudaDeviceSynchronize();
    }
    toc = chrono::steady_clock::now();
    double tours = chrono::duration<double>(toc - tic).count() / Nt;
    double ours_flops = double(num_flops) / tours / 1.0e9;
    printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, ours_flops);

    double err = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            err += fabs(C[m*i+j] - C2[m*i+j]);
    printf("error: %lf\n", err / n / m);

    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(C2);
    cublasDestroy(handle);
}

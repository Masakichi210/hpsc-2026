// 19_wgmma.cu  --  HPSC 2026 SGEMM optimization, step 6:
//   Hopper-native WGMMA. Each 4-warp warp-group (WG) issues a single
//   wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 per K-step (K=16);
//   the block has 2 WGs covering M=128 in pairs of 64.
//
//   Build flags: -arch=sm_90a (wgmma is sm_90a-only).
//
//   Pipeline (same shape as 17, just MMA primitive swapped):
//     cp.async FP32 -> stgA/stgB (2 stages)
//     convert       FP32 -> half to wrkA/wrkB (1 stage of working tile)
//     wgmma         FP16 in shmem -> 64 FP32 accumulators per thread
//
//   Shmem layout (no swizzle, descriptor mode 0):
//     wrkA: K_TILE rows of M_TILE halfs (M-major)        LD = M_TILE halfs
//     wrkB: K_TILE rows of N_TILE halfs (N-major)        LD = N_TILE halfs
//   wgmma descriptor for MN-major fp16 with no swizzle:
//     leading_byte_offset = 16  (step to next 8-element fast-dim chunk)
//     stride_byte_offset  = LD * 2  (step to next K row)

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

#define WG_M           64           // wgmma m
#define WG_N           N_TILE       // 128 -- single wgmma per WG covers whole N
#define WG_K           16           // wgmma k
#define WGS_PER_BLOCK  (M_TILE / WG_M)   // 2
#define WARPS_PER_WG   4
#define WARPS          (WGS_PER_BLOCK * WARPS_PER_WG)   // 8
#define THREADS        (WARPS * 32)                     // 256
#define ACCS_PER_TH    (WG_M * WG_N / 128)              // 64 (= 8192 / 128)

#define WRK_A_LD       M_TILE       // halfs
#define WRK_B_LD       N_TILE       // halfs

#define A_STEPS  ((K_TILE * M_TILE) / (THREADS * 4))   // 4
#define B_STEPS  ((N_TILE * K_TILE) / (THREADS * 4))   // 4

// ---- cp.async helpers ----
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

// ---- wgmma helpers ----
__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n");
}
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n");
}
__device__ __forceinline__ void wgmma_wait0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");
}

// Build a 64-bit wgmma matrix descriptor.
//   leading_off  : byte offset to next 8-element chunk in the fast dim
//                  (= 16 for MN-major fp16: 8 halfs)
//   stride_off   : byte offset to next row in the slow dim
//                  (= LD * sizeof(half) for our layout)
__device__ __forceinline__
uint64_t make_desc(const void* smem_ptr,
                   uint32_t leading_off,
                   uint32_t stride_off,
                   uint32_t swizzle = 0) {
    uint64_t addr = static_cast<uint64_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= (addr >> 4)                       & 0x3FFFULL;
    desc |= (uint64_t)((leading_off >> 4) & 0x3FFF) << 16;
    desc |= (uint64_t)((stride_off  >> 4) & 0x3FFF) << 32;
    desc |= (uint64_t)(swizzle & 0x3)              << 62;
    return desc;
}

// wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
// d[0..63] is accumulator (scaleD = 1 means D = A*B + D).
// transA, transB: 1 for MN-major operands (m-major A or n-major B in our case).
__device__ __forceinline__
void wgmma_m64n128k16(float* d, uint64_t descA, uint64_t descB) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        " %8,%9,%10,%11,%12,%13,%14,%15,"
        " %16,%17,%18,%19,%20,%21,%22,%23,"
        " %24,%25,%26,%27,%28,%29,%30,%31,"
        " %32,%33,%34,%35,%36,%37,%38,%39,"
        " %40,%41,%42,%43,%44,%45,%46,%47,"
        " %48,%49,%50,%51,%52,%53,%54,%55,"
        " %56,%57,%58,%59,%60,%61,%62,%63}, "
        " %64, %65, 1, 1, 1, 1, 1;\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(descA), "l"(descB));
}

__global__ void sgemm_v19(int M, int N, int K,
                          const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C) {
    const int bm = blockIdx.x * M_TILE;
    const int bn = blockIdx.y * N_TILE;
    const int tid = threadIdx.x;
    const int wg_id = tid / (WARPS_PER_WG * 32);   // 0..1
    const int t_in_wg = tid - wg_id * (WARPS_PER_WG * 32);
    const int warp_in_wg = t_in_wg / 32;            // 0..3
    const int lane = t_in_wg & 31;

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

    // Accumulators (64 fp32 per thread). WGMMA accumulates into these.
    float acc[ACCS_PER_TH];
    #pragma unroll
    for (int i = 0; i < ACCS_PER_TH; i++) acc[i] = 0.0f;

    // Prologue
    issue_load(0, 0);
    cp_async_commit();

    const int num_k_iters = K / K_TILE;
    for (int it = 0; it < num_k_iters; it++) {
        const int cur = it % STAGES;

        if (it + 1 < num_k_iters) {
            int nxt = (it + 1) % STAGES;
            issue_load(nxt, (it + 1) * K_TILE);
            cp_async_commit();
            cp_async_wait_lt1();
        } else {
            cp_async_wait_all();
        }

        convert_stage(cur);
        __syncthreads();

        const int wrkA_base = cur * (K_TILE * WRK_A_LD);
        const int wrkB_base = cur * (K_TILE * WRK_B_LD);

        // Issue WGMMAs for this iter's K_TILE in chunks of WG_K=16.
        // This WG covers M rows [wg_m_off, wg_m_off + WG_M).
        const int wg_m_off = wg_id * WG_M;

        wgmma_fence();
        #pragma unroll
        for (int kk = 0; kk < K_TILE; kk += WG_K) {
            // MN-major fp16 descriptor: LBO = K-row stride, SBO = 8-elem M chunk stride
            uint64_t descA = make_desc(
                &wrkA[wrkA_base + kk * WRK_A_LD + wg_m_off],
                WRK_A_LD * sizeof(half),                  // LBO = 256
                16);                                       // SBO = 16
            uint64_t descB = make_desc(
                &wrkB[wrkB_base + kk * WRK_B_LD + 0],
                WRK_B_LD * sizeof(half),
                16);
            wgmma_m64n128k16(acc, descA, descB);
        }
        wgmma_commit();
        wgmma_wait0();
    }

    // ---- Store C ----
    // Per-thread accumulator layout for wgmma.m64nN.f32 (per PTX/CUTLASS):
    //   For each mma block c in 0..N/8-1 (along N), the four regs are
    //     reg[c*4 + 0] = element at (warp*16 + lane/4,     c*8 + (lane%4)*2 + 0)
    //     reg[c*4 + 1] = element at (warp*16 + lane/4,     c*8 + (lane%4)*2 + 1)
    //     reg[c*4 + 2] = element at (warp*16 + lane/4 + 8, c*8 + (lane%4)*2 + 0)
    //     reg[c*4 + 3] = element at (warp*16 + lane/4 + 8, c*8 + (lane%4)*2 + 1)
    const int row_base = wg_id * WG_M + warp_in_wg * 16;
    const int row_lo   = row_base + lane / 4;
    const int row_hi   = row_lo + 8;
    const int col_base = (lane & 3) * 2;
    #pragma unroll
    for (int c = 0; c < WG_N / 8; c++) {
        int g_col0 = bn + c * 8 + col_base + 0;
        int g_col1 = bn + c * 8 + col_base + 1;
        int g_row_lo = bm + row_lo;
        int g_row_hi = bm + row_hi;
        C[(size_t)g_col0 * M + g_row_lo] = acc[c * 4 + 0];
        C[(size_t)g_col1 * M + g_row_lo] = acc[c * 4 + 1];
        C[(size_t)g_col0 * M + g_row_hi] = acc[c * 4 + 2];
        C[(size_t)g_col1 * M + g_row_hi] = acc[c * 4 + 3];
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
    cudaFuncSetAttribute(sgemm_v19,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);
    for (int i = 0; i < Nt + 2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        sgemm_v19<<<grid, block, smem_bytes>>>(m, n, k, A, B, C2);
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

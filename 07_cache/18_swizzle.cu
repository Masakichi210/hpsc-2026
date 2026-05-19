// 18_swizzle.cu  --  HPSC 2026 SGEMM optimization, step 5:
//   Builds on 17_mma by adding an XOR-based swizzle to wrkA / wrkB shmem so
//   that ldmatrix.trans.x4 hits distinct banks within each 8-lane sub-tile.
//
//   Physical offset for logical (k_row, m_col) in wrkA:
//       k_row * LD + (m_col XOR ((k_row & 7) << 3))     (units: halfs)
//   Same shape for wrkB with n_col.
//
//   LD = M_TILE (= 128 halfs); padding dropped because the swizzle plays
//   the role of bank-conflict avoidance.
//
//   The 4-bit XOR mask `(k_row & 7) << 3` permutes 8-half chunks within a
//   row. For 8-aligned base columns (m_base, n_base) the swizzle preserves
//   the 4- and 8-half contiguity needed for half2 stores (in convert_stage)
//   and 8-half ldmatrix reads.

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

#define MMA_M    16
#define MMA_N    8
#define MMA_K    16
#define WTM      (M_WARP / MMA_M)           // 4
#define WTN      (N_WARP / MMA_N)           // 4

// NO padding; bank conflicts handled by swizzle.
#define WRK_A_LD M_TILE
#define WRK_B_LD N_TILE

#define A_STEPS  ((K_TILE * M_TILE) / (THREADS * 4))   // 4
#define B_STEPS  ((N_TILE * K_TILE) / (THREADS * 4))   // 4

__device__ __forceinline__ int swz(int col, int row) {
    return col ^ ((row & 7) << 3);
}

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

__device__ __forceinline__
void ldmatrix_trans_x4(uint32_t (&r)[4], uint32_t smem_int_ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smem_int_ptr));
}

__device__ __forceinline__
void mma_m16n8k16(float (&d)[4],
                  const uint32_t (&a)[4],
                  const uint32_t (&b)[2],
                  const float (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__global__ void sgemm_v18(int M, int N, int K,
                          const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C) {
    const int bm = blockIdx.x * M_TILE;
    const int bn = blockIdx.y * N_TILE;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
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
            // Swizzle: m_phys = m_off XOR ((k_off & 7) * 8).
            // m_off is a multiple of 4; XOR by multiples of 8 preserves the
            // 4-element run because (m_off & 3) == (m_phys & 3).
            int m_phys = swz(m_off, k_off);
            half* dst = &wrkA[wrkA_base + k_off * WRK_A_LD + m_phys];
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
            // Each of the 4 halfs is written to a different k row; the
            // swizzle therefore must be recomputed per i.
            wrkB[wrkB_base + (k_off + 0) * WRK_B_LD + swz(n_off, k_off + 0)] = __float2half(v.x);
            wrkB[wrkB_base + (k_off + 1) * WRK_B_LD + swz(n_off, k_off + 1)] = __float2half(v.y);
            wrkB[wrkB_base + (k_off + 2) * WRK_B_LD + swz(n_off, k_off + 2)] = __float2half(v.z);
            wrkB[wrkB_base + (k_off + 3) * WRK_B_LD + swz(n_off, k_off + 3)] = __float2half(v.w);
        }
    };

    float acc[WTM][WTN][4];
    #pragma unroll
    for (int i = 0; i < WTM; i++)
        #pragma unroll
        for (int j = 0; j < WTN; j++)
            #pragma unroll
            for (int x = 0; x < 4; x++)
                acc[i][j][x] = 0.0f;

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

        // Lane decomposition for ldmatrix.x4 (matches v17).
        const int t_row     = lane & 7;
        const int sub_k_off = (lane >> 4) << 3;     // 0 or 8
        const int sub_c_off = ((lane >> 3) & 1) << 3; // 0 or 8

        uint32_t a_regs[WTM][4];
        uint32_t b_regs[WTN][2];

        #pragma unroll
        for (int kk = 0; kk < K_TILE; kk += MMA_K) {
            int k_row    = kk + sub_k_off + t_row;
            int shift    = (k_row & 7) << 3;        // = t_row * 8

            #pragma unroll
            for (int i = 0; i < WTM; i++) {
                int m_base = wm * M_WARP + i * MMA_M;
                int m_col  = m_base + sub_c_off;
                int m_phys = m_col ^ shift;
                uint32_t sptr = __cvta_generic_to_shared(
                    &wrkA[wrkA_base + k_row * WRK_A_LD + m_phys]);
                ldmatrix_trans_x4(a_regs[i], sptr);
            }

            uint32_t b_x4[2][4];
            #pragma unroll
            for (int g = 0; g < 2; g++) {
                int n_base = wn * N_WARP + g * 16;
                int n_col  = n_base + sub_c_off;
                int n_phys = n_col ^ shift;
                uint32_t sptr = __cvta_generic_to_shared(
                    &wrkB[wrkB_base + k_row * WRK_B_LD + n_phys]);
                ldmatrix_trans_x4(b_x4[g], sptr);
            }
            b_regs[0][0] = b_x4[0][0]; b_regs[0][1] = b_x4[0][2];
            b_regs[1][0] = b_x4[0][1]; b_regs[1][1] = b_x4[0][3];
            b_regs[2][0] = b_x4[1][0]; b_regs[2][1] = b_x4[1][2];
            b_regs[3][0] = b_x4[1][1]; b_regs[3][1] = b_x4[1][3];

            #pragma unroll
            for (int i = 0; i < WTM; i++) {
                #pragma unroll
                for (int j = 0; j < WTN; j++) {
                    mma_m16n8k16(acc[i][j], a_regs[i], b_regs[j], acc[i][j]);
                }
            }
        }
    }

    // ---- Store C ----
    const int t_row_c = lane / 4;
    const int t_col_c = (lane & 3) * 2;
    #pragma unroll
    for (int i = 0; i < WTM; i++) {
        int m_base = bm + wm * M_WARP + i * MMA_M;
        #pragma unroll
        for (int j = 0; j < WTN; j++) {
            int n_base = bn + wn * N_WARP + j * MMA_N;
            int r0 = m_base + t_row_c + 0;
            int r1 = m_base + t_row_c + 8;
            int c0 = n_base + t_col_c + 0;
            int c1 = n_base + t_col_c + 1;
            C[(size_t)c0 * M + r0] = acc[i][j][0];
            C[(size_t)c1 * M + r0] = acc[i][j][1];
            C[(size_t)c0 * M + r1] = acc[i][j][2];
            C[(size_t)c1 * M + r1] = acc[i][j][3];
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
    cudaFuncSetAttribute(sgemm_v18,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);
    for (int i = 0; i < Nt + 2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        sgemm_v18<<<grid, block, smem_bytes>>>(m, n, k, A, B, C2);
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

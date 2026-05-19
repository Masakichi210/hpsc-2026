// 14_tile128.cu  --  HPSC 2026 SGEMM optimization, step 1:
//   - block tile 128 x 128, K-block = 32
//   - 256 threads / 8 warps per block, warp grid 2 (M) x 4 (N)
//   - each warp accumulates 4 x 2 = 8 WMMA 16x16x16 tiles in registers
//   - global -> shared loads via float4 (4 floats/thread, 4 steps)
//   - shared layout: half shA[K][M+pad], half shB[K][N+pad]
//   - no async/double-buffer yet (that comes in 15)
//
// Matrices follow the cublasGemmEx(N, N, m, n, k, A(lda=m), B(ldb=k), C(ldc=m))
// convention used in 13_tensorcore.cu, so A and B are column-major
// (A is m x k with leading dim m; B is k x n with leading dim k; C is m x n
// with leading dim m).

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

#define WARPS_M  2
#define WARPS_N  4
#define WARPS    (WARPS_M * WARPS_N)        // 8
#define THREADS  (WARPS * 32)               // 256

#define M_WARP   (M_TILE / WARPS_M)         // 64
#define N_WARP   (N_TILE / WARPS_N)         // 32

#define WMMA_M   16
#define WMMA_N   16
#define WMMA_K   16
#define WTM      (M_WARP / WMMA_M)          // 4
#define WTN      (N_WARP / WMMA_N)          // 2

// +8 half pad to break 128-byte bank patterns
#define SHA_LD   (M_TILE + 8)
#define SHB_LD   (N_TILE + 8)

__global__ void sgemm_v14(int M, int N, int K,
                          const float * __restrict__ A,
                          const float * __restrict__ B,
                          float * __restrict__ C) {
  const int bm = blockIdx.x * M_TILE;
  const int bn = blockIdx.y * N_TILE;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int wm = warp_id / WARPS_N;         // 0..1
  const int wn = warp_id % WARPS_N;         // 0..3

  __shared__ half shA[K_TILE][SHA_LD];
  __shared__ half shB[K_TILE][SHB_LD];

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WTM][WTN];
  #pragma unroll
  for (int i = 0; i < WTM; i++)
    #pragma unroll
    for (int j = 0; j < WTN; j++)
      wmma::fill_fragment(acc[i][j], 0.0f);

  // Per K-iter we need:
  //   A tile: K_TILE * M_TILE = 32*128 = 4096 floats = 1024 float4
  //   B tile: K_TILE * N_TILE = 32*128 = 4096 floats = 1024 float4
  // With 256 threads each issuing 1 float4 we cover 1024 floats per step
  // -> 4 steps per side.
  const int A_STEPS = (K_TILE * M_TILE) / (THREADS * 4);   // 4
  const int B_STEPS = (K_TILE * N_TILE) / (THREADS * 4);   // 4

  for (int kbase = 0; kbase < K; kbase += K_TILE) {
    __syncthreads();

    // ---- Load A tile (col-major, contiguous in M) ----
    // shA[k_off][m_off] = A[bm + m_off, kbase + k_off]
    //                   = A[(kbase + k_off) * M + bm + m_off]
    #pragma unroll
    for (int s = 0; s < A_STEPS; s++) {
      int idx4   = s * THREADS + tid;            // 0..1023
      int m_off4 = idx4 & ((M_TILE / 4) - 1);    // idx4 % 32
      int k_off  = idx4 / (M_TILE / 4);          // 0..31
      int m_off  = m_off4 * 4;
      float4 v = *reinterpret_cast<const float4*>(
                    &A[(kbase + k_off) * M + bm + m_off]);
      shA[k_off][m_off + 0] = __float2half(v.x);
      shA[k_off][m_off + 1] = __float2half(v.y);
      shA[k_off][m_off + 2] = __float2half(v.z);
      shA[k_off][m_off + 3] = __float2half(v.w);
    }

    // ---- Load B tile (col-major over n*K, contiguous in K) ----
    // shB[k_off][n_off] = B[kbase + k_off, bn + n_off]
    //                   = B[(bn + n_off) * K + kbase + k_off]
    #pragma unroll
    for (int s = 0; s < B_STEPS; s++) {
      int idx4   = s * THREADS + tid;            // 0..1023
      int k_off4 = idx4 & ((K_TILE / 4) - 1);    // 0..7
      int n_off  = idx4 / (K_TILE / 4);          // 0..127
      int k_off  = k_off4 * 4;
      float4 v = *reinterpret_cast<const float4*>(
                    &B[(bn + n_off) * K + kbase + k_off]);
      shB[k_off + 0][n_off] = __float2half(v.x);
      shB[k_off + 1][n_off] = __float2half(v.y);
      shB[k_off + 2][n_off] = __float2half(v.z);
      shB[k_off + 3][n_off] = __float2half(v.w);
    }

    __syncthreads();

    // ---- MMA over K_TILE in 2 steps of WMMA_K=16 ----
    #pragma unroll
    for (int kk = 0; kk < K_TILE; kk += WMMA_K) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag[WTM];
      #pragma unroll
      for (int i = 0; i < WTM; i++) {
        int m_row = wm * M_WARP + i * WMMA_M;
        wmma::load_matrix_sync(a_frag[i], &shA[kk][m_row], SHA_LD);
      }
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WTN];
      #pragma unroll
      for (int j = 0; j < WTN; j++) {
        int n_col = wn * N_WARP + j * WMMA_N;
        wmma::load_matrix_sync(b_frag[j], &shB[kk][n_col], SHB_LD);
      }
      #pragma unroll
      for (int i = 0; i < WTM; i++)
        #pragma unroll
        for (int j = 0; j < WTN; j++)
          wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
    }
  }

  // ---- Store C (col-major, ld = M) ----
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
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta  = 0.0;
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

  // ---- cuBLAS reference ----
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt + 2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 m, n, k,
                 &alpha,
                 A, CUDA_R_32F, m,
                 B, CUDA_R_32F, k,
                 &beta,
                 C, CUDA_R_32F, m,
                 CUBLAS_COMPUTE_32F_FAST_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;

  // ---- Custom kernel ----
  dim3 block(THREADS);
  dim3 grid((m + M_TILE - 1) / M_TILE, (n + N_TILE - 1) / N_TILE);
  for (int i = 0; i < Nt + 2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    sgemm_v14<<<grid, block>>>(m, n, k, A, B, C2);
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

# HPSC 2026 Final Project: SGEMM on H100

Target: maximize Gflops of a hand-rolled CUDA SGEMM kernel relative to
`cublasGemmEx(CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`.

- Problem size: M=10240, K=4096, N=8192, FP32 I/O.
- Reference kernel: `13_tensorcore.cu` (WMMA 16x16x16, 64x64 tile, 2 warps/block).
- Measurement: warm-up 2 + average over 10, mean absolute error vs cuBLAS.
- Build: `nvcc -arch=sm_90 -O3 ...` on TSUBAME 4.0 H100 (sm_90).

## Environment

- TSUBAME 4.0 H100 node (r6n11), NVIDIA H100 (single GPU)
- CUDA 12.8 (release 12.8.61), `nvcc -arch=sm_90 -O3 -lineinfo`
- Driver / SXM details: see `bench_logs/13_tensorcore_*.log`

## Results

| ver | file | optimization | Gflops | cublas Gflops | ratio | err | notes |
|----:|------|--------------|-------:|--------------:|------:|----:|-------|
|  13 | 13_tensorcore.cu | WMMA 16x16x16, 64x64 tile, 2 warps/block | 20,883 | 361,995 | 5.77% | 0.003980 | baseline |
|  14 | 14_tile128.cu | 128x128 tile, K=32, 8 warps/block (2x4), 4x2 WMMA/warp, float4 loads | 121,515 | 359,439 | 33.81% | 0.003980 | 5.82x over baseline |
|  15 | 15_async.cu | + cp.async (cg.shared.global, 16B) 2-stage pipeline, FP32 staging + FP16 work tile | 168,553 | 360,376 | 46.77% | 0.003980 | 8.07x over baseline, 1.39x over 14 |
|  16 | 16_dbuf.cu | + double-buffered wrkA/wrkB, dropped one __syncthreads/iter, half2 conversion for A | 166,836 | 361,724 | 46.12% | 0.003980 | basically tied with 15; sync/conversion no longer bottleneck |
|  17 | 17_mma.cu | wmma replaced by inline PTX ldmatrix.trans.x4 + mma.sync.m16n8k16, +8 pad kept | 167,553 | 361,181 | 46.39% | 0.003980 | parity with 16; foundation for 18 to add swizzle |
|  18 | 18_swizzle.cu | + XOR swizzle on wrkA/wrkB (m XOR ((k & 7) << 3)), padding dropped | 122,142 | 360,618 | 33.87% | 0.003980 | regressed: simple swizzle didn't cut sub0/sub2 conflicts, and B-side per-i XOR added compute overhead |

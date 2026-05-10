#include <cstdio>
#include <cstdlib>

__global__ void init_bucket(int *bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range) bucket[i] = 0;
}

__global__ void count_bucket(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&bucket[key[i]], 1);
}

__global__ void scan_bucket(int *bucket, int *tmp, int range) {
  int i = threadIdx.x;
  if (i >= range) return;
  for (int j=1; j<range; j<<=1) {
    tmp[i] = bucket[i];
    __syncthreads();
    if (i >= j) bucket[i] += tmp[i-j];
    __syncthreads();
  }
}

__global__ void write_keys(int *key, int *offset, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range) {
    int start = (i == 0) ? 0 : offset[i-1];
    int end = offset[i];
    for (int k = start; k < end; k++) {
      key[k] = i;
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *key, *bucket, *tmp;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&tmp, range*sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  init_bucket<<<1,range>>>(bucket, range);
  cudaDeviceSynchronize();
  count_bucket<<<1,n>>>(key, bucket, n);
  cudaDeviceSynchronize();
  scan_bucket<<<1,range>>>(bucket, tmp, range);
  cudaDeviceSynchronize();
  write_keys<<<1,range>>>(key, bucket, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
  cudaFree(tmp);
}

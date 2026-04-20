#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0);
#pragma omp parallel for
  for (int i=0; i<n; i++)
#pragma omp atomic update
    bucket[key[i]]++;

  std::vector<int> offset(range,0);
#pragma omp parallel for
  for (int i=1; i<range; i++) offset[i] = bucket[i-1];

  std::vector<int> b(range);
#pragma omp parallel
  for (int j=1; j<range; j<<=1) {
#pragma omp for
    for (int i=0; i<range; i++)
      b[i] = offset[i];
#pragma omp for
    for (int i=j; i<range; i++)
      offset[i] += b[i-j];
  }

#pragma omp parallel for
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (int k=0; k<bucket[i]; k++) {
      key[j+k] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}

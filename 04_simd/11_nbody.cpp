#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);
  __m512 zero = _mm512_setzero_ps();
  for(int i=0; i<N; i++) {
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);
    __m512 rx = _mm512_sub_ps(xi, xvec);
    __m512 ry = _mm512_sub_ps(yi, yvec);
    __m512 r2 = _mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry));
    __m512 r  = _mm512_sqrt_ps(r2);
    __m512 r3 = _mm512_mul_ps(_mm512_mul_ps(r, r), r);
    __mmask16 mask = _mm512_cmp_ps_mask(r2, zero, _MM_CMPINT_NE);
    __m512 fxc = _mm512_div_ps(_mm512_mul_ps(rx, mvec), r3);
    __m512 fyc = _mm512_div_ps(_mm512_mul_ps(ry, mvec), r3);
    fxc = _mm512_mask_blend_ps(mask, zero, fxc);
    fyc = _mm512_mask_blend_ps(mask, zero, fyc);
    fx[i] -= _mm512_reduce_add_ps(fxc);
    fy[i] -= _mm512_reduce_add_ps(fyc);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}

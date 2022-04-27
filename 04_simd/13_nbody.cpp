#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/types.h>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m256 zero=_mm256_setzero_ps();
  __m256 x_vec=_mm256_load_ps(x);
  __m256 y_vec=_mm256_load_ps(y);
  __m256 m_vec=_mm256_load_ps(m);
  __m256 fx_vec=_mm256_load_ps(fx);
  __m256 fy_vec=_mm256_load_ps(fy);

  for(int i=0; i<N; i++) {
    __m256 x_i=_mm256_set1_ps(x[i]);
    __m256 y_i=_mm256_set1_ps(y[i]);
    __m256 fx_i=_mm256_set1_ps(fx[i]);
    __m256 fy_i=_mm256_set1_ps(fy[i]);
    __m256 r=zero;

    // rx (i==j: 0, else: x_i-x_j)
    __m256 rx=_mm256_sub_ps(x_i,x_vec);
    // ry (i==j: 0, else: y_i,y_j)
    __m256 ry=_mm256_sub_ps(y_i,y_vec);

    // rx^2+ry^2
    r=_mm256_fmadd_ps(rx,rx,r);
    r=_mm256_fmadd_ps(ry,ry,r);

    // i==j => 0, i!=j => rx^2+ry^2
    __m256 mask=_mm256_cmp_ps(r,zero,_CMP_GT_OQ);
    __m256 inv_r=_mm256_rsqrt_ps(r);
    inv_r=_mm256_blendv_ps(zero,inv_r,mask);

    // calc r^{-3}
    __m256 inv_r2=_mm256_mul_ps(inv_r,inv_r);
    __m256 inv_r3=_mm256_mul_ps(inv_r2,inv_r);
    
    // fx, fy
    m_vec=_mm256_mul_ps(m_vec,inv_r3);
    fx_i=_mm256_mul_ps(rx,m_vec);
    fy_i=_mm256_mul_ps(ry,m_vec);
    fx_vec=_mm256_sub_ps(fx_vec,fx_i);
    fy_vec=_mm256_sub_ps(fy_vec,fy_i);
    //}
    _mm256_store_ps(fx,fx_vec);
    _mm256_store_ps(fy,fy_vec);
    /*
    for(int j=0; j<N; j++) {
      if(i != j) {
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];
        float r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }
    }*/
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}

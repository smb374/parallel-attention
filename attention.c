//
// Created by poyehchen on 5/19/25.
//
#include <immintrin.h>
#include <math.h>
#include <smmintrin.h>
#include <stdalign.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <time.h>

// NOTE: feel free to include any header you need, but we will not
// link libraries other than C's math library for you.

// NOTE: feel free to add new macros
#define SLICE 8

// NOTE: feel free to add new functions

/*
 * Q: m by dk
 * K: n by dk
 * V: n by dv
 * result: m by dv, containing the attention result
 */

static inline __attribute__((always_inline)) __m256i create_mask(const int rem) {
    const __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    const __m256i remv = _mm256_set1_epi64x(MIN(rem, 4));
    return _mm256_cmpgt_epi64(remv, indices);
}

static inline __attribute__((always_inline)) __m256d hmax(__m256d x) {
    __m256d y = _mm256_permute2f128_pd(x, x, 1);
    __m256d m1 = _mm256_max_pd(x, y);
    __m256d m2 = _mm256_permute_pd(m1, 5);
    return _mm256_max_pd(m1, m2);
}

static inline __attribute__((always_inline)) __m256d hsum(__m256d x) {
    __m256d y = _mm256_permute2f128_pd(x, x, 1);
    __m256d m1 = _mm256_add_pd(x, y);
    __m256d m2 = _mm256_permute_pd(m1, 5);
    return _mm256_add_pd(m1, m2);
}

static inline __attribute__((always_inline)) double reduce_sum(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    const __m128d vhigh = _mm256_extractf128_pd(v, 1);
    vlow = _mm_add_pd(vlow, vhigh);

    const __m128d h64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_add_sd(vlow, h64));
}

static inline __attribute__((always_inline)) double reduce_max(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    const __m128d vhigh = _mm256_extractf128_pd(v, 1);
    vlow = _mm_max_pd(vlow, vhigh);

    const __m128d h64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_max_sd(vlow, h64));
}

static inline __attribute__((always_inline)) __m256d vmax_avx2(const double *z, const int n) {
    const __m256d neg_inf = _mm256_set1_pd(-HUGE_VAL);
    __m256d maxv = neg_inf;

    const int nsteps = (n / 4) * 4, nrem = n % 4;
    int i = 0;
    for (; i < nsteps; i += 4) {
        __m256d x = _mm256_loadu_pd(z + i);
        maxv = _mm256_max_pd(maxv, x);
    }
    if (nrem) {
        __m256i mask = create_mask(n - i);
        __m256d x = _mm256_maskload_pd(z + i, mask);
        __m256d y = _mm256_blendv_pd(neg_inf, x, _mm256_castsi256_pd(mask));
        maxv = _mm256_max_pd(maxv, y);
    }

    return hmax(maxv);
}

static inline __attribute__((always_inline)) __m256d vsum_avx2(const double *z, const int n) {
    const __m256i fmask = _mm256_set1_epi8(-1);
    __m256d sum = _mm256_setzero_pd();

    const int nsteps = (n / 4) * 4, nrem = n % 4;
    int i = 0;
    for (; i < nsteps; i += 4) {
        __m256d x = _mm256_loadu_pd(z + i);
        sum = _mm256_add_pd(sum, x);
    }
    if (nrem) {
        __m256i mask = create_mask(n - i);
        __m256d x = _mm256_maskload_pd(z + i, mask);
        sum = _mm256_add_pd(sum, x);
    }

    return hsum(sum);
}

static inline __attribute__((always_inline)) void vdiv_avx2(double *out, const double *z, const __m256d basev,
                                                            const int n) {
    const int nsteps = (n / 4) * 4, nrem = n % 4;
    int i = 0;
    for (; i < nsteps; i += 4) {
        __m256d x = _mm256_loadu_pd(z + i);
        __m256d y = _mm256_div_pd(x, basev);
        _mm256_storeu_pd(out + i, y);
    }
    if (nrem) {
        __m256i mask = create_mask(n - i);
        __m256d x = _mm256_maskload_pd(z + i, mask);
        __m256d y = _mm256_div_pd(x, basev);
        _mm256_maskstore_pd(out + i, mask, y);
    }
}

static inline __attribute__((always_inline)) void vmul_avx2(double *out, const double *z, const __m256d basev,
                                                            const int n) {
    const int nsteps = (n / 4) * 4, nrem = n % 4;
    int i = 0;
    for (; i < nsteps; i += 4) {
        __m256d x = _mm256_loadu_pd(z + i);
        __m256d y = _mm256_mul_pd(x, basev);
        _mm256_storeu_pd(out + i, y);
    }
    if (nrem) {
        __m256i mask = create_mask(n - i);
        __m256d x = _mm256_maskload_pd(z + i, mask);
        __m256d y = _mm256_mul_pd(x, basev);
        _mm256_maskstore_pd(out + i, mask, y);
    }
}

static inline __attribute__((always_inline)) void vadd_avx2(double *out, const double *a, const double *b,
                                                            const int n) {
    const int nsteps = (n / 4) * 4, nrem = n % 4;
    int i = 0;
    for (; i < nsteps; i += 4) {
        __m256d x = _mm256_loadu_pd(a + i);
        __m256d y = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_add_pd(x, y));
    }
    if (nrem) {
        __m256i mask = create_mask(n - i);
        __m256d x = _mm256_maskload_pd(a + i, mask);
        __m256d y = _mm256_maskload_pd(b + i, mask);
        _mm256_maskstore_pd(out + i, mask, _mm256_add_pd(x, y));
    }
}

#define EXP_P0 1.0
#define EXP_P1 1.0
#define EXP_P2 0.5
#define EXP_P3 0.16666666666666666
#define EXP_P4 0.041666666666666664
#define EXP_P5 0.008333333333333333
#define EXP_P6 0.001388888888888889
#define EXP_P7 0.0001984126984126984
#define EXP_P8 2.48015873015873e-05

#define LOG2_E 1.4426950408889634073599
#define LN2_HI 0.69314718036912381649
#define LN2_LO 1.90821492927058770002e-10

static inline __m256d exp256_pd(__m256d x) {
    // Constants
    const __m256d inv_ln2 = _mm256_set1_pd(LOG2_E);  // 1/ln(2)

    // Split ln2 into high/low parts for extra precision: ln2 = ln2_hi + ln2_lo
    const __m256d ln2_hi = _mm256_set1_pd(LN2_HI);
    const __m256d ln2_lo = _mm256_set1_pd(LN2_LO);

    // Polynomial coefficients for exp(r) on r in [-ln2/2, ln2/2]
    const __m256d exp_p0 = _mm256_set1_pd(EXP_P0);
    const __m256d exp_p1 = _mm256_set1_pd(EXP_P1);
    const __m256d exp_p2 = _mm256_set1_pd(EXP_P2);
    const __m256d exp_p3 = _mm256_set1_pd(EXP_P3);
    const __m256d exp_p4 = _mm256_set1_pd(EXP_P4);
    const __m256d exp_p5 = _mm256_set1_pd(EXP_P5);
    const __m256d exp_p6 = _mm256_set1_pd(EXP_P6);

    // 1. Range reduction: n = floor(x / ln2)
    __m256d t = _mm256_mul_pd(x, inv_ln2);
    __m256d n_real = _mm256_floor_pd(t);

    // 2. r = x - n*ln2_hi - n*ln2_lo
    __m256d r = _mm256_fnmadd_pd(n_real, ln2_hi, x);
    r = _mm256_fnmadd_pd(n_real, ln2_lo, r);

    // 3. Taylor series of exp(r) to the 7th term (0 - 6)
    __m256d poly = exp_p6;
    poly = _mm256_fmadd_pd(poly, r, exp_p5);
    poly = _mm256_fmadd_pd(poly, r, exp_p4);
    poly = _mm256_fmadd_pd(poly, r, exp_p3);
    poly = _mm256_fmadd_pd(poly, r, exp_p2);
    poly = _mm256_fmadd_pd(poly, r, exp_p1);
    poly = _mm256_fmadd_pd(poly, r, exp_p0);

    // 4. Reconstruct: result = p * 2^n
    //    Build exponent bits: e = (n + bias) << 52
    //    where bias = 1023 for IEEE‑754 double
    const __m128i bias32 = _mm_set1_epi32(1023);
    __m128i n_lo = _mm256_cvttpd_epi32(n_real);  // round->i32 low 4 lanes
    __m256i n64 = _mm256_cvtepi32_epi64(n_lo);   // widen to i64
    __m256i e64 = _mm256_add_epi64(n64, _mm256_cvtepi32_epi64(bias32));
    e64 = _mm256_slli_epi64(e64, 52);  // shift into exponent field

    __m256d pow2_n = _mm256_castsi256_pd(e64);

    __m256d res = _mm256_mul_pd(poly, pow2_n);

    for (int i = 0; i < 4; i++) {
        if (x[i] == -INFINITY) {
            res[i] = 0;
        }
    }

    return res;
}

void softmax_avx2(double *z, double *exp_z, const __m256d maxv, const int n) {
    const int nsteps = (n / 4) * 4, nrem = n % 4;
    int i = 0;
    for (; i < nsteps; i += 4) {
        __m256d x = _mm256_loadu_pd(z + i);
        __m256d exp_y = exp256_pd(_mm256_sub_pd(x, maxv));
        _mm256_storeu_pd(exp_z + i, exp_y);
    }
    if (nrem) {
        __m256i mask = create_mask(n - i);
        __m256d x = _mm256_maskload_pd(z + i, mask);
        __m256d exp_y = exp256_pd(_mm256_sub_pd(x, maxv));
        _mm256_maskstore_pd(exp_z + i, mask, exp_y);
    }

    __m256d sum = vsum_avx2(exp_z, n);
    vdiv_avx2(z, exp_z, sum, n);
}

void dot_product_avx2(double *out, const double *a, const double *b, const int n) {
    const int nsteps = (n / 4) * 4, nrem = n % 4;
    int i = 0;
    for (; i < nsteps; i += 4) {
        __m256d x = _mm256_loadu_pd(a + i);
        __m256d y = _mm256_loadu_pd(b + i);
        __m256d sum = _mm256_add_pd(x, y);
        _mm256_storeu_pd(out + i, sum);
    }
    if (nrem) {
        __m256i mask = create_mask(n - i);
        __m256d x = _mm256_maskload_pd(a + i, mask);
        __m256d y = _mm256_maskload_pd(b + i, mask);
        __m256d sum = _mm256_add_pd(x, y);
        _mm256_maskstore_pd(out + i, mask, sum);
    }
}

void matmulT_kernel(double *out, const double *A, const double *B, const int out_n, const int p, const int ms,
                    const int msize, const int ns, const int nsize, const int ps, const int psize) {
    const int nsteps = ns + (nsize / 4) * 4, nrem = nsize % 4;
    const int psteps = ps + (psize / 4) * 4, prem = psize % 4;
    const int me = ms + msize, ne = ns + nsize, pe = ps + psize;
    for (int i = ms; i < me; i++) {
        int j = ns;
        for (; j < nsteps; j += 4) {
            __m256d sum0 = _mm256_setzero_pd();
            __m256d sum1 = _mm256_setzero_pd();
            __m256d sum2 = _mm256_setzero_pd();
            __m256d sum3 = _mm256_setzero_pd();

            int k = ps;
            for (; k < psteps; k += 4) {
                __m256d x = _mm256_loadu_pd(A + i * p + k);

                __m256d y0 = _mm256_loadu_pd(B + j * p + k);
                __m256d y1 = _mm256_loadu_pd(B + (j + 1) * p + k);
                __m256d y2 = _mm256_loadu_pd(B + (j + 2) * p + k);
                __m256d y3 = _mm256_loadu_pd(B + (j + 3) * p + k);

                sum0 = _mm256_fmadd_pd(x, y0, sum0);
                sum1 = _mm256_fmadd_pd(x, y1, sum1);
                sum2 = _mm256_fmadd_pd(x, y2, sum2);
                sum3 = _mm256_fmadd_pd(x, y3, sum3);
            }
            if (prem) {
                const __m256i mask = create_mask(pe - k);
                __m256d x = _mm256_maskload_pd(A + i * p + k, mask);

                __m256d y0 = _mm256_maskload_pd(B + j * p + k, mask);
                __m256d y1 = _mm256_maskload_pd(B + (j + 1) * p + k, mask);
                __m256d y2 = _mm256_maskload_pd(B + (j + 2) * p + k, mask);
                __m256d y3 = _mm256_maskload_pd(B + (j + 3) * p + k, mask);

                sum0 = _mm256_fmadd_pd(x, y0, sum0);
                sum1 = _mm256_fmadd_pd(x, y1, sum1);
                sum2 = _mm256_fmadd_pd(x, y2, sum2);
                sum3 = _mm256_fmadd_pd(x, y3, sum3);
            }
            out[i * out_n + j] = reduce_sum(sum0);
            out[i * out_n + j + 1] = reduce_sum(sum1);
            out[i * out_n + j + 2] = reduce_sum(sum2);
            out[i * out_n + j + 3] = reduce_sum(sum3);
        }
        if (nrem) {
            for (; j < ne; j++) {
                __m256d sum = _mm256_setzero_pd();
                int k = ps;
                for (; k < psteps; k += 4) {
                    __m256d x = _mm256_loadu_pd(A + i * p + k);
                    __m256d y = _mm256_loadu_pd(B + j * p + k);
                    sum = _mm256_fmadd_pd(x, y, sum);
                }
                if (prem) {
                    __m256i mask = create_mask(pe - k);
                    __m256d x = _mm256_maskload_pd(A + i * p + k, mask);
                    __m256d y = _mm256_maskload_pd(B + j * p + k, mask);
                    sum = _mm256_fmadd_pd(x, y, sum);
                }
                out[i * out_n + j] = reduce_sum(sum);
            }
        }
    }
}

void matmul_kernel(double *out, const double *A, const double *B, const int n, const int p, const int ms,
                   const int msize, const int ns, const int nsize, const int ps, const int psize) {
    const int msteps = ms + (msize / 2) * 2, mrem = msize % 2;
    const int psteps = ps + (psize / 8) * 8, prem = psize % 8;
    const int me = ms + msize, ne = ns + nsize, pe = ps + psize;
    int i = ms;
    for (; i < msteps; i += 2) {
        int j = ps;
        for (; j < psteps; j += 8) {
            __m256d sum00 = _mm256_setzero_pd();
            __m256d sum01 = _mm256_setzero_pd();
            __m256d sum10 = _mm256_setzero_pd();
            __m256d sum11 = _mm256_setzero_pd();

            for (int k = ns; k < ne; k++) {
                __m256d a0 = _mm256_set1_pd(A[i * n + k]);
                __m256d a1 = _mm256_set1_pd(A[(i + 1) * n + k]);

                __m256d br0 = _mm256_loadu_pd(B + k * p + j);
                __m256d br1 = _mm256_loadu_pd(B + k * p + (j + 4));

                sum00 = _mm256_fmadd_pd(a0, br0, sum00);
                sum01 = _mm256_fmadd_pd(a0, br1, sum01);
                sum10 = _mm256_fmadd_pd(a1, br0, sum10);
                sum11 = _mm256_fmadd_pd(a1, br1, sum11);
            }
            _mm256_storeu_pd(out + i * p + j, sum00);
            _mm256_storeu_pd(out + i * p + (j + 4), sum01);
            _mm256_storeu_pd(out + (i + 1) * p + j, sum10);
            _mm256_storeu_pd(out + (i + 1) * p + (j + 4), sum11);
        }
        if (prem) {
            for (; j < pe; j += 4) {
                __m256d sum0 = _mm256_setzero_pd();
                __m256d sum1 = _mm256_setzero_pd();
                __m256i mask = create_mask(pe - j);
                for (int k = ns; k < ne; k++) {
                    __m256d a0 = _mm256_set1_pd(A[i * n + k]);
                    __m256d a1 = _mm256_set1_pd(A[(i + 1) * n + k]);

                    __m256d br = _mm256_maskload_pd(B + k * p + j, mask);

                    sum0 = _mm256_fmadd_pd(a0, br, sum0);
                    sum1 = _mm256_fmadd_pd(a1, br, sum1);
                }
                _mm256_maskstore_pd(out + i * p + j, mask, sum0);
                _mm256_maskstore_pd(out + (i + 1) * p + j, mask, sum1);
            }
        }
    }
    if (mrem) {
        int j = ps;
        for (; j < psteps; j += 8) {
            __m256d sum0 = _mm256_setzero_pd();
            __m256d sum1 = _mm256_setzero_pd();

            for (int k = ns; k < ne; k++) {
                __m256d a = _mm256_set1_pd(A[i * n + k]);

                __m256d br0 = _mm256_loadu_pd(B + k * p + j);
                __m256d br1 = _mm256_loadu_pd(B + k * p + (j + 4));

                sum0 = _mm256_fmadd_pd(a, br0, sum0);
                sum1 = _mm256_fmadd_pd(a, br1, sum1);
            }
            _mm256_storeu_pd(out + i * p + j, sum0);
            _mm256_storeu_pd(out + i * p + (j + 4), sum1);
        }
        if (prem) {
            for (; j < pe; j += 4) {
                __m256d sum = _mm256_setzero_pd();
                __m256i mask = create_mask(pe - j);
                for (int k = ns; k < ne; k++) {
                    __m256d a = _mm256_set1_pd(A[i * n + k]);

                    __m256d br = _mm256_maskload_pd(B + k * p + j, mask);

                    sum = _mm256_fmadd_pd(a, br, sum);
                }
                _mm256_maskstore_pd(out + i * p + j, mask, sum);
            }
        }
    }
}

double invsqrt(double number) {
    double y = number;
    double x2 = y * 0.5;
    int64_t i;
    memcpy(&i, &y, sizeof(int64_t));
    i = 0x5fe6eb50c7b537a9 - (i >> 1);
    memcpy(&y, &i, sizeof(int64_t));
    y = y * (1.5 - (x2 * y * y));
    return y;
}

// Flash attention core with 4x4 tile
static inline __attribute__((always_inline)) void flash_atten_4x4core(double *s_r, double *p_r, double *O_r,
                                                                      double *PK_r, __m256d *m_r, __m256d *l_r,
                                                                      const double *Qb, const double *Kb,
                                                                      const double *Vb, const __m256d inv_dk_sqrt,
                                                                      const int jsize, const int dv, const int dk) {
    matmulT_kernel(s_r, Qb, Kb, 4, dk, 0, 4, 0, jsize, 0, dk);
    vmul_avx2(s_r, s_r, inv_dk_sqrt, 16);
    __m256d s_rv0 = _mm256_load_pd(s_r);
    __m256d s_rv1 = _mm256_load_pd(s_r + 4);
    __m256d s_rv2 = _mm256_load_pd(s_r + 8);
    __m256d s_rv3 = _mm256_load_pd(s_r + 12);

    __m256d s_rmax = _mm256_set_pd(reduce_max(s_rv3), reduce_max(s_rv2), reduce_max(s_rv1), reduce_max(s_rv0));
    __m256d m_rn = _mm256_max_pd(*m_r, s_rmax);
    __m256d m_rdiff = _mm256_sub_pd(*m_r, m_rn);
    // Need to use acurate exp.

    __m256d drate = exp256_pd(m_rdiff);
    *l_r = _mm256_mul_pd(*l_r, drate);

    vmul_avx2(O_r, O_r, _mm256_set1_pd(drate[0]), dv);
    vmul_avx2(O_r + dv, O_r + dv, _mm256_set1_pd(drate[1]), dv);
    vmul_avx2(O_r + 2 * dv, O_r + 2 * dv, _mm256_set1_pd(drate[2]), dv);
    vmul_avx2(O_r + 3 * dv, O_r + 3 * dv, _mm256_set1_pd(drate[3]), dv);
    // Softmax start
    __m256d m_rnv0 = _mm256_set1_pd(m_rn[0]);
    __m256d m_rnv1 = _mm256_set1_pd(m_rn[1]);
    __m256d m_rnv2 = _mm256_set1_pd(m_rn[2]);
    __m256d m_rnv3 = _mm256_set1_pd(m_rn[3]);
    __m256d exp_y0 = exp256_pd(_mm256_sub_pd(s_rv0, m_rnv0));
    __m256d exp_y1 = exp256_pd(_mm256_sub_pd(s_rv1, m_rnv1));
    __m256d exp_y2 = exp256_pd(_mm256_sub_pd(s_rv2, m_rnv2));
    __m256d exp_y3 = exp256_pd(_mm256_sub_pd(s_rv3, m_rnv3));
    _mm256_store_pd(p_r, exp_y0);
    _mm256_store_pd(p_r + 4, exp_y1);
    _mm256_store_pd(p_r + 8, exp_y2);
    _mm256_store_pd(p_r + 12, exp_y3);
    // Softmax end
    matmul_kernel(PK_r, p_r, Vb, 4, dv, 0, 4, 0, jsize, 0, dv);
    vadd_avx2(O_r, O_r, PK_r, 4 * dv);
    __m256d exp_sums = _mm256_set_pd(reduce_sum(exp_y3), reduce_sum(exp_y2), reduce_sum(exp_y1), reduce_sum(exp_y0));
    *l_r = _mm256_add_pd(*l_r, exp_sums);
    *m_r = m_rn;
}

static inline __attribute__((always_inline)) void flash_atten_1x4core(double *s_r, double *p_r, double *O_r,
                                                                      double *PK_r, double *m_r, double *l_r,
                                                                      const double *q_r, const double *Kb,
                                                                      const double *Vb, const __m256d inv_dk_sqrt,
                                                                      const int jsize, const int dv, const int dk) {
    // s_r is at least 1x4.
    matmulT_kernel(s_r, q_r, Kb, 4, dk, 0, 1, 0, jsize, 0, dk);
    __m256d s_rv = _mm256_loadu_pd(s_r);
    s_rv = _mm256_mul_pd(s_rv, inv_dk_sqrt);

    double m_rn = MAX(*m_r, reduce_max(s_rv));
    double drate = exp(*m_r - m_rn);
    *l_r *= drate;
    vmul_avx2(O_r, O_r, _mm256_set1_pd(drate), dv);

    // Softmax start
    const __m256d m_rnv = _mm256_set1_pd(m_rn);
    __m256d exp_y = exp256_pd(_mm256_sub_pd(s_rv, m_rnv));
    _mm256_storeu_pd(p_r, exp_y);
    // Softmax end

    matmul_kernel(PK_r, p_r, Vb, 4, dv, 0, 1, 0, jsize, 0, dv);
    vadd_avx2(O_r, O_r, PK_r, dv);

    *l_r += reduce_sum(exp_y);
    *m_r = m_rn;
}

void attention(const double *Q, const double *K, const double *V, double *result, const int m, const int n,
               const int dk, const int dv) {
    alignas(32) double s_r[16];
    alignas(32) double p_r[16];
    double *O_r = calloc(4 * dv, sizeof(double));
    double *PK_r = calloc(4 * dv, sizeof(double));

    const __m256d inv_dk_sqrt = _mm256_set1_pd(1 / sqrt(dk));
    const int msteps = (m / 4) * 4, mrem = m % 4;
    const int nsteps = (n / 4) * 4, nrem = n % 4;

    int i = 0;
    for (; i < msteps; i += 4) {
        memset(O_r, 0, 4 * dv * sizeof(double));
        const double *Qb = Q + i * dk;
        __m256d l_r = _mm256_setzero_pd();
        __m256d m_r = _mm256_set1_pd(-HUGE_VAL);
        int j = 0;
        for (; j < nsteps; j += 4) {
            const double *Kb = K + j * dk;
            const double *Vb = V + j * dv;
            flash_atten_4x4core(s_r, p_r, O_r, PK_r, &m_r, &l_r, Qb, Kb, Vb, inv_dk_sqrt, 4, dv, dk);
        }
        if (nrem) {
            const double *Kb = K + j * dk;
            const double *Vb = V + j * dv;
            memset(s_r, 0, 16 * sizeof(double));
            flash_atten_4x4core(s_r, p_r, O_r, PK_r, &m_r, &l_r, Qb, Kb, Vb, inv_dk_sqrt, nrem, dv, dk);
        }
        vdiv_avx2(result + i * dv, O_r, _mm256_set1_pd(l_r[0]), dv);
        vdiv_avx2(result + (i + 1) * dv, O_r + dv, _mm256_set1_pd(l_r[1]), dv);
        vdiv_avx2(result + (i + 2) * dv, O_r + 2 * dv, _mm256_set1_pd(l_r[2]), dv);
        vdiv_avx2(result + (i + 3) * dv, O_r + 3 * dv, _mm256_set1_pd(l_r[3]), dv);
    }
    if (mrem) {
        for (i = msteps; i < m; i++) {
            double l_r = 0;
            double m_r = -HUGE_VAL;
            const double *q_r = Q + i * dk;
            memset(O_r, 0, dv * sizeof(double));
            int j = 0;
            for (; j < nsteps; j += 4) {
                const double *Kb = K + j * dk;
                const double *Vb = V + j * dv;
                flash_atten_1x4core(s_r, p_r, O_r, PK_r, &m_r, &l_r, q_r, Kb, Vb, inv_dk_sqrt, 4, dv, dk);
            }
            if (nrem) {
                const double *Kb = K + j * dk;
                const double *Vb = V + j * dv;
                memset(s_r, 0, 4 * sizeof(double));
                flash_atten_1x4core(s_r, p_r, O_r, PK_r, &m_r, &l_r, q_r, Kb, Vb, inv_dk_sqrt, nrem, dv, dk);
            }
            vdiv_avx2(result + i * dv, O_r, _mm256_set1_pd(l_r), dv);
        }
    }

    free(O_r);
    free(PK_r);
}

// WARN: You are forbidden to modify the codes after the line in your submission.
// Before submitting your code, the output of running the following command
// should be empty: `diff <(tail -n 115 <template code>) <(tail -n 115 <your code>)`

// ----------------------------- You shall not pass! ----------------------------- //

void read_matrix(double **M, size_t len, FILE *file) {
    *M = (double *)malloc(len * sizeof(double));
    if (fread(*M, sizeof(double), len, file) != len) {
        fprintf(stderr, "Invalid testing data.\n");
        exit(1);
    }
}

/*
 * Reads Q, K, and V matrices from the testing data file
 * File format:
 *   1. 4 integers: m, n, dk, dv
 *   2. m*dk doubles -> Q
 *   3. n*dk doubles -> K
 *   4. n*dv doubles -> V
 */
void read_matrices(const char *file_path, double **Q, double **K, double **V, int *m, int *n, int *dk, int *dv) {
    FILE *file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", file_path);
        exit(1);
    }

    if (fread(m, sizeof(int), 1, file) != 1 || fread(n, sizeof(int), 1, file) != 1 ||
        fread(dk, sizeof(int), 1, file) != 1 || fread(dv, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Invalid testing data.\n");
        exit(1);
    }

    read_matrix(Q, (*m) * (*dk), file);
    read_matrix(K, (*n) * (*dk), file);
    read_matrix(V, (*n) * (*dv), file);

    fclose(file);
}

bool verify(const char *file_path, const double *result) {
    FILE *file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open answer file: %s\n", file_path);
        return false;
    }

    int m, n, dk, dv;
    if (fread(&m, sizeof(int), 1, file) != 1 || fread(&n, sizeof(int), 1, file) != 1 ||
        fread(&dk, sizeof(int), 1, file) != 1 || fread(&dv, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Invalid testing data.\n");
        exit(1);
    }

    int offset = sizeof(int) * 4 + sizeof(double) * (m * dk + n * dk + n * dv);
    fseek(file, offset, SEEK_SET);

    bool res = true;
    double threshold = 0.02;
    double *row = (double *)malloc(sizeof(double) * dv);

    for (int i = 0; i < m; i++) {
        int base = i * dv;
        fread(row, sizeof(double), dv, file);
        for (int j = 0; j < dv; j++) {
            if (isnan(result[base + 1]) || fabs(result[base + j] - row[j]) > threshold) {
                printf("Expect result[%d][%d] to be %lf, but it is %lf\n", i, j, row[j], result[base + j]);
                res = false;
                goto end;
            }
        }
    }

end:
    free(row);
    fclose(file);
    return res;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <testing data>\n", argv[0]);
        return 1;
    }

    double *Q = NULL;
    double *K = NULL;
    double *V = NULL;
    double *result = NULL;
    int m, n, dk, dv;

    read_matrices(argv[1], &Q, &K, &V, &m, &n, &dk, &dv);
    result = malloc(sizeof(double) * m * dv);

    struct timespec beg, end;
    clock_gettime(CLOCK_MONOTONIC, &beg);
    attention(Q, K, V, result, m, n, dk, dv);
    clock_gettime(CLOCK_MONOTONIC, &end);

    if (verify(argv[1], result)) {
        double elapsed_time = (end.tv_sec - beg.tv_sec) * 1e6 + (end.tv_nsec - beg.tv_nsec) / 1e3;
        printf("Correct!\nElapsed time: %.2lf us\n", elapsed_time);
    } else {
        puts("Wrong!");
    }

    free(Q);
    free(K);
    free(V);
    free(result);
    return 0;
}

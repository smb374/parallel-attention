//
// Created by poyehchen on 5/19/25.
//
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <smmintrin.h>
#include <stdalign.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <time.h>
#include <unistd.h>

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
    int i = 0;

    for (; i < n / 4; i++) {
        __m256d x = _mm256_loadu_pd(z + i * 4);
        maxv = _mm256_max_pd(maxv, x);
    }
    if (n % 4) {
        __m256i mask = create_mask(n - i * 4);
        __m256d x = _mm256_maskload_pd(z + i * 4, mask);
        __m256d y = _mm256_blendv_pd(neg_inf, x, _mm256_castsi256_pd(mask));
        maxv = _mm256_max_pd(maxv, y);
    }

    return hmax(maxv);
}

static inline __attribute__((always_inline)) __m256d vsum_avx2(const double *z, const int n) {
    const __m256i fmask = _mm256_set1_epi8(-1);
    __m256d sum = _mm256_setzero_pd();

    for (int i = 0; i < n; i += 4) {
        __m256i mask = (i + 4 <= n) ? fmask : create_mask(n - i);
        __m256d x = _mm256_maskload_pd(z + i, mask);
        sum = _mm256_add_pd(sum, x);
    }

    return hsum(sum);
}

static inline __attribute__((always_inline)) void vsub_avx2(double *out, const double *z, const double rhs,
                                                            const int n) {
    const __m256i fmask = _mm256_set1_epi8(-1);
    const __m256d rhv = _mm256_set1_pd(rhs);

    for (int i = 0; i < n; i += 4) {
        __m256i mask = (i + 4 <= n) ? fmask : create_mask(n - i);
        __m256d x = _mm256_maskload_pd(z + i, mask);
        __m256d y = _mm256_sub_pd(x, rhv);
        _mm256_maskstore_pd(out + i, mask, y);
    }
}

static inline __attribute__((always_inline)) void vdiv_avx2(double *out, const double *z, const __m256d basev,
                                                            const int n) {
    const __m256i fmask = _mm256_set1_epi8(-1);

    for (int i = 0; i < n; i += 4) {
        __m256i mask = (i + 4 <= n) ? fmask : create_mask(n - i);
        __m256d x = _mm256_maskload_pd(z + i, mask);
        __m256d y = _mm256_div_pd(x, basev);
        _mm256_maskstore_pd(out + i, mask, y);
    }
}

static inline __attribute__((always_inline)) void vmul_avx2(double *out, const double *z, const __m256d basev,
                                                            const int n) {
    const __m256i fmask = _mm256_set1_epi8(-1);

    for (int i = 0; i < n; i += 4) {
        __m256i mask = (i + 4 <= n) ? fmask : create_mask(n - i);
        __m256d x = _mm256_maskload_pd(z + i, mask);
        __m256d y = _mm256_mul_pd(x, basev);
        _mm256_maskstore_pd(out + i, mask, y);
    }
}

#define EXP_P0 1.0
#define EXP_P1 1.0
#define EXP_P2 0.5
#define EXP_P3 0.16666666666666666
#define EXP_P4 0.041666666666666664
#define EXP_P5 0.008333333333333333
#define EXP_P6 0.001388888888888889

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

    return _mm256_mul_pd(poly, pow2_n);
}

void softmax_avx2(double *z, double *exp_z, const int n) {
    __m256d maxv = vmax_avx2(z, n);

    const __m256i fmask = _mm256_set1_epi8(-1);
    for (int i = 0; i < n; i += 4) {
        __m256i mask = (i + 4 <= n) ? fmask : create_mask(n - i);
        __m256d x = _mm256_maskload_pd(z + i, mask);
        __m256d exp_y = exp256_pd(_mm256_sub_pd(x, maxv));
        _mm256_maskstore_pd(exp_z + i, mask, exp_y);
    }

    __m256d sum = vsum_avx2(exp_z, n);
    vdiv_avx2(z, exp_z, sum, n);
}

double dot_product_avx2(const double *a, const double *b, const int n) {
    double total = 0;
    const __m256i fmask = _mm256_set1_epi8(-1);
    __m256d sum = _mm256_setzero_pd();

    for (int i = 0; i < n; i += 4) {
        __m256i mask = (i + 4 <= n) ? fmask : create_mask(n - i);
        __m256d x = _mm256_maskload_pd(a + i, mask);
        __m256d y = _mm256_maskload_pd(b + i, mask);
        sum = _mm256_fmadd_pd(x, y, sum);
    }

    return reduce_sum(sum);
}

struct attention_arg {
    const double *Q, *K, *V;
    double *result;
    const int rank, size;
    const int m, n, dk, dv;
    const int ms, msize;
};

void *attention_worker(void *arg) {
    const struct attention_arg *args = (struct attention_arg *)arg;

    double *buf = calloc(args->m * args->n + args->n, sizeof(double));
    double *Q_Kt = buf;
    double *exp_z = buf + args->m * args->n;

    // Optimize for smaller inner loop.
    if (args->m < args->n) {
        for (int j = 0; j < args->n; j++) {
            for (int i = args->ms; i < args->ms + args->msize; i++) {
                Q_Kt[i * args->n + j] = dot_product_avx2(args->Q + i * args->dk, args->K + j * args->dk, args->dk);
            }
        }
    } else {
        for (int i = args->ms; i < args->ms + args->msize; i++) {
            for (int j = 0; j < args->n; j++) {
                Q_Kt[i * args->n + j] = dot_product_avx2(args->Q + i * args->dk, args->K + j * args->dk, args->dk);
            }
        }
    }

    const __m256i fmask = _mm256_set1_epi8(-1);
    const __m256d inv_dk_sqrt = _mm256_set1_pd(1 / sqrt(args->dk));
    for (int i = args->ms; i < args->ms + args->msize; i++) {
        vmul_avx2(Q_Kt + i * args->n, Q_Kt + i * args->n, inv_dk_sqrt, args->n);
        softmax_avx2(Q_Kt + i * args->n, exp_z, args->n);
        for (int j = 0; j < args->dv; j += 4) {
            __m256d sumv = _mm256_setzero_pd();
            __m256i mask = (j + 4 <= args->dv) ? fmask : create_mask(args->dv - j);
            // Doing 4 columns at once.
            for (int k = 0; k < args->n; k++) {
                __m256d q_kt_bcast = _mm256_set1_pd(Q_Kt[i * args->n + k]);
                __m256d v_row = _mm256_maskload_pd(args->V + k * args->dv + j, mask);
                sumv = _mm256_fmadd_pd(q_kt_bcast, v_row, sumv);
            }
            _mm256_maskstore_pd(args->result + i * args->dv + j, mask, sumv);
        }
    }

    free(buf);
    return NULL;
}

void attention(const double *Q, const double *K, const double *V, double *result, const int m, const int n,
               const int dk, const int dv) {
    const int size = (int)sysconf(_SC_NPROCESSORS_ONLN);
    pthread_t *threads = calloc(size - 1, sizeof(pthread_t));
    struct attention_arg *args = calloc(size - 1, sizeof(struct attention_arg));

    const int wrow = m / size, rrem = m % size;
    int rs = 0;

    for (int i = 0; i < size - 1; i++) {
        const int rsize = wrow + (i < rrem ? 1 : 0);
        struct attention_arg arg = {
            .rank = i,
            .size = size,
            .Q = Q,
            .K = K,
            .V = V,
            .result = result,
            .m = m,
            .n = n,
            .dk = dk,
            .dv = dv,
            .ms = rs,
            .msize = rsize,
        };
        memcpy(&args[i], &arg, sizeof(struct attention_arg));
        rs += rsize;
        pthread_create(&threads[i], NULL, attention_worker, &args[i]);
    }

    struct attention_arg arg = {
        .rank = size - 1,
        .size = size,
        .Q = Q,
        .K = K,
        .V = V,
        .result = result,
        .m = m,
        .n = n,
        .dk = dk,
        .dv = dv,
        .ms = rs,
        .msize = wrow,
    };

    attention_worker(&arg);

    for (int i = 0; i < size - 1; i++) {
        pthread_join(threads[i], NULL);
    }
    free(threads);
    free(args);
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

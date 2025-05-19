//
// Created by poyehchen on 5/19/25.
//
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <sys/param.h>

// NOTE: feel free to include any header you need, but we will not
// link libraries other than C's math library for you.

// NOTE: feel free to add new macros

// NOTE: feel free to add new functions

/*
 * Q: m by dk
 * K: n by dk
 * V: n by dv
 * result: m by dv, containing the attention result
 */
static double reduce_sum(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    const __m128d vhigh = _mm256_extractf128_pd(v, 1);
    vlow = _mm_add_pd(vlow, vhigh);

    const __m128d h64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_add_sd(vlow, h64));
}

static double reduce_max(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    const __m128d vhigh = _mm256_extractf128_pd(v, 1);
    vlow = _mm_max_pd(vlow, vhigh);

    const __m128d h64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_max_sd(vlow, h64));
}

static double vmax_avx2(const double *z, const int n) {
    const int steps = n / 4;

    __m256d maxv = _mm256_loadu_pd(z);

    for (int i = 1; i < steps; i++) {
        maxv = _mm256_max_pd(maxv, _mm256_loadu_pd(z + i * 4));
    }

    double rmax = reduce_max(maxv);
    for (int i = steps * 4; i < n; i++) {
        rmax = MAX(rmax, z[i]);
    }

    return rmax;
}

static double vsum_avx2(const double *z, const int n) {
    const int steps = n / 4;

    __m256d sum = _mm256_loadu_pd(z);

    for (int i = 1; i < steps; i++) {
        sum = _mm256_add_pd(sum, _mm256_loadu_pd(z + i * 4));
    }

    double remain = 0;
    for (int i = steps * 4; i < n; i++) {
        remain += z[i];
    }

    return reduce_sum(sum) + remain;
}

static void vsub_avx2(double *out, const double *z, const double rhs, const int n) {
    const int steps = n / 4;

    const __m256d rhv = _mm256_set1_pd(rhs);
    for (int i = 0; i < steps; i++) {
        const __m256d x = _mm256_loadu_pd(z + i * 4);
        const __m256d y = _mm256_sub_pd(x, rhv);
        _mm256_storeu_pd(out + i * 4, y);
    }

    for (int i = steps * 4; i < n; i++) {
        out[i] = z[i] - rhs;
    }
}

static void vdiv_avx2(double *out, const double *z, const double base, const int n) {
    const int steps = n / 4;

    const __m256d basev = _mm256_set1_pd(base);
    for (int i = 0; i < steps; i++) {
        const __m256d x = _mm256_loadu_pd(z + i * 4);
        const __m256d y = _mm256_div_pd(x, basev);
        _mm256_storeu_pd(out + i * 4, y);
    }

    for (int i = steps * 4; i < n; i++) {
        out[i] = z[i] / base;
    }
}

void softmax_avx2(double *z, double *exp_z, const int len) {
    const double max = vmax_avx2(z, len);
    vsub_avx2(exp_z, z, max, len);
    for (int i = 0; i < len; i++) {
        exp_z[i] = exp(exp_z[i]);
    }
    const double sum = vsum_avx2(exp_z, len);
    vdiv_avx2(z, exp_z, sum, len);
}

double dot_product_avx2(const double *a, const double *b, const int n) {
    double remain = 0.0;
    const int steps = n / 4;
    __m256d sum = _mm256_setzero_pd();

    for (int i = 0; i < steps; i++) {
        const __m256d x = _mm256_loadu_pd(a + i * 4);
        const __m256d y = _mm256_loadu_pd(b + i * 4);
        sum = _mm256_add_pd(sum, _mm256_mul_pd(x, y));
    }

    for (int i = steps * 4; i < n; i++) {
        remain += a[i] * b[i];
    }


    return reduce_sum(sum) + remain;
}

double* Q_Kt = NULL;

void attention(double* Q, double* K, double* V, double* result,
               int m, int n, int dk, int dv) {
    // TODO: your serial attention implementation

    Q_Kt = calloc(m * n, sizeof(double));
    const double dk_sqrt = sqrt(dk);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            Q_Kt[i * n + j] = dot_product_avx2(Q + i * dk, K + j * dk, dk) / dk_sqrt;
        }
    }

    double *buffer = calloc(n, sizeof(double));

    for (int i = 0; i < m; i++) {
        softmax_avx2(Q_Kt + i * n, buffer, n);
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < dv; j++) {
            __m256d sumv = _mm256_setzero_pd();
            const int steps = n / 4;
            for (int k = 0; k < steps; k++) {
                const int k2 = k * 4;
                const __m256d x = _mm256_set_pd(Q_Kt[i * n + k2], Q_Kt[i * n + k2 + 1], Q_Kt[i * n + k2 + 2],
                                                Q_Kt[i * n + k2 + 3]);
                const __m256d y = _mm256_set_pd(V[k2 * dv + j], V[(k2 + 1) * dv + j], V[(k2 + 2) * dv + j],
                                                V[(k2 + 3) * dv + j]);
                sumv = _mm256_add_pd(sumv, _mm256_mul_pd(x, y));
            }
            double sum = reduce_sum(sumv);
            for (int k = steps * 4; k < n; k++) {
                sum += Q_Kt[i * n + k] * V[k * dv + j];
            }
            result[i * dv + j] = sum;
        }
    }

    free(Q_Kt);
    free(buffer);
}

// WARN: You are forbidden to modify the codes after the line in your submission.
// Before submitting your code, the output of running the following command
// should be empty: `diff <(tail -n 115 <template code>) <(tail -n 115 <your code>)`

// ----------------------------- You shall not pass! ----------------------------- //

void read_matrix(double** M, size_t len, FILE* file) {
    *M = (double*) malloc(len * sizeof(double));
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
void read_matrices(const char* file_path, double** Q, double** K, double** V,
                  int *m, int *n, int *dk, int *dv) {
    FILE* file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", file_path);
        exit(1);
    }

    if (fread(m, sizeof(int), 1, file) != 1 ||
        fread(n, sizeof(int), 1, file) != 1 ||
        fread(dk, sizeof(int), 1, file) != 1 ||
        fread(dv, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Invalid testing data.\n");
        exit(1);
    }

    read_matrix(Q, (*m) * (*dk), file);
    read_matrix(K, (*n) * (*dk), file);
    read_matrix(V, (*n) * (*dv), file);

    fclose(file);
}

bool verify(const char* file_path, const double* result) {
    FILE* file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open answer file: %s\n", file_path);
        return false;
    }

    int m, n, dk, dv;
    if (fread(&m, sizeof(int), 1, file) != 1 ||
        fread(&n, sizeof(int), 1, file) != 1 ||
        fread(&dk, sizeof(int), 1, file) != 1 ||
        fread(&dv, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Invalid testing data.\n");
        exit(1);
    }

    int offset = sizeof(int) * 4 + sizeof(double) * (m * dk + n * dk + n * dv);
    fseek(file, offset, SEEK_SET);

    bool res = true;
    double threshold = 0.02;
    double* row = (double*) malloc(sizeof(double) * dv);

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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <testing data>\n", argv[0]);
        return 1;
    }

    double* Q = NULL;
    double* K = NULL;
    double* V = NULL;
    double* result = NULL;
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

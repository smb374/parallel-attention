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
#include <unistd.h>
#include <pthread.h>
#include <string.h>

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
    double rmax = -HUGE_VAL;

    if (steps) {
        __m256d maxv = _mm256_loadu_pd(z);
        for (int i = 1; i < steps; i++) {
            maxv = _mm256_max_pd(maxv, _mm256_loadu_pd(z + i * 4));
        }
        rmax = reduce_max(maxv);
    }

    for (int i = steps * 4; i < n; i++) {
        rmax = MAX(rmax, z[i]);
    }

    return rmax;
}

static double vsum_avx2(const double *z, const int n) {
    const int steps = n / 4;
    double total = 0.0;

    if (steps) {
        __m256d sum = _mm256_loadu_pd(z);
        for (int i = 1; i < steps; i++) {
            sum = _mm256_add_pd(sum, _mm256_loadu_pd(z + i * 4));
        }
        total += reduce_sum(sum);
    }

    for (int i = steps * 4; i < n; i++) {
        total += z[i];
    }

    return total;
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
        sum = _mm256_fmadd_pd(x, y, sum);
    }

    for (int i = steps * 4; i < n; i++) {
        remain += a[i] * b[i];
    }


    return reduce_sum(sum) + remain;
}

struct attention_arg {
    const double *Q, *K, *V;
    double *result, *Q_Kt;
    const int rank, size;
    const int m, n, dk, dv;
    const int ms, msize;
};

void *attention_worker(void *arg) {
    const struct attention_arg *args = (struct attention_arg*)arg;

    const double dk_sqrt = sqrt(args->dk);

    for (int i = args->ms; i < args->ms + args->msize; i++) {
        for (int j = 0; j < args->n; j++) {
            args->Q_Kt[i * args->n + j] = dot_product_avx2(args->Q + i * args->dk, args->K + j * args->dk, args->dk) / dk_sqrt;
        }
    }

    double *buffer = calloc(args->n, sizeof(double));

    for (int i = args->ms; i < args->ms + args->msize; i++) {
        softmax_avx2(args->Q_Kt + i * args->n, buffer, args->n);
    }

    for (int i = args->ms; i < args->ms + args->msize; i++) {
        for (int j = 0; j < args->dv; j += 4) {
            __m256d sumv = _mm256_setzero_pd();
            if (j + 3 < args->dv) {
                for (int k = 0; k < args->n; k++) {
                    __m256d q_kt_bcast = _mm256_set1_pd(args->Q_Kt[i * args->n + k]);
                    __m256d v_row = _mm256_loadu_pd(args->V + k * args->dv + j);
                    sumv = _mm256_fmadd_pd(q_kt_bcast, v_row, sumv);
                }

                _mm256_storeu_pd(args->result + i * args->dv + j, sumv);
            } else {
                for (int jj = j; jj < args->dv; jj ++) {
                    double sum = 0.0;
                    for (int k = 0; k < args->n; k++) {
                        sum += args->Q_Kt[i * args->n + k] * args->V[k * args->dv + jj];
                    }
                    args->result[i * args->dv + jj] = sum;
                }
            }
        }
    }

    free(buffer);
    return NULL;
}

void attention(const double *Q, const double *K, const double *V, double *result,
               const int m, const int n, const int dk, const int dv) {
    double *Q_Kt = calloc(m * n, sizeof(double));
    const int size = sysconf(_SC_NPROCESSORS_ONLN);
    pthread_t *threads = calloc(size - 1, sizeof(pthread_t));
    struct attention_arg* args = calloc(size - 1, sizeof(struct attention_arg));

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
            .Q_Kt = Q_Kt,
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
        .Q_Kt = Q_Kt,
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
    free(Q_Kt);
}

// WARN: You are forbidden to modify the codes after the line in your submission.
// Before submitting your code, the output of running the following command
// should be empty: `diff <(tail -n 115 <template code>) <(tail -n 115 <your code>)`

// ----------------------------- You shall not pass! ----------------------------- //

void read_matrix(double **M, size_t len, FILE *file) {
    *M = (double *) malloc(len * sizeof(double));
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
void read_matrices(const char *file_path, double **Q, double **K, double **V,
                   int *m, int *n, int *dk, int *dv) {
    FILE *file = fopen(file_path, "rb");
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

bool verify(const char *file_path, const double *result) {
    FILE *file = fopen(file_path, "rb");
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
    double *row = (double *) malloc(sizeof(double) * dv);

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

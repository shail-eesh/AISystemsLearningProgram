/* T16A · matmul on a CPU, five versions of the same 2*N^3 flops.
 *
 * Every kernel here computes C = A*B for row-major, contiguous, float64
 * matrices. They differ only in the *order they touch memory*, and that is the
 * entire lesson: the arithmetic is identical, the runtime is not.
 *
 * Build (done for you by native.py):
 *   cc -O3 -march=native -fopenmp -shared -fPIC kernels.c -o libmatmul.so
 *
 * Note on fairness: all five are compiled with the same flags. It would be
 * easy to make the "naive" version look worse by handicapping it with -O0;
 * the point is that the *algorithm* is what moves the number, so the compiler
 * gets to do its best everywhere.
 */

#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ---------------------------------------------------------------- 1. naive
 * The textbook triple loop, i-j-k. The inner loop walks B down a column:
 * B[k*n + j] with k varying, so each access is n*8 bytes from the last. Every
 * one of them is a fresh cache line of which we use 8 bytes and discard 56.
 */
void matmul_naive_ijk(const double *A, const double *B, double *C,
                      int m, int n, int k_dim) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int k = 0; k < k_dim; k++) {
                acc += A[i * k_dim + k] * B[k * n + j];
            }
            C[i * n + j] = acc;
        }
    }
}

/* ------------------------------------------------------------ 2. loop order
 * Swap the two inner loops. Now the innermost loop varies j, so both B[k*n+j]
 * and C[i*n+j] walk *along* rows — contiguous, prefetchable, vectorisable.
 * The accumulator moves out of a register and into C, which is the price.
 * Same flops, same result, one changed line.
 */
void matmul_ikj(const double *restrict A, const double *restrict B,
                double *restrict C, int m, int n, int k_dim) {
    memset(C, 0, (size_t)m * n * sizeof(double));
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < k_dim; k++) {
            const double a = A[i * k_dim + k];
            const double *brow = &B[(size_t)k * n];
            double *crow = &C[(size_t)i * n];
            for (int j = 0; j < n; j++) {
                crow[j] += a * brow[j];
            }
        }
    }
}

/* --------------------------------------------------------------- 3. blocked
 * ikj fixes the *stride*, but not the *volume*: for each row of A we stream
 * all of B past the cache. At n=1024 that is 8 MB per row, ~8 GB in total.
 *
 * Blocking fixes the volume. Work on a KC x NC tile of B and reuse it for MC
 * rows of A while it is resident. The defaults below aim the B tile at L2
 * (128 x 256 x 8 B = 256 KB) rather than at any specific chip; the benchmark
 * sweeps them so the choice is measured, not asserted.
 */
void matmul_blocked(const double *restrict A, const double *restrict B,
                    double *restrict C, int m, int n, int k_dim,
                    int mc, int kc, int nc) {
    memset(C, 0, (size_t)m * n * sizeof(double));
    for (int kk = 0; kk < k_dim; kk += kc) {
        const int kmax = (kk + kc < k_dim) ? kk + kc : k_dim;
        for (int jj = 0; jj < n; jj += nc) {
            const int jmax = (jj + nc < n) ? jj + nc : n;
            for (int ii = 0; ii < m; ii += mc) {
                const int imax = (ii + mc < m) ? ii + mc : m;
                for (int i = ii; i < imax; i++) {
                    double *restrict crow = &C[(size_t)i * n];
                    for (int k = kk; k < kmax; k++) {
                        const double a = A[(size_t)i * k_dim + k];
                        const double *restrict brow = &B[(size_t)k * n];
                        for (int j = jj; j < jmax; j++) {
                            crow[j] += a * brow[j];
                        }
                    }
                }
            }
        }
    }
}

/* ---------------------------------------------------- 4. blocked + threads
 * The j-tiles are independent: each writes a disjoint column band of C, so
 * there is no sharing and no false sharing at tile granularity. Parallelising
 * over j (rather than over k, which would need a reduction) is the version
 * that needs no locks at all.
 */
void matmul_blocked_omp(const double *restrict A, const double *restrict B,
                        double *restrict C, int m, int n, int k_dim,
                        int mc, int kc, int nc) {
    memset(C, 0, (size_t)m * n * sizeof(double));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int jj = 0; jj < n; jj += nc) {
        const int jmax = (jj + nc < n) ? jj + nc : n;
        for (int kk = 0; kk < k_dim; kk += kc) {
            const int kmax = (kk + kc < k_dim) ? kk + kc : k_dim;
            for (int ii = 0; ii < m; ii += mc) {
                const int imax = (ii + mc < m) ? ii + mc : m;
                for (int i = ii; i < imax; i++) {
                    double *restrict crow = &C[(size_t)i * n];
                    for (int k = kk; k < kmax; k++) {
                        const double a = A[(size_t)i * k_dim + k];
                        const double *restrict brow = &B[(size_t)k * n];
                        for (int j = jj; j < jmax; j++) {
                            crow[j] += a * brow[j];
                        }
                    }
                }
            }
        }
    }
}

/* ------------------------------------------- 5. blocked + 4-row register tile
 * The remaining inefficiency in (3) is that each element of B is loaded once
 * per row of A. Process four rows of A at a time and one load of brow[j]
 * serves four fused multiply-adds. This is the germ of a real micro-kernel:
 * BLIS and OpenBLAS do the same thing with hand-written assembly and much
 * larger register tiles.
 */
void matmul_blocked_regtile(const double *restrict A, const double *restrict B,
                            double *restrict C, int m, int n, int k_dim,
                            int mc, int kc, int nc) {
    memset(C, 0, (size_t)m * n * sizeof(double));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int jj = 0; jj < n; jj += nc) {
        const int jmax = (jj + nc < n) ? jj + nc : n;
        for (int kk = 0; kk < k_dim; kk += kc) {
            const int kmax = (kk + kc < k_dim) ? kk + kc : k_dim;
            for (int ii = 0; ii < m; ii += mc) {
                const int imax = (ii + mc < m) ? ii + mc : m;
                int i = ii;
                for (; i + 3 < imax; i += 4) {
                    double *restrict c0 = &C[(size_t)(i + 0) * n];
                    double *restrict c1 = &C[(size_t)(i + 1) * n];
                    double *restrict c2 = &C[(size_t)(i + 2) * n];
                    double *restrict c3 = &C[(size_t)(i + 3) * n];
                    for (int k = kk; k < kmax; k++) {
                        const double a0 = A[(size_t)(i + 0) * k_dim + k];
                        const double a1 = A[(size_t)(i + 1) * k_dim + k];
                        const double a2 = A[(size_t)(i + 2) * k_dim + k];
                        const double a3 = A[(size_t)(i + 3) * k_dim + k];
                        const double *restrict brow = &B[(size_t)k * n];
                        for (int j = jj; j < jmax; j++) {
                            const double b = brow[j];
                            c0[j] += a0 * b;
                            c1[j] += a1 * b;
                            c2[j] += a2 * b;
                            c3[j] += a3 * b;
                        }
                    }
                }
                for (; i < imax; i++) {   /* the tail, when m % 4 != 0 */
                    double *restrict crow = &C[(size_t)i * n];
                    for (int k = kk; k < kmax; k++) {
                        const double a = A[(size_t)i * k_dim + k];
                        const double *restrict brow = &B[(size_t)k * n];
                        for (int j = jj; j < jmax; j++) {
                            crow[j] += a * brow[j];
                        }
                    }
                }
            }
        }
    }
}

int matmul_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

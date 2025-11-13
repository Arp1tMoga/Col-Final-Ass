
// matmul_branch_ilp.cpp
#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include <cassert>

using namespace std;

// === Tunables (keep them configurable later if you want) ===
#ifndef BLOCK_I
#define BLOCK_I 64
#endif
#ifndef BLOCK_J
#define BLOCK_J 256
#endif
#ifndef BLOCK_K
#define BLOCK_K 64
#endif
#ifndef UNROLL
#define UNROLL 8
#endif

// Helper branch prediction hints
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

// Ensure alignment for AVX loads/stores
constexpr size_t ALIGN = 32;

// Micro-kernel: full BLOCK_K (no k_block checks inside) - assumes k_block == BLOCK_K
static inline void micro_kernel_8x4_fullk(const double* __restrict__ A, const double* __restrict__ B,
                                          double* __restrict__ C, int N, int ldc) {
    // Load C into registers (aligned or unaligned depending on pointer)
    __m256d c0 = _mm256_loadu_pd(&C[0*ldc]);
    __m256d c1 = _mm256_loadu_pd(&C[1*ldc]);
    __m256d c2 = _mm256_loadu_pd(&C[2*ldc]);
    __m256d c3 = _mm256_loadu_pd(&C[3*ldc]);
    __m256d c4 = _mm256_loadu_pd(&C[4*ldc]);
    __m256d c5 = _mm256_loadu_pd(&C[5*ldc]);
    __m256d c6 = _mm256_loadu_pd(&C[6*ldc]);
    __m256d c7 = _mm256_loadu_pd(&C[7*ldc]);

    // Main k loop - no branching on k_block here
    for (int k = 0; k < BLOCK_K; ++k) {
        // Prefetch A and B a bit ahead (bounded prefetch)
        int pf = k + 8;
        if (pf < BLOCK_K) {
            __builtin_prefetch(&A[pf * N], 0, 3);
            __builtin_prefetch(&B[pf * N], 0, 3);
        }

        // load B's 4 doubles (j..j+3) for this k
        __m256d b = _mm256_loadu_pd(&B[k * N]); // B pointer assumed to point at B[0] of this micro-block

        // Broadcast A scaled elements for 8 rows (A row offsets are 0..7)
        // Note: using _mm256_set1_pd is fine; alternative: load into scalar then broadcast
        __m256d a0 = _mm256_set1_pd(A[0 * N + k]);
        __m256d a1 = _mm256_set1_pd(A[1 * N + k]);
        __m256d a2 = _mm256_set1_pd(A[2 * N + k]);
        __m256d a3 = _mm256_set1_pd(A[3 * N + k]);
        __m256d a4 = _mm256_set1_pd(A[4 * N + k]);
        __m256d a5 = _mm256_set1_pd(A[5 * N + k]);
        __m256d a6 = _mm256_set1_pd(A[6 * N + k]);
        __m256d a7 = _mm256_set1_pd(A[7 * N + k]);

        // FMA accumulate (many independent ops -> good ILP)
        c0 = _mm256_fmadd_pd(a0, b, c0);
        c1 = _mm256_fmadd_pd(a1, b, c1);
        c2 = _mm256_fmadd_pd(a2, b, c2);
        c3 = _mm256_fmadd_pd(a3, b, c3);
        c4 = _mm256_fmadd_pd(a4, b, c4);
        c5 = _mm256_fmadd_pd(a5, b, c5);
        c6 = _mm256_fmadd_pd(a6, b, c6);
        c7 = _mm256_fmadd_pd(a7, b, c7);
    }

    // Store results back
    _mm256_storeu_pd(&C[0*ldc], c0);
    _mm256_storeu_pd(&C[1*ldc], c1);
    _mm256_storeu_pd(&C[2*ldc], c2);
    _mm256_storeu_pd(&C[3*ldc], c3);
    _mm256_storeu_pd(&C[4*ldc], c4);
    _mm256_storeu_pd(&C[5*ldc], c5);
    _mm256_storeu_pd(&C[6*ldc], c6);
    _mm256_storeu_pd(&C[7*ldc], c7);
}

// Micro-kernel: tail version handles k_block < BLOCK_K safely
static inline void micro_kernel_8x4_tail(const double* __restrict__ A, const double* __restrict__ B,
                                         double* __restrict__ C, int N, int ldc, int k_block) {
    __m256d c0 = _mm256_loadu_pd(&C[0*ldc]);
    __m256d c1 = _mm256_loadu_pd(&C[1*ldc]);
    __m256d c2 = _mm256_loadu_pd(&C[2*ldc]);
    __m256d c3 = _mm256_loadu_pd(&C[3*ldc]);
    __m256d c4 = _mm256_loadu_pd(&C[4*ldc]);
    __m256d c5 = _mm256_loadu_pd(&C[5*ldc]);
    __m256d c6 = _mm256_loadu_pd(&C[6*ldc]);
    __m256d c7 = _mm256_loadu_pd(&C[7*ldc]);

    for (int k = 0; k < k_block; ++k) {
        int pf = k + 8;
        if (pf < k_block) {
            __builtin_prefetch(&A[pf * N], 0, 3);
            __builtin_prefetch(&B[pf * N], 0, 3);
        }
        __m256d b = _mm256_loadu_pd(&B[k * N]);
        __m256d a0 = _mm256_set1_pd(A[0 * N + k]);
        __m256d a1 = _mm256_set1_pd(A[1 * N + k]);
        __m256d a2 = _mm256_set1_pd(A[2 * N + k]);
        __m256d a3 = _mm256_set1_pd(A[3 * N + k]);
        __m256d a4 = _mm256_set1_pd(A[4 * N + k]);
        __m256d a5 = _mm256_set1_pd(A[5 * N + k]);
        __m256d a6 = _mm256_set1_pd(A[6 * N + k]);
        __m256d a7 = _mm256_set1_pd(A[7 * N + k]);

        c0 = _mm256_fmadd_pd(a0, b, c0);
        c1 = _mm256_fmadd_pd(a1, b, c1);
        c2 = _mm256_fmadd_pd(a2, b, c2);
        c3 = _mm256_fmadd_pd(a3, b, c3);
        c4 = _mm256_fmadd_pd(a4, b, c4);
        c5 = _mm256_fmadd_pd(a5, b, c5);
        c6 = _mm256_fmadd_pd(a6, b, c6);
        c7 = _mm256_fmadd_pd(a7, b, c7);
    }

    _mm256_storeu_pd(&C[0*ldc], c0);
    _mm256_storeu_pd(&C[1*ldc], c1);
    _mm256_storeu_pd(&C[2*ldc], c2);
    _mm256_storeu_pd(&C[3*ldc], c3);
    _mm256_storeu_pd(&C[4*ldc], c4);
    _mm256_storeu_pd(&C[5*ldc], c5);
    _mm256_storeu_pd(&C[6*ldc], c6);
    _mm256_storeu_pd(&C[7*ldc], c7);
}

// Main optimized matmul with branch-minimized inner path & ILP hints
static void matmul_ultra_branch_ilp(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int N) {
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < N; ii += BLOCK_I) {
        int i_block = std::min(BLOCK_I, N - ii);

        for (int jj = 0; jj < N; jj += BLOCK_J) {
            int j_block = std::min(BLOCK_J, N - jj);

            for (int kk = 0; kk < N; kk += BLOCK_K) {
                int k_block = std::min(BLOCK_K, N - kk);

                int i = ii;
                for (; i <= ii + i_block - 8; i += 8) {
                    int j = jj;
                    for (; j <= jj + j_block - 4; j += 4) {
                        // Pointers into sub-blocks
                        const double* Ablk = &A[i * N + kk];
                        const double* Bblk = &B[kk * N + j];
                        double* Cblk = &C[i * N + j];

                        // Branch only once per micro-block whether we have full k-block
                        if (likely(k_block == BLOCK_K)) {
                            micro_kernel_8x4_fullk(Ablk, Bblk, Cblk, N, N);
                        } else {
                            micro_kernel_8x4_tail(Ablk, Bblk, Cblk, N, N, k_block);
                        }
                    }

                    // Remaining j columns (scalar but vectorized)
                    for (; j < jj + j_block; ++j) {
                        // Try to make inner loop vectorizable / high-ILP
                        // accumulate 4 independent accumulators (unrolled)
                        double r0 = 0.0, r1 = 0.0, r2 = 0.0, r3 = 0.0;
                        int k = kk;
                        // Unroll the k loop to expose ILP
                        for (; k <= kk + k_block - 4; k += 4) {
                            r0 += A[(i+0)*N + k] * B[k*N + j];
                            r1 += A[(i+1)*N + k] * B[k*N + j];
                            r2 += A[(i+2)*N + k] * B[k*N + j];
                            r3 += A[(i+3)*N + k] * B[k*N + j];

                            r0 += A[(i+0)*N + k+1] * B[(k+1)*N + j];
                            r1 += A[(i+1)*N + k+1] * B[(k+1)*N + j];
                            r2 += A[(i+2)*N + k+1] * B[(k+1)*N + j];
                            r3 += A[(i+3)*N + k+1] * B[(k+1)*N + j];

                            r0 += A[(i+0)*N + k+2] * B[(k+2)*N + j];
                            r1 += A[(i+1)*N + k+2] * B[(k+2)*N + j];
                            r2 += A[(i+2)*N + k+2] * B[(k+2)*N + j];
                            r3 += A[(i+3)*N + k+2] * B[(k+2)*N + j];

                            r0 += A[(i+0)*N + k+3] * B[(k+3)*N + j];
                            r1 += A[(i+1)*N + k+3] * B[(k+3)*N + j];
                            r2 += A[(i+2)*N + k+3] * B[(k+3)*N + j];
                            r3 += A[(i+3)*N + k+3] * B[(k+3)*N + j];
                        }
                        for (; k < kk + k_block; ++k) {
                            r0 += A[(i+0)*N + k] * B[k*N + j];
                            r1 += A[(i+1)*N + k] * B[k*N + j];
                            r2 += A[(i+2)*N + k] * B[k*N + j];
                            r3 += A[(i+3)*N + k] * B[k*N + j];
                        }
                        // write back
                        C[(i+0)*N + j] += r0;
                        C[(i+1)*N + j] += r1;
                        C[(i+2)*N + j] += r2;
                        C[(i+3)*N + j] += r3;

                        // next group of 4 rows (i+4..i+7)
                        double s0=0.0,s1=0.0,s2=0.0,s3=0.0;
                        k = kk;
                        for (; k <= kk + k_block - 4; k += 4) {
                            s0 += A[(i+4)*N + k] * B[k*N + j];
                            s1 += A[(i+5)*N + k] * B[k*N + j];
                            s2 += A[(i+6)*N + k] * B[k*N + j];
                            s3 += A[(i+7)*N + k] * B[k*N + j];

                            s0 += A[(i+4)*N + k+1] * B[(k+1)*N + j];
                            s1 += A[(i+5)*N + k+1] * B[(k+1)*N + j];
                            s2 += A[(i+6)*N + k+1] * B[(k+1)*N + j];
                            s3 += A[(i+7)*N + k+1] * B[(k+1)*N + j];

                            s0 += A[(i+4)*N + k+2] * B[(k+2)*N + j];
                            s1 += A[(i+5)*N + k+2] * B[(k+2)*N + j];
                            s2 += A[(i+6)*N + k+2] * B[(k+2)*N + j];
                            s3 += A[(i+7)*N + k+2] * B[(k+2)*N + j];

                            s0 += A[(i+4)*N + k+3] * B[(k+3)*N + j];
                            s1 += A[(i+5)*N + k+3] * B[(k+3)*N + j];
                            s2 += A[(i+6)*N + k+3] * B[(k+3)*N + j];
                            s3 += A[(i+7)*N + k+3] * B[(k+3)*N + j];
                        }
                        for (; k < kk + k_block; ++k) {
                            s0 += A[(i+4)*N + k] * B[k*N + j];
                            s1 += A[(i+5)*N + k] * B[k*N + j];
                            s2 += A[(i+6)*N + k] * B[k*N + j];
                            s3 += A[(i+7)*N + k] * B[k*N + j];
                        }
                        C[(i+4)*N + j] += s0;
                        C[(i+5)*N + j] += s1;
                        C[(i+6)*N + j] += s2;
                        C[(i+7)*N + j] += s3;
                    }
                }

                // Handle remaining i rows (scalar but simd-hinted)
                for (; i < ii + i_block; ++i) {
                    for (int k = kk; k < kk + k_block; ++k) {
                        double a = A[i * N + k];
                        // hint vectorizer
                        #pragma omp simd
                        for (int j = jj; j < jj + j_block; ++j) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " N num_threads\n";
        return 1;
    }
    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    omp_set_num_threads(T);

    // aligned allocations for better vector loads/stores
    double *A = nullptr, *B = nullptr, *C = nullptr;
    if (posix_memalign((void**)&A, ALIGN, sizeof(double) * (size_t)N * N) != 0) { cerr<<"alloc fail\n"; return 1; }
    if (posix_memalign((void**)&B, ALIGN, sizeof(double) * (size_t)N * N) != 0) { cerr<<"alloc fail\n"; return 1; }
    if (posix_memalign((void**)&C, ALIGN, sizeof(double) * (size_t)N * N) != 0) { cerr<<"alloc fail\n"; return 1; }

    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < (size_t)N * N; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
        C[i] = 0.0;
    }

    double t0 = omp_get_wtime();
    matmul_ultra_branch_ilp(A, B, C, N);
    double t1 = omp_get_wtime();

    cout << "N="<<N<<" T="<<T<<" time="<<(t1-t0)<<" seconds\n";

    double s = 0.0;
    #pragma omp parallel for reduction(+:s)
    for (size_t i = 0; i < (size_t)N * N; ++i) s += C[i];
    cout << "checksum=" << s << "\n";

    free(A); free(B); free(C);
    return 0;
}
#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <immintrin.h>  // AVX2 intrinsics
#include <cstdlib>

using namespace std;

// ============================================================================
// ULTRA-OPTIMIZED MATRIX MULTIPLICATION
// Combines: Blocking + SIMD + OpenMP + Loop Unrolling + Prefetching
// ============================================================================

#define BLOCK_I 64   // L2 cache blocking
#define BLOCK_J 256  // Larger j-block
#define BLOCK_K 64   // K-block
#define UNROLL 8     // Unroll factor

// Micro-kernel: Highly optimized 8x4 block computation
static inline void micro_kernel_8x4(const double* __restrict__ A, const double* __restrict__ B, 
                                     double* __restrict__ C, int N, int ldc) {
    // Load C into registers
    __m256d c0 = _mm256_loadu_pd(&C[0*ldc]);
    __m256d c1 = _mm256_loadu_pd(&C[1*ldc]);
    __m256d c2 = _mm256_loadu_pd(&C[2*ldc]);
    __m256d c3 = _mm256_loadu_pd(&C[3*ldc]);
    __m256d c4 = _mm256_loadu_pd(&C[4*ldc]);
    __m256d c5 = _mm256_loadu_pd(&C[5*ldc]);
    __m256d c6 = _mm256_loadu_pd(&C[6*ldc]);
    __m256d c7 = _mm256_loadu_pd(&C[7*ldc]);
    
    for (int k = 0; k < BLOCK_K; ++k) {
        // Prefetch future data
        __builtin_prefetch(&A[(k+8)*N], 0, 3);
        __builtin_prefetch(&B[(k+8)*N], 0, 3);
        
        // Load B once (broadcast will happen)
        __m256d b = _mm256_loadu_pd(&B[k*N]);
        
        // Broadcast A and compute FMA
        c0 = _mm256_fmadd_pd(_mm256_set1_pd(A[0*N + k]), b, c0);
        c1 = _mm256_fmadd_pd(_mm256_set1_pd(A[1*N + k]), b, c1);
        c2 = _mm256_fmadd_pd(_mm256_set1_pd(A[2*N + k]), b, c2);
        c3 = _mm256_fmadd_pd(_mm256_set1_pd(A[3*N + k]), b, c3);
        c4 = _mm256_fmadd_pd(_mm256_set1_pd(A[4*N + k]), b, c4);
        c5 = _mm256_fmadd_pd(_mm256_set1_pd(A[5*N + k]), b, c5);
        c6 = _mm256_fmadd_pd(_mm256_set1_pd(A[6*N + k]), b, c6);
        c7 = _mm256_fmadd_pd(_mm256_set1_pd(A[7*N + k]), b, c7);
    }
    
    // Store back
    _mm256_storeu_pd(&C[0*ldc], c0);
    _mm256_storeu_pd(&C[1*ldc], c1);
    _mm256_storeu_pd(&C[2*ldc], c2);
    _mm256_storeu_pd(&C[3*ldc], c3);
    _mm256_storeu_pd(&C[4*ldc], c4);
    _mm256_storeu_pd(&C[5*ldc], c5);
    _mm256_storeu_pd(&C[6*ldc], c6);
    _mm256_storeu_pd(&C[7*ldc], c7);
}

// Main optimized GEMM function
static void matmul_ultra_optimized(const double* A, const double* B, double* C, int N) {
    #pragma omp parallel for schedule(static) collapse(1)
    for (int ii = 0; ii < N; ii += BLOCK_I) {
        int i_block = min(BLOCK_I, N - ii);
        
        for (int jj = 0; jj < N; jj += BLOCK_J) {
            int j_block = min(BLOCK_J, N - jj);
            
            for (int kk = 0; kk < N; kk += BLOCK_K) {
                int k_block = min(BLOCK_K, N - kk);
                
                // Process 8x4 blocks with micro-kernel
                int i = ii;
                for (; i <= ii + i_block - 8; i += 8) {
                    int j = jj;
                    for (; j <= jj + j_block - 4; j += 4) {
                        micro_kernel_8x4(&A[i*N + kk], &B[kk*N + j], 
                                        &C[i*N + j], N, N);
                    }
                    
                    // Handle remaining j
                    for (; j < jj + j_block; ++j) {
                        for (int k = kk; k < kk + k_block; ++k) {
                            for (int i2 = i; i2 < i + 8; ++i2) {
                                C[i2*N + j] += A[i2*N + k] * B[k*N + j];
                            }
                        }
                    }
                }
                
                // Handle remaining i rows
                for (; i < ii + i_block; ++i) {
                    for (int k = kk; k < kk + k_block; ++k) {
                        double a = A[i*N + k];
                        __m256d a_vec = _mm256_set1_pd(a);
                        
                        int j = jj;
                        for (; j <= jj + j_block - 4; j += 4) {
                            __m256d b_vec = _mm256_loadu_pd(&B[k*N + j]);
                            __m256d c_vec = _mm256_loadu_pd(&C[i*N + j]);
                            c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                            _mm256_storeu_pd(&C[i*N + j], c_vec);
                        }
                        
                        for (; j < jj + j_block; ++j) {
                            C[i*N + j] += a * B[k*N + j];
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

    vector<double> A(N*N), B(N*N), C(N*N);
    
    // reproducible RNG
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i=0;i<N*N;i++) { 
        A[i] = dist(rng); 
        B[i] = dist(rng); 
        C[i] = 0.0; 
    }

    double t0 = omp_get_wtime();
    matmul_ultra_optimized(A.data(), B.data(), C.data(), N);
    double t1 = omp_get_wtime();
    
    cout << "N="<<N<<" T="<<T<<" time="<<(t1-t0)<<" seconds\n";

    // simple checksum to validate
    double s = 0; 
    #pragma omp parallel for reduction(+:s)
    for (int i=0;i<N*N;i++) s += C[i];
    cout << "checksum=" << s << "\n";
    
    return 0;
}











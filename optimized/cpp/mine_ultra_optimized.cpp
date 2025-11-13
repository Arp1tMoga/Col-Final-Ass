// #include <iostream>
// #include <vector>
// #include <random>
// #include <omp.h>
// #include <immintrin.h>  // AVX2 intrinsics
// #include <cstdlib>

// using namespace std;

// // ============================================================================
// // ULTRA-OPTIMIZED MATRIX MULTIPLICATION v3 - OpenBLAS-Inspired
// // Critical techniques from OpenBLAS assembly analysis:
// // - Explicit manual loop unrolling (no compiler dependency)
// // - Broadcast loads (_mm256_broadcast_sd) instead of set1
// // - Prefetch with _mm_prefetch at optimal distances
// // - 16x4 micro-kernel (larger register blocking)
// // - Quad accumulator chains (4-way ILP)
// // ============================================================================

// #define BLOCK_I 128  // Larger I block
// #define BLOCK_J 512  // Much larger J block (OpenBLAS uses big NC)
// #define BLOCK_K 256  // Larger K block
// #define MR 16        // Micro-kernel rows (like OpenBLAS 16x2)
// #define NR 4         // Micro-kernel cols

// // Micro-kernel: 16x4 with EXPLICIT MANUAL UNROLLING (OpenBLAS style)
// // Uses broadcast loads and quad accumulator chains
// static inline void micro_kernel_16x4(const double* __restrict__ A, const double* __restrict__ B, 
//                                       double* __restrict__ C, int N, int ldc) {
//     // Load all 16 C registers
//     __m256d c0 = _mm256_loadu_pd(&C[0*ldc]);
//     __m256d c1 = _mm256_loadu_pd(&C[1*ldc]);
//     __m256d c2 = _mm256_loadu_pd(&C[2*ldc]);
//     __m256d c3 = _mm256_loadu_pd(&C[3*ldc]);
//     __m256d c4 = _mm256_loadu_pd(&C[4*ldc]);
//     __m256d c5 = _mm256_loadu_pd(&C[5*ldc]);
//     __m256d c6 = _mm256_loadu_pd(&C[6*ldc]);
//     __m256d c7 = _mm256_loadu_pd(&C[7*ldc]);
//     __m256d c8 = _mm256_loadu_pd(&C[8*ldc]);
//     __m256d c9 = _mm256_loadu_pd(&C[9*ldc]);
//     __m256d c10 = _mm256_loadu_pd(&C[10*ldc]);
//     __m256d c11 = _mm256_loadu_pd(&C[11*ldc]);
//     __m256d c12 = _mm256_loadu_pd(&C[12*ldc]);
//     __m256d c13 = _mm256_loadu_pd(&C[13*ldc]);
//     __m256d c14 = _mm256_loadu_pd(&C[14*ldc]);
//     __m256d c15 = _mm256_loadu_pd(&C[15*ldc]);
    
//     // EXPLICIT MANUAL UNROLLING - Process 4 k iterations at once (quad accumulator chains)
//     for (int k = 0; k < BLOCK_K; k += 4) {
//         // Aggressive prefetching
//         _mm_prefetch((const char*)&A[(k+32)*N], _MM_HINT_T0);
//         _mm_prefetch((const char*)&B[(k+32)*N], _MM_HINT_T0);
        
//         // ===== Iteration k+0 =====
//         __m256d b0 = _mm256_loadu_pd(&B[k*N]);
//         __m256d a0, a1, a2, a3;
        
//         // Unroll rows 0-3
//         a0 = _mm256_broadcast_sd(&A[0*N + k]);
//         a1 = _mm256_broadcast_sd(&A[1*N + k]);
//         a2 = _mm256_broadcast_sd(&A[2*N + k]);
//         a3 = _mm256_broadcast_sd(&A[3*N + k]);
//         c0 = _mm256_fmadd_pd(a0, b0, c0);
//         c1 = _mm256_fmadd_pd(a1, b0, c1);
//         c2 = _mm256_fmadd_pd(a2, b0, c2);
//         c3 = _mm256_fmadd_pd(a3, b0, c3);
        
//         // Unroll rows 4-7
//         a0 = _mm256_broadcast_sd(&A[4*N + k]);
//         a1 = _mm256_broadcast_sd(&A[5*N + k]);
//         a2 = _mm256_broadcast_sd(&A[6*N + k]);
//         a3 = _mm256_broadcast_sd(&A[7*N + k]);
//         c4 = _mm256_fmadd_pd(a0, b0, c4);
//         c5 = _mm256_fmadd_pd(a1, b0, c5);
//         c6 = _mm256_fmadd_pd(a2, b0, c6);
//         c7 = _mm256_fmadd_pd(a3, b0, c7);
        
//         // Unroll rows 8-11
//         a0 = _mm256_broadcast_sd(&A[8*N + k]);
//         a1 = _mm256_broadcast_sd(&A[9*N + k]);
//         a2 = _mm256_broadcast_sd(&A[10*N + k]);
//         a3 = _mm256_broadcast_sd(&A[11*N + k]);
//         c8 = _mm256_fmadd_pd(a0, b0, c8);
//         c9 = _mm256_fmadd_pd(a1, b0, c9);
//         c10 = _mm256_fmadd_pd(a2, b0, c10);
//         c11 = _mm256_fmadd_pd(a3, b0, c11);
        
//         // Unroll rows 12-15
//         a0 = _mm256_broadcast_sd(&A[12*N + k]);
//         a1 = _mm256_broadcast_sd(&A[13*N + k]);
//         a2 = _mm256_broadcast_sd(&A[14*N + k]);
//         a3 = _mm256_broadcast_sd(&A[15*N + k]);
//         c12 = _mm256_fmadd_pd(a0, b0, c12);
//         c13 = _mm256_fmadd_pd(a1, b0, c13);
//         c14 = _mm256_fmadd_pd(a2, b0, c14);
//         c15 = _mm256_fmadd_pd(a3, b0, c15);
        
//         if (k + 1 >= BLOCK_K) break;
        
//         // ===== Iteration k+1 =====
//         __m256d b1 = _mm256_loadu_pd(&B[(k+1)*N]);
        
//         a0 = _mm256_broadcast_sd(&A[0*N + k + 1]);
//         a1 = _mm256_broadcast_sd(&A[1*N + k + 1]);
//         a2 = _mm256_broadcast_sd(&A[2*N + k + 1]);
//         a3 = _mm256_broadcast_sd(&A[3*N + k + 1]);
//         c0 = _mm256_fmadd_pd(a0, b1, c0);
//         c1 = _mm256_fmadd_pd(a1, b1, c1);
//         c2 = _mm256_fmadd_pd(a2, b1, c2);
//         c3 = _mm256_fmadd_pd(a3, b1, c3);
        
//         a0 = _mm256_broadcast_sd(&A[4*N + k + 1]);
//         a1 = _mm256_broadcast_sd(&A[5*N + k + 1]);
//         a2 = _mm256_broadcast_sd(&A[6*N + k + 1]);
//         a3 = _mm256_broadcast_sd(&A[7*N + k + 1]);
//         c4 = _mm256_fmadd_pd(a0, b1, c4);
//         c5 = _mm256_fmadd_pd(a1, b1, c5);
//         c6 = _mm256_fmadd_pd(a2, b1, c6);
//         c7 = _mm256_fmadd_pd(a3, b1, c7);
        
//         a0 = _mm256_broadcast_sd(&A[8*N + k + 1]);
//         a1 = _mm256_broadcast_sd(&A[9*N + k + 1]);
//         a2 = _mm256_broadcast_sd(&A[10*N + k + 1]);
//         a3 = _mm256_broadcast_sd(&A[11*N + k + 1]);
//         c8 = _mm256_fmadd_pd(a0, b1, c8);
//         c9 = _mm256_fmadd_pd(a1, b1, c9);
//         c10 = _mm256_fmadd_pd(a2, b1, c10);
//         c11 = _mm256_fmadd_pd(a3, b1, c11);
        
//         a0 = _mm256_broadcast_sd(&A[12*N + k + 1]);
//         a1 = _mm256_broadcast_sd(&A[13*N + k + 1]);
//         a2 = _mm256_broadcast_sd(&A[14*N + k + 1]);
//         a3 = _mm256_broadcast_sd(&A[15*N + k + 1]);
//         c12 = _mm256_fmadd_pd(a0, b1, c12);
//         c13 = _mm256_fmadd_pd(a1, b1, c13);
//         c14 = _mm256_fmadd_pd(a2, b1, c14);
//         c15 = _mm256_fmadd_pd(a3, b1, c15);
        
//         if (k + 2 >= BLOCK_K) break;
        
//         // ===== Iteration k+2 =====
//         __m256d b2 = _mm256_loadu_pd(&B[(k+2)*N]);
        
//         a0 = _mm256_broadcast_sd(&A[0*N + k + 2]);
//         a1 = _mm256_broadcast_sd(&A[1*N + k + 2]);
//         a2 = _mm256_broadcast_sd(&A[2*N + k + 2]);
//         a3 = _mm256_broadcast_sd(&A[3*N + k + 2]);
//         c0 = _mm256_fmadd_pd(a0, b2, c0);
//         c1 = _mm256_fmadd_pd(a1, b2, c1);
//         c2 = _mm256_fmadd_pd(a2, b2, c2);
//         c3 = _mm256_fmadd_pd(a3, b2, c3);
        
//         a0 = _mm256_broadcast_sd(&A[4*N + k + 2]);
//         a1 = _mm256_broadcast_sd(&A[5*N + k + 2]);
//         a2 = _mm256_broadcast_sd(&A[6*N + k + 2]);
//         a3 = _mm256_broadcast_sd(&A[7*N + k + 2]);
//         c4 = _mm256_fmadd_pd(a0, b2, c4);
//         c5 = _mm256_fmadd_pd(a1, b2, c5);
//         c6 = _mm256_fmadd_pd(a2, b2, c6);
//         c7 = _mm256_fmadd_pd(a3, b2, c7);
        
//         a0 = _mm256_broadcast_sd(&A[8*N + k + 2]);
//         a1 = _mm256_broadcast_sd(&A[9*N + k + 2]);
//         a2 = _mm256_broadcast_sd(&A[10*N + k + 2]);
//         a3 = _mm256_broadcast_sd(&A[11*N + k + 2]);
//         c8 = _mm256_fmadd_pd(a0, b2, c8);
//         c9 = _mm256_fmadd_pd(a1, b2, c9);
//         c10 = _mm256_fmadd_pd(a2, b2, c10);
//         c11 = _mm256_fmadd_pd(a3, b2, c11);
        
//         a0 = _mm256_broadcast_sd(&A[12*N + k + 2]);
//         a1 = _mm256_broadcast_sd(&A[13*N + k + 2]);
//         a2 = _mm256_broadcast_sd(&A[14*N + k + 2]);
//         a3 = _mm256_broadcast_sd(&A[15*N + k + 2]);
//         c12 = _mm256_fmadd_pd(a0, b2, c12);
//         c13 = _mm256_fmadd_pd(a1, b2, c13);
//         c14 = _mm256_fmadd_pd(a2, b2, c14);
//         c15 = _mm256_fmadd_pd(a3, b2, c15);
        
//         if (k + 3 >= BLOCK_K) break;
        
//         // ===== Iteration k+3 =====
//         __m256d b3 = _mm256_loadu_pd(&B[(k+3)*N]);
        
//         a0 = _mm256_broadcast_sd(&A[0*N + k + 3]);
//         a1 = _mm256_broadcast_sd(&A[1*N + k + 3]);
//         a2 = _mm256_broadcast_sd(&A[2*N + k + 3]);
//         a3 = _mm256_broadcast_sd(&A[3*N + k + 3]);
//         c0 = _mm256_fmadd_pd(a0, b3, c0);
//         c1 = _mm256_fmadd_pd(a1, b3, c1);
//         c2 = _mm256_fmadd_pd(a2, b3, c2);
//         c3 = _mm256_fmadd_pd(a3, b3, c3);
        
//         a0 = _mm256_broadcast_sd(&A[4*N + k + 3]);
//         a1 = _mm256_broadcast_sd(&A[5*N + k + 3]);
//         a2 = _mm256_broadcast_sd(&A[6*N + k + 3]);
//         a3 = _mm256_broadcast_sd(&A[7*N + k + 3]);
//         c4 = _mm256_fmadd_pd(a0, b3, c4);
//         c5 = _mm256_fmadd_pd(a1, b3, c5);
//         c6 = _mm256_fmadd_pd(a2, b3, c6);
//         c7 = _mm256_fmadd_pd(a3, b3, c7);
        
//         a0 = _mm256_broadcast_sd(&A[8*N + k + 3]);
//         a1 = _mm256_broadcast_sd(&A[9*N + k + 3]);
//         a2 = _mm256_broadcast_sd(&A[10*N + k + 3]);
//         a3 = _mm256_broadcast_sd(&A[11*N + k + 3]);
//         c8 = _mm256_fmadd_pd(a0, b3, c8);
//         c9 = _mm256_fmadd_pd(a1, b3, c9);
//         c10 = _mm256_fmadd_pd(a2, b3, c10);
//         c11 = _mm256_fmadd_pd(a3, b3, c11);
        
//         a0 = _mm256_broadcast_sd(&A[12*N + k + 3]);
//         a1 = _mm256_broadcast_sd(&A[13*N + k + 3]);
//         a2 = _mm256_broadcast_sd(&A[14*N + k + 3]);
//         a3 = _mm256_broadcast_sd(&A[15*N + k + 3]);
//         c12 = _mm256_fmadd_pd(a0, b3, c12);
//         c13 = _mm256_fmadd_pd(a1, b3, c13);
//         c14 = _mm256_fmadd_pd(a2, b3, c14);
//         c15 = _mm256_fmadd_pd(a3, b3, c15);
//     }
    
//     // Store all 16 results
//     _mm256_storeu_pd(&C[0*ldc], c0);
//     _mm256_storeu_pd(&C[1*ldc], c1);
//     _mm256_storeu_pd(&C[2*ldc], c2);
//     _mm256_storeu_pd(&C[3*ldc], c3);
//     _mm256_storeu_pd(&C[4*ldc], c4);
//     _mm256_storeu_pd(&C[5*ldc], c5);
//     _mm256_storeu_pd(&C[6*ldc], c6);
//     _mm256_storeu_pd(&C[7*ldc], c7);
//     _mm256_storeu_pd(&C[8*ldc], c8);
//     _mm256_storeu_pd(&C[9*ldc], c9);
//     _mm256_storeu_pd(&C[10*ldc], c10);
//     _mm256_storeu_pd(&C[11*ldc], c11);
//     _mm256_storeu_pd(&C[12*ldc], c12);
//     _mm256_storeu_pd(&C[13*ldc], c13);
//     _mm256_storeu_pd(&C[14*ldc], c14);
//     _mm256_storeu_pd(&C[15*ldc], c15);
// }

// // Micro-kernel: 12x4 with dual accumulator chains for better ILP (fallback)
// static inline void micro_kernel_12x4(const double* __restrict__ A, const double* __restrict__ B, 
//                                       double* __restrict__ C, int N, int ldc) {
//     // Load C into 12 registers
//     __m256d c0 = _mm256_loadu_pd(&C[0*ldc]);
//     __m256d c1 = _mm256_loadu_pd(&C[1*ldc]);
//     __m256d c2 = _mm256_loadu_pd(&C[2*ldc]);
//     __m256d c3 = _mm256_loadu_pd(&C[3*ldc]);
//     __m256d c4 = _mm256_loadu_pd(&C[4*ldc]);
//     __m256d c5 = _mm256_loadu_pd(&C[5*ldc]);
//     __m256d c6 = _mm256_loadu_pd(&C[6*ldc]);
//     __m256d c7 = _mm256_loadu_pd(&C[7*ldc]);
//     __m256d c8 = _mm256_loadu_pd(&C[8*ldc]);
//     __m256d c9 = _mm256_loadu_pd(&C[9*ldc]);
//     __m256d c10 = _mm256_loadu_pd(&C[10*ldc]);
//     __m256d c11 = _mm256_loadu_pd(&C[11*ldc]);
    
//     // Dual accumulator chains - process 2 k iterations at once for better ILP
//     for (int k = 0; k < BLOCK_K; k += 2) {
//         // Prefetch ahead (larger distance)
//         __builtin_prefetch(&A[(k+16)*N], 0, 3);
//         __builtin_prefetch(&B[(k+16)*N], 0, 3);
        
//         // First accumulator chain (k)
//         __m256d b0 = _mm256_loadu_pd(&B[k*N]);
//         c0 = _mm256_fmadd_pd(_mm256_set1_pd(A[0*N + k]), b0, c0);
//         c1 = _mm256_fmadd_pd(_mm256_set1_pd(A[1*N + k]), b0, c1);
//         c2 = _mm256_fmadd_pd(_mm256_set1_pd(A[2*N + k]), b0, c2);
//         c3 = _mm256_fmadd_pd(_mm256_set1_pd(A[3*N + k]), b0, c3);
//         c4 = _mm256_fmadd_pd(_mm256_set1_pd(A[4*N + k]), b0, c4);
//         c5 = _mm256_fmadd_pd(_mm256_set1_pd(A[5*N + k]), b0, c5);
//         c6 = _mm256_fmadd_pd(_mm256_set1_pd(A[6*N + k]), b0, c6);
//         c7 = _mm256_fmadd_pd(_mm256_set1_pd(A[7*N + k]), b0, c7);
//         c8 = _mm256_fmadd_pd(_mm256_set1_pd(A[8*N + k]), b0, c8);
//         c9 = _mm256_fmadd_pd(_mm256_set1_pd(A[9*N + k]), b0, c9);
//         c10 = _mm256_fmadd_pd(_mm256_set1_pd(A[10*N + k]), b0, c10);
//         c11 = _mm256_fmadd_pd(_mm256_set1_pd(A[11*N + k]), b0, c11);
        
//         // Second accumulator chain (k+1) - independent from first
//         if (k + 1 < BLOCK_K) {
//             __m256d b1 = _mm256_loadu_pd(&B[(k+1)*N]);
//             c0 = _mm256_fmadd_pd(_mm256_set1_pd(A[0*N + k + 1]), b1, c0);
//             c1 = _mm256_fmadd_pd(_mm256_set1_pd(A[1*N + k + 1]), b1, c1);
//             c2 = _mm256_fmadd_pd(_mm256_set1_pd(A[2*N + k + 1]), b1, c2);
//             c3 = _mm256_fmadd_pd(_mm256_set1_pd(A[3*N + k + 1]), b1, c3);
//             c4 = _mm256_fmadd_pd(_mm256_set1_pd(A[4*N + k + 1]), b1, c4);
//             c5 = _mm256_fmadd_pd(_mm256_set1_pd(A[5*N + k + 1]), b1, c5);
//             c6 = _mm256_fmadd_pd(_mm256_set1_pd(A[6*N + k + 1]), b1, c6);
//             c7 = _mm256_fmadd_pd(_mm256_set1_pd(A[7*N + k + 1]), b1, c7);
//             c8 = _mm256_fmadd_pd(_mm256_set1_pd(A[8*N + k + 1]), b1, c8);
//             c9 = _mm256_fmadd_pd(_mm256_set1_pd(A[9*N + k + 1]), b1, c9);
//             c10 = _mm256_fmadd_pd(_mm256_set1_pd(A[10*N + k + 1]), b1, c10);
//             c11 = _mm256_fmadd_pd(_mm256_set1_pd(A[11*N + k + 1]), b1, c11);
//         }
//     }
    
//     // Store back
//     _mm256_storeu_pd(&C[0*ldc], c0);
//     _mm256_storeu_pd(&C[1*ldc], c1);
//     _mm256_storeu_pd(&C[2*ldc], c2);
//     _mm256_storeu_pd(&C[3*ldc], c3);
//     _mm256_storeu_pd(&C[4*ldc], c4);
//     _mm256_storeu_pd(&C[5*ldc], c5);
//     _mm256_storeu_pd(&C[6*ldc], c6);
//     _mm256_storeu_pd(&C[7*ldc], c7);
//     _mm256_storeu_pd(&C[8*ldc], c8);
//     _mm256_storeu_pd(&C[9*ldc], c9);
//     _mm256_storeu_pd(&C[10*ldc], c10);
//     _mm256_storeu_pd(&C[11*ldc], c11);
// }

// // Micro-kernel: Highly optimized 8x4 block computation (for edge cases)
// static inline void micro_kernel_8x4(const double* __restrict__ A, const double* __restrict__ B, 
//                                      double* __restrict__ C, int N, int ldc) {
//     // Load C into registers
//     __m256d c0 = _mm256_loadu_pd(&C[0*ldc]);
//     __m256d c1 = _mm256_loadu_pd(&C[1*ldc]);
//     __m256d c2 = _mm256_loadu_pd(&C[2*ldc]);
//     __m256d c3 = _mm256_loadu_pd(&C[3*ldc]);
//     __m256d c4 = _mm256_loadu_pd(&C[4*ldc]);
//     __m256d c5 = _mm256_loadu_pd(&C[5*ldc]);
//     __m256d c6 = _mm256_loadu_pd(&C[6*ldc]);
//     __m256d c7 = _mm256_loadu_pd(&C[7*ldc]);
    
//     for (int k = 0; k < BLOCK_K; ++k) {
//         // Prefetch future data
//         __builtin_prefetch(&A[(k+8)*N], 0, 3);
//         __builtin_prefetch(&B[(k+8)*N], 0, 3);
        
//         // Load B once (broadcast will happen)
//         __m256d b = _mm256_loadu_pd(&B[k*N]);
        
//         // Broadcast A and compute FMA
//         c0 = _mm256_fmadd_pd(_mm256_set1_pd(A[0*N + k]), b, c0);
//         c1 = _mm256_fmadd_pd(_mm256_set1_pd(A[1*N + k]), b, c1);
//         c2 = _mm256_fmadd_pd(_mm256_set1_pd(A[2*N + k]), b, c2);
//         c3 = _mm256_fmadd_pd(_mm256_set1_pd(A[3*N + k]), b, c3);
//         c4 = _mm256_fmadd_pd(_mm256_set1_pd(A[4*N + k]), b, c4);
//         c5 = _mm256_fmadd_pd(_mm256_set1_pd(A[5*N + k]), b, c5);
//         c6 = _mm256_fmadd_pd(_mm256_set1_pd(A[6*N + k]), b, c6);
//         c7 = _mm256_fmadd_pd(_mm256_set1_pd(A[7*N + k]), b, c7);
//     }
    
//     // Store back
//     _mm256_storeu_pd(&C[0*ldc], c0);
//     _mm256_storeu_pd(&C[1*ldc], c1);
//     _mm256_storeu_pd(&C[2*ldc], c2);
//     _mm256_storeu_pd(&C[3*ldc], c3);
//     _mm256_storeu_pd(&C[4*ldc], c4);
//     _mm256_storeu_pd(&C[5*ldc], c5);
//     _mm256_storeu_pd(&C[6*ldc], c6);
//     _mm256_storeu_pd(&C[7*ldc], c7);
// }

// // Main optimized GEMM function
// static void matmul_ultra_optimized(const double* A, const double* B, double* C, int N) {
//     #pragma omp parallel for schedule(static) collapse(1)
//     for (int ii = 0; ii < N; ii += BLOCK_I) {
//         int i_block = min(BLOCK_I, N - ii);
        
//         for (int jj = 0; jj < N; jj += BLOCK_J) {
//             int j_block = min(BLOCK_J, N - jj);
            
//             for (int kk = 0; kk < N; kk += BLOCK_K) {
//                 int k_block = min(BLOCK_K, N - kk);
                
//                 // Process 16x4 blocks with OpenBLAS-style micro-kernel
//                 int i = ii;
//                 for (; i <= ii + i_block - 16; i += 16) {
//                     int j = jj;
//                     for (; j <= jj + j_block - 4; j += 4) {
//                         micro_kernel_16x4(&A[i*N + kk], &B[kk*N + j], 
//                                          &C[i*N + j], N, N);
//                     }
                    
//                     // Handle remaining j for this 16-row block
//                     for (; j < jj + j_block; ++j) {
//                         for (int k = kk; k < kk + k_block; ++k) {
//                             for (int i2 = i; i2 < i + 16; ++i2) {
//                                 C[i2*N + j] += A[i2*N + k] * B[k*N + j];
//                             }
//                         }
//                     }
//                 }
                
//                 // Process 12x4 blocks if we have at least 12 rows remaining
//                 if (i <= ii + i_block - 12) {
//                     int j = jj;
//                     for (; j <= jj + j_block - 4; j += 4) {
//                         micro_kernel_12x4(&A[i*N + kk], &B[kk*N + j], 
//                                          &C[i*N + j], N, N);
//                     }
                    
//                     // Handle remaining j for this 12-row block
//                     for (; j < jj + j_block; ++j) {
//                         for (int k = kk; k < kk + k_block; ++k) {
//                             for (int i2 = i; i2 < i + 12; ++i2) {
//                                 C[i2*N + j] += A[i2*N + k] * B[k*N + j];
//                             }
//                         }
//                     }
//                     i += 12;
//                 }
                
//                 // Process 8x4 blocks if we have at least 8 rows remaining
//                 if (i <= ii + i_block - 8) {
//                     int j = jj;
//                     for (; j <= jj + j_block - 4; j += 4) {
//                         micro_kernel_8x4(&A[i*N + kk], &B[kk*N + j], 
//                                         &C[i*N + j], N, N);
//                     }
                    
//                     // Handle remaining j for this 8-row block
//                     for (; j < jj + j_block; ++j) {
//                         for (int k = kk; k < kk + k_block; ++k) {
//                             for (int i2 = i; i2 < i + 8; ++i2) {
//                                 C[i2*N + j] += A[i2*N + k] * B[k*N + j];
//                             }
//                         }
//                     }
//                     i += 8;
//                 }
                
//                 // Handle remaining i rows (< 8)
//                 for (; i < ii + i_block; ++i) {
//                     for (int k = kk; k < kk + k_block; ++k) {
//                         double a = A[i*N + k];
//                         __m256d a_vec = _mm256_set1_pd(a);
                        
//                         int j = jj;
//                         for (; j <= jj + j_block - 4; j += 4) {
//                             __m256d b_vec = _mm256_loadu_pd(&B[k*N + j]);
//                             __m256d c_vec = _mm256_loadu_pd(&C[i*N + j]);
//                             c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
//                             _mm256_storeu_pd(&C[i*N + j], c_vec);
//                         }
                        
//                         for (; j < jj + j_block; ++j) {
//                             C[i*N + j] += a * B[k*N + j];
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// int main(int argc, char** argv) {
//     if (argc < 3) {
//         cerr << "Usage: " << argv[0] << " N num_threads\n";
//         return 1;
//     }
//     int N = atoi(argv[1]);
//     int T = atoi(argv[2]);
//     omp_set_num_threads(T);

//     vector<double> A(N*N), B(N*N), C(N*N);
    
//     // reproducible RNG
//     std::mt19937_64 rng(12345);
//     std::normal_distribution<double> dist(0.0, 1.0);
//     for (int i=0;i<N*N;i++) { 
//         A[i] = dist(rng); 
//         B[i] = dist(rng); 
//         C[i] = 0.0; 
//     }

//     double t0 = omp_get_wtime();
//     matmul_ultra_optimized(A.data(), B.data(), C.data(), N);
//     double t1 = omp_get_wtime();
    
//     cout << "N="<<N<<" T="<<T<<" time="<<(t1-t0)<<" seconds\n";

//     // simple checksum to validate
//     double s = 0; 
//     for (int i=0;i<N*N;i++) s += C[i];
//     cout << "checksum=" << s << "\n";
    
//     return 0;
// }





































#include <immintrin.h>
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

using namespace std;

/*
 High-Performance DGEMM (C = A * B) for CPUs WITHOUT AVX-512.
 Optimized for AVX2 + FMA; falls back to scalar if AVX2/FMA unavailable.

 Core strategy (GotoBLAS / OpenBLAS inspired):
   - 3-level blocking: (nc, kc, mc) chosen for typical x86 cache hierarchies.
   - Packing of panels:
       * Pack B (kc x 8) micro-panels contiguous in memory to enable aligned SIMD loads.
       * Pack A (kc x 6) micro-panels so each k iteration touches 6 contiguous scalars.
   - AVX2 micro-kernel: MR=6, NR=8 (widely used performant shape for double).
       * Each row has two __m256d accumulators (8 columns = 2 * 4 doubles).
       * Loop over k dimension: load 8 B doubles (two vectors), broadcast A scalars, FMA.
   - OpenMP parallelization across i-blocks after packing B (reduces redundant B packing).
   - Careful alignment (64 bytes) for packed buffers.
   - Edge handling for leftover rows/cols uses a scalar cleanup path.

 Typical tuning values (double precision, AVX2):
   nc = 4096
   kc = 256
   mc = 144
 These can be overridden via environment variables: NC, KC, MC

 Compile:
   g++ -O3 -march=native -fopenmp -std=c++17 fast_dgemm_avx2.cpp -o matmul

 Run:
   ./matmul 2048 16
*/

static inline bool has_avx2_fma() {
#if defined(__x86_64__) || defined(__i386)
#if defined(__GNUC__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
#else
    return false;
#endif
}

static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
    void* p = nullptr;
#if defined(_MSC_VER)
    p = _aligned_malloc(bytes, align);
    if (!p) throw std::bad_alloc();
#else
    if (posix_memalign(&p, align, bytes) != 0) throw std::bad_alloc();
#endif
    return p;
}

static inline void aligned_free(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

/*
 Pack B panel:
 Original B is row-major (N x N).
 We pack a kc x NR micro-panel for each NR=8 column block:
   For p in [0, kc):
      store B[(k0 + p), (j0 + c)] for c=0..7 contiguous.
 Layout result: sequence of blocks of size kc*NR.
*/
static void pack_B(const double* __restrict B,
                   double* __restrict Bp,
                   int N, int k0, int kc, int j0, int jb, int NR)
{
    int full_col_blocks = jb / NR;
    for (int blk = 0; blk < full_col_blocks; ++blk) {
        int j_start = j0 + blk * NR;
        double* out = Bp + blk * (kc * NR);
        for (int p = 0; p < kc; ++p) {
            const double* src = B + (k0 + p) * N + j_start;
            // Copy NR=8 doubles
            _mm_prefetch((const char*)(src + 16), _MM_HINT_T0);
            std::memcpy(out + p * NR, src, sizeof(double) * NR);
        }
    }
    // Remainder columns
    int rem = jb - full_col_blocks * NR;
    if (rem) {
        int j_start = j0 + full_col_blocks * NR;
        double* out = Bp + full_col_blocks * (kc * NR);
        for (int p = 0; p < kc; ++p) {
            const double* src = B + (k0 + p) * N + j_start;
            for (int c = 0; c < rem; ++c) out[p * NR + c] = src[c];
            for (int c = rem; c < NR; ++c) out[p * NR + c] = 0.0; // pad
        }
    }
}

/*
 Pack A micro-panel of size (ic..ic+MR, k0..k0+kc):
 Layout: for each p in [0,kc) store A[(ic + r), (k0 + p)] for r=0..MR-1 contiguous.
 Thus memory order: p-major; inside each p we have MR consecutive doubles.
 Size: kc * MR doubles.
*/
static void pack_A_micro(const double* __restrict A,
                         double* __restrict Ap,
                         int N, int ic, int i_rem, int k0, int kc, int MR)
{
    for (int p = 0; p < kc; ++p) {
        const double* src = A + (k0 + p);
        double* dst = Ap + p * MR;
        for (int r = 0; r < i_rem; ++r) {
            dst[r] = src[(ic + r) * N];
        }
        for (int r = i_rem; r < MR; ++r) dst[r] = 0.0; // pad unused rows
    }
}

/*
 AVX2 Micro-kernel: MR=6, NR=8
 Ap: packed A micro-panel (kc * MR)
 Bp_block: pointer to packed B micro-panel for current column block (kc * NR)
 C: base pointer to output matrix
 ic, jc: starting row/col in C
 ldc: leading dimension (N)
 kc: depth
*/
static inline void dgemm_kernel_6x8(int kc,
                                    const double* __restrict Ap,
                                    const double* __restrict Bp_block,
                                    double* __restrict C,
                                    int ldc, int ic, int jc)
{
    // 6 rows x 8 columns => each row has two 256-bit accumulators
    __m256d acc0[6]; // columns 0..3
    __m256d acc1[6]; // columns 4..7
#pragma unroll
    for (int r = 0; r < 6; ++r) {
        acc0[r] = _mm256_setzero_pd();
        acc1[r] = _mm256_setzero_pd();
    }

    for (int p = 0; p < kc; ++p) {
        const double* arow = Ap + p * 6;
        __m256d b0 = _mm256_load_pd(Bp_block + p * 8);     // cols 0..3
        __m256d b1 = _mm256_load_pd(Bp_block + p * 8 + 4); // cols 4..7
#pragma unroll
        for (int r = 0; r < 6; ++r) {
            __m256d a_bcast = _mm256_set1_pd(arow[r]);
            acc0[r] = _mm256_fmadd_pd(a_bcast, b0, acc0[r]);
            acc1[r] = _mm256_fmadd_pd(a_bcast, b1, acc1[r]);
        }
    }

    // Accumulate into C
#pragma unroll
    for (int r = 0; r < 6; ++r) {
        double* cptr = C + (ic + r) * ldc + jc;
        __m256d cold0 = _mm256_loadu_pd(cptr);
        __m256d cold1 = _mm256_loadu_pd(cptr + 4);
        _mm256_storeu_pd(cptr,     _mm256_add_pd(cold0, acc0[r]));
        _mm256_storeu_pd(cptr + 4, _mm256_add_pd(cold1, acc1[r]));
    }
}

/*
 Scalar cleanup micro-kernel for arbitrary (i_rem x j_rem).
 Directly multiplies un-packed A row segment with B row segment.
 Used for edge cases (rows < MR or cols < NR).
*/
static inline void dgemm_scalar_edge(const double* A, const double* B, double* C,
                                     int N, int ic, int jc, int i_rem, int j_rem,
                                     int k0, int kc)
{
    for (int p = 0; p < kc; ++p) {
        const double* brow = B + (k0 + p) * N + jc;
        for (int r = 0; r < i_rem; ++r) {
            double aval = A[(ic + r) * N + (k0 + p)];
            double* cptr = C + (ic + r) * N + jc;
            for (int c = 0; c < j_rem; ++c) {
                cptr[c] += aval * brow[c];
            }
        }
    }
}

void CUSTOM_MATMUL(const double* __restrict A,
                   const double* __restrict B,
                   double* __restrict C,
                   int N)
{
    const bool use_avx2 = has_avx2_fma();

    // Block sizes (override with environment if set)
    int nc = 4096;
    int kc = 256;
    int mc = 144;

    if (const char* e = getenv("NC")) nc = atoi(e);
    if (const char* e = getenv("KC")) kc = atoi(e);
    if (const char* e = getenv("MC")) mc = atoi(e);

    // Micro-kernel sizes
    const int MR = use_avx2 ? 6 : 1;
    const int NR = use_avx2 ? 8 : 1;

#pragma omp parallel for
    for (int i = 0; i < N * N; ++i) C[i] = 0.0;

    // Allocate packed buffers once (shared) for B panel.
    double* Bpack = (double*)aligned_malloc(sizeof(double) * kc * ((nc / NR + 1) * NR));
    // A micro-panel buffer (thread-local) allocated inside parallel region.

    for (int j0 = 0; j0 < N; j0 += nc) {
        int jb = min(nc, N - j0);
        for (int k0 = 0; k0 < N; k0 += kc) {
            int kb = min(kc, N - k0);

            // Pack B panel for columns j0..j0+jb
            pack_B(B, Bpack, N, k0, kb, j0, jb, NR);

            // Parallelize over i-blocks after B packed
#pragma omp parallel
            {
                double* Apack = (double*)aligned_malloc(sizeof(double) * kb * MR);

#pragma omp for schedule(static)
                for (int i0 = 0; i0 < N; i0 += mc) {
                    int ib = min(mc, N - i0);

                    for (int ic = i0; ic < i0 + ib; ic += MR) {
                        int i_rem = min(MR, i0 + ib - ic);

                        // Pack A micro-panel if using AVX2 path
                        if (use_avx2) {
                            pack_A_micro(A, Apack, N, ic, i_rem, k0, kb, MR);
                        }

                        // Iterate over column micro-panels
                        for (int jc = j0; jc < j0 + jb; jc += NR) {
                            int j_rem = min(NR, j0 + jb - jc);

                            if (use_avx2 && i_rem == MR && j_rem == NR) {
                                // Locate packed B block
                                int col_block_index = (jc - j0) / NR;
                                const double* Bpanel_block = Bpack + col_block_index * (kb * NR);

                                dgemm_kernel_6x8(kb, Apack, Bpanel_block, C, N, ic, jc);
                            } else {
                                // Scalar edge
                                dgemm_scalar_edge(A, B, C, N, ic, jc, i_rem, j_rem, k0, kb);
                            }
                        } // jc
                    } // ic
                } // i0

                aligned_free(Apack);
            } // omp parallel
        } // k0
    } // j0

    aligned_free(Bpack);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " N num_threads\n";
        return 1;
    }
    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    omp_set_num_threads(T);

    vector<double> A(N * N), B(N * N), C(N * N);
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < N * N; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
        C[i] = 0.0;
    }

    double t0 = omp_get_wtime();
    CUSTOM_MATMUL(A.data(), B.data(), C.data(), N);
    double t1 = omp_get_wtime();

    cout << "N=" << N << " T=" << T << " time=" << (t1 - t0) << " seconds\n";

    double checksum = 0.0;
#pragma omp parallel for reduction(+:checksum)
    for (int i = 0; i < N * N; ++i) checksum += C[i];
    cout << "checksum=" << checksum << "\n";
    return 0;
}
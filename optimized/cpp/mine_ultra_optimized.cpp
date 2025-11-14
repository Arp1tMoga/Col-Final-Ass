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
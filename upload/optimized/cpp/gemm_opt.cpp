// #include <immintrin.h>
// #include <omp.h>
// #include <cstdlib>
// #include <cstring>
// #include <iostream>
// #include <vector>
// #include <random>
// #include <algorithm>
// #include <fstream>
// #include <sstream>

// using namespace std;

// /*
//  Portable High-Performance DGEMM for x86-64 CPUs
 
//  Features:
//  - Runtime CPU detection (AVX512, AVX2+FMA, SSE, or scalar fallback)
//  - Auto-tuned cache blocking parameters based on detected cache sizes
//  - Works across different x86-64 CPUs without recompilation
 
//  Compile:
//    g++ -O3 -march=x86-64 -fopenmp -std=c++17 portable_dgemm.cpp -o matmul
//    (Note: -march=x86-64 for maximum compatibility, runtime dispatch handles features)
// */

// // ============================================================================
// // CPU Feature Detection
// // ============================================================================

// enum ISA_LEVEL {
//     ISA_SCALAR = 0,
//     ISA_SSE = 1,
//     ISA_AVX2_FMA = 2,
//     ISA_AVX512 = 3
// };

// struct CPUInfo {
//     ISA_LEVEL isa_level;
//     int L1_cache;   // bytes
//     int L2_cache;   // bytes
//     int L3_cache;   // bytes
//     int cache_line; // bytes
//     string cpu_name;
// };

// static CPUInfo detect_cpu() {
//     CPUInfo info;
//     info.isa_level = ISA_SCALAR;
//     info.L1_cache = 32 * 1024;
//     info.L2_cache = 256 * 1024;
//     info.L3_cache = 8 * 1024 * 1024;
//     info.cache_line = 64;
//     info.cpu_name = "Unknown";

// #if defined(__x86_64__) || defined(__i386__)
// #if defined(__GNUC__) || defined(__clang__)
//     // Detect ISA level
//     __builtin_cpu_init();
    
//     if (__builtin_cpu_supports("avx512f") && 
//         __builtin_cpu_supports("avx512dq") &&
//         __builtin_cpu_supports("avx512vl")) {
//         info.isa_level = ISA_AVX512;
//     } else if (__builtin_cpu_supports("avx2") && 
//                __builtin_cpu_supports("fma")) {
//         info.isa_level = ISA_AVX2_FMA;
//     } else if (__builtin_cpu_supports("sse4.2")) {
//         info.isa_level = ISA_SSE;
//     }
    
//     // Read CPU info from /proc/cpuinfo
//     ifstream cpuinfo("/proc/cpuinfo");
//     string line;
//     while (getline(cpuinfo, line)) {
//         if (line.find("model name") != string::npos) {
//             size_t pos = line.find(":");
//             if (pos != string::npos) {
//                 info.cpu_name = line.substr(pos + 2);
//             }
//             break;
//         }
//     }
    
//     // Try to read cache sizes from sysfs (Linux)
//     auto read_cache_size = [](const char* path) -> int {
//         ifstream f(path);
//         if (f) {
//             string val;
//             getline(f, val);
//             // Remove 'K' suffix if present
//             if (!val.empty() && val.back() == 'K') {
//                 val.pop_back();
//                 return stoi(val) * 1024;
//             }
//             return stoi(val);
//         }
//         return 0;
//     };
    
//     int l1d = read_cache_size("/sys/devices/system/cpu/cpu0/cache/index0/size");
//     int l2 = read_cache_size("/sys/devices/system/cpu/cpu0/cache/index2/size");
//     int l3 = read_cache_size("/sys/devices/system/cpu/cpu0/cache/index3/size");
    
//     if (l1d > 0) info.L1_cache = l1d;
//     if (l2 > 0) info.L2_cache = l2;
//     if (l3 > 0) info.L3_cache = l3;
    
//     int line_size = read_cache_size("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");
//     if (line_size > 0) info.cache_line = line_size;
// #endif
// #endif
    
//     return info;
// }

// // ============================================================================
// // Cache-Aware Parameter Selection
// // ============================================================================

// struct BlockParams {
//     int nc, kc, mc;
//     int MR, NR;
// };

// static BlockParams compute_block_params(const CPUInfo& cpu) {
//     BlockParams p;
    
//     // Set micro-kernel sizes based on ISA
//     switch (cpu.isa_level) {
//         case ISA_AVX512:
//             p.MR = 8;
//             p.NR = 16; // AVX512 can handle 8 doubles per register
//             break;
//         case ISA_AVX2_FMA:
//             p.MR = 8;
//             p.NR = 8;  // AVX2 handles 4 doubles, but we use 2 registers
//             break;
//         case ISA_SSE:
//             p.MR = 4;
//             p.NR = 4;
//             break;
//         default:
//             p.MR = 1;
//             p.NR = 1;
//     }
    
//     // kc: should fit A_panel (kc*MR) + B_panel (kc*NR) in L1
//     // Target: use ~60% of L1 for data, leave room for code/stack
//     int L1_target = (cpu.L1_cache * 6) / 10;
//     int kc_from_L1 = L1_target / (sizeof(double) * (p.MR + p.NR));
    
//     // kc should also fit reasonably in L2 for the full B panel
//     // Target: kc * nc should fit in ~60% of L2
//     int L2_target = (cpu.L2_cache * 6) / 10;
    
//     // mc: multiple rows that reuse kc*NR B panel
//     // Target: mc*kc A data + kc*nc B data should fit in ~80% of L3
//     int L3_target = (cpu.L3_cache * 8) / 10;
    
//     // Heuristic tuning based on cache sizes
//     if (cpu.L3_cache >= 16 * 1024 * 1024) {
//         // Large cache (server/HEDT)
//         p.kc = min(512, kc_from_L1);
//         p.nc = 4096;
//         p.mc = 256;
//     } else if (cpu.L3_cache >= 6 * 1024 * 1024) {
//         // Medium cache (desktop)
//         p.kc = min(384, kc_from_L1);
//         p.nc = 4096;
//         p.mc = 192;
//     } else {
//         // Small cache (mobile/old)
//         p.kc = min(256, kc_from_L1);
//         p.nc = 2048;
//         p.mc = 144;
//     }
    
//     // Round to multiples of MR/NR for cleaner blocking
//     p.kc = (p.kc / 16) * 16;
//     p.mc = (p.mc / p.MR) * p.MR;
//     p.nc = (p.nc / p.NR) * p.NR;
    
//     // Environment variable overrides
//     if (const char* e = getenv("NC")) p.nc = atoi(e);
//     if (const char* e = getenv("KC")) p.kc = atoi(e);
//     if (const char* e = getenv("MC")) p.mc = atoi(e);
    
//     return p;
// }

// // ============================================================================
// // Memory Utilities
// // ============================================================================

// static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
//     void* p = nullptr;
// #if defined(_MSC_VER)
//     p = _aligned_malloc(bytes, align);
//     if (!p) throw std::bad_alloc();
// #else
//     if (posix_memalign(&p, align, bytes) != 0) throw std::bad_alloc();
// #endif
//     return p;
// }

// static inline void aligned_free(void* p) {
// #if defined(_MSC_VER)
//     _aligned_free(p);
// #else
//     free(p);
// #endif
// }

// // ============================================================================
// // Packing Functions
// // ============================================================================

// static void pack_B_avx2(const double* __restrict B,
//                         double* __restrict Bp,
//                         int N, int k0, int kc, int j0, int jb, int NR)
// {
//     int full_col_blocks = jb / NR;
//     for (int blk = 0; blk < full_col_blocks; ++blk) {
//         int j_start = j0 + blk * NR;
//         double* out = Bp + blk * (kc * NR);
        
//         for (int p = 0; p < kc; ++p) {
//             const double* src = B + (k0 + p) * N + j_start;
//             if (p + 8 < kc) {
//                 _mm_prefetch((const char*)(B + (k0 + p + 8) * N + j_start), _MM_HINT_T0);
//             }
//             __m256d v0 = _mm256_loadu_pd(src);
//             __m256d v1 = _mm256_loadu_pd(src + 4);
//             _mm256_store_pd(out + p * NR, v0);
//             _mm256_store_pd(out + p * NR + 4, v1);
//         }
//     }
    
//     int rem = jb - full_col_blocks * NR;
//     if (rem) {
//         int j_start = j0 + full_col_blocks * NR;
//         double* out = Bp + full_col_blocks * (kc * NR);
//         for (int p = 0; p < kc; ++p) {
//             const double* src = B + (k0 + p) * N + j_start;
//             for (int c = 0; c < rem; ++c) out[p * NR + c] = src[c];
//             for (int c = rem; c < NR; ++c) out[p * NR + c] = 0.0;
//         }
//     }
// }

// static void pack_B_scalar(const double* __restrict B,
//                           double* __restrict Bp,
//                           int N, int k0, int kc, int j0, int jb, int NR)
// {
//     int full_col_blocks = jb / NR;
//     for (int blk = 0; blk < full_col_blocks; ++blk) {
//         int j_start = j0 + blk * NR;
//         double* out = Bp + blk * (kc * NR);
//         for (int p = 0; p < kc; ++p) {
//             const double* src = B + (k0 + p) * N + j_start;
//             memcpy(out + p * NR, src, sizeof(double) * NR);
//         }
//     }
    
//     int rem = jb - full_col_blocks * NR;
//     if (rem) {
//         int j_start = j0 + full_col_blocks * NR;
//         double* out = Bp + full_col_blocks * (kc * NR);
//         for (int p = 0; p < kc; ++p) {
//             const double* src = B + (k0 + p) * N + j_start;
//             for (int c = 0; c < rem; ++c) out[p * NR + c] = src[c];
//             for (int c = rem; c < NR; ++c) out[p * NR + c] = 0.0;
//         }
//     }
// }

// static void pack_A_micro(const double* __restrict A,
//                          double* __restrict Ap,
//                          int N, int ic, int i_rem, int k0, int kc, int MR)
// {
//     for (int p = 0; p < kc; ++p) {
//         const double* src = A + (k0 + p);
//         double* dst = Ap + p * MR;
//         for (int r = 0; r < i_rem; ++r) {
//             dst[r] = src[(ic + r) * N];
//         }
//         for (int r = i_rem; r < MR; ++r) dst[r] = 0.0;
//     }
// }

// // ============================================================================
// // Micro-kernels
// // ============================================================================

// // AVX2 8x8 kernel with k-unrolling
// static inline void dgemm_kernel_8x8_avx2(int kc,
//                                          const double* __restrict Ap,
//                                          const double* __restrict Bp_block,
//                                          double* __restrict C,
//                                          int ldc, int ic, int jc)
// {
//     __m256d acc0[8], acc1[8];
    
//     for (int r = 0; r < 8; ++r) {
//         acc0[r] = _mm256_setzero_pd();
//         acc1[r] = _mm256_setzero_pd();
//     }

//     int p = 0;
//     for (; p + 3 < kc; p += 4) {
//         _mm_prefetch((const char*)(Bp_block + (p + 16) * 8), _MM_HINT_T0);
//         _mm_prefetch((const char*)(Ap + (p + 16) * 8), _MM_HINT_T0);
        
//         const double* arow0 = Ap + p * 8;
//         __m256d b0_0 = _mm256_load_pd(Bp_block + p * 8);
//         __m256d b1_0 = _mm256_load_pd(Bp_block + p * 8 + 4);
        
//         const double* arow1 = Ap + (p + 1) * 8;
//         __m256d b0_1 = _mm256_load_pd(Bp_block + (p + 1) * 8);
//         __m256d b1_1 = _mm256_load_pd(Bp_block + (p + 1) * 8 + 4);
        
//         const double* arow2 = Ap + (p + 2) * 8;
//         __m256d b0_2 = _mm256_load_pd(Bp_block + (p + 2) * 8);
//         __m256d b1_2 = _mm256_load_pd(Bp_block + (p + 2) * 8 + 4);
        
//         const double* arow3 = Ap + (p + 3) * 8;
//         __m256d b0_3 = _mm256_load_pd(Bp_block + (p + 3) * 8);
//         __m256d b1_3 = _mm256_load_pd(Bp_block + (p + 3) * 8 + 4);
        
//         for (int r = 0; r < 8; ++r) {
//             __m256d a0 = _mm256_set1_pd(arow0[r]);
//             __m256d a1 = _mm256_set1_pd(arow1[r]);
//             __m256d a2 = _mm256_set1_pd(arow2[r]);
//             __m256d a3 = _mm256_set1_pd(arow3[r]);
            
//             acc0[r] = _mm256_fmadd_pd(a0, b0_0, acc0[r]);
//             acc1[r] = _mm256_fmadd_pd(a0, b1_0, acc1[r]);
            
//             acc0[r] = _mm256_fmadd_pd(a1, b0_1, acc0[r]);
//             acc1[r] = _mm256_fmadd_pd(a1, b1_1, acc1[r]);
            
//             acc0[r] = _mm256_fmadd_pd(a2, b0_2, acc0[r]);
//             acc1[r] = _mm256_fmadd_pd(a2, b1_2, acc1[r]);
            
//             acc0[r] = _mm256_fmadd_pd(a3, b0_3, acc0[r]);
//             acc1[r] = _mm256_fmadd_pd(a3, b1_3, acc1[r]);
//         }
//     }
    
//     for (; p < kc; ++p) {
//         const double* arow = Ap + p * 8;
//         __m256d b0 = _mm256_load_pd(Bp_block + p * 8);
//         __m256d b1 = _mm256_load_pd(Bp_block + p * 8 + 4);
        
//         for (int r = 0; r < 8; ++r) {
//             __m256d a_bcast = _mm256_set1_pd(arow[r]);
//             acc0[r] = _mm256_fmadd_pd(a_bcast, b0, acc0[r]);
//             acc1[r] = _mm256_fmadd_pd(a_bcast, b1, acc1[r]);
//         }
//     }

//     for (int r = 0; r < 8; ++r) {
//         double* cptr = C + (ic + r) * ldc + jc;
//         __m256d cold0 = _mm256_loadu_pd(cptr);
//         __m256d cold1 = _mm256_loadu_pd(cptr + 4);
//         _mm256_storeu_pd(cptr,     _mm256_add_pd(cold0, acc0[r]));
//         _mm256_storeu_pd(cptr + 4, _mm256_add_pd(cold1, acc1[r]));
//     }
// }

// // Scalar edge kernel
// static inline void dgemm_scalar_edge(const double* A, const double* B, double* C,
//                                      int N, int ic, int jc, int i_rem, int j_rem,
//                                      int k0, int kc)
// {
//     for (int p = 0; p < kc; ++p) {
//         const double* brow = B + (k0 + p) * N + jc;
//         for (int r = 0; r < i_rem; ++r) {
//             double aval = A[(ic + r) * N + (k0 + p)];
//             double* cptr = C + (ic + r) * N + jc;
//             for (int c = 0; c < j_rem; ++c) {
//                 cptr[c] += aval * brow[c];
//             }
//         }
//     }
// }

// // ============================================================================
// // Main GEMM Function
// // ============================================================================

// void CUSTOM_MATMUL(const double* __restrict A,
//                    const double* __restrict B,
//                    double* __restrict C,
//                    int N)
// {
//     // Detect CPU and compute optimal parameters
//     static CPUInfo cpu_info = detect_cpu();
//     static BlockParams params = compute_block_params(cpu_info);
//     static bool info_printed = false;
    
//     if (!info_printed) {
//         const char* isa_names[] = {"Scalar", "SSE", "AVX2+FMA", "AVX512"};
//         // cout << "CPU: " << cpu_info.cpu_name << "\n";
//         // cout << "ISA: " << isa_names[cpu_info.isa_level] << "\n";
//         // cout << "Cache: L1=" << (cpu_info.L1_cache/1024) << "KB, "
//         //      << "L2=" << (cpu_info.L2_cache/1024) << "KB, "
//         //      << "L3=" << (cpu_info.L3_cache/1024/1024) << "MB\n";
//         // cout << "Blocks: nc=" << params.nc << ", kc=" << params.kc 
//         //      << ", mc=" << params.mc << ", MR=" << params.MR 
//         //      << ", NR=" << params.NR << "\n";
//         info_printed = true;
//     }
    
//     const int nc = params.nc;
//     const int kc = params.kc;
//     const int mc = params.mc;
//     const int MR = params.MR;
//     const int NR = params.NR;
    
//     const bool use_avx2 = (cpu_info.isa_level >= ISA_AVX2_FMA);

//     #pragma omp parallel for schedule(static)
//     for (int i = 0; i < N * N; ++i) C[i] = 0.0;

//     double* Bpack = (double*)aligned_malloc(sizeof(double) * kc * ((nc / NR + 1) * NR));

//     for (int j0 = 0; j0 < N; j0 += nc) {
//         int jb = min(nc, N - j0);
//         for (int k0 = 0; k0 < N; k0 += kc) {
//             int kb = min(kc, N - k0);

//             if (use_avx2) {
//                 pack_B_avx2(B, Bpack, N, k0, kb, j0, jb, NR);
//             } else {
//                 pack_B_scalar(B, Bpack, N, k0, kb, j0, jb, NR);
//             }

//             #pragma omp parallel
//             {
//                 double* Apack = (double*)aligned_malloc(sizeof(double) * kb * MR);

//                 #pragma omp for schedule(dynamic, 1) nowait
//                 for (int i0 = 0; i0 < N; i0 += mc) {
//                     int ib = min(mc, N - i0);

//                     for (int ic = i0; ic < i0 + ib; ic += MR) {
//                         int i_rem = min(MR, i0 + ib - ic);

//                         if (use_avx2) {
//                             pack_A_micro(A, Apack, N, ic, i_rem, k0, kb, MR);
//                         }

//                         for (int jc = j0; jc < j0 + jb; jc += NR) {
//                             int j_rem = min(NR, j0 + jb - jc);

//                             if (use_avx2 && i_rem == MR && j_rem == NR) {
//                                 int col_block_index = (jc - j0) / NR;
//                                 const double* Bpanel_block = Bpack + col_block_index * (kb * NR);

//                                 dgemm_kernel_8x8_avx2(kb, Apack, Bpanel_block, C, N, ic, jc);
//                             } else {
//                                 dgemm_scalar_edge(A, B, C, N, ic, jc, i_rem, j_rem, k0, kb);
//                             }
//                         }
//                     }
//                 }

//                 aligned_free(Apack);
//             }
//         }
//     }

//     aligned_free(Bpack);
// }

// // ============================================================================
// // Main
// // ============================================================================

// int main(int argc, char** argv) {
//     if (argc < 3) {
//         cerr << "Usage: " << argv[0] << " N num_threads\n";
//         return 1;
//     }
//     int N = atoi(argv[1]);
//     int T = atoi(argv[2]);
//     omp_set_num_threads(T);

//     vector<double> A(N * N), B(N * N), C(N * N);
//     std::mt19937_64 rng(12345);
//     std::normal_distribution<double> dist(0.0, 1.0);

//     for (int i = 0; i < N * N; ++i) {
//         A[i] = dist(rng);
//         B[i] = dist(rng);
//         C[i] = 0.0;
//     }

//     double t0 = omp_get_wtime();
//     CUSTOM_MATMUL(A.data(), B.data(), C.data(), N);
//     double t1 = omp_get_wtime();

//     cout << "N=" << N << " T=" << T << " time=" << (t1 - t0) << " seconds\n";

//     double checksum = 0.0;
//     #pragma omp parallel for reduction(+:checksum)
//     for (int i = 0; i < N * N; ++i) checksum += C[i];
//     cout << "checksum=" << checksum << "\n";
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
#include <fstream>
#include <sstream>

using namespace std;

enum ISA_LEVEL {
    ISA_SCALAR = 0,
    ISA_SSE = 1,
    ISA_AVX2_FMA = 2,
    ISA_AVX512 = 3
};

struct CPUInfo {
    ISA_LEVEL isa_level;
    int L1_cache;
    int L2_cache;
    int L3_cache;
    int cache_line;
    string cpu_name;
};

static CPUInfo detect_cpu() {
    CPUInfo info;
    info.isa_level = ISA_SCALAR;
    info.L1_cache = 32 * 1024;
    info.L2_cache = 256 * 1024;
    info.L3_cache = 8 * 1024 * 1024;
    info.cache_line = 64;
    info.cpu_name = "Unknown";

#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("avx512vl")) {
        info.isa_level = ISA_AVX512;
    } else if (__builtin_cpu_supports("avx2") &&
               __builtin_cpu_supports("fma")) {
        info.isa_level = ISA_AVX2_FMA;
    } else if (__builtin_cpu_supports("sse4.2")) {
        info.isa_level = ISA_SSE;
    }

    ifstream cpuinfo("/proc/cpuinfo");
    string line;
    while (getline(cpuinfo, line)) {
        if (line.find("model name") != string::npos) {
            size_t pos = line.find(":");
            if (pos != string::npos) {
                info.cpu_name = line.substr(pos + 2);
            }
            break;
        }
    }

    auto read_cache_size = [](const char* path) -> int {
        ifstream f(path);
        if (f) {
            string val;
            getline(f, val);
            if (!val.empty() && val.back() == 'K') {
                val.pop_back();
                return stoi(val) * 1024;
            }
            return stoi(val);
        }
        return 0;
    };

    int l1d = read_cache_size("/sys/devices/system/cpu/cpu0/cache/index0/size");
    int l2  = read_cache_size("/sys/devices/system/cpu/cpu0/cache/index2/size");
    int l3  = read_cache_size("/sys/devices/system/cpu/cpu0/cache/index3/size");

    if (l1d > 0) info.L1_cache = l1d;
    if (l2  > 0) info.L2_cache = l2;
    if (l3  > 0) info.L3_cache = l3;

    int line_size = read_cache_size("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");
    if (line_size > 0) info.cache_line = line_size;
#endif
#endif

    return info;
}

struct BlockParams {
    int nc, kc, mc;
    int MR, NR;
};

static BlockParams compute_block_params(const CPUInfo& cpu) {
    BlockParams p;

    switch (cpu.isa_level) {
        case ISA_AVX512:
            p.MR = 8;
            p.NR = 16;
            break;
        case ISA_AVX2_FMA:
            p.MR = 8;
            p.NR = 8;
            break;
        case ISA_SSE:
            p.MR = 4;
            p.NR = 4;
            break;
        default:
            p.MR = 1;
            p.NR = 1;
    }

    int L1_target = (cpu.L1_cache * 6) / 10;
    int kc_from_L1 = L1_target / (int)(sizeof(double) * (p.MR + p.NR));

    int L2_target = (cpu.L2_cache * 6) / 10;
    (void)L2_target; // not used, keep minimal

    int L3_target = (cpu.L3_cache * 8) / 10;
    (void)L3_target; // not used, keep minimal

    if (cpu.L3_cache >= 16 * 1024 * 1024) {
        p.kc = min(512, kc_from_L1);
        p.nc = 4096;
        p.mc = 256;
    } else if (cpu.L3_cache >= 6 * 1024 * 1024) {
        p.kc = min(384, kc_from_L1);
        p.nc = 4096;
        p.mc = 192;
    } else {
        p.kc = min(256, kc_from_L1);
        p.nc = 2048;
        p.mc = 144;
    }

    p.kc = (p.kc / 16) * 16;
    if (p.kc <= 0) p.kc = 16;

    p.mc = (p.mc / p.MR) * p.MR;
    if (p.mc <= 0) p.mc = p.MR;

    p.nc = (p.nc / p.NR) * p.NR;
    if (p.nc <= 0) p.nc = p.NR;

    if (const char* e = getenv("NC")) p.nc = atoi(e);
    if (const char* e = getenv("KC")) p.kc = atoi(e);
    if (const char* e = getenv("MC")) p.mc = atoi(e);

    return p;
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

static void pack_B_avx2(const double* __restrict B,
                        double* __restrict Bp,
                        int N, int k0, int kc, int j0, int jb, int NR)
{
    int full_col_blocks = jb / NR;
    for (int blk = 0; blk < full_col_blocks; ++blk) {
        int j_start = j0 + blk * NR;
        double* out = Bp + blk * (kc * NR);

        for (int p = 0; p < kc; ++p) {
            const double* src = B + (k0 + p) * N + j_start;
            if (p + 8 < kc) {
                _mm_prefetch((const char*)(B + (k0 + p + 8) * N + j_start), _MM_HINT_T0);
            }
            __m256d v0 = _mm256_loadu_pd(src);
            __m256d v1 = _mm256_loadu_pd(src + 4);
            _mm256_store_pd(out + p * NR, v0);
            _mm256_store_pd(out + p * NR + 4, v1);
        }
    }

    int rem = jb - full_col_blocks * NR;
    if (rem) {
        int j_start = j0 + full_col_blocks * NR;
        double* out = Bp + full_col_blocks * (kc * NR);
        for (int p = 0; p < kc; ++p) {
            const double* src = B + (k0 + p) * N + j_start;
            for (int c = 0; c < rem; ++c) out[p * NR + c] = src[c];
            for (int c = rem; c < NR; ++c) out[p * NR + c] = 0.0;
        }
    }
}

static void pack_B_scalar(const double* __restrict B,
                          double* __restrict Bp,
                          int N, int k0, int kc, int j0, int jb, int NR)
{
    int full_col_blocks = jb / NR;
    for (int blk = 0; blk < full_col_blocks; ++blk) {
        int j_start = j0 + blk * NR;
        double* out = Bp + blk * (kc * NR);
        for (int p = 0; p < kc; ++p) {
            const double* src = B + (k0 + p) * N + j_start;
            memcpy(out + p * NR, src, sizeof(double) * NR);
        }
    }

    int rem = jb - full_col_blocks * NR;
    if (rem) {
        int j_start = j0 + full_col_blocks * NR;
        double* out = Bp + full_col_blocks * (kc * NR);
        for (int p = 0; p < kc; ++p) {
            const double* src = B + (k0 + p) * N + j_start;
            for (int c = 0; c < rem; ++c) out[p * NR + c] = src[c];
            for (int c = rem; c < NR; ++c) out[p * NR + c] = 0.0;
        }
    }
}

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
        for (int r = i_rem; r < MR; ++r) dst[r] = 0.0;
    }
}

static inline void dgemm_kernel_8x8_avx2(int kc,
                                         const double* __restrict Ap,
                                         const double* __restrict Bp_block,
                                         double* __restrict C,
                                         int ldc, int ic, int jc)
{
    __m256d acc0[8], acc1[8];

    for (int r = 0; r < 8; ++r) {
        acc0[r] = _mm256_setzero_pd();
        acc1[r] = _mm256_setzero_pd();
    }

    int p = 0;
    for (; p + 3 < kc; p += 4) {
        _mm_prefetch((const char*)(Bp_block + (p + 16) * 8), _MM_HINT_T0);
        _mm_prefetch((const char*)(Ap      + (p + 16) * 8), _MM_HINT_T0);

        const double* arow0 = Ap + p * 8;
        __m256d b0_0 = _mm256_load_pd(Bp_block + p * 8);
        __m256d b1_0 = _mm256_load_pd(Bp_block + p * 8 + 4);

        const double* arow1 = Ap + (p + 1) * 8;
        __m256d b0_1 = _mm256_load_pd(Bp_block + (p + 1) * 8);
        __m256d b1_1 = _mm256_load_pd(Bp_block + (p + 1) * 8 + 4);

        const double* arow2 = Ap + (p + 2) * 8;
        __m256d b0_2 = _mm256_load_pd(Bp_block + (p + 2) * 8);
        __m256d b1_2 = _mm256_load_pd(Bp_block + (p + 2) * 8 + 4);

        const double* arow3 = Ap + (p + 3) * 8;
        __m256d b0_3 = _mm256_load_pd(Bp_block + (p + 3) * 8);
        __m256d b1_3 = _mm256_load_pd(Bp_block + (p + 3) * 8 + 4);

        for (int r = 0; r < 8; ++r) {
            __m256d a0 = _mm256_set1_pd(arow0[r]);
            __m256d a1 = _mm256_set1_pd(arow1[r]);
            __m256d a2 = _mm256_set1_pd(arow2[r]);
            __m256d a3 = _mm256_set1_pd(arow3[r]);

            acc0[r] = _mm256_fmadd_pd(a0, b0_0, acc0[r]);
            acc1[r] = _mm256_fmadd_pd(a0, b1_0, acc1[r]);

            acc0[r] = _mm256_fmadd_pd(a1, b0_1, acc0[r]);
            acc1[r] = _mm256_fmadd_pd(a1, b1_1, acc1[r]);

            acc0[r] = _mm256_fmadd_pd(a2, b0_2, acc0[r]);
            acc1[r] = _mm256_fmadd_pd(a2, b1_2, acc1[r]);

            acc0[r] = _mm256_fmadd_pd(a3, b0_3, acc0[r]);
            acc1[r] = _mm256_fmadd_pd(a3, b1_3, acc1[r]);
        }
    }

    for (; p < kc; ++p) {
        const double* arow = Ap + p * 8;
        __m256d b0 = _mm256_load_pd(Bp_block + p * 8);
        __m256d b1 = _mm256_load_pd(Bp_block + p * 8 + 4);

        for (int r = 0; r < 8; ++r) {
            __m256d a_bcast = _mm256_set1_pd(arow[r]);
            acc0[r] = _mm256_fmadd_pd(a_bcast, b0, acc0[r]);
            acc1[r] = _mm256_fmadd_pd(a_bcast, b1, acc1[r]);
        }
    }

    for (int r = 0; r < 8; ++r) {
        double* cptr = C + (ic + r) * ldc + jc;
        __m256d cold0 = _mm256_loadu_pd(cptr);
        __m256d cold1 = _mm256_loadu_pd(cptr + 4);
        _mm256_storeu_pd(cptr,     _mm256_add_pd(cold0, acc0[r]));
        _mm256_storeu_pd(cptr + 4, _mm256_add_pd(cold1, acc1[r]));
    }
}

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
    static CPUInfo cpu_info = detect_cpu();
    static BlockParams params = compute_block_params(cpu_info);

    static bool info_printed = false;
    if (!info_printed) {
        const char* isa_names[] = {"Scalar", "SSE", "AVX2+FMA", "AVX512"};
        cout << "CPU: " << cpu_info.cpu_name << "\n";
        cout << "ISA: " << isa_names[(int)cpu_info.isa_level] << "\n";
        cout << "Cache line: " << cpu_info.cache_line << " B\n";
        cout << "L1d: " << (cpu_info.L1_cache / 1024) << " KB\n";
        cout << "L2: " << (cpu_info.L2_cache / 1024) << " KB\n";
        cout << "L3: " << (cpu_info.L3_cache / (1024 * 1024)) << " MB\n";
        cout << "Block params: "
             << "MR=" << params.MR << " "
             << "NR=" << params.NR << " "
             << "mc=" << params.mc << " "
             << "kc=" << params.kc << " "
             << "nc=" << params.nc << "\n";
        info_printed = true;
    }

    const int nc = params.nc;
    const int kc = params.kc;
    const int mc = params.mc;
    const int MR = params.MR;
    const int NR = params.NR;

    const bool use_avx2 = (cpu_info.isa_level >= ISA_AVX2_FMA);

    double* Bpack = (double*)aligned_malloc(sizeof(double) * kc * ((nc / NR + 1) * NR));

#pragma omp parallel
    {
        double* Apack = (double*)aligned_malloc(sizeof(double) * kc * MR);

        #pragma omp for schedule(static)
        for (int i = 0; i < N * N; ++i) {
            C[i] = 0.0;
        }

        for (int j0 = 0; j0 < N; j0 += nc) {
            int jb = std::min(nc, N - j0);
            for (int k0 = 0; k0 < N; k0 += kc) {
                int kb = std::min(kc, N - k0);

                #pragma omp single
                {
                    if (use_avx2) {
                        pack_B_avx2(B, Bpack, N, k0, kb, j0, jb, NR);
                    } else {
                        pack_B_scalar(B, Bpack, N, k0, kb, j0, jb, NR);
                    }
                }
                #pragma omp for schedule(dynamic, 1)
                for (int i0 = 0; i0 < N; i0 += mc) {
                    int ib = std::min(mc, N - i0);

                    for (int ic = i0; ic < i0 + ib; ic += MR) {
                        int i_rem = std::min(MR, i0 + ib - ic);

                        if (use_avx2) {
                            pack_A_micro(A, Apack, N, ic, i_rem, k0, kb, MR);
                        }

                        for (int jc = j0; jc < j0 + jb; jc += NR) {
                            int j_rem = std::min(NR, j0 + jb - jc);

                            if (use_avx2 && i_rem == MR && j_rem == NR) {
                                int col_block_index = (jc - j0) / NR;
                                const double* Bpanel_block = Bpack + col_block_index * (kb * NR);
                                dgemm_kernel_8x8_avx2(kb, Apack, Bpanel_block, C, N, ic, jc);
                            } else {
                                dgemm_scalar_edge(A, B, C, N, ic, jc, i_rem, j_rem, k0, kb);
                            }
                        }
                    }
                }
            }
        }

        aligned_free(Apack);
    }

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
        // C is initialized inside CUSTOM_MATMUL
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
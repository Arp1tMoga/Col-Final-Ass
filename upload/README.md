# GEMM Accelerator — Quick Start

Minimal instructions to build, run, and measure this repository.

## Clone
```bash
git clone <your-repo-url>
cd <repo-directory>
```

## Build && Run 
```bash
make clean && make all && ./run.sh baseline 1024 8
```
- 1024 = square matrix size N
- 8 = thread count

## Current best implementations
- mine_ultra_optimized: current most optimized file.
- mine.cpp: contains multiple approaches; the fastest is the 6th strategy (matmul_blocked_simd_omp).

## Measure performance with perf
Example at larger size:
```bash
perf stat ./optimized/gemm_ultra 2048 8
perf stat ./optimized/gemm_mine  2048 8
```
- First argument: matrix size N (e.g., 2048)
- Second argument: thread count (e.g., 8)

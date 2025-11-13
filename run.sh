#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-baseline}
N=${2:-1024}
P=${3:-4}

echo " "
echo "=============================================== "
echo " "

echo "Running baseline python implementation"
python3 baseline/gemm_baseline.py $N $P
echo " "

echo "Running old C++ binary"
    if [ ! -x optimized/gemm_mine ]; then
        echo "Old binary not found. Run make first."
        exit 1
    fi
    optimized/gemm_mine $N $P
    echo " "

echo "Running new C++ binary"
    if [ ! -x optimized/gemm_new ]; then
        echo "Improved binary not found. Run make first."
        exit 1
    fi
    optimized/gemm_new $N $P
    echo " "

echo "Running Optimized C++ binary"
    if [ ! -x optimized/gemm_ultra ]; then
        echo "Optimized binary not found. Run make first."
        exit 1
    fi
    optimized/gemm_ultra $N $P
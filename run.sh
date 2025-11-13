
#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-baseline}
N=${2:-1024}
P=${3:-4}

echo "Running baseline python implementation"
python3 baseline/gemm_baseline.py $N $P
echo " "

    # echo "Running Baseline C++ binary"
    # if [ ! -x optimized/gemm_opt ]; then
    #     echo "Baseline binary not found. Run make first."
    #     exit 1
    # fi
    # optimized/gemm_opt $N $P
    # echo " "

echo "Running old C++ binary"
    if [ ! -x optimized/gemm_mine ]; then
        echo "Old binary not found. Run make first."
        exit 1
    fi
    optimized/gemm_mine $N $P
    echo " "

echo "Running Optimized C++ binary"
    if [ ! -x optimized/gemm_ultra ]; then
        echo "Optimized binary not found. Run make first."
        exit 1
    fi
    optimized/gemm_ultra $N $P

# if [ "$MODE" = "baseline" ]; then
#     echo "Running baseline python implementation"
#     python3 baseline/gemm_baseline.py $N $P
# elif [ "$MODE" = "optimized" ]; then
#     echo "Running optimized binary"
#     if [ ! -x optimized/gemm_opt ]; then
#         echo "Optimized binary not found. Run make first."
#         exit 1
#     fi
#     optimized/gemm_opt $N $P
# else
#     echo "Unknown mode: $MODE"
#     exit 1
# fi

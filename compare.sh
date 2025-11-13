#!/usr/bin/env bash
set -euo pipefail

MATRIX_SIZES=(512 1024 2048 4096)
THREADS=(1 2 4 6 8)
RUNS=10

PYTHON_BASELINE="python3 baseline/gemm_baseline.py"
# CPP_OLD="optimized/gemm_mine"
CPP="optimized/gemm_ultra"

extract_time() {
    echo "$1" | sed -n 's/.*time=\([0-9\.eE+-]*\).*/\1/p'
}

compute_avg() {
    awk '{sum+=$1} END {print sum/NR}'
}

compute_median() {
    sort -n | awk '
        {a[NR]=$1}
        END {
            if (NR%2==1) print a[(NR+1)/2];
            else print (a[NR/2] + a[NR/2+1]) / 2;
        }'
}

compute_std() {
    awk '
        {x[NR]=$1; sum+=$1}
        END {
            mean=sum/NR
            for(i=1;i<=NR;i++){sq+=(x[i]-mean)^2}
            print sqrt(sq/NR)
        }'
}

# ------------------------------------
# Create CSV file header (PROGRAM FIRST)
# ------------------------------------
CSV_FILE="results.csv"
echo "Program,Matrix,Threads,Avg,Median,StdDev" > "$CSV_FILE"

printf "%-10s %-8s %-12s %-12s %-12s %-12s\n" \
    "Matrix" "Threads" "Program" "Avg" "Median" "StdDev"
printf "%-10s %-8s %-12s %-12s %-12s %-12s\n" \
    "------" "-------" "--------" "--------" "--------" "--------"

for N in "${MATRIX_SIZES[@]}"; do
    for P in "${THREADS[@]}"; do
        for PROGRAM in "PYTHON" "CPP"; do

            TIMES=()

            for ((i=1; i<=RUNS; i++)); do
                case "$PROGRAM" in
                    "PYTHON")   RAW=$($PYTHON_BASELINE "$N" "$P") ;;
                    "CPP")  RAW=$($CPP "$N" "$P") ;;
                esac

                T=$(extract_time "$RAW")
                TIMES+=("$T")
            done

            AVG=$(printf "%s\n" "${TIMES[@]}" | compute_avg)
            MED=$(printf "%s\n" "${TIMES[@]}" | compute_median)
            STD=$(printf "%s\n" "${TIMES[@]}" | compute_std)

            printf "%-10s %-8s %-12s %-12.6f %-12.6f %-12.6f\n" \
                "$N" "$P" "$PROGRAM" "$AVG" "$MED" "$STD"

            echo "$PROGRAM,$N,$P,$AVG,$MED,$STD" >> "$CSV_FILE"

        done
    done
done

echo "CSV saved to: $CSV_FILE"

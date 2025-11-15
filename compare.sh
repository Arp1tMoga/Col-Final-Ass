# #!/usr/bin/env bash
# set -euo pipefail

# MATRIX_SIZES=(512 1024 2048 4096)
# THREADS=(1 2 4 6 8)
# RUNS=10

# PYTHON_BASELINE="python3 baseline/gemm_baseline.py"
# # CPP_OLD="optimized/gemm_mine"
# CPP="optimized/gemm_mine"

# extract_time() {
#     echo "$1" | sed -n 's/.*time=\([0-9\.eE+-]*\).*/\1/p'
# }

# compute_avg() {
#     awk '{sum+=$1} END {print sum/NR}'
# }

# compute_median() {
#     sort -n | awk '
#         {a[NR]=$1}
#         END {
#             if (NR%2==1) print a[(NR+1)/2];
#             else print (a[NR/2] + a[NR/2+1]) / 2;
#         }'
# }

# compute_std() {
#     awk '
#         {x[NR]=$1; sum+=$1}
#         END {
#             mean=sum/NR
#             for(i=1;i<=NR;i++){sq+=(x[i]-mean)^2}
#             print sqrt(sq/NR)
#         }'
# }

# # ------------------------------------
# # Create CSV file header (PROGRAM FIRST)
# # ------------------------------------
# CSV_FILE="results.csv"
# echo "Program,Matrix,Threads,Avg,Median,StdDev" > "$CSV_FILE"

# printf "%-10s %-8s %-12s %-12s %-12s %-12s\n" \
#     "Matrix" "Threads" "Program" "Avg" "Median" "StdDev"
# printf "%-10s %-8s %-12s %-12s %-12s %-12s\n" \
#     "------" "-------" "--------" "--------" "--------" "--------"

# for N in "${MATRIX_SIZES[@]}"; do
#     for P in "${THREADS[@]}"; do
#         for PROGRAM in "PYTHON" "CPP"; do

#             TIMES=()

#             for ((i=1; i<=RUNS; i++)); do
#                 case "$PROGRAM" in
#                     "PYTHON")   RAW=$($PYTHON_BASELINE "$N" "$P") ;;
#                     "CPP")  RAW=$($CPP "$N" "$P") ;;
#                 esac

#                 T=$(extract_time "$RAW")
#                 TIMES+=("$T")
#             done

#             AVG=$(printf "%s\n" "${TIMES[@]}" | compute_avg)
#             MED=$(printf "%s\n" "${TIMES[@]}" | compute_median)
#             STD=$(printf "%s\n" "${TIMES[@]}" | compute_std)

#             printf "%-10s %-8s %-12s %-12.6f %-12.6f %-12.6f\n" \
#                 "$N" "$P" "$PROGRAM" "$AVG" "$MED" "$STD"

#             echo "$PROGRAM,$N,$P,$AVG,$MED,$STD" >> "$CSV_FILE"

#         done
#     done
# done

# echo "CSV saved to: $CSV_FILE"











#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
MATRIX_SIZES=(512 1024 2048 4096)
THREADS=(1 2 4 6 8)
RUNS=10 # Number of runs for each configuration to calculate averages

# --- Programs to Benchmark ---
PYTHON_BASELINE="python3 baseline/gemm_baseline.py"
CPP="optimized/gemm_mine"

# --- Output File ---
CSV_FILE="results_wall_time.csv"

# ==============================================================================
# Helper functions for statistical analysis
# ==============================================================================

compute_avg() {
    awk '{sum+=$1} END {if (NR > 0) print sum/NR; else print 0}'
}

compute_median() {
    sort -n | awk '
        { a[NR]=$1 }
        END {
            if (NR == 0) { print 0; exit }
            if (NR % 2 == 1) {
                print a[(NR+1)/2]
            } else {
                print (a[NR/2] + a[NR/2+1]) / 2
            }
        }'
}

compute_std() {
    awk '
        { x[NR]=$1; sum+=$1 }
        END {
            if (NR == 0) { print 0; exit }
            mean = sum / NR
            for(i=1; i<=NR; i++) {
                sq_diff += (x[i] - mean)^2
            }
            print sqrt(sq_diff / NR)
        }'
}

# ==============================================================================
# Main benchmarking logic
# ==============================================================================

# Check for bc dependency
if ! command -v bc &> /dev/null
then
    echo "Error: 'bc' (basic calculator) is not installed. Please install it to run this script."
    echo "On Debian/Ubuntu: sudo apt-get install bc"
    echo "On RHEL/CentOS: sudo yum install bc"
    exit 1
fi


# Create CSV file header
echo "Program,Matrix,Threads,Avg,Median,StdDev" > "$CSV_FILE"

# Print table header to the console
printf "%-10s %-8s %-12s %-12s %-12s %-12s\n" \
    "Matrix" "Threads" "Program" "Avg" "Median" "StdDev"
printf -- '-%.0s' {1..70} && printf "\n"

# Start the main loop
for N in "${MATRIX_SIZES[@]}"; do
    for P in "${THREADS[@]}"; do
        for PROGRAM in "PYTHON" "CPP"; do

            TIMES=() # Array to store the execution times for the current configuration

            # Run the benchmark multiple times
            for ((i=1; i<=RUNS; i++)); do
                
                # Use date for high-precision start/end timestamps
                start_time=$(date +%s.%N)

                case "$PROGRAM" in
                    "PYTHON")
                        $PYTHON_BASELINE "$N" "$P" >/dev/null 2>&1
                        ;;
                    "CPP")
                        $CPP "$N" "$P" >/dev/null 2>&1
                        ;;
                esac
                
                end_time=$(date +%s.%N)

                # Use bc for floating-point arithmetic to get the duration
                T=$(echo "$end_time - $start_time" | bc -l)

                TIMES+=("$T") # Add the measured time to our list
            done

            # Calculate statistics
            AVG=$(printf "%s\n" "${TIMES[@]}" | compute_avg)
            MED=$(printf "%s\n" "${TIMES[@]}" | compute_median)
            STD=$(printf "%s\n" "${TIMES[@]}" | compute_std)

            # Print results to the console
            printf "%-10s %-8s %-12s %-12.6f %-12.6f %-12.6f\n" \
                "$N" "$P" "$PROGRAM" "$AVG" "$MED" "$STD"

            # Append results to the CSV file
            echo "$PROGRAM,$N,$P,$AVG,$MED,$STD" >> "$CSV_FILE"

        done
    done
done

printf -- '-%.0s' {1..70} && printf "\n"
echo "Benchmark complete. Results saved to: $CSV_FILE"
#!/bin/bash
# =========================================================
# run_scaling.sh  --  MPI Traffic Simulation Scalability Test
# Group 10
#
# Compiles traffic_mpi.c, then runs it with np = 1, 2, 4, 8
# processes, collecting timing data into mpi_performance.txt.
#
# Usage:
#   bash run_scaling.sh
#   bash run_scaling.sh --no-compile    (skip recompilation)
# =========================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${SCRIPT_DIR}/traffic_mpi.c"
BINARY="${SCRIPT_DIR}/traffic_mpi"
PERF_FILE="${SCRIPT_DIR}/mpi_performance.txt"

NP_LIST="1 2 4 8"

# ---- Compiler and flags ----
MPI_CC="mpicc"
CFLAGS="-O2"
LIBS="-lm"

echo "==================================================="
echo "  MPI Traffic Simulation - Scalability Test"
echo "  Group 10"
echo "==================================================="
echo ""

# ---- Compile (unless --no-compile is passed) ----
if [[ "$1" != "--no-compile" ]]; then
    echo "[Compiling traffic_mpi.c ...]"
    ${MPI_CC} ${CFLAGS} -o "${BINARY}" "${SRC}" ${LIBS}
    echo "  OK -> ${BINARY}"
    echo ""
fi

# ---- Verify binary exists ----
if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: Binary not found: ${BINARY}"
    echo "       Run without --no-compile to build first."
    exit 1
fi

# ---- Write performance file header ----
# (Each mpirun invocation appends one data row via append_perf_line())
cat > "${PERF_FILE}" << 'HEADER'
=================================================
  MPI Traffic Simulation - Performance Results
  Group 10
=================================================
Grid: 200x200  Lanes: 3  Time Steps: 200

  Procs    Time(s)        Min      Max      Avg     
  ---------------------------------------------------
HEADER

echo "[Performance file initialised: ${PERF_FILE}]"
echo ""

# ---- Run scaling tests ----
echo "[Running scaling tests ...]"
echo ""

for np in ${NP_LIST}; do
    echo "---------------------------------------------------"
    echo "  mpirun -np ${np} ./traffic_mpi"
    echo "---------------------------------------------------"
    mpirun --oversubscribe -np "${np}" "${BINARY}"
    echo ""
done

# ---- Append speedup/efficiency footer (computed from logged times) ----
# Read data rows from the performance file into an array BEFORE appending,
# to avoid the read-and-write-to-same-file infinite loop.
mapfile -t PERF_LINES < "${PERF_FILE}"

{
    echo ""
    echo "  Speedup and Efficiency (relative to np=1):"
    echo ""
    echo "  Procs    Speedup    Efficiency"
    echo "  --------------------------------"
} >> "${PERF_FILE}"

BASE_TIME=""
for line in "${PERF_LINES[@]}"; do
    # Match data rows like "  1        0.019810  ..."
    if [[ "$line" =~ ^[[:space:]]+([0-9]+)[[:space:]]+([0-9]+\.[0-9]+) ]]; then
        np_val="${BASH_REMATCH[1]}"
        t="${BASH_REMATCH[2]}"
        if [[ -z "$BASE_TIME" ]]; then
            BASE_TIME="$t"
        fi
        awk -v np="$np_val" -v t="$t" -v base="$BASE_TIME" \
            'BEGIN {
                speedup    = base / t;
                efficiency = speedup / np;
                printf "  %-8d %-10.4f %-10.4f (%.2f%%)\n",
                       np, speedup, efficiency, efficiency*100;
            }' >> "${PERF_FILE}"
    fi
done

# ---- Display final performance file ----
echo "==================================================="
echo "  Performance Summary"
echo "==================================================="
cat "${PERF_FILE}"
echo ""
echo "==================================================="
echo "  Scaling tests complete."
echo "  Performance log : ${PERF_FILE}"
echo "  Run analysis_mpi.py to generate plots."
echo "==================================================="

#!/bin/bash
# =============================================================
# run_scaling_hybrid.sh  — Hybrid MPI+OpenMP Scaling Test
# Group 10 | EE7218/EC7207 High Performance Computing
#
# Tests all combinations of MPI processes and OpenMP threads.
# Total workers = MPI_procs × OMP_threads.
#
# Usage: bash run_scaling_hybrid.sh
# =============================================================

cd "$(dirname "$0")"

echo ""
echo "=========================================================="
echo "  Hybrid MPI+OpenMP Traffic Simulation — Scaling Test"
echo "  Group 10"
echo "=========================================================="

# ── Compile ──────────────────────────────────────────────────
echo ""
echo "[Compiling] traffic_hybrid.c ..."
mpicc -O2 -fopenmp -o traffic_hybrid traffic_hybrid.c -lm

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi
echo "  Compilation successful."

# ── Write performance file header ────────────────────────────
cat > hybrid_performance.txt << 'EOF'
=================================================
  Hybrid MPI+OpenMP Traffic Simulation - Performance Results
  Group 10
=================================================
Grid: 200x200  Lanes: 3  Time Steps: 200

  Procs    Threads  Total    Time(s)        Min      Max      Avg
  --------------------------------------------------------------------------
EOF

echo ""
echo "[Scaling Configurations]"
echo "  Configs: (MPI procs) x (OMP threads) = total workers"
echo ""

# ── Test configurations ───────────────────────────────────────
# Format: "np nt"   (MPI processes, OMP threads per process)
CONFIGS=(
    "1 1"
    "1 2"
    "1 4"
    "2 1"
    "2 2"
    "2 4"
    "4 1"
    "4 2"
)

for config in "${CONFIGS[@]}"; do
    np=$(echo "$config" | awk '{print $1}')
    nt=$(echo "$config" | awk '{print $2}')
    total=$(( np * nt ))
    echo "  Running: MPI=$np × OMP=$nt = $total total workers ..."
    export OMP_NUM_THREADS=$nt
    mpirun --oversubscribe -np "$np" ./traffic_hybrid "$nt"
done

echo ""
echo "=========================================================="
echo "  All configurations complete."
echo "  Results written to: hybrid_performance.txt"
echo "=========================================================="
echo ""

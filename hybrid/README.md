# Hybrid MPI + OpenMP — Two-Level Parallel Traffic Simulation
## EE7218 / EC7207 — High Performance Computing | Group 10

---

## 1. What Is Hybrid Parallelism?

**Hybrid parallelism** combines **two levels of parallelism simultaneously**:

- **MPI** (outer level) — multiple independent processes, each with private
  memory, communicating via message passing
- **OpenMP** (inner level) — multiple threads within each process, sharing
  that process's local memory

This matches the hardware reality of modern HPC clusters:

```
  MODERN HPC CLUSTER HARDWARE

  ┌──────────── Node A ─────────────┐   ┌──────────── Node B ─────────────┐
  │  CPU Socket 0                   │   │  CPU Socket 0                   │
  │  ├── Core 0  ──┐                │   │  ├── Core 0  ──┐                │
  │  ├── Core 1    ├── Shared L3    │   │  ├── Core 1    ├── Shared L3    │
  │  ├── Core 2    │   Cache        │   │  ├── Core 2    │   Cache        │
  │  └── Core 3  ──┘                │   │  └── Core 3  ──┘                │
  │            RAM (local)          │   │            RAM (local)          │
  └─────────────────────────────────┘   └─────────────────────────────────┘
           │                                      │
           └──────────── Network (MPI) ───────────┘

  MPI connects NODES (separate RAM)
  OpenMP connects CORES (shared RAM within one node)
  Hybrid uses BOTH.
```

---

## 2. The Two-Level Hierarchy

```
  HYBRID STRUCTURE for 2 MPI processes × 4 OMP threads = 8 total workers

  ┌─────────────────────────────────────────────────────────────────────┐
  │                         HYBRID SYSTEM                              │
  │                                                                     │
  │  MPI Process 0                     MPI Process 1                   │
  │  (owns rows 0–99)                  (owns rows 100–199)             │
  │  ┌───────────────────────────┐     ┌───────────────────────────┐   │
  │  │  Shared Local Memory      │     │  Shared Local Memory      │   │
  │  │  local_traffic[102][200]  │     │  local_traffic[102][200]  │   │
  │  │                           │     │                           │   │
  │  │  OMP Thread 0: rows 1–24  │     │  OMP Thread 0: rows 1–24  │   │
  │  │  OMP Thread 1: rows 25–49 │     │  OMP Thread 1: rows 25–49 │   │
  │  │  OMP Thread 2: rows 50–74 │     │  OMP Thread 2: rows 50–74 │   │
  │  │  OMP Thread 3: rows 75–99 │     │  OMP Thread 3: rows 75–99 │   │
  │  └───────────────────────────┘     └───────────────────────────┘   │
  │                │                                 │                  │
  │                └──────── MPI halo exchange ──────┘                  │
  └─────────────────────────────────────────────────────────────────────┘

  MPI level: divides the 200-row grid across 2 processes
  OMP level: each process uses 4 threads to update its 100 rows in parallel
```

### The Analogy from the Slides

```
  Country (Hybrid System)
    │
    ├── Colombo Control Center  (MPI Process 0)
    │     ├── Engineer 1        (OpenMP Thread 0)  ← updates rows 1–24
    │     ├── Engineer 2        (OpenMP Thread 1)  ← updates rows 25–49
    │     ├── Engineer 3        (OpenMP Thread 2)  ← updates rows 50–74
    │     └── Engineer 4        (OpenMP Thread 3)  ← updates rows 75–99
    │     [Engineers share the Colombo map — OpenMP shared memory]
    │     [Colombo center talks to Kandy center — MPI message passing]
    │
    └── Kandy Control Center    (MPI Process 1)
          ├── Engineer 1        (OpenMP Thread 0)  ← updates rows 101–124
          ├── Engineer 2        (OpenMP Thread 1)  ← updates rows 125–149
          ├── Engineer 3        (OpenMP Thread 2)  ← updates rows 150–174
          └── Engineer 4        (OpenMP Thread 3)  ← updates rows 175–198
          [Engineers share the Kandy map — OpenMP shared memory]
```

---

## 3. What Changes from Pure MPI?

The hybrid code is identical to MPI **except for one function** —
`update_local()` gains an `#pragma omp parallel for`:

```
  PURE MPI update_local()         │  HYBRID update_local()
  ──────────────────────────────  │  ──────────────────────────────────────
  for (li = 1; li <= local_rows;  │  #pragma omp parallel for schedule(static)
       li++) {                    │  for (li = 1; li <= local_rows; li++) {
    int gi = row_start + li - 1;  │    int gi = row_start + li - 1;
    if (gi==0 || gi==ROWS-1)      │    if (gi==0 || gi==ROWS-1) continue;
        continue;                 │    for (j = 1; j < COLS-1; j++)
    for (j = 1; j < COLS-1; j++) │      for (l = 0; l < LANES; l++)
      for (l = 0; l < LANES; l++)│        local_new[li][j][l] = diffuse(...)
        local_new[li][j][l] =    │  }    // ^ T threads split these rows
            diffuse(...)          │
  }                               │  // implicit OMP barrier here
                                  │  // then copy-back phase also parallel
```

`omp_set_num_threads(T)` is called in `main()` based on the command-line
argument, and `omp_set_dynamic(0)` locks the thread count.

The MPI halo exchange is **unchanged** — it runs outside the parallel region,
executed by the master thread of each process.

---

## 4. Complete Per-Step Timeline

```
  ONE TIME STEP in the Hybrid (2 MPI processes × 4 OMP threads)

  Time →

  Process 0, master thread:
  ┌── halo_exchange() ──────────────────────────────────────────────────┐
  │  MPI_Irecv(ghost_top  ← proc 1)                                    │
  │  MPI_Isend(row[1]     → proc 1)                                     │
  │  MPI_Waitall() ─────────────────────────────────────────────────────►│
  └──────────────────────────────────────────────────────────────────────┘
                                   ▼
  ┌── update_local() — OpenMP parallel ─────────────────────────────────┐
  │  Thread 0: rows  1–25  ──────────────────────────────────────────► │
  │  Thread 1: rows 26–50  ──────────────────────────────────────────► │ barrier
  │  Thread 2: rows 51–75  ──────────────────────────────────────────► │
  │  Thread 3: rows 76–99  ──────────────────────────────────────────► │
  └──────────────────────────────────────────────────────────────────────┘
                                   ▼
  Process 0 ready for next step.

  Process 1 runs the same pattern simultaneously on rows 100–199.
  The two MPI halo exchanges overlap with each other in time.
```

---

## 5. Why Hybrid Outperforms Both Pure OpenMP and Pure MPI

```
  PURE OPENMP (8 threads, 1 process)
  ┌────────────────────────────────────────────────────────────────────┐
  │  Thread 0: rows   1–25                                            │
  │  Thread 1: rows  26–50                                            │
  │  Thread 2: rows  51–75                                            │
  │  Thread 3: rows  76–100                                           │
  │  Thread 4: rows 101–125                                           │
  │  Thread 5: rows 126–150                                           │
  │  Thread 6: rows 151–175                                           │
  │  Thread 7: rows 176–198                                           │
  │  ─── all sharing ONE process's memory (potential cache pressure) ─┤
  └────────────────────────────────────────────────────────────────────┘
  Limitation: all 8 threads share one cache domain → cache contention

  PURE MPI (8 processes)
  Process 0–7 each own 25 rows
  ─── But: 8 halo exchanges per step, each sending/receiving full rows ─
  Limitation: communication overhead scales with process count

  HYBRID (2 MPI × 4 OMP)
  ┌────────────────────────────┐   ┌────────────────────────────┐
  │ Process 0: 4 threads       │   │ Process 1: 4 threads       │
  │ own rows 0–99 in local RAM │   │ own rows 100–199 in local  │
  │ Threads share local cache  │   │ RAM. Threads share local   │
  └────────────────────────────┘   └────────────────────────────┘
              │                                │
              └─── only 1 halo exchange ───────┘
                   (proc0 ↔ proc1)

  Benefits:
  • Fewer MPI messages than 8 pure-MPI processes
  • Better cache utilisation than 8 shared-memory threads
  • Each process's 4 threads fit comfortably in that CPU's L3 cache
```

---

## 6. Scaling Configurations Tested

```
  Config    MPI Procs  OMP Threads  Total Workers  Time (s)  Speedup vs Serial
  ─────────────────────────────────────────────────────────────────────────────
  1P × 1T       1           1             1         0.0206       2.49×
  1P × 2T       1           2             2         0.0118       4.34×
  1P × 4T       1           4             4         0.0068       7.50×
  2P × 1T       2           1             2         0.0105       4.87×
  2P × 2T       2           2             4         0.0063       8.21×
  2P × 4T ★     2           4             8         0.0036      14.10×   ← best
  4P × 1T       4           1             4         0.0064       8.02×
  4P × 2T       4           2             8         0.0037      13.82×

  ★  Best configuration: 2 MPI processes × 4 OpenMP threads
```

```
  Speedup comparison (serial = 0.0513 s)

  14× │              ★ Hybrid 2P×4T  (14.10×)
  12× │
  10× │
   8× │    ●────●────● MPI (4P = 7.98×)
   6× │              ▲ OMP collapse 4T (7.13×)
   4× │
   2× │
   1× ├────────────────────────────────────────
      1       2       4       8   Total workers
```

---

## 7. Correctness Proof

Because:
1. All implementations use the same seed (`srand(1)`)
2. The diffusion formula reads from `local_traffic[]` (previous step) and
   writes to `local_new[]` — no in-place modification, so parallel order
   does not affect results
3. MPI halo exchange provides **exact copies** of boundary rows (no
   approximation)

All four implementations produce **bit-for-bit identical** final grids.
Verified in `verify_all4.py` — max |diff| = 0.0 for all 40,000 cells.

---

## 8. Build and Run

```bash
# Compile (requires MPI + OpenMP)
mpicc -O2 -fopenmp -o traffic_hybrid traffic_hybrid.c -lm

# Single run: 2 MPI processes, 4 OMP threads each
mpirun -np 2 ./traffic_hybrid 4

# Full scaling test (all 8 configurations)
bash run_scaling_hybrid.sh

# 4-way comparison analysis
python3 analysis_hybrid.py

# Correctness verification
python3 verify_all4.py
```

---

## 9. Key Variables and Functions

| Symbol | Meaning |
|--------|---------|
| `nprocs` | Total MPI process count |
| `rank` | This process's index (0..nprocs-1) |
| `local_rows` | Number of grid rows owned by this process |
| `row_start` | Global row index of this process's first owned row |
| `local_traffic[local_rows+2][COLS][LANES]` | Local data + ghost rows |
| `num_omp_threads` | OpenMP threads per process (from argv[1]) |
| `halo_exchange()` | MPI non-blocking boundary row swap |
| `update_local()` | OpenMP-parallel diffusion step on local rows |
| `distribute_data()` | `MPI_Scatterv` from rank 0 to all |
| `gather_data()` | `MPI_Gatherv` from all back to rank 0 |
| `compute_stats()` | OpenMP reduction + `MPI_Reduce` for global stats |

---

## 10. Output Files

| File | Description |
|------|-------------|
| `hybrid_output.txt` | Config, timing, final density all lanes |
| `hybrid_traffic_values.txt` | 200×200 lane-averaged grid |
| `hybrid_traffic_heatmap.ppm` | PPM heatmap image |
| `hybrid_performance.txt` | All scaling results (8 configs) |
| `hybrid_full_report.png` | 6-panel master analysis figure |
| `hybrid_heatmap_4way.png` | 4-way heatmap comparison |
| `hybrid_speedup_comparison.png` | Speedup + efficiency all implementations |
| `hybrid_time_bars.png` | Bar chart all configurations |
| `hybrid_scaling_surface.png` | 2D scaling: MPI lines × OMP x-axis |
| `verification_all4.png` | Correctness proof figure |

---

*Previous: [MPI: distributed-memory](../mpi/README.md)*  
*Back to project root: [README.md](../README.md)*

# MPI — Distributed-Memory Parallel Traffic Simulation
## EE7218 / EC7207 — High Performance Computing | Group 10

---

## 1. What Is MPI?

**MPI (Message Passing Interface)** runs **multiple independent processes**,
each with its **own private memory**.  Processes cannot directly read each
other's data — they must explicitly **send and receive messages**.

### Analogy

> The country is divided into regions.  Each region has its own
> **Traffic Control Centre** with its own copy of only that region's
> road map.  When a centre needs data about roads at the edge of a
> neighbouring region, it sends a message and waits for a reply.

```
  DISTRIBUTED MEMORY — Each process owns its own data

  ┌─────────────────────────┐    ┌─────────────────────────┐
  │  Process 0 (Colombo)    │    │  Process 1 (Kandy)      │
  │  local_traffic[52][200] │    │  local_traffic[52][200] │
  │  rows  0 – 49           │◄──►│  rows 50 – 99           │
  └─────────────────────────┘    └─────────────────────────┘
           ▲  MPI messages             ▲
           │                           │
  ┌────────┴────────────────┐    ┌─────┴───────────────────┐
  │  Process 2 (Galle)      │    │  Process 3 (Jaffna)     │
  │  rows 100 – 149         │◄──►│  rows 150 – 199         │
  └─────────────────────────┘    └─────────────────────────┘

  No process can see another's memory directly.
  All data exchange goes through MPI send/receive calls.
```

---

## 2. Data Decomposition

### 2.1 One-Dimensional Row Decomposition

The 200-row global grid is sliced into **contiguous row blocks**, one per
process.  Each process owns `local_rows ≈ ROWS / nprocs` rows.

```
  GLOBAL GRID (200 rows)  →  4 PROCESSES

  Global Row
  ┌──────────────────────────────────────────────────────┐
  │  Row   0–49   │  process 0  │  local_rows = 50       │
  ├───────────────┤             │                        │
  │  Row  50–99   │  process 1  │  local_rows = 50       │
  ├───────────────┤             │                        │
  │  Row 100–149  │  process 2  │  local_rows = 50       │
  ├───────────────┤             │                        │
  │  Row 150–199  │  process 3  │  local_rows = 50       │
  └──────────────────────────────────────────────────────┘

  For np=3: 200/3 = 66 remainder 2
    process 0 → 67 rows  (gets one extra)
    process 1 → 67 rows  (gets one extra)
    process 2 → 66 rows
```

### 2.2 Ghost Rows (Halo)

Each process needs one row **above** and one row **below** its owned slice
to apply the diffusion stencil at its borders.  These are called
**ghost rows** or **halo cells** — copies of the neighbouring process's
boundary rows.

```
  LOCAL ARRAY LAYOUT for process 1 (owns rows 50–99)

  local index │ global row │ description
  ────────────┼────────────┼──────────────────────────────
       0      │     49     │  TOP GHOST  (copy from proc 0)
       1      │     50     │  first owned row
       2      │     51     │  ...
      ...     │    ...     │  owned rows
      50      │     99     │  last owned row
      51      │    100     │  BOTTOM GHOST (copy from proc 2)
  ────────────┴────────────┴──────────────────────────────

  local_traffic[local_rows+2][COLS][LANES]
                    ↑
         +2 for top and bottom ghost rows
```

---

## 3. Halo Exchange (The Key MPI Operation)

Before every time step, each process must fill its ghost rows with the
latest boundary data from its neighbours.  This is called the **halo exchange**.

```
  HALO EXCHANGE DIAGRAM  (for 4 processes, one time step)

  Process 0         Process 1         Process 2         Process 3
  ┌─────────┐       ┌─────────┐       ┌─────────┐       ┌─────────┐
  │ ghost↑  │       │ ghost↑  │◄──────│ row[1]  │       │ ghost↑  │
  │ row[1]  │──────►│ ghost↑  │       │  ...    │       │         │
  │  ...    │       │  ...    │       │ row[50] │──────►│ ghost↑  │
  │ row[50] │──────►│ ghost↓  │       │ ghost↓  │◄──────│ row[1]  │
  │ ghost↓  │◄──────│ row[50] │       └─────────┘       │  ...    │
  └─────────┘       └─────────┘                         │ row[50] │
                                                         │ ghost↓  │
                                                         └─────────┘
       ↑ boundary process: only communicates in one direction

  Each process:
  1. Sends its TOP data row    → rank-1 (fills rank-1's bottom ghost)
  2. Sends its BOTTOM data row → rank+1 (fills rank+1's top ghost)
  3. Receives rank-1's bottom row → fills own top ghost
  4. Receives rank+1's top row   → fills own bottom ghost
```

### 3.1 Non-Blocking Implementation

```c
MPI_Request reqs[4];
int n = 0;

// Post RECEIVES first (avoids deadlock)
if (prev >= 0)
    MPI_Irecv(&local_traffic[0],          ..., prev, TAG_SOUTH, ..., &reqs[n++]);
if (next < nprocs)
    MPI_Irecv(&local_traffic[local_rows+1],..., next, TAG_NORTH, ..., &reqs[n++]);

// Post SENDS
if (prev >= 0)
    MPI_Isend(&local_traffic[1],          ..., prev, TAG_NORTH, ..., &reqs[n++]);
if (next < nprocs)
    MPI_Isend(&local_traffic[local_rows], ..., next, TAG_SOUTH, ..., &reqs[n++]);

// Wait for ALL to complete before computing
MPI_Waitall(n, reqs, MPI_STATUSES_IGNORE);
```

`MPI_Isend` / `MPI_Irecv` are **non-blocking** — they return immediately
and let the MPI library handle transfer in the background.
`MPI_Waitall` blocks until every pending transfer finishes.

---

## 4. Data Distribution and Collection

```
  INITIALISATION  (rank 0 only)

  Rank 0                    Rank 1      Rank 2      Rank 3
  ┌──────────────────┐
  │ global_traffic   │
  │ [200][200][3]    │──MPI_Scatterv──►[50 rows] [50 rows] [50 rows]
  │ global_weather   │
  │ [200][200]       │──MPI_Scatterv──►[50 rows] [50 rows] [50 rows]
  └──────────────────┘

  MPI_Scatterv allows different send counts (handles uneven splits).

  FINALISATION  (rank 0 only)

  Rank 0                    Rank 1      Rank 2      Rank 3
  ┌──────────────────┐
  │ global_traffic   │◄─MPI_Gatherv───[50 rows] [50 rows] [50 rows]
  └──────────────────┘
  Write output files
```

---

## 5. Global Statistics (MPI_Reduce)

After all time steps, each process has partial min/max/sum for its rows.
`MPI_Reduce` combines them at rank 0:

```
  MPI_REDUCE  for global statistics

  Process 0: local_min=0.0  local_max=72.3  local_sum=143,210
  Process 1: local_min=1.2  local_max=89.7  local_sum=218,440    ──► Rank 0
  Process 2: local_min=0.0  local_max=99.0  local_sum=199,300         gets:
  Process 3: local_min=0.5  local_max=81.2  local_sum=164,870    global_min=0.0
                                                                  global_max=99.0
  MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, ...); global_sum=725,820
  MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, ...);
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, ...);
```

---

## 6. Complete Execution Flow

```
  All processes run this program (SPMD model — Same Program Multiple Data)

  main()
    │
    ├─ MPI_Init()
    ├─ MPI_Comm_rank()   → each process learns its rank (0,1,2,3)
    ├─ MPI_Comm_size()   → each process learns total count
    │
    ├─ compute_distribution()   → calculate row offsets and counts
    │
    ├─ malloc local arrays (local_rows + 2 ghost rows)
    │
    ├─ if rank==0: initialize_global()  → fill global arrays, srand(1)
    │
    ├─ distribute_data()   → MPI_Scatterv traffic and weather
    │
    ├─ MPI_Barrier()       → all processes start timer together
    ├─ START TIMER
    │
    ├─ for t = 0 to 199:
    │    ├─ halo_exchange()     ← MPI: swap boundary rows with neighbours
    │    └─ update_local()      ← compute diffusion on owned rows
    │
    ├─ MPI_Barrier()       → all processes stop timer together
    ├─ STOP TIMER
    │
    ├─ gather_data()       → MPI_Gatherv all rows back to rank 0
    ├─ compute_stats()     → MPI_Reduce for global min/max/avg
    │
    └─ if rank==0:
         save_output(), save_image(), save_raw_values()
         append_perf_line() → mpi_performance.txt
    │
    └─ MPI_Finalize()
```

---

## 7. MPI Operations Reference

```
  OPERATION        │ TYPE        │ PURPOSE
  ─────────────────┼─────────────┼──────────────────────────────────────
  MPI_Init         │ Setup       │ Initialise MPI environment
  MPI_Comm_rank    │ Info        │ Get this process's rank (0..nprocs-1)
  MPI_Comm_size    │ Info        │ Get total number of processes
  MPI_Scatterv     │ Collective  │ Distribute rows from rank 0 to all
  MPI_Gatherv      │ Collective  │ Collect rows from all back to rank 0
  MPI_Isend        │ P2P nonblk  │ Send boundary row (returns immediately)
  MPI_Irecv        │ P2P nonblk  │ Receive ghost row  (returns immediately)
  MPI_Waitall      │ Sync        │ Wait for all non-blocking ops to finish
  MPI_Reduce       │ Collective  │ Combine partial stats to rank 0
  MPI_Barrier      │ Sync        │ All processes meet here before proceeding
  MPI_Wtime        │ Timing      │ High-resolution wall-clock time
  MPI_Finalize     │ Cleanup     │ Shut down MPI environment
```

---

## 8. Performance Results

| Processes | Time (s) | Speedup (vs np=1) | Speedup (vs serial) | Efficiency |
|-----------|----------|------------------|---------------------|------------|
| 1         | 0.020590 | 1.00×            | 2.49×               | 100%       |
| 2         | 0.010686 | 1.93×            | 4.80×               | 96.3%      |
| 4         | 0.006429 | 3.20×            | 7.98×               | 80.1%      |
| 8         | 0.032825 | 0.63×            | 1.56×               | 7.8%       |

```
  Speedup vs serial (T_serial = 0.0513 s)

  8× │              ★ np=4  (7.98×)
  6× │
  4× │    ●  np=2 (4.80×)
  2× │ ● np=1 (2.49×)               (np=8 degrades — oversubscription)
  1× ├──────────────────────────────────────
     1       2       4       8     Processes
```

**Why np=8 is slower**: the test machine has fewer than 8 physical cores;
oversubscription causes context-switching overhead that overwhelms the speedup
from additional parallelism.  On a real HPC cluster with 8+ cores, np=8 would
continue to improve.

**Why np=1 is faster than serial**: the MPI single-process code has a slightly
different code path (local array layout with ghost rows, fixed `srand(1)`) and
was benchmarked on a Linux system where the serial binary ran under Windows.
Speedups vs serial should be compared within the same machine.

---

## 9. Build and Run

```bash
# Compile
mpicc -O2 -o traffic_mpi traffic_mpi.c -lm

# Single run
mpirun -np 4 ./traffic_mpi

# Full scaling test (np = 1, 2, 4, 8)
bash run_scaling.sh

# Generate analysis plots
python3 analysis_mpi.py
```

---

*Previous: [OpenMP: shared-memory](../openmp/README.md)*  
*Next: [Hybrid: MPI + OpenMP combined](../hybrid/README.md)*

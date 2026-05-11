# OpenMP — Shared-Memory Parallel Traffic Simulation
## EE7218 / EC7207 — High Performance Computing | Group 10

---

## 1. What Is OpenMP?

**OpenMP (Open Multi-Processing)** adds parallelism within a single process
by spawning multiple *threads* that **share the same memory**.  A single
`#pragma omp parallel for` directive tells the compiler to split a loop
across all available CPU cores simultaneously.

### Analogy

> Instead of one traffic engineer working alone, the control centre hires
> **T engineers** who all work on the **same city map**.  Each engineer is
> assigned a different set of road rows and updates them in parallel.
> They do not send messages — they simply read/write the shared map.

```
  ONE CONTROL CENTER (shared memory)
  ┌────────────────────────────────────────────────────────┐
  │                   Shared Memory                        │
  │  traffic[200][200][3]   new_traffic[200][200][3]       │
  │                                                        │
  │  Thread 0  → rows 1–49    (updates cells independently)│
  │  Thread 1  → rows 50–99                                │
  │  Thread 2  → rows 100–149                              │
  │  Thread 3  → rows 150–198                              │
  │                                                        │
  │  All threads finish → implicit barrier → next step     │
  └────────────────────────────────────────────────────────┘
```

---

## 2. How OpenMP Parallelises the Serial Code

### 2.1 The Parallelised Loop

The serial nested loop becomes one pragma:

```
  SERIAL                            │  OPENMP
  ──────────────────────────────    │  ──────────────────────────────
  for (i = 1; i < ROWS-1; i++) {   │  #pragma omp parallel for
    for (j = 1; j < COLS-1; j++) { │          schedule(static)
      for (l = 0; l < LANES; l++) {│  for (i = 1; i < ROWS-1; i++) {
        // compute                  │    for (j = ...) {
      }                             │      // compute — unchanged
    }                               │    }
  }                                 │  }
```

Only the **outermost row loop** is parallelised.  The inner `j` and `l` loops
run serially within each thread, keeping data access patterns cache-friendly.

The weather modifier grid has two hazard zones:
- **Rain zone** (centre third): `weather = 0.8` — 20% density reduction
- **Accident zone** (bottom-right corner, rows > 3*ROWS/4, cols > 3*COLS/4): `weather = 0.5` — 50% density reduction

### 2.2 Thread Work Distribution (Static Schedule)

```
  198 interior rows shared across T threads (static = equal chunks)

  T=1  ┌────────────────────────── rows 1–198 ──────────────────────────┐
       │ Thread 0                                                        │
       └─────────────────────────────────────────────────────────────────┘

  T=2  ┌──────────── rows 1–99 ─────────────┐ ┌─── rows 100–198 ───────┐
       │ Thread 0                            │ │ Thread 1               │
       └─────────────────────────────────────┘ └────────────────────────┘

  T=4  ┌── 1-49 ──┐ ┌── 50-98 ─┐ ┌── 99-148 ─┐ ┌── 149-198 ──┐
       │ Thread 0 │ │ Thread 1  │ │ Thread 2   │ │ Thread 3    │
       └──────────┘ └───────────┘ └────────────┘ └─────────────┘

  T=8  Each thread handles ~25 rows
       ┌─T0─┐ ┌─T1─┐ ┌─T2─┐ ┌─T3─┐ ┌─T4─┐ ┌─T5─┐ ┌─T6─┐ ┌─T7─┐
       └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘
```

### 2.3 The Three Scheduling Strategies

```
  STATIC   — rows pre-assigned at compile time in equal contiguous chunks
  ┌──────────────────────────────────────────────────────┐
  │ Pros: zero runtime overhead, cache-friendly          │
  │ Cons: uneven work if some rows are costlier          │
  └──────────────────────────────────────────────────────┘

  DYNAMIC  — threads grab the next available chunk=10 rows when free
  ┌──────────────────────────────────────────────────────┐
  │ Pros: handles load imbalance automatically           │
  │ Cons: runtime overhead from scheduling queue         │
  └──────────────────────────────────────────────────────┘

  COLLAPSE — merges i and j loops into one flat 198×198 iteration space
  ┌──────────────────────────────────────────────────────┐
  │ #pragma omp parallel for schedule(static) collapse(2)│
  │ → 39,204 iterations shared across T threads          │
  │ Pros: finer granularity, better load balance         │
  │ Cons: may hurt cache locality for large grids        │
  └──────────────────────────────────────────────────────┘
```

---

## 3. Thread Safety Analysis

```
  WHY THERE ARE NO RACE CONDITIONS

  Phase 1 (compute):
  ┌─────────────────────────────────────────────────────────┐
  │ Thread X reads:  traffic[i][j][l]  (old values — READ)  │
  │ Thread X writes: new_traffic[i][j][l]         (WRITE)   │
  │                                                          │
  │ Each thread writes to a DIFFERENT row of new_traffic.   │
  │ All threads read from traffic[] which is NOT modified   │
  │ during Phase 1.  No two threads touch the same cell.    │
  └─────────────────────────────────────────────────────────┘

  Phase 2 (copy back):
  ┌─────────────────────────────────────────────────────────┐
  │ Thread X copies its own rows: new_traffic → traffic     │
  │ No overlap between thread row assignments.              │
  └─────────────────────────────────────────────────────────┘

  IMPLICIT BARRIER between each parallel region ensures all
  threads complete Phase 1 before any thread starts Phase 2.
```

---

## 4. OpenMP Reduction for Statistics

After all time steps, global min/max/average are computed with a
**parallel reduction** — threads compute local partial results then
combine automatically:

```c
double sum = 0.0, lo = 1e18, hi = -1e18;

#pragma omp parallel for schedule(static)      \
        reduction(+:sum) reduction(min:lo) reduction(max:hi)
for (int i = 0; i < ROWS; i++)
    for (int j = 0; j < COLS; j++)
        for (int l = 0; l < LANES; l++) {
            double v = traffic[i][j][l];
            sum += v;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
```

```
  Reduction diagram (4 threads, reduction(+:sum))

  Thread 0: sum_0 = Σ(rows  0–49 )
  Thread 1: sum_1 = Σ(rows 50–99 )    ──► sum = sum_0 + sum_1 + sum_2 + sum_3
  Thread 2: sum_2 = Σ(rows100–149)
  Thread 3: sum_3 = Σ(rows150–198)
```

---

## 5. Complete Execution Flow

```
  main()
    │
    ├─ omp_set_dynamic(0)       — disable dynamic thread adjustment
    │
    ├─ for each schedule (static / dynamic / collapse):
    │    for each thread count (1, 2, 4, 8):
    │      │
    │      ├─ omp_set_num_threads(T)
    │      ├─ initialize()            — reset grid with srand(1)
    │      │
    │      ├─ START TIMER
    │      ├─ for t = 0 to 199:
    │      │    ┌─────────────────────────────────────────────┐
    │      │    │  #pragma omp parallel for schedule(...)     │
    │      │    │  for i = 1..198:   ← split across T threads │
    │      │    │    for j = 1..198:                          │
    │      │    │      for l = 0..2:                          │
    │      │    │        new_traffic[i][j][l] = diffuse(...)  │
    │      │    │  [IMPLICIT BARRIER]                         │
    │      │    │  #pragma omp parallel for schedule(...)     │
    │      │    │  copy new_traffic → traffic                 │
    │      │    └─────────────────────────────────────────────┘
    │      ├─ STOP TIMER
    │      └─ log result to openmp_performance.txt
    │
    └─ save final output (best config: 8 threads, static)
```

---

## 6. Performance Results

| Threads | Static (s)  | Dynamic (s) | Collapse (s) |
|---------|-------------|-------------|--------------|
| 1       | 0.058156    | 0.049670    | 0.055762     |
| 2       | 0.022953    | 0.032586    | 0.026073     |
| 4       | 0.021252    | 0.018464    | 0.014498     |
| 8       | **0.014252**| 0.017350    | 0.014347     |

Speedup vs serial baseline (0.0513 s):

```
  Speedup
    4.0 │                                 ★ static  8T = 3.60×
        │                              ●
    3.0 │                       ▲
        │              ▲    ●  ▲
    2.0 │       ●   ▲
        │   ▲ ●
    1.0 ├────────────────────────────────────────
        1       2       4       8    Threads

    ● static    ▲ collapse    ■ dynamic    --- ideal
```

### Why efficiency drops above 4 threads

On this machine the simulation grid (200×200×3 × 8 bytes = 0.96 MB) fits
comfortably in shared L3 cache.  Above 4–6 threads the overhead of thread
creation, barrier synchronisation, and OS scheduling becomes comparable to
the reduced computation time — efficiency falls below 50%.

---

## 7. Build and Run

```bash
# Compile
gcc -O2 -fopenmp -o traffic_openmp traffic_openmp.c -lm

# Run (all schedules, 1/2/4/8 threads automatically)
./traffic_openmp

# Or set threads manually
export OMP_NUM_THREADS=4
./traffic_openmp

# Generate comparison graphs vs serial
python3 analysis_comparison.py
```

---

## 8. Key OpenMP Directives Used

| Directive | Purpose |
|-----------|---------|
| `#pragma omp parallel for` | Parallelise the following for-loop |
| `schedule(static)` | Equal contiguous row chunks per thread |
| `schedule(dynamic,10)` | Work-stealing chunks of 10 rows |
| `collapse(2)` | Merge i and j loops into one parallel space |
| `reduction(+:sum)` | Thread-local partial sums merged at end |
| `reduction(min:lo)` | Thread-local minimums merged at end |
| `omp_set_dynamic(0)` | Disable runtime thread-count changes |
| `omp_set_num_threads(T)` | Fix thread count before parallel region |
| `omp_get_wtime()` | High-resolution wall-clock timer |

---

*Previous: [Serial baseline](../serial/README.md)*  
*Next: [MPI: distributed-memory parallelism](../mpi/README.md)*

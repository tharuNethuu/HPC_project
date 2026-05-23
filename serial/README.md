# Serial Traffic Density Simulation — Deep Explanation
## EE7218 / EC7207 — High Performance Computing | Group 10

---

## 1. What Is Serial Execution?

In serial execution **one CPU core does all the work, one step at a time**.
There is no parallelism — every road segment is processed in strict order
before moving to the next.

### Analogy

> Imagine a single traffic engineer hired to inspect every road in a
> 200 × 200 city grid.  She starts at segment (0,0) and works through
> every row, every column, every lane — one after another — before
> moving on to the next time step.  200 steps × 200 × 200 × 3 lanes =
> **24,000,000 updates**, all done by one person.

```
  THE TRAFFIC ENGINEER (serial)
  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  Time Step 1:                                    │
  │  → update (1,1)                                  │
  │  → update (1,2)                                  │
  │  → update (1,3) ...                              │
  │  → update (1,199)                                │
  │  → update (2,1)                                  │
  │  ...                                             │
  │  → update (198,198)   ← last interior cell       │
  │                                                  │
  │  Time Step 2:  repeat from (1,1)                 │
  │  ...                                             │
  │  Time Step 200: done.                            │
  └──────────────────────────────────────────────────┘
```

---

## 2. The Computational Model

### 2.1 Grid Layout

```
  GLOBAL GRID  (200 rows × 200 cols × 3 lanes)

  Col →    0    1    2   ...  199
  Row
   0   [  B  ][ B  ][ B  ]...[ B  ]   ← boundary row (never updated)
   1   [  B  ][ C  ][ C  ]...[ B  ]
   2   [  B  ][ C  ][ C  ]...[ B  ]
   .          interior cells (C)
   .          updated each step
  198  [  B  ][ C  ][ C  ]...[ B  ]
  199  [  B  ][ B  ][ B  ]...[ B  ]   ← boundary row (never updated)
         ↑                       ↑
     boundary col           boundary col

  Each cell holds 3 lane values: traffic[i][j][0..2]
```

### 2.2 Weather Zone

Two distinct hazard zones modify traffic density each step:

```
  WEATHER MODIFIER MATRIX

  Col →  0       50      150     199
  Row
   0    [1.0  1.0  1.0  1.0  1.0  1.0]
  ...   [1.0  1.0  1.0  1.0  1.0  1.0]
  67    [1.0  1.0 ╔══════════╗ 1.0  1.0]
  ...   [1.0  1.0 ║  0.8     ║ 1.0  1.0]   ← rain zone (20% reduction)
  133   [1.0  1.0 ╚══════════╝ 1.0  1.0]
  ...   [1.0  1.0  1.0  1.0  1.0  1.0]
  150   [1.0  1.0  1.0 ╔═════╗ 1.0  1.0]
  ...   [1.0  1.0  1.0 ║ 0.5 ║ ← accident zone (50% reduction)
  199   [1.0  1.0  1.0 ╚═════╝]

  weather[i][j] = 1.0  (normal conditions)
                = 0.8  if ROWS/3 < i < 2*ROWS/3  AND  COLS/3 < j < 2*COLS/3
                = 0.5  if i > 3*ROWS/4           AND  j > 3*COLS/4
```

### 2.3 The Diffusion Stencil

Each interior cell `(i,j)` uses its **four cardinal neighbours** from the
*previous* time step to compute the *new* value.

```
  DIFFUSION STENCIL  (5-point, read from previous time step)

                    traffic[i-1][j][l]
                          │
                          │  (north neighbour)
                          ▼
  traffic[i][j-1][l] ──► [i][j] ◄── traffic[i][j+1][l]
         (west)           │            (east)
                          │
                          ▼
                    traffic[i+1][j][l]
                          (south)

  avg_neighbours = (N + S + E + W) / 4.0

  new_traffic[i][j][l] = ( current + 0.1 × (avg_neighbours − current) )
                         × weather[i][j]

  The 0.1 factor is the diffusion coefficient — vehicles spread
  gently toward less-congested neighbours each time step.
```

### 2.4 Two-Phase Update

The code uses a **two-buffer** approach to avoid using partially-updated
values as inputs within the same time step:

```
  Phase 1 — COMPUTE (read from traffic[], write to new_traffic[])
  ┌────────────────────────────────────────────┐
  │  for i = 1 to ROWS-2:                      │
  │    for j = 1 to COLS-2:                    │
  │      for l = 0 to LANES-1:                 │
  │        new_traffic[i][j][l] = diffuse(...) │
  └────────────────────────────────────────────┘
           ↓
  Phase 2 — COPY BACK
  ┌────────────────────────────────────────────┐
  │  traffic[i][j][l] = new_traffic[i][j][l]  │
  └────────────────────────────────────────────┘

  Without two buffers, a cell updated early in Phase 1 could
  contaminate its neighbours' calculations in the same step.
```

---

## 3. Code Structure

```
traffic_serial.c
│
├── Global arrays (static, on BSS segment)
│   ├── double traffic[200][200][3]
│   ├── double new_traffic[200][200][3]
│   └── double weather[200][200]
│
├── initialize()
│   ├── Fill weather[][] (1.0 normal, 0.8 rain, 0.5 accident)
│   └── Fill traffic[][][] with rand() % 100
│
├── update()          ← called 200 times
│   ├── Phase 1: compute new densities using stencil
│   └── Phase 2: copy new_traffic → traffic
│
├── save_output()     ← write timing + all lanes to serial_output.txt
├── save_raw_traffic_data() ← write lane-averaged grid to traffic_values.txt
└── save_image()      ← write PPM heatmap
```

### Execution Flow

```
  main()
    │
    ├─ initialize()
    │     Fill all arrays with initial values
    │
    ├─ START TIMER  (omp_get_wtime)
    │
    ├─ for t = 0 to 199:
    │     update()
    │       ├── for i = 1..198, j = 1..198, l = 0..2:
    │       │       new_traffic[i][j][l] = diffuse(...)
    │       └── copy new_traffic → traffic
    │
    ├─ STOP TIMER
    │
    ├─ save_output()        → serial_output.txt
    ├─ save_image()         → traffic_heatmap.ppm
    └─ save_raw_traffic_data() → traffic_values.txt
```

---

## 4. Build and Run

```bash
# Compile
gcc -O2 -fopenmp -o traffic_serial traffic_serial.c -lm
#        ^^^^^^^^ used only for omp_get_wtime() timing — not for parallelism

# Run
./traffic_serial
```

### Expected Output

```
Initializing traffic simulation...
Simulation completed.
Execution Time: 0.051329 seconds
Results saved to serial_output.txt
Heatmap image saved to traffic_heatmap.ppm
Raw traffic values saved to traffic_values.txt
```

---

## 5. Output Files

| File | Contents |
|------|----------|
| `serial_output.txt` | Grid size, time, final density for all 3 lanes |
| `traffic_values.txt` | 200×200 lane-averaged values (for Python analysis) |
| `traffic_heatmap.ppm` | Raw PPM image — red = high density, blue = low |
| `traffic_grid.png` | PNG version with colorbar and statistics |
| `traffic_zoom.png` | 20×20 zoomed region with cell values |

### Visualisation

```bash
python3 convert_to_png.py
```

---

## 6. Performance Characteristics

| Metric | Value |
|--------|-------|
| Grid size | 200 × 200 × 3 lanes |
| Time steps | 200 |
| Operations | ~24 million per step |
| Total FP ops | ~4.8 billion |
| Execution time | ~0.051 s (measured) |
| Threads | 1 |
| Speedup | 1.00× (baseline) |

The serial version is the **reference baseline** for all speedup and
efficiency calculations in OpenMP, MPI, and Hybrid analyses.

---

## 7. Why Serial First?

1. **Simplest to understand** — no parallel concepts needed
2. **Correctness reference** — OpenMP, MPI, and Hybrid results are verified
   against this output (max |diff| = 0.0 across all cells)
3. **Timing baseline** — `T_serial = 0.0513 s` is the denominator in every
   speedup formula: `Speedup = T_serial / T_parallel`

---

*Next step → [OpenMP: shared-memory parallelism](../openmp/README.md)*

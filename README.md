# Parallel Traffic Density Simulation
## EE7218 / EC7207 — High Performance Computing | Group 10

A multi-lane, weather-aware traffic diffusion simulation implemented in four
progressively parallel versions, from single-threaded baseline to a full
Hybrid MPI+OpenMP system.

---

## Project Structure

```
HPC_project/
│
├── serial/                    ← Single-threaded baseline
│   ├── traffic_serial.c       ← Source code
│   ├── serial_output.txt      ← Results
│   ├── traffic_values.txt     ← Grid data (for analysis)
│   ├── traffic_heatmap.ppm    ← Heatmap image
│   ├── convert_to_png.py      ← Visualisation helper
│   └── README.md              ← Deep explanation (this implementation)
│
├── openmp/                    ← Shared-memory parallelism
│   ├── traffic_openmp.c
│   ├── openmp_performance.txt
│   ├── analysis_comparison.py
│   └── README.md
│
├── mpi/                       ← Distributed-memory parallelism
│   ├── traffic_mpi.c
│   ├── run_scaling.sh
│   ├── mpi_performance.txt
│   ├── analysis_mpi.py
│   └── README.md
│
├── hybrid/                    ← MPI + OpenMP combined
│   ├── traffic_hybrid.c
│   ├── run_scaling_hybrid.sh
│   ├── hybrid_performance.txt
│   ├── analysis_hybrid.py
│   ├── verify_all4.py
│   └── README.md
│
└── README.md                  ← This file
```

---

## The Problem: City Traffic Simulation

In real cities, traffic conditions change continuously due to:
- **Number of vehicles** on each road
- **Multiple lanes** on every road segment
- **External conditions** such as rain or accidents

We model a city road network as a **200 × 200 matrix** (40,000 road segments)
with **3 lanes per segment**, updated over **200 time steps**.

### Real World → Computational Model

| Real World         | Computational Model            |
|--------------------|-------------------------------|
| City road network  | 2D grid / matrix               |
| Road segment       | One cell `traffic[i][j]`       |
| Cars on the road   | Traffic density value (0–100)  |
| Multiple lanes     | Third array dimension `[LANES]`|
| Weather / rain     | Modifier matrix `weather[i][j]`|
| Time passing       | Iterative update loop          |

### Traffic Update Formula

Every time step, each interior cell is updated using a **4-neighbour diffusion**:

```
new[i][j][l] = ( current + 0.1 × (avg_neighbours − current) ) × weather[i][j]

where avg_neighbours = ( traffic[i-1][j][l] + traffic[i+1][j][l]
                       + traffic[i][j-1][l] + traffic[i][j+1][l] ) / 4
```

This models vehicles gradually flowing toward less-congested roads.  
The `weather` modifier reduces density each step:
- `1.0` = normal conditions (no effect)
- `0.8` = rain zone (centre third of grid) — 20% density reduction
- `0.5` = accident / obstruction zone (bottom-right corner) — 50% density reduction

---

## The Four Implementations

```
┌─────────────────────────────────────────────────────────────────┐
│                    IMPLEMENTATION HIERARCHY                     │
│                                                                 │
│   Serial          OpenMP           MPI           Hybrid         │
│   ──────          ──────           ───           ──────         │
│   1 process       1 process        N processes   N processes    │
│   1 thread        T threads        1 thread/proc T threads/proc │
│                   shared mem       dist. memory  both           │
│                                                                 │
│   Speedup:  1×      ~3.6×           ~8×           ~14×          │
└─────────────────────────────────────────────────────────────────┘
```

### Analogy: Traffic Control Authority

```
SERIAL          — One traffic engineer updates every road, one by one

OPENMP          — One control center, multiple engineers share the same
                  city map and each updates different road sections

MPI             — Multiple cities, each with its own control center
                  and its own copy of the map; centres exchange data
                  at their shared border each time step

HYBRID          — Multiple cities (MPI), each with multiple engineers
  (MPI+OpenMP)    sharing that city's map (OpenMP) — both levels at once
```

---

## Performance Summary

| Implementation | Best Config | Time (s) | Speedup vs Serial |
|----------------|-------------|----------|-------------------|
| Serial         | 1P · 1T     | 0.0513   | 1.00×             |
| OpenMP         | 8T static   | 0.0143   | 3.60×             |
| MPI            | 4 processes | 0.0064   | 7.98×             |
| **Hybrid**     | **2P × 4T** | **0.0036** | **14.10×**      |

### Correctness

All four implementations produce **bit-for-bit identical** final grids
(max |difference| = 0.0 across all 40,000 cells × 3 lanes).
See `hybrid/verification_all4.png`.

---

## Quick Start

```bash
# Serial
cd serial
gcc -O2 -fopenmp -o traffic_serial traffic_serial.c -lm
./traffic_serial

# OpenMP
cd openmp
gcc -O2 -fopenmp -o traffic_openmp traffic_openmp.c -lm
./traffic_openmp

# MPI
cd mpi
bash run_scaling.sh

# Hybrid
cd hybrid
bash run_scaling_hybrid.sh

# Analysis (4-way comparison)
cd hybrid
python3 analysis_hybrid.py
```

---

## Key Output Files

| File | Description |
|------|-------------|
| `*/serial_output.txt` | Execution time + final density matrices |
| `*/openmp_performance.txt` | Speedup & efficiency for 1/2/4/8 threads |
| `*/mpi_performance.txt` | Speedup & efficiency for 1/2/4/8 processes |
| `*/hybrid_performance.txt` | All 8 MPI×OMP combinations |
| `hybrid/hybrid_full_report.png` | Master 6-panel comparison figure |
| `hybrid/verification_all4.png` | Correctness proof across all implementations |

---

*EE7218/EC7207 High Performance Computing — Group 10*

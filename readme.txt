# HPC Traffic Simulation Project Guide

This is an excellent HPC project. Let me guide you through the complete implementation step by step.

## Project Overview

You'll build a traffic simulation that models a city grid where:
- Each road segment has multiple lanes with traffic density values
- Weather affects traffic flow
- Traffic updates based on neighboring cells
- The simulation runs for multiple time steps

## Phase 1: Understanding the Core Algorithm

### 1.1 Data Structures

First, decide on your data representation:

```c
// For a city of size N x N with L lanes per road
// Traffic density: [time][row][col][lane] or just [row][col][lane] for current state
// Weather: [row][col] - multiplier values (1.0 normal, 0.8 rain, 0.5 accident)
```

### 1.2 Update Rule (Simplified)

For each time step, each cell's new density = function of:
- Current density in this cell
- Densities in neighboring cells (up/down/left/right)
- Weather modifier for this cell

A simple formula:
```
new_density[row][col][lane] = weather[row][col] * 
    (current[row][col][lane] * 0.5 + 
     average_of_neighbors * 0.5)
```

## Phase 2: Implementation Plan

### Step 1: Serial Implementation (Baseline)

**File: `traffic_serial.c`**

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Function prototypes
float*** allocate_3d(int rows, int cols, int lanes);
void free_3d(float*** array, int rows, int cols);
float** allocate_2d(int rows, int cols);
void initialize_traffic(float*** traffic, int rows, int cols, int lanes);
void initialize_weather(float** weather, int rows, int cols);
void update_traffic_serial(float*** traffic, float** weather, 
                           int rows, int cols, int lanes);
float get_neighbor_avg(float*** traffic, int r, int c, int l, 
                       int rows, int cols, int lanes);

int main(int argc, char* argv[]) {
    int rows = 100, cols = 100, lanes = 3, time_steps = 100;
    
    // Allocate memory
    float*** traffic = allocate_3d(rows, cols, lanes);
    float** weather = allocate_2d(rows, cols);
    
    // Initialize data
    initialize_traffic(traffic, rows, cols, lanes);
    initialize_weather(weather, rows, cols);
    
    // Timing
    clock_t start = clock();
    
    // Main simulation loop
    for (int t = 0; t < time_steps; t++) {
        update_traffic_serial(traffic, weather, rows, cols, lanes);
    }
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Serial execution time: %f seconds\n", time_taken);
    
    // Save results for verification
    // free memory
    return 0;
}

void update_traffic_serial(float*** traffic, float** weather, 
                           int rows, int cols, int lanes) {
    // Create temporary array for new values
    float*** new_traffic = allocate_3d(rows, cols, lanes);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < lanes; k++) {
                float neighbor_avg = get_neighbor_avg(traffic, i, j, k, 
                                                      rows, cols, lanes);
                float current = traffic[i][j][k];
                
                // Update formula: weather * (current * 0.5 + neighbor_avg * 0.5)
                new_traffic[i][j][k] = weather[i][j] * (0.5 * current + 0.5 * neighbor_avg);
            }
        }
    }
    
    // Copy back
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < lanes; k++) {
                traffic[i][j][k] = new_traffic[i][j][k];
            }
        }
    }
    
    free_3d(new_traffic, rows, cols);
}
```

### Step 2: OpenMP Implementation (Shared Memory)

**File: `traffic_openmp.c`**

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Similar structure but with OpenMP parallelization

void update_traffic_omp(float*** traffic, float** weather, 
                        int rows, int cols, int lanes, int num_threads) {
    float*** new_traffic = allocate_3d(rows, cols, lanes);
    
    omp_set_num_threads(num_threads);
    
    double start = omp_get_wtime();
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < lanes; k++) {
                float neighbor_avg = get_neighbor_avg(traffic, i, j, k, 
                                                      rows, cols, lanes);
                float current = traffic[i][j][k];
                new_traffic[i][j][k] = weather[i][j] * (0.5 * current + 0.5 * neighbor_avg);
            }
        }
    }
    
    // Copy back (can also be parallelized)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < lanes; k++) {
                traffic[i][j][k] = new_traffic[i][j][k];
            }
        }
    }
    
    double end = omp_get_wtime();
    
    free_3d(new_traffic, rows, cols);
}
```

### Step 3: MPI Implementation (Distributed Memory)

**File: `traffic_mpi.c`**

This is more complex. The key concept is domain decomposition:

```
Global grid (e.g., 100x100) divided among P processes:
Process 0: rows 0-24
Process 1: rows 25-49
Process 2: rows 50-74  
Process 3: rows 75-99

Each process needs halo exchange with neighbors
```

Key components:

```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int local_rows;
    int local_cols;
    int lanes;
    float**** traffic;  // [local_rows][local_cols][lanes]
    float*** weather;   // [local_rows][local_cols]
    int rank, size;
    int top_neighbor, bottom_neighbor;
} Domain;

void exchange_boundaries(Domain* domain) {
    // Send bottom row to next process
    // Receive top row from next process as bottom halo
    // Send top row to previous process
    // Receive bottom row from previous process as top halo
    
    MPI_Request requests[4];
    
    // For each lane, send/receive boundary data
    // Use MPI_Sendrecv or non-blocking MPI_Isend/MPI_Irecv
}

void update_traffic_mpi(Domain* domain) {
    // First exchange boundary data
    exchange_boundaries(domain);
    
    // Update interior points (no communication needed)
    // Update boundary points (using received halo data)
}
```

### Step 4: Hybrid MPI+OpenMP

**File: `traffic_hybrid.c`**

```c
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void update_traffic_hybrid(Domain* domain, int num_threads) {
    // Exchange boundaries with MPI
    exchange_boundaries(domain);
    
    // Parallel update within process using OpenMP
    #pragma omp parallel for collapse(2) num_threads(num_threads)
    for (int i = 0; i < domain->local_rows; i++) {
        for (int j = 0; j < domain->local_cols; j++) {
            for (int k = 0; k < domain->lanes; k++) {
                // Update using local data + halo data
                // ...
            }
        }
    }
}
```

## Phase 3: Adding Realistic Features

### Directional Flow (Rush Hour)

```c
float get_directional_factor(int row, int col, int center_row, int center_col, int time_step) {
    // Morning: traffic toward center (7-9 AM)
    // Evening: traffic away from center (4-7 PM)
    int hour = (time_step % 24);  // Assuming 1 step = 1 hour
    
    float distance_to_center = sqrt(pow(row - center_row, 2) + pow(col - center_col, 2));
    
    if (hour >= 7 && hour <= 9) {
        // Morning rush - toward center
        return 1.0 / (1.0 + distance_to_center/10.0);
    } else if (hour >= 16 && hour <= 19) {
        // Evening rush - away from center
        return 1.0 / (1.0 + (50.0 - distance_to_center)/10.0);
    }
    return 1.0;
}
```

### Dynamic Hotspots

```c
void add_hotspot(float*** traffic, int rows, int cols, int lanes, 
                 int center_r, int center_c, float intensity) {
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            int r = center_r + i;
            int c = center_c + j;
            if (r >= 0 && r < rows && c >= 0 && c < cols) {
                float distance = sqrt(i*i + j*j);
                float factor = intensity * exp(-distance/2.0);
                for (int k = 0; k < lanes; k++) {
                    traffic[r][c][k] += factor;
                }
            }
        }
    }
}
```

## Phase 4: Performance Analysis

### Create a Test Harness

**File: `run_experiments.sh`**

```bash
#!/bin/bash

# Compile all versions
gcc -o serial traffic_serial.c -lm
gcc -fopenmp -o omp traffic_openmp.c -lm
mpicc -o mpi traffic_mpi.c -lm
mpicc -fopenmp -o hybrid traffic_hybrid.c -lm

# Grid sizes to test
sizes="50 100 200 400"

# Thread counts for OpenMP
threads="1 2 4 8 16"

# Process counts for MPI
procs="1 2 4 8"

echo "Running experiments..."

# Test serial
for size in $sizes; do
    ./serial $size >> results_serial.csv
done

# Test OpenMP
for size in $sizes; do
    for t in $threads; do
        export OMP_NUM_THREADS=$t
        ./omp $size >> results_omp.csv
    done
done

# Test MPI
for size in $sizes; do
    for p in $procs; do
        mpirun -np $p ./mpi $size >> results_mpi.csv
    done
done
```

## Phase 5: Analysis and Report

### Required Measurements

1. **Execution Time**: Run each configuration multiple times, take average
2. **Speedup**: `Speedup = T_serial / T_parallel`
3. **Efficiency**: `Efficiency = Speedup / Number of cores`
4. **Accuracy (RMSE)**: 
   ```c
   float rmse = 0;
   for each cell/lane:
       diff = serial_value - parallel_value;
       rmse += diff * diff;
   rmse = sqrt(rmse / total_cells);
   ```

### Report Structure

1. **Introduction** (from your proposal)
2. **Methodology** 
   - Include diagram showing domain decomposition
   - Explain parallelization strategy
3. **Results Tables/Graphs**
   - Execution time vs grid size
   - Speedup vs threads/processes
   - Scaling analysis (strong and weak scaling)
4. **Accuracy Verification**
   - RMSE values (should be near zero for deterministic updates)
5. **Discussion**
   - Why OpenMP scales well for this problem
   - Communication overhead in MPI
   - Best configuration for different problem sizes
6. **Conclusion**

## Common Pitfalls to Avoid

1. **Race conditions in OpenMP**: Use `#pragma omp parallel for` correctly
2. **Deadlocks in MPI**: Ensure matching send/receive pairs
3. **Memory leaks**: Free all allocated memory
4. **Load imbalance**: Use dynamic scheduling if needed
5. **Boundary conditions**: Handle edges carefully in neighbor calculations

## Getting Started

1. **First**: Implement and debug serial version
2. **Second**: Add OpenMP, verify correctness
3. **Third**: Implement MPI with simple 1D decomposition
4. **Fourth**: Add hybrid version
5. **Finally**: Run experiments and analyze results

Start with small grids (like 10x10) for debugging, then scale up for performance testing. Good luck!
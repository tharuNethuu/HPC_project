/*
=========================================================
 FILE: traffic_mpi.c
 GROUP: 10

 DESCRIPTION:
   MPI Parallel Traffic Density Simulation

   Based on the serial baseline (traffic_serial.c).
   Parallelizes the 2D multi-lane traffic diffusion model
   using MPI distributed-memory message passing.

   Strategy: 1-D row decomposition.
   Each MPI process owns a contiguous block of rows.
   Neighboring processes exchange boundary rows (halo
   exchange) before every time step so the diffusion
   stencil can be evaluated without gaps.

 MPI Features Demonstrated:
   1.  MPI_Init / MPI_Finalize           (setup / teardown)
   2.  MPI_Comm_rank / MPI_Comm_size     (process identity)
   3.  MPI_Scatterv                      (distribute grid rows to processes)
   4.  MPI_Gatherv                       (collect final rows to rank 0)
   5.  MPI_Isend / MPI_Irecv             (non-blocking halo exchange)
   6.  MPI_Waitall                       (complete non-blocking ops)
   7.  MPI_Reduce  (MPI_SUM/MIN/MAX)     (global statistics aggregation)
   8.  MPI_Barrier                       (synchronisation for accurate timing)
   9.  MPI_Wtime                         (high-resolution wall-clock timer)

 Grid decomposition:
   Global grid : ROWS x COLS x LANES  (all rows in rank 0's global arrays)
   Local slice : local_rows x COLS x LANES  (+2 ghost/halo rows at top & bottom)
   Ghost rows  : filled by halo exchange each time step

 Outputs (written by rank 0):
   mpi_output.txt          : Final traffic density matrices (all lanes)
   mpi_traffic_heatmap.ppm : Heatmap image (Red=dense, Blue=sparse)
   mpi_traffic_values.txt  : Lane-averaged values for external visualization
   mpi_performance.txt     : Performance data row appended each run
                             (run_scaling.sh writes header before first run)

 Compile:
   mpicc -O2 -o traffic_mpi traffic_mpi.c -lm

 Run:
   mpirun -np 4 ./traffic_mpi
=========================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

/* ---- Simulation Parameters (match serial/OpenMP baselines) ---- */
#define ROWS        200
#define COLS        200
#define LANES       3
#define TIME_STEPS  200

/* ---- Output File Names ---- */
#define OUTPUT_FILE  "mpi_output.txt"
#define IMAGE_FILE   "mpi_traffic_heatmap.ppm"
#define PERF_FILE    "mpi_performance.txt"
#define VALUES_FILE  "mpi_traffic_values.txt"

/* ---- Halo Exchange Message Tags ---- */
/* TAG_NORTH : message carrying a row travelling northward (to lower rank)  */
/* TAG_SOUTH : message carrying a row travelling southward (to higher rank) */
#define TAG_NORTH  10
#define TAG_SOUTH  11

/* ================================================================
 * Global State
 * ================================================================ */

int rank, nprocs;

/* Rows owned by this process and their global offset */
int local_rows;
int row_start;

/* Local arrays: dimension [local_rows + 2][COLS][LANES]
 * Index 0            = top ghost row    (from rank-1)
 * Index 1..local_rows = owned data rows
 * Index local_rows+1  = bottom ghost row (from rank+1)       */
double (*local_traffic)[COLS][LANES];
double (*local_new)[COLS][LANES];
double (*local_weather)[COLS];

/* Full-grid arrays live on rank 0; used for init, gather, and output.
 * All processes allocate them (global/BSS) because MPI scatter/gather
 * signatures require valid pointers on all ranks; for non-root the
 * send/recv buffers of MPI_Scatterv/Gatherv are simply ignored.      */
double global_traffic[ROWS][COLS][LANES];
double global_weather[ROWS][COLS];

/* Per-process row distribution metadata */
int *row_counts;   /* row_counts[p] = number of rows assigned to process p */
int *row_displs;   /* row_displs[p] = global row index of process p's first row */

/* MPI_Scatterv / MPI_Gatherv counts and displacements (in doubles) */
int *tc, *td;      /* traffic : counts / displs */
int *wc, *wd;      /* weather : counts / displs */


/* ================================================================
 * compute_distribution()
 * Splits ROWS as evenly as possible across nprocs.
 * Remainder rows (ROWS % nprocs) are given one each to the
 * first (ROWS % nprocs) processes.
 * ================================================================ */
void compute_distribution(void)
{
    row_counts = (int *)malloc(nprocs * sizeof(int));
    row_displs = (int *)malloc(nprocs * sizeof(int));
    tc         = (int *)malloc(nprocs * sizeof(int));
    td         = (int *)malloc(nprocs * sizeof(int));
    wc         = (int *)malloc(nprocs * sizeof(int));
    wd         = (int *)malloc(nprocs * sizeof(int));

    int base = ROWS / nprocs;
    int rem  = ROWS % nprocs;
    int off  = 0;

    for (int p = 0; p < nprocs; p++) {
        row_counts[p] = base + (p < rem ? 1 : 0);
        row_displs[p] = off;

        tc[p] = row_counts[p] * COLS * LANES;   /* doubles for traffic */
        td[p] = off * COLS * LANES;
        wc[p] = row_counts[p] * COLS;            /* doubles for weather */
        wd[p] = off * COLS;

        off += row_counts[p];
    }

    local_rows = row_counts[rank];
    row_start  = row_displs[rank];
}


/* ================================================================
 * initialize_global()
 * Called on rank 0 only.
 * Fixed seed (srand(1)) to match serial / OpenMP baselines so that
 * accuracy comparisons across implementations are meaningful.
 * ================================================================ */
void initialize_global(void)
{
    srand(1);
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {

            global_weather[i][j] = 1.0;

            /* Rain / slow-down region in the centre of the grid */
            if (i > ROWS/3 && i < 2*ROWS/3 &&
                j > COLS/3 && j < 2*COLS/3)
                global_weather[i][j] = 0.7;

            for (int l = 0; l < LANES; l++)
                global_traffic[i][j][l] = (double)(rand() % 100);
        }
    }
}


/* ================================================================
 * distribute_data()
 * MPI_Scatterv sends each process its slice of rows from rank 0.
 * Received data is placed starting at local index 1 (index 0 is
 * reserved for the top ghost row).
 * ================================================================ */
void distribute_data(void)
{
    /* --- Traffic grid --- */
    MPI_Scatterv(
        &global_traffic[0][0][0], tc, td, MPI_DOUBLE,   /* send (rank 0) */
        &local_traffic[1][0][0],                          /* recv buffer   */
        local_rows * COLS * LANES, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    /* --- Weather grid --- */
    MPI_Scatterv(
        &global_weather[0][0], wc, wd, MPI_DOUBLE,
        &local_weather[1][0],
        local_rows * COLS, MPI_DOUBLE,
        0, MPI_COMM_WORLD);
}


/* ================================================================
 * halo_exchange()
 * Non-blocking send/receive to swap boundary rows with neighbours.
 *
 * Each process:
 *   - Sends its top data row    (local[1])         to rank-1
 *   - Sends its bottom data row (local[local_rows]) to rank+1
 *   - Receives rank-1's bottom row into local[0]          (top ghost)
 *   - Receives rank+1's top row  into local[local_rows+1] (bottom ghost)
 *
 * Using MPI_Isend/MPI_Irecv avoids deadlock and allows the MPI
 * library to overlap communication with independent work.
 * MPI_Waitall ensures all transfers complete before the update step.
 * ================================================================ */
void halo_exchange(void)
{
    MPI_Request reqs[4];
    int n = 0;

    int prev = rank - 1;   /* rank above (lower index) */
    int next = rank + 1;   /* rank below (higher index) */

    /* Post receives first to minimise latency */
    if (prev >= 0)
        MPI_Irecv(&local_traffic[0][0][0],            COLS * LANES, MPI_DOUBLE,
                  prev, TAG_SOUTH, MPI_COMM_WORLD, &reqs[n++]);

    if (next < nprocs)
        MPI_Irecv(&local_traffic[local_rows + 1][0][0], COLS * LANES, MPI_DOUBLE,
                  next, TAG_NORTH, MPI_COMM_WORLD, &reqs[n++]);

    /* Send top data row northward to prev */
    if (prev >= 0)
        MPI_Isend(&local_traffic[1][0][0],            COLS * LANES, MPI_DOUBLE,
                  prev, TAG_NORTH, MPI_COMM_WORLD, &reqs[n++]);

    /* Send bottom data row southward to next */
    if (next < nprocs)
        MPI_Isend(&local_traffic[local_rows][0][0],   COLS * LANES, MPI_DOUBLE,
                  next, TAG_SOUTH, MPI_COMM_WORLD, &reqs[n++]);

    MPI_Waitall(n, reqs, MPI_STATUSES_IGNORE);
}


/* ================================================================
 * update_local()
 * Apply one diffusion step to this process's owned rows.
 *
 * Diffusion formula (matches serial baseline exactly):
 *   new[i][j][l] = (current + 0.1*(avg_neighbours - current))
 *                  * weather[i][j]
 *
 * Global boundary rows (gi == 0 or gi == ROWS-1) are left unchanged,
 * matching the serial baseline which iterates i from 1 to ROWS-2.
 * Column boundaries (j == 0 and j == COLS-1) are also skipped.
 *
 * Two-phase approach (read from local_traffic, write to local_new,
 * then copy back) avoids read-after-write races.
 * ================================================================ */
void update_local(void)
{
    /* Phase 1: compute new densities */
    for (int li = 1; li <= local_rows; li++) {
        int gi = row_start + li - 1;        /* global row index */
        if (gi == 0 || gi == ROWS - 1) continue;

        for (int j = 1; j < COLS - 1; j++) {
            for (int l = 0; l < LANES; l++) {
                double cur  = local_traffic[li][j][l];
                double nbrs = local_traffic[li-1][j][l]
                            + local_traffic[li+1][j][l]
                            + local_traffic[li][j-1][l]
                            + local_traffic[li][j+1][l];
                local_new[li][j][l] =
                    (cur + 0.1 * (nbrs / 4.0 - cur)) * local_weather[li][j];
            }
        }
    }

    /* Phase 2: copy new values back */
    for (int li = 1; li <= local_rows; li++) {
        int gi = row_start + li - 1;
        if (gi == 0 || gi == ROWS - 1) continue;

        for (int j = 1; j < COLS - 1; j++)
            for (int l = 0; l < LANES; l++)
                local_traffic[li][j][l] = local_new[li][j][l];
    }
}


/* ================================================================
 * gather_data()
 * MPI_Gatherv collects each process's owned rows back to rank 0's
 * global_traffic array for output.
 * ================================================================ */
void gather_data(void)
{
    MPI_Gatherv(
        &local_traffic[1][0][0],
        local_rows * COLS * LANES, MPI_DOUBLE,
        &global_traffic[0][0][0], tc, td, MPI_DOUBLE,
        0, MPI_COMM_WORLD);
}


/* ================================================================
 * compute_stats()
 * Each process computes local partial min/max/sum; MPI_Reduce
 * aggregates them to rank 0.  Output pointers are only set on
 * rank 0.
 * ================================================================ */
void compute_stats(double *out_min, double *out_max, double *out_avg)
{
    double lsum = 0.0, lmin = 1e18, lmax = -1e18;

    for (int li = 1; li <= local_rows; li++) {
        for (int j = 0; j < COLS; j++) {
            for (int l = 0; l < LANES; l++) {
                double v = local_traffic[li][j][l];
                lsum += v;
                if (v < lmin) lmin = v;
                if (v > lmax) lmax = v;
            }
        }
    }

    double gsum, gmin, gmax;
    MPI_Reduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lmin, &gmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lmax, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        *out_min = gmin;
        *out_max = gmax;
        *out_avg = gsum / (double)(ROWS * COLS * LANES);
    }
}


/* ================================================================
 * save_output()  -- rank 0 only
 * Saves final traffic density matrices for all lanes.
 * ================================================================ */
void save_output(double exec_time)
{
    FILE *fp = fopen(OUTPUT_FILE, "w");
    if (!fp) { fprintf(stderr, "ERROR: Cannot open %s\n", OUTPUT_FILE); return; }

    fprintf(fp, "=================================================\n");
    fprintf(fp, "  MPI Parallel Traffic Simulation Output\n");
    fprintf(fp, "  Group 10\n");
    fprintf(fp, "=================================================\n");
    fprintf(fp, "Grid Size  : %d x %d\n",       ROWS, COLS);
    fprintf(fp, "Lanes      : %d\n",             LANES);
    fprintf(fp, "Time Steps : %d\n",             TIME_STEPS);
    fprintf(fp, "Processes  : %d\n",             nprocs);
    fprintf(fp, "Exec Time  : %.6f seconds\n\n", exec_time);

    for (int l = 0; l < LANES; l++) {
        fprintf(fp, "--- Lane %d ---\n", l);
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++)
                fprintf(fp, "%.2f ", global_traffic[i][j][l]);
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}


/* ================================================================
 * save_image()  -- rank 0 only
 * Saves a PPM heatmap.  Red = high density, Blue = low density.
 * Matches the serial baseline format (traffic_heatmap.ppm).
 * ================================================================ */
void save_image(void)
{
    FILE *img = fopen(IMAGE_FILE, "w");
    if (!img) { fprintf(stderr, "ERROR: Cannot open %s\n", IMAGE_FILE); return; }

    fprintf(img, "P3\n%d %d\n255\n", COLS, ROWS);
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            double sum = 0.0;
            for (int l = 0; l < LANES; l++) sum += global_traffic[i][j][l];
            double avg = sum / LANES;
            int v = (int)(255.0 * avg / 100.0);
            if (v > 255) v = 255;
            if (v < 0)   v = 0;
            fprintf(img, "%d %d %d ", v, 0, 255 - v);
        }
        fprintf(img, "\n");
    }
    fclose(img);
}


/* ================================================================
 * save_raw_values()  -- rank 0 only
 * Saves lane-averaged traffic densities.
 * Same format as serial traffic_values.txt for direct comparison.
 * ================================================================ */
void save_raw_values(void)
{
    FILE *fp = fopen(VALUES_FILE, "w");
    if (!fp) { fprintf(stderr, "ERROR: Cannot open %s\n", VALUES_FILE); return; }

    fprintf(fp, "%d %d %d\n", ROWS, COLS, LANES);
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            double sum = 0.0;
            for (int l = 0; l < LANES; l++) sum += global_traffic[i][j][l];
            fprintf(fp, "%.2f ", sum / LANES);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}


/* ================================================================
 * append_perf_line()  -- rank 0 only
 * Appends one data row to mpi_performance.txt.
 * run_scaling.sh writes the file header before the first invocation
 * so that the final file contains all np configurations together.
 * ================================================================ */
void append_perf_line(double exec_time,
                      double min_d, double max_d, double avg_d)
{
    FILE *fp = fopen(PERF_FILE, "a");
    if (!fp) { fprintf(stderr, "ERROR: Cannot open %s\n", PERF_FILE); return; }

    fprintf(fp, "  %-8d %-14.6f %-8.4f %-8.4f %-8.4f\n",
            nprocs, exec_time, min_d, max_d, avg_d);

    fclose(fp);
}


/* ================================================================
 * main()
 * ================================================================ */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* ---- Compute how many rows each process owns ---- */
    compute_distribution();

    /* ---- Allocate local arrays (+2 ghost rows) ---- */
    local_traffic = (double (*)[COLS][LANES])
                    malloc((local_rows + 2) * sizeof(*local_traffic));
    local_new     = (double (*)[COLS][LANES])
                    malloc((local_rows + 2) * sizeof(*local_new));
    local_weather = (double (*)[COLS])
                    malloc((local_rows + 2) * sizeof(*local_weather));

    memset(local_traffic, 0, (local_rows + 2) * sizeof(*local_traffic));
    memset(local_new,     0, (local_rows + 2) * sizeof(*local_new));
    memset(local_weather, 0, (local_rows + 2) * sizeof(*local_weather));

    /* ---- Rank 0 initialises the full grid ---- */
    if (rank == 0)
        initialize_global();

    /* ---- Distribute rows and weather to all processes ---- */
    distribute_data();

    /* ---- Print header (rank 0 only) ---- */
    if (rank == 0) {
        printf("\n==========================================================\n");
        printf("  MPI Parallel Traffic Density Simulation - Group 10\n");
        printf("==========================================================\n");
        printf("[MPI Environment]\n");
        printf("  Processes  : %d\n", nprocs);
        printf("[Simulation Parameters]\n");
        printf("  Grid Size  : %d x %d\n", ROWS, COLS);
        printf("  Lanes      : %d\n",       LANES);
        printf("  Time Steps : %d\n",       TIME_STEPS);
        printf("  Local rows : ~%d per process\n", ROWS / nprocs);
        printf("\n[Running simulation...]\n");
    }

    /* ---- Synchronise all processes, then start timer ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* ================================================================
     * MAIN SIMULATION LOOP
     * Each iteration:
     *   1. halo_exchange() — fill ghost rows from neighbours
     *   2. update_local()  — compute one diffusion step on owned rows
     * ================================================================ */
    for (int t = 0; t < TIME_STEPS; t++) {
        halo_exchange();
        update_local();
    }

    /* ---- Synchronise then stop timer ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double exec_time = MPI_Wtime() - t_start;

    /* ---- Collect all rows back to rank 0 ---- */
    gather_data();

    /* ---- Compute global statistics via MPI_Reduce ---- */
    double min_d = 0.0, max_d = 0.0, avg_d = 0.0;
    compute_stats(&min_d, &max_d, &avg_d);

    /* ---- Rank 0: write outputs and print summary ---- */
    if (rank == 0) {
        printf("[Performance]\n");
        printf("  Exec Time  : %.6f seconds\n", exec_time);
        printf("  Min Density: %.4f\n", min_d);
        printf("  Max Density: %.4f\n", max_d);
        printf("  Avg Density: %.4f\n", avg_d);

        printf("\n[Saving Output Files]\n");
        save_output(exec_time);
        printf("  %-42s -> Traffic density data (all lanes)\n", OUTPUT_FILE);

        save_image();
        printf("  %-42s -> Heatmap image (PPM format)\n", IMAGE_FILE);

        save_raw_values();
        printf("  %-42s -> Raw lane-averaged values\n", VALUES_FILE);

        append_perf_line(exec_time, min_d, max_d, avg_d);
        printf("  %-42s -> Performance log (appended)\n", PERF_FILE);

        printf("\n==========================================================\n");
        printf("  Simulation Complete  |  %d process(es)\n", nprocs);
        printf("==========================================================\n\n");
    }

    /* ---- Cleanup ---- */
    free(local_traffic);
    free(local_new);
    free(local_weather);
    free(row_counts);
    free(row_displs);
    free(tc); free(td);
    free(wc); free(wd);

    MPI_Finalize();
    return 0;
}

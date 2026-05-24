/*
=========================================================
 FILE: traffic_hybrid.c
 GROUP: 10

 DESCRIPTION:
   Hybrid MPI + OpenMP Traffic Density Simulation

   Combines distributed-memory (MPI) and shared-memory
   (OpenMP) parallelism in a two-level hierarchy:

     MPI level  : 1-D row decomposition — each MPI process
                  owns a contiguous block of grid rows.
                  Processes communicate via halo exchange.

     OpenMP level: Within each MPI process, the per-row
                  diffusion loop is parallelised across
                  OMP threads sharing the process's memory.

   Analogy (from slides):
     Country
       ├── Colombo Control Center  (MPI Process)
       │     ├── Engineer 1        (OpenMP Thread)
       │     └── Engineer 2        (OpenMP Thread)
       └── Kandy Control Center    (MPI Process)
             ├── Engineer 1        (OpenMP Thread)
             └── Engineer 2        (OpenMP Thread)

 MPI Features:
   MPI_Init / MPI_Finalize, MPI_Comm_rank / MPI_Comm_size,
   MPI_Scatterv, MPI_Gatherv, MPI_Isend / MPI_Irecv,
   MPI_Waitall, MPI_Reduce, MPI_Barrier, MPI_Wtime

 OpenMP Features:
   #pragma omp parallel for  (parallelise row loop)
   schedule(static)          (even work distribution)
   reduction(+:), reduction(min:), reduction(max:)
   omp_set_num_threads()     (runtime thread control)

 Compile:
   mpicc -O2 -fopenmp -o traffic_hybrid traffic_hybrid.c -lm

 Run:
   mpirun -np 2 ./traffic_hybrid 4    (2 MPI x 4 OMP = 8 workers)
   mpirun -np 4 ./traffic_hybrid 2    (4 MPI x 2 OMP = 8 workers)
=========================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

/* ── Simulation parameters (identical to serial / OMP / MPI baselines) ── */
#define ROWS        200
#define COLS        200
#define LANES       3
#define TIME_STEPS  200

/* ── Output files ── */
#define OUTPUT_FILE  "hybrid_output.txt"
#define IMAGE_FILE   "hybrid_traffic_heatmap.ppm"
#define PERF_FILE    "hybrid_performance.txt"
#define VALUES_FILE  "hybrid_traffic_values.txt"

/* ── Halo-exchange tags ── */
#define TAG_NORTH  10
#define TAG_SOUTH  11

/* ================================================================
 * Performance tracking — upserts one row per (nprocs, threads) across runs
 * ================================================================ */
extern int nprocs;   /* defined in Global State below */
#define MAX_CONFIGS 32
typedef struct { int procs, threads; double time, lo, hi, avg; } HPerfEntry;
static int        hperf_n;
static HPerfEntry hperf_buf[MAX_CONFIGS];

static void hperf_read(void) {
    hperf_n = 0;
    FILE *fp = fopen(PERF_FILE, "r");
    if (!fp) return;
    char line[256];
    while (fgets(line, sizeof(line), fp) && hperf_n < MAX_CONFIGS) {
        int p, th, tot; double t, a, b, c;
        if (sscanf(line, " %d %d %d %lf %lf %lf %lf", &p, &th, &tot, &t, &a, &b, &c) == 7
                && p > 0 && t > 0)
            hperf_buf[hperf_n++] = (HPerfEntry){p, th, t, a, b, c};
    }
    fclose(fp);
}

static void hperf_write(void) {
    FILE *fp = fopen(PERF_FILE, "w");
    if (!fp) return;
    fprintf(fp, "=================================================\n");
    fprintf(fp, "  Hybrid MPI+OpenMP Traffic Simulation - Performance Results\n");
    fprintf(fp, "  Group 10\n");
    fprintf(fp, "=================================================\n");
    fprintf(fp, "Grid: %dx%d  Lanes: %d  Time Steps: %d\n\n", ROWS, COLS, LANES, TIME_STEPS);
    fprintf(fp, "  %-8s %-8s %-8s %-14s %-8s %-8s %-8s\n",
            "Procs", "Threads", "Total", "Time(s)", "Min", "Max", "Avg");
    fprintf(fp, "  --------------------------------------------------------------------------\n");
    for (int i = 0; i < hperf_n; i++)
        fprintf(fp, "  %-8d %-8d %-8d %-14.6f %-8.4f %-8.4f %-8.4f\n",
                hperf_buf[i].procs, hperf_buf[i].threads,
                hperf_buf[i].procs * hperf_buf[i].threads,
                hperf_buf[i].time, hperf_buf[i].lo, hperf_buf[i].hi, hperf_buf[i].avg);
    fclose(fp);
}

static void hperf_upsert(int omp_th, double t, double lo, double hi, double av) {
    hperf_read();
    for (int i = 0; i < hperf_n; i++) {
        if (hperf_buf[i].procs == nprocs && hperf_buf[i].threads == omp_th) {
            hperf_buf[i] = (HPerfEntry){nprocs, omp_th, t, lo, hi, av};
            hperf_write(); return;
        }
    }
    if (hperf_n < MAX_CONFIGS)
        hperf_buf[hperf_n++] = (HPerfEntry){nprocs, omp_th, t, lo, hi, av};
    for (int i = hperf_n-1; i > 0; i--) {
        HPerfEntry *a = &hperf_buf[i], *b = &hperf_buf[i-1];
        if (a->procs < b->procs || (a->procs == b->procs && a->threads < b->threads)) {
            HPerfEntry tmp = *a; *a = *b; *b = tmp;
        } else break;
    }
    hperf_write();
}

static void hperf_print(int omp_th, double exec_time) {
    hperf_read();
    double base = -1.0;
    for (int i = 0; i < hperf_n; i++)
        if (hperf_buf[i].procs == 1 && hperf_buf[i].threads == 1) { base = hperf_buf[i].time; break; }
    int total = nprocs * omp_th;
    printf("\nSelected config: %d MPI process%s x %d thread%s (%d total workers)\n",
           nprocs, nprocs != 1 ? "es" : "",
           omp_th, omp_th != 1 ? "s" : "", total);
    printf("  Execution time : %.4f s\n", exec_time);
    if (base > 0) {
        double sp = base / exec_time, ef = sp / total;
        printf("  Speedup        : %.4fx\n", sp);
        printf("  Efficiency     : %.2f%%\n", ef * 100.0);
    }
    printf("\n=================================================\n");
    printf("  Hybrid MPI+OpenMP Traffic Simulation - Performance Results\n");
    printf("  Group 10\n");
    printf("=================================================\n");
    printf("Grid: %dx%d  Lanes: %d  Time Steps: %d\n\n", ROWS, COLS, LANES, TIME_STEPS);
    printf("  %-8s %-8s %-8s %-14s %-8s %-8s %-8s\n",
           "Procs", "Threads", "Total", "Time(s)", "Min", "Max", "Avg");
    printf("  --------------------------------------------------------------------------\n");
    for (int i = 0; i < hperf_n; i++)
        printf("  %-8d %-8d %-8d %-14.6f %-8.4f %-8.4f %-8.4f\n",
               hperf_buf[i].procs, hperf_buf[i].threads,
               hperf_buf[i].procs * hperf_buf[i].threads,
               hperf_buf[i].time, hperf_buf[i].lo, hperf_buf[i].hi, hperf_buf[i].avg);
}


/* ================================================================
 * Global State
 * ================================================================ */

int rank, nprocs;
int local_rows;   /* rows owned by this process                    */
int row_start;    /* global row index of this process's first row  */

/* Local arrays: [local_rows + 2][COLS][LANES]
 * index 0              = top ghost row    (from rank-1)
 * index 1..local_rows  = owned data rows
 * index local_rows+1   = bottom ghost row (from rank+1) */
double (*local_traffic)[COLS][LANES];
double (*local_new)[COLS][LANES];
double (*local_weather)[COLS];

/* Full-grid arrays on rank 0 (scatter/gather source/destination) */
double global_traffic[ROWS][COLS][LANES];
double global_weather[ROWS][COLS];

/* Per-process row distribution metadata */
int *row_counts, *row_displs;
int *tc, *td;   /* traffic scatter/gather counts and displacements */
int *wc, *wd;   /* weather  scatter/gather counts and displacements */


/* ================================================================
 * compute_distribution()
 * Split ROWS evenly; remainder rows go to first (ROWS % nprocs)
 * processes one each.
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
        tc[p] = row_counts[p] * COLS * LANES;
        td[p] = off * COLS * LANES;
        wc[p] = row_counts[p] * COLS;
        wd[p] = off * COLS;
        off  += row_counts[p];
    }

    local_rows = row_counts[rank];
    row_start  = row_displs[rank];
}


/* ================================================================
 * initialize_global()
 * Rank 0 only. Fixed seed (srand(1)) matches all other baselines.
 * ================================================================ */
void initialize_global(void)
{
    srand(1);
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            global_weather[i][j] = 1.0;

            /* Rain zone in centre: 0.8 = 20% density reduction */
            if (i > ROWS/3 && i < 2*ROWS/3 &&
                j > COLS/3 && j < 2*COLS/3)
                global_weather[i][j] = 0.8;

            /* Accident zone in bottom-right corner: 0.5 = 50% density reduction */
            if (i > 3*ROWS/4 && j > 3*COLS/4)
                global_weather[i][j] = 0.5;
            for (int l = 0; l < LANES; l++)
                global_traffic[i][j][l] = (double)(rand() % 100);
        }
    }
}


/* ================================================================
 * distribute_data()
 * Rank 0 scatters rows to all processes via MPI_Scatterv.
 * Data lands at local index 1 (index 0 is reserved for ghost row).
 * ================================================================ */
void distribute_data(void)
{
    MPI_Scatterv(
        &global_traffic[0][0][0], tc, td, MPI_DOUBLE,
        &local_traffic[1][0][0],
        local_rows * COLS * LANES, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    MPI_Scatterv(
        &global_weather[0][0], wc, wd, MPI_DOUBLE,
        &local_weather[1][0],
        local_rows * COLS, MPI_DOUBLE,
        0, MPI_COMM_WORLD);
}


/* ================================================================
 * halo_exchange()
 * Non-blocking exchange of boundary rows with neighbouring ranks.
 * Identical communication pattern to the pure-MPI baseline.
 * ================================================================ */
void halo_exchange(void)
{
    MPI_Request reqs[4];
    int n    = 0;
    int prev = rank - 1;
    int next = rank + 1;

    if (prev >= 0)
        MPI_Irecv(&local_traffic[0][0][0],
                  COLS * LANES, MPI_DOUBLE,
                  prev, TAG_SOUTH, MPI_COMM_WORLD, &reqs[n++]);

    if (next < nprocs)
        MPI_Irecv(&local_traffic[local_rows + 1][0][0],
                  COLS * LANES, MPI_DOUBLE,
                  next, TAG_NORTH, MPI_COMM_WORLD, &reqs[n++]);

    if (prev >= 0)
        MPI_Isend(&local_traffic[1][0][0],
                  COLS * LANES, MPI_DOUBLE,
                  prev, TAG_NORTH, MPI_COMM_WORLD, &reqs[n++]);

    if (next < nprocs)
        MPI_Isend(&local_traffic[local_rows][0][0],
                  COLS * LANES, MPI_DOUBLE,
                  next, TAG_SOUTH, MPI_COMM_WORLD, &reqs[n++]);

    MPI_Waitall(n, reqs, MPI_STATUSES_IGNORE);
}


/* ================================================================
 * update_local()
 * KEY HYBRID FUNCTION
 *
 * MPI provides the row slice; OpenMP threads parallelise the
 * inner row loop so multiple threads update different rows
 * of this process's local partition simultaneously.
 *
 * Two-phase approach (read → local_new, copy back) prevents
 * read-after-write races across threads.
 * ================================================================ */
void update_local(void)
{
    /* Phase 1: compute new densities — OpenMP parallelises rows */
    #pragma omp parallel for schedule(static)
    for (int li = 1; li <= local_rows; li++) {
        int gi = row_start + li - 1;        /* global row index */
        if (gi == 0 || gi == ROWS - 1) continue;   /* global boundary */

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

    /* Phase 2: copy back — also OpenMP-parallel */
    #pragma omp parallel for schedule(static)
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
 * Collect all process slices back to rank 0.
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
 * Each process uses OpenMP reduction for local partial stats;
 * MPI_Reduce aggregates to rank 0 for global values.
 * ================================================================ */
void compute_stats(double *out_min, double *out_max, double *out_avg)
{
    double lsum = 0.0, lmin = 1e18, lmax = -1e18;

    #pragma omp parallel for schedule(static) \
            reduction(+:lsum) reduction(min:lmin) reduction(max:lmax)
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
 * save_output()  — rank 0 only
 * ================================================================ */
void save_output(double exec_time, int num_omp_threads)
{
    FILE *fp = fopen(OUTPUT_FILE, "w");
    if (!fp) { fprintf(stderr, "ERROR: Cannot open %s\n", OUTPUT_FILE); return; }

    fprintf(fp, "=================================================\n");
    fprintf(fp, "  Hybrid MPI+OpenMP Traffic Simulation Output\n");
    fprintf(fp, "  Group 10\n");
    fprintf(fp, "=================================================\n");
    fprintf(fp, "Grid Size      : %d x %d\n",  ROWS, COLS);
    fprintf(fp, "Lanes          : %d\n",         LANES);
    fprintf(fp, "Time Steps     : %d\n",         TIME_STEPS);
    fprintf(fp, "MPI Processes  : %d\n",         nprocs);
    fprintf(fp, "OMP Threads    : %d  (per process)\n", num_omp_threads);
    fprintf(fp, "Total Workers  : %d\n",         nprocs * num_omp_threads);
    fprintf(fp, "Exec Time      : %.6f seconds\n\n", exec_time);

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
 * save_image()  — rank 0 only
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
 * save_raw_values()  — rank 0 only
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
 * main()
 * Usage: mpirun -np <P> ./traffic_hybrid <T>
 *        P = number of MPI processes
 *        T = number of OpenMP threads per process (default: 1)
 * ================================================================ */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* ---- Parse OMP thread count from command line ---- */
    int num_omp_threads = 1;
    if (argc > 1) num_omp_threads = atoi(argv[1]);
    if (num_omp_threads < 1) num_omp_threads = 1;

    omp_set_dynamic(0);                       /* disable dynamic adjustment */
    omp_set_num_threads(num_omp_threads);

    /* ---- Compute row distribution ---- */
    compute_distribution();

    /* ---- Allocate local arrays (owned rows + 2 ghost rows) ---- */
    local_traffic = (double (*)[COLS][LANES])
                    malloc((local_rows + 2) * sizeof(*local_traffic));
    local_new     = (double (*)[COLS][LANES])
                    malloc((local_rows + 2) * sizeof(*local_new));
    local_weather = (double (*)[COLS])
                    malloc((local_rows + 2) * sizeof(*local_weather));

    memset(local_traffic, 0, (local_rows + 2) * sizeof(*local_traffic));
    memset(local_new,     0, (local_rows + 2) * sizeof(*local_new));
    memset(local_weather, 0, (local_rows + 2) * sizeof(*local_weather));

    /* ---- Rank 0 initialises full grid ---- */
    if (rank == 0)
        initialize_global();

    /* ---- Distribute rows and weather to all processes ---- */
    distribute_data();

    /* ---- Print configuration (rank 0 only) ---- */
    if (rank == 0) {
        printf("\n==========================================================\n");
        printf("  Hybrid MPI+OpenMP Traffic Density Simulation — Group 10\n");
        printf("==========================================================\n");
        printf("[Hybrid Configuration]\n");
        printf("  MPI Processes  : %d\n",     nprocs);
        printf("  OMP Threads    : %d  (per process)\n", num_omp_threads);
        printf("  Total Workers  : %d\n",     nprocs * num_omp_threads);
        printf("[Simulation Parameters]\n");
        printf("  Grid Size      : %d x %d\n", ROWS, COLS);
        printf("  Lanes          : %d\n",        LANES);
        printf("  Time Steps     : %d\n",        TIME_STEPS);
        printf("\n[Process & Thread Assignments]\n");
        printf("  %-8s %-12s %-20s %-10s\n",
               "Rank", "Rows Owned", "Global Row Range", "OMP Threads");
        printf("  ------------------------------------------------\n");
        fflush(stdout);
    }

    /* ---- Each rank announces itself in order ---- */
    for (int p = 0; p < nprocs; p++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == p) {
            printf("  %-8d %-12d [%3d .. %3d]           %-10d\n",
                   rank, local_rows,
                   row_start, row_start + local_rows - 1,
                   num_omp_threads);
            fflush(stdout);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ---- Each rank shows its OpenMP threads ---- */
    if (rank == 0) {
        printf("\n[OpenMP Threads per Process]\n");
        fflush(stdout);
    }
    for (int p = 0; p < nprocs; p++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == p) {
            printf("  Rank %d threads:\n", rank);
            #pragma omp parallel
            {
                int tid   = omp_get_thread_num();
                int total = omp_get_num_threads();
                int rows_per_thread = local_rows / total;
                int t_row_start = row_start + tid * rows_per_thread;
                int t_row_end   = (tid == total - 1)
                                  ? row_start + local_rows - 1
                                  : t_row_start + rows_per_thread - 1;
                #pragma omp critical
                printf("    Thread %d/%d  ->  global rows [%3d .. %3d]\n",
                       tid, total - 1, t_row_start, t_row_end);
            }
            fflush(stdout);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n[Running simulation...]\n");
        fflush(stdout);
    }

    /* ---- Synchronise, then start wall-clock timer ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* ================================================================
     * MAIN SIMULATION LOOP
     *   Step 1 : halo_exchange()  — MPI boundary row swap
     *   Step 2 : update_local()   — OpenMP-parallel diffusion step
     * ================================================================ */
    for (int t = 0; t < TIME_STEPS; t++) {
        halo_exchange();
        update_local();
    }

    /* ---- Synchronise, stop timer ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double exec_time = MPI_Wtime() - t_start;

    /* ---- Each rank reports its local time ---- */
    if (rank == 0) {
        printf("\n[Per-Process Timing]\n");
        printf("  %-8s %-14s %-10s\n", "Rank", "Local Time(s)", "Rows");
        printf("  --------------------------------\n");
        fflush(stdout);
    }
    for (int p = 0; p < nprocs; p++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == p) {
            printf("  %-8d %-14.6f %-10d\n", rank, exec_time, local_rows);
            fflush(stdout);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ---- Collect results to rank 0 ---- */
    gather_data();

    /* ---- Global statistics ---- */
    double min_d = 0.0, max_d = 0.0, avg_d = 0.0;
    compute_stats(&min_d, &max_d, &avg_d);

    /* ---- Rank 0: output files and summary ---- */
    if (rank == 0) {
        printf("[Performance]\n");
        printf("  Exec Time      : %.6f seconds\n", exec_time);
        printf("  Min Density    : %.4f\n", min_d);
        printf("  Max Density    : %.4f\n", max_d);
        printf("  Avg Density    : %.4f\n", avg_d);

        printf("\n[Saving Output Files]\n");
        save_output(exec_time, num_omp_threads);
        printf("  %-42s -> Traffic density (all lanes)\n", OUTPUT_FILE);

        save_image();
        printf("  %-42s -> Heatmap PPM image\n", IMAGE_FILE);

        save_raw_values();
        printf("  %-42s -> Lane-averaged values\n", VALUES_FILE);

        hperf_upsert(num_omp_threads, exec_time, min_d, max_d, avg_d);
        printf("  %-42s -> Performance log (updated)\n", PERF_FILE);

        hperf_print(num_omp_threads, exec_time);

        printf("\n==========================================================\n");
        printf("  Simulation Complete  |  MPI=%d  OMP=%d  Total=%d workers\n",
               nprocs, num_omp_threads, nprocs * num_omp_threads);
        printf("==========================================================\n\n");
    }

    /* ---- Cleanup ---- */
    free(local_traffic);
    free(local_new);
    free(local_weather);
    free(row_counts); free(row_displs);
    free(tc); free(td);
    free(wc); free(wd);

    MPI_Finalize();
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define ROWS        200
#define COLS        200
#define LANES       3
#define TIME_STEPS  200
#define CHUNK       10


#define OUTPUT_FILE  "openmp_output.txt"
#define IMAGE_FILE   "openmp_traffic_heatmap.ppm"
#define PERF_FILE    "openmp_performance.txt"
#define VALUES_FILE  "openmp_traffic_values.txt"

#define NUM_CONFIGS  4
static int thread_counts[NUM_CONFIGS] = {1, 2, 4, 8};

double traffic[ROWS][COLS][LANES];
double new_traffic[ROWS][COLS][LANES];
double weather[ROWS][COLS];
double ref_traffic[ROWS][COLS][LANES];

void initialize() {
    srand(1);
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            weather[i][j] = 1.0;

            /* Rain zone in center: 0.8 = 20% density reduction */
            if (i > ROWS/3 && i < 2*ROWS/3 &&
                j > COLS/3 && j < 2*COLS/3)
                weather[i][j] = 0.8;

            /* Accident zone in bottom-right corner: 0.5 = 50% density reduction */
            if (i > 3*ROWS/4 && j > 3*COLS/4)
                weather[i][j] = 0.5;

            for (int l = 0; l < LANES; l++)
                traffic[i][j][l] = (double)(rand() % 100);
        }
    }
}

void update_static() {
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < ROWS-1; i++) {
        for (int j = 1; j < COLS-1; j++) {
            for (int l = 0; l < LANES; l++) {
                double current   = traffic[i][j][l];
                double neighbors = traffic[i-1][j][l] + traffic[i+1][j][l]
                                 + traffic[i][j-1][l] + traffic[i][j+1][l];
                double updated   = current + 0.1 * (neighbors / 4.0 - current);
                new_traffic[i][j][l] = updated * weather[i][j];
            }
        }
    }
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < ROWS-1; i++)
        for (int j = 1; j < COLS-1; j++)
            for (int l = 0; l < LANES; l++)
                traffic[i][j][l] = new_traffic[i][j][l];
}

void update_dynamic() {
    #pragma omp parallel for schedule(dynamic, CHUNK)
    for (int i = 1; i < ROWS-1; i++) {
        for (int j = 1; j < COLS-1; j++) {
            for (int l = 0; l < LANES; l++) {
                double current   = traffic[i][j][l];
                double neighbors = traffic[i-1][j][l] + traffic[i+1][j][l]
                                 + traffic[i][j-1][l] + traffic[i][j+1][l];
                double updated   = current + 0.1 * (neighbors / 4.0 - current);
                new_traffic[i][j][l] = updated * weather[i][j];
            }
        }
    }
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < ROWS-1; i++)
        for (int j = 1; j < COLS-1; j++)
            for (int l = 0; l < LANES; l++)
                traffic[i][j][l] = new_traffic[i][j][l];
}

void update_collapse() {
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 1; i < ROWS-1; i++) {
        for (int j = 1; j < COLS-1; j++) {
            for (int l = 0; l < LANES; l++) {
                double current   = traffic[i][j][l];
                double neighbors = traffic[i-1][j][l] + traffic[i+1][j][l]
                                 + traffic[i][j-1][l] + traffic[i][j+1][l];
                double updated   = current + 0.1 * (neighbors / 4.0 - current);
                new_traffic[i][j][l] = updated * weather[i][j];
            }
        }
    }
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 1; i < ROWS-1; i++)
        for (int j = 1; j < COLS-1; j++)
            for (int l = 0; l < LANES; l++)
                traffic[i][j][l] = new_traffic[i][j][l];
}

void compute_stats(double *out_min, double *out_max, double *out_avg) {
    double sum = 0.0;
    double lo  = 1e18;
    double hi  = -1e18;

    #pragma omp parallel for schedule(static) \
            reduction(+:sum) reduction(min:lo) reduction(max:hi)
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            for (int l = 0; l < LANES; l++) {
                double v = traffic[i][j][l];
                sum += v;
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }
        }
    }

    *out_min = lo;
    *out_max = hi;
    *out_avg = sum / (double)(ROWS * COLS * LANES);
}

double accuracy_check() {
    double max_diff = 0.0;
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++)
            for (int l = 0; l < LANES; l++) {
                double d = fabs(traffic[i][j][l] - ref_traffic[i][j][l]);
                if (d > max_diff) max_diff = d;
            }
    return max_diff;
}

void save_output(double exec_time, int num_threads, const char *sched,
                 double speedup, double efficiency) {
    FILE *fp = fopen(OUTPUT_FILE, "w");
    if (!fp) { fprintf(stderr, "ERROR: Cannot open %s\n", OUTPUT_FILE); return; }

    fprintf(fp, "=================================================\n");
    fprintf(fp, "  OpenMP Parallel Traffic Simulation Output\n");
    fprintf(fp, "  Group 10\n");
    fprintf(fp, "=================================================\n");
    fprintf(fp, "Grid Size   : %d x %d\n",   ROWS, COLS);
    fprintf(fp, "Lanes       : %d\n",         LANES);
    fprintf(fp, "Time Steps  : %d\n",         TIME_STEPS);
    fprintf(fp, "Threads     : %d\n",         num_threads);
    fprintf(fp, "Schedule    : %s\n",         sched);
    fprintf(fp, "Exec Time   : %.6f seconds\n", exec_time);
    fprintf(fp, "Speedup     : %.4f x\n",     speedup);
    fprintf(fp, "Efficiency  : %.4f (%.2f%%)\n\n", efficiency, efficiency * 100.0);

    for (int l = 0; l < LANES; l++) {
        fprintf(fp, "--- Lane %d ---\n", l);
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++)
                fprintf(fp, "%.2f ", traffic[i][j][l]);
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

void save_image(const char *filename) {
    FILE *img = fopen(filename, "w");
    if (!img) { fprintf(stderr, "ERROR: Cannot open %s\n", filename); return; }

    fprintf(img, "P3\n%d %d\n255\n", COLS, ROWS);
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            double sum = 0.0;
            for (int l = 0; l < LANES; l++)
                sum += traffic[i][j][l];
            double avg = sum / LANES;
            int intensity = (int)(255.0 * avg / 100.0);
            if (intensity > 255) intensity = 255;
            if (intensity < 0)   intensity = 0;
            fprintf(img, "%d %d %d ", intensity, 0, 255 - intensity);
        }
        fprintf(img, "\n");
    }
    fclose(img);
}

void save_raw_values(const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "ERROR: Cannot open %s\n", filename); return; }

    fprintf(fp, "%d %d %d\n", ROWS, COLS, LANES);
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            double sum = 0.0;
            for (int l = 0; l < LANES; l++)
                sum += traffic[i][j][l];
            fprintf(fp, "%.2f ", sum / LANES);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void run_scalability_test(FILE *perf_fp,
                          void (*update_fn)(void),
                          const char *sched_name,
                          double *base_time_out) {
    double base_time = 0.0;
    double start, end, exec_time;
    double min_d, max_d, avg_d;

    printf("\n  [Schedule: %-8s]\n", sched_name);
    printf("  %-8s %-14s %-10s %-12s %-18s\n",
           "Threads", "Time(s)", "Speedup", "Efficiency", "Max Diff vs 1T");
    printf("  ----------------------------------------------------------------\n");

    for (int tc = 0; tc < NUM_CONFIGS; tc++) {
        int nt = thread_counts[tc];
        omp_set_num_threads(nt);

        initialize();

        start = omp_get_wtime();
        for (int t = 0; t < TIME_STEPS; t++)
            update_fn();
        end = omp_get_wtime();
        exec_time = end - start;

        if (tc == 0) {
            base_time = exec_time;
            memcpy(ref_traffic, traffic, sizeof(traffic));
        }

        double speedup    = base_time / exec_time;
        double efficiency = speedup / (double)nt;
        double max_diff   = (tc > 0) ? accuracy_check() : 0.0;

        compute_stats(&min_d, &max_d, &avg_d);

        printf("  %-8d %-14.6f %-10.4f %-12.4f %-18.2e\n",
               nt, exec_time, speedup, efficiency, max_diff);

        fprintf(perf_fp, "  %-8d %-10s %-14.6f %-10.4f %-12.4f %-8.4f %-8.4f %-8.4f %-14.2e\n",
                nt, sched_name, exec_time, speedup, efficiency,
                min_d, max_d, avg_d, max_diff);
    }

    *base_time_out = base_time;
}

int main() {
    double start, end, exec_time;
    double min_d, max_d, avg_d;

    printf("\n==========================================================\n");
    printf("  OpenMP Parallel Traffic Density Simulation - Group 10\n");
    printf("==========================================================\n");

    omp_set_dynamic(0);
    #pragma omp parallel
    {
        #pragma omp master
        {
            printf("\n[OpenMP Environment Info]\n");
            printf("  Available Processors : %d\n", omp_get_num_procs());
            printf("  Active Threads       : %d\n", omp_get_num_threads());
            printf("  Max Threads          : %d\n", omp_get_max_threads());
            printf("  Dynamic Threads      : %s\n",
                   omp_get_dynamic() ? "Enabled" : "Disabled");
            printf("  Nested Parallelism   : %s\n",
                   omp_get_nested() ? "Enabled" : "Disabled");
        }
    }

    printf("\n[Simulation Parameters]\n");
    printf("  Grid Size      : %d x %d\n", ROWS, COLS);
    printf("  Lanes          : %d\n",       LANES);
    printf("  Time Steps     : %d\n",       TIME_STEPS);
    printf("  Chunk Size     : %d  (dynamic schedule)\n", CHUNK);
    printf("  Thread Configs : ");
    for (int i = 0; i < NUM_CONFIGS; i++)
        printf("%d%s", thread_counts[i], i < NUM_CONFIGS - 1 ? ", " : "\n");

    FILE *perf_fp = fopen(PERF_FILE, "w");
    if (!perf_fp) { fprintf(stderr, "ERROR: Cannot open %s\n", PERF_FILE); return 1; }

    fprintf(perf_fp, "=================================================\n");
    fprintf(perf_fp, "  OpenMP Traffic Simulation - Performance Results\n");
    fprintf(perf_fp, "  Group 10\n");
    fprintf(perf_fp, "=================================================\n");
    fprintf(perf_fp, "Grid: %dx%d  Lanes: %d  Time Steps: %d\n\n",
            ROWS, COLS, LANES, TIME_STEPS);
    fprintf(perf_fp, "  %-8s %-10s %-14s %-10s %-12s %-8s %-8s %-8s %-14s\n",
            "Threads", "Schedule", "Time(s)", "Speedup",
            "Efficiency", "Min", "Max", "Avg", "MaxDiff(vs1T)");
    fprintf(perf_fp, "  ------------------------------------------------------------------------------------\n");

    printf("\n[Scalability Analysis]\n");
    printf("  Grid: %dx%d | Lanes: %d | Time Steps: %d\n",
           ROWS, COLS, LANES, TIME_STEPS);

    double base_static, base_dynamic, base_collapse;

    run_scalability_test(perf_fp, update_static,   "static",   &base_static);
    run_scalability_test(perf_fp, update_dynamic,  "dynamic",  &base_dynamic);
    run_scalability_test(perf_fp, update_collapse, "collapse", &base_collapse);

    fprintf(perf_fp, "\n[Performance Notes]\n");
    fprintf(perf_fp, "  Speedup    = T(1 thread) / T(N threads)\n");
    fprintf(perf_fp, "  Efficiency = Speedup / N_threads  [ideal = 1.0 = 100%%]\n");
    fprintf(perf_fp, "  MaxDiff    = max|traffic_N[i][j][l] - traffic_1[i][j][l]|\n");
    fprintf(perf_fp, "               Expected: 0.00e+00 (deterministic computation)\n");
    fclose(perf_fp);

    int best_threads = thread_counts[NUM_CONFIGS - 1];
    omp_set_num_threads(best_threads);
    initialize();

    start = omp_get_wtime();
    for (int t = 0; t < TIME_STEPS; t++)
        update_static();
    end = omp_get_wtime();
    exec_time = end - start;

    double final_speedup    = base_static / exec_time;
    double final_efficiency = final_speedup / (double)best_threads;

    compute_stats(&min_d, &max_d, &avg_d);

    printf("\n[Saving Output Files]\n");
    save_output(exec_time, best_threads, "static", final_speedup, final_efficiency);
    printf("  %-42s -> Traffic density data (all lanes)\n", OUTPUT_FILE);

    save_image(IMAGE_FILE);
    printf("  %-42s -> Heatmap image (PPM format)\n", IMAGE_FILE);

    save_raw_values(VALUES_FILE);
    printf("  %-42s -> Raw lane-averaged values\n", VALUES_FILE);

    printf("  %-42s -> Scalability log (all thread configs)\n", PERF_FILE);

    printf("\n[Final Summary : %d threads, static schedule]\n", best_threads);
    printf("  Execution Time : %.6f seconds\n",      exec_time);
    printf("  Speedup        : %.4f x\n",             final_speedup);
    printf("  Efficiency     : %.4f  (%.2f%%)\n",     final_efficiency,
                                                       final_efficiency * 100.0);
    printf("  Min Density    : %.4f\n",               min_d);
    printf("  Max Density    : %.4f\n",               max_d);
    printf("  Avg Density    : %.4f\n",               avg_d);

    printf("\n==========================================================\n");
    printf("  Simulation Complete\n");
    printf("==========================================================\n\n");

    return 0;
}

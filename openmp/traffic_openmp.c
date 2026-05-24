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

/* ================================================================
 * Performance tracking — upserts one row per (threads, schedule) across runs
 * ================================================================ */
#define OMAX_CONFIGS 32
typedef struct {
    int    threads;
    char   sched[16];
    double time, speedup, efficiency, lo, hi, avg, max_diff;
} OPerfEntry;
static int        operf_n;
static OPerfEntry operf_buf[OMAX_CONFIGS];

static int sched_order(const char *s) {
    if (strcmp(s, "static")   == 0) return 0;
    if (strcmp(s, "dynamic")  == 0) return 1;
    if (strcmp(s, "collapse") == 0) return 2;
    return 3;
}

static void operf_read(void) {
    operf_n = 0;
    FILE *fp = fopen(PERF_FILE, "r");
    if (!fp) return;
    char line[512];
    while (fgets(line, sizeof(line), fp) && operf_n < OMAX_CONFIGS) {
        int th; char sc[16]; double t, sp, ef, mn, mx, av, md;
        if (sscanf(line, " %d %15s %lf %lf %lf %lf %lf %lf %le",
                   &th, sc, &t, &sp, &ef, &mn, &mx, &av, &md) == 9 && th > 0 && t > 0) {
            OPerfEntry *e = &operf_buf[operf_n++];
            e->threads = th;
            strncpy(e->sched, sc, 15); e->sched[15] = '\0';
            e->time = t; e->speedup = sp; e->efficiency = ef;
            e->lo = mn; e->hi = mx; e->avg = av; e->max_diff = md;
        }
    }
    fclose(fp);
}

static void operf_write(void) {
    FILE *fp = fopen(PERF_FILE, "w");
    if (!fp) return;
    fprintf(fp, "=================================================\n");
    fprintf(fp, "  OpenMP Traffic Simulation - Performance Results\n");
    fprintf(fp, "  Group 10\n");
    fprintf(fp, "=================================================\n");
    fprintf(fp, "Grid: %dx%d  Lanes: %d  Time Steps: %d\n\n", ROWS, COLS, LANES, TIME_STEPS);
    fprintf(fp, "  %-8s %-10s %-14s %-10s %-12s %-8s %-8s %-8s %-14s\n",
            "Threads", "Schedule", "Time(s)", "Speedup", "Efficiency",
            "Min", "Max", "Avg", "MaxDiff(vs1T)");
    fprintf(fp, "  ------------------------------------------------------------------------------------\n");
    for (int i = 0; i < operf_n; i++)
        fprintf(fp, "  %-8d %-10s %-14.6f %-10.4f %-12.4f %-8.4f %-8.4f %-8.4f %-14.2e\n",
                operf_buf[i].threads, operf_buf[i].sched,
                operf_buf[i].time, operf_buf[i].speedup, operf_buf[i].efficiency,
                operf_buf[i].lo, operf_buf[i].hi, operf_buf[i].avg, operf_buf[i].max_diff);
    fclose(fp);
}

static void operf_upsert(int th, const char *sc, double t, double sp, double ef,
                          double lo, double hi, double av, double md) {
    operf_read();
    int idx = -1;
    for (int i = 0; i < operf_n; i++) {
        if (operf_buf[i].threads == th && strcmp(operf_buf[i].sched, sc) == 0) {
            idx = i; break;
        }
    }
    if (idx < 0 && operf_n < OMAX_CONFIGS) idx = operf_n++;
    if (idx >= 0) {
        OPerfEntry *e = &operf_buf[idx];
        e->threads = th; strncpy(e->sched, sc, 15); e->sched[15] = '\0';
        e->time = t; e->speedup = sp; e->efficiency = ef;
        e->lo = lo; e->hi = hi; e->avg = av; e->max_diff = md;
    }
    /* Insertion sort by (schedule order, threads) */
    for (int i = 1; i < operf_n; i++) {
        OPerfEntry key = operf_buf[i];
        int ki = sched_order(key.sched), j = i - 1;
        while (j >= 0) {
            int ji = sched_order(operf_buf[j].sched);
            if (ji > ki || (ji == ki && operf_buf[j].threads > key.threads)) {
                operf_buf[j+1] = operf_buf[j]; j--;
            } else break;
        }
        operf_buf[j+1] = key;
    }
    operf_write();
}

static void operf_print(int sel_threads, const char *sel_sched,
                         double sel_time, double sel_speedup, double sel_efficiency) {
    operf_read();
    if (sel_threads > 0) {
        printf("\nSelected config: %d thread%s / %s schedule\n",
               sel_threads, sel_threads != 1 ? "s" : "", sel_sched);
        printf("  Execution time : %.4f s\n", sel_time);
        printf("  Speedup        : %.4fx\n",  sel_speedup);
        printf("  Efficiency     : %.2f%%\n", sel_efficiency * 100.0);
    }
    printf("\n=================================================\n");
    printf("  OpenMP Traffic Simulation - Performance Results\n");
    printf("  Group 10\n");
    printf("=================================================\n");
    printf("Grid: %dx%d  Lanes: %d  Time Steps: %d\n\n", ROWS, COLS, LANES, TIME_STEPS);
    printf("  %-8s %-10s %-14s %-10s %-12s %-8s %-8s %-8s %-14s\n",
           "Threads", "Schedule", "Time(s)", "Speedup", "Efficiency",
           "Min", "Max", "Avg", "MaxDiff(vs1T)");
    printf("  ------------------------------------------------------------------------------------\n");
    for (int i = 0; i < operf_n; i++)
        printf("  %-8d %-10s %-14.6f %-10.4f %-12.4f %-8.4f %-8.4f %-8.4f %-14.2e\n",
               operf_buf[i].threads, operf_buf[i].sched,
               operf_buf[i].time, operf_buf[i].speedup, operf_buf[i].efficiency,
               operf_buf[i].lo, operf_buf[i].hi, operf_buf[i].avg, operf_buf[i].max_diff);
}

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

void run_scalability_test(void (*update_fn)(void),
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

        operf_upsert(nt, sched_name, exec_time, speedup, efficiency, min_d, max_d, avg_d, max_diff);
    }

    *base_time_out = base_time;
}

void run_single_configuration(int num_threads, const char *sched_name) {
    void (*update_fn)(void) = update_static;
    if (strcmp(sched_name, "dynamic") == 0) {
        update_fn = update_dynamic;
    } else if (strcmp(sched_name, "collapse") == 0) {
        update_fn = update_collapse;
    }

    double base_time, start, end, exec_time;
    double min_d, max_d, avg_d;

    omp_set_num_threads(1);
    initialize();
    start = omp_get_wtime();
    for (int t = 0; t < TIME_STEPS; t++)
        update_fn();
    end = omp_get_wtime();
    base_time = end - start;

    omp_set_num_threads(num_threads);
    initialize();
    start = omp_get_wtime();
    for (int t = 0; t < TIME_STEPS; t++)
        update_fn();
    end = omp_get_wtime();
    exec_time = end - start;

    double speedup = base_time / exec_time;
    double efficiency = speedup / (double)num_threads;
    compute_stats(&min_d, &max_d, &avg_d);

    operf_upsert(num_threads, sched_name, exec_time, speedup, efficiency, min_d, max_d, avg_d, 0.0);

    save_output(exec_time, num_threads, sched_name, speedup, efficiency);
    save_image(IMAGE_FILE);
    save_raw_values(VALUES_FILE);

    operf_print(num_threads, sched_name, exec_time, speedup, efficiency);
    printf("\nOutput files updated: %s, %s, %s, %s\n",
           OUTPUT_FILE, VALUES_FILE, IMAGE_FILE, PERF_FILE);
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        int selected_threads = atoi(argv[1]);
        if (selected_threads < 1) selected_threads = 1;
        const char *selected_schedule = "static";
        if (argc > 2) {
            if (strcmp(argv[2], "dynamic") == 0 || strcmp(argv[2], "collapse") == 0 || strcmp(argv[2], "static") == 0) {
                selected_schedule = argv[2];
            }
        }

        printf("\n==========================================================\n");
        printf("  OpenMP Traffic Density Simulation - Single Config Run\n");
        printf("==========================================================\n");
        omp_set_dynamic(0);
        run_single_configuration(selected_threads, selected_schedule);
        return 0;
    }

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

    printf("\n[Scalability Analysis]\n");
    printf("  Grid: %dx%d | Lanes: %d | Time Steps: %d\n",
           ROWS, COLS, LANES, TIME_STEPS);

    double base_static, base_dynamic, base_collapse;

    run_scalability_test(update_static,   "static",   &base_static);
    run_scalability_test(update_dynamic,  "dynamic",  &base_dynamic);
    run_scalability_test(update_collapse, "collapse", &base_collapse);

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

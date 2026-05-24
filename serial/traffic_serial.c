/*
=========================================================
Serial Traffic Density Simulation (Baseline)
=========================================================
Features:
- Multi-lane traffic grid
- Weather effects
- Iterative time stepping
 - Execution time measurement using a portable wall-clock timer
- Output saved to file
- Heatmap image output (PPM format)
=========================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define ROWS 200
#define COLS 200
#define LANES 3
#define TIME_STEPS 200

#define OUTPUT_FILE "serial_output.txt"
#define IMAGE_FILE  "traffic_heatmap.ppm"

// Allocate 3D traffic matrix
double traffic[ROWS][COLS][LANES];
double new_traffic[ROWS][COLS][LANES];
double weather[ROWS][COLS];

double now_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

// Initialize traffic and weather
void initialize() {
    for(int i = 0; i < ROWS; i++) {
        for(int j = 0; j < COLS; j++) {

            // Weather modifier: 1.0=normal, 0.8=rain, 0.5=accident
            weather[i][j] = 1.0;

            // Rain zone: center third of grid
            if (i > ROWS/3 && i < 2*ROWS/3 &&
                j > COLS/3 && j < 2*COLS/3)
                weather[i][j] = 0.8;

            // Accident zone: bottom-right corner
            if (i > 3*ROWS/4 && j > 3*COLS/4)
                weather[i][j] = 0.5;

            for(int l = 0; l < LANES; l++) {
                // Random initial density (0–100)
                traffic[i][j][l] = rand() % 100;
            }
        }
    }
}

// Update traffic using simple diffusion model
void update() {
    for(int i = 1; i < ROWS-1; i++) {
        for(int j = 1; j < COLS-1; j++) {
            for(int l = 0; l < LANES; l++) {

                double current = traffic[i][j][l];

                // Average of neighbors
                double neighbors =
                    traffic[i-1][j][l] +
                    traffic[i+1][j][l] +
                    traffic[i][j-1][l] +
                    traffic[i][j+1][l];

                double updated =
                    current + 0.1 * (neighbors/4.0 - current);

                // Apply weather modifier
                updated *= weather[i][j];

                new_traffic[i][j][l] = updated;
            }
        }
    }

    // Copy back
    for(int i = 1; i < ROWS-1; i++)
        for(int j = 1; j < COLS-1; j++)
            for(int l = 0; l < LANES; l++)
                traffic[i][j][l] = new_traffic[i][j][l];
}

// Save matrix to file
void save_output(double exec_time) {

    FILE *fp = fopen(OUTPUT_FILE, "w");

    fprintf(fp, "Serial Traffic Simulation Output\n");
    fprintf(fp, "Grid Size: %d x %d\n", ROWS, COLS);
    fprintf(fp, "Lanes: %d\n", LANES);
    fprintf(fp, "Time Steps: %d\n", TIME_STEPS);
    fprintf(fp, "Execution Time: %f seconds\n\n", exec_time);

    for(int l = 0; l < LANES; l++) {
        fprintf(fp, "Lane %d\n", l);
        for(int i = 0; i < ROWS; i++) {
            for(int j = 0; j < COLS; j++) {
                fprintf(fp, "%.2f ", traffic[i][j][l]);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

// Add this function to your C code
void save_raw_traffic_data() {
    FILE *fp = fopen("traffic_values.txt", "w");
    
    fprintf(fp, "%d %d %d\n", ROWS, COLS, LANES);  // Header
    
    for(int i = 0; i < ROWS; i++) {
        for(int j = 0; j < COLS; j++) {
            // Average over lanes for this cell
            double sum = 0.0;
            for(int l = 0; l < LANES; l++) {
                sum += traffic[i][j][l];
            }
            double avg = sum / LANES;
            fprintf(fp, "%.2f ", avg);
        }
        fprintf(fp, "\n");
    }
    
    fclose(fp);
    printf("Raw traffic values saved to traffic_values.txt\n");
}



// Save heatmap image (PPM)
void save_image() {

    FILE *img = fopen(IMAGE_FILE, "w");

    fprintf(img, "P3\n%d %d\n255\n", COLS, ROWS);

    for(int i = 0; i < ROWS; i++) {
        for(int j = 0; j < COLS; j++) {

            // Average over lanes
            double sum = 0.0;
            for(int l = 0; l < LANES; l++)
                sum += traffic[i][j][l];

            double avg = sum / LANES;

            // Normalize (0–100 range assumed)
            int intensity = (int)(255.0 * avg / 100.0);
            if(intensity > 255) intensity = 255;
            if(intensity < 0) intensity = 0;

            // Heatmap coloring: Red = High traffic
            int r = intensity;
            int g = 0;
            int b = 255 - intensity;

            fprintf(img, "%d %d %d ", r, g, b);
        }
        fprintf(img, "\n");
    }

    fclose(img);
}

// Main function
int main() {

    printf("Initializing traffic simulation...\n");
    initialize();

    double start = now_seconds();

    for(int t = 0; t < TIME_STEPS; t++) {
        update();
    }

    double end = now_seconds();
    double exec_time = end - start;

    printf("Simulation completed.\n");
    printf("Execution Time: %f seconds\n", exec_time);

    save_output(exec_time);
    save_image();
    save_raw_traffic_data();

    printf("Results saved to %s\n", OUTPUT_FILE);
    printf("Heatmap image saved to %s\n", IMAGE_FILE);

    return 0;
}
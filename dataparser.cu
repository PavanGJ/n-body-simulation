#include "global.h"
// This file is used to parse the input data and get it to the form that is required.
void parseCSVData(float values[N_SAMPLES][N_FEATURES]){
    /*
     * This subroutine parses data from an input file defined by INPUT.
     * It parses a csv file and extracts x, y, z, vx, vy, vz, m in the same order & discards an id field defined in the input file.
     */
    FILE* stream;
    int idx = 0;
    stream = fopen(INPUT, "r");
    for(idx = 0; fscanf(stream,"%f,%f,%f,%f,%f,%f,%f,%*f\n",
                        &values[idx][INDEX_X],
                        &values[idx][INDEX_Y],
                        &values[idx][INDEX_Z],
                        &values[idx][INDEX_VX],
                        &values[idx][INDEX_VY],
                        &values[idx][INDEX_VZ],
                        &values[idx][INDEX_M]) != EOF && idx < N_SAMPLES; idx++);
    return;
}
void generate3DData(float values[N_SAMPLES][N_FEATURES], int x[2], int y[2], int z[2], int vx[2], int vy[2], int vz[2], int m[2]){
    /*
     * This subroutine generates 3-dimensional data given the min and max values for spatial coordinates x, y, z, velocities vx, vy, vz and mass m.
     */
    return;
}
/*
 *  Usage example:
 *
 *  int main(){
 *      float arr[N_SAMPLES][N_FEATURES];
 *      int i;
 *      parseCSVData(arr);
 *      // CODE USING `arr`
 *      return 0;
 *  }
 *
 */

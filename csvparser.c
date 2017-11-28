#pragma once

#include <stdio.h>

#include "global.h"

// This file defines subroutines to read or write data

void readFromCSV(float4 *phy_attributes, float3 *velocities){
    /*
     * This subroutine reads data from an input file defined by INPUT.
     * It parses a csv file and extracts x, y, z, vx, vy, vz, m in the same order defined in the input file.
     */
    FILE* stream;
    int idx = 0;
    stream = fopen(INPUT_PATH, "r");
    for(idx = 0; idx < N_SAMPLES && fscanf(stream,"%f,%f,%f,%f,%f,%f,%f\n",
                        &phy_attributes[idx].x,
                        &phy_attributes[idx].y,
                        &phy_attributes[idx].z,
                        &velocities[idx].x,
                        &velocities[idx].y,
                        &velocities[idx].z,
                        &phy_attributes[idx].w) != EOF; idx++);
    return;
}
void writeToCSV(float4 *phy_attributes, float3 *velocities, int step){
    /*
     * This subroutine writes data to an output file.
     */
    FILE* stream;
    int idx = 0;
    char output[256] = OUTPUT_DIR;
    char fileName[128];
    sprintf(fileName, "output_%04d.csv", step);
    strcat(output, fileName);
    stream = fopen(output, "w");
    for(idx = 0; idx < N_SAMPLES && fprintf(stream,"%f,%f,%f,%f,%f,%f,%f\n",
                            phy_attributes[idx].x,
                            phy_attributes[idx].y,
                            phy_attributes[idx].z,
                            velocities[idx].x,
                            velocities[idx].y,
                            velocities[idx].z,
                            phy_attributes[idx].w) != EOF; idx++);
    return;
}

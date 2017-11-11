#include "global.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// This file is used to parse the input data and get it to the form that is required.

void parseCSVData(float4 *phy_attributes, float3 *velocities){
    /*
     * This subroutine parses data from an input file defined by INPUT.
     * It parses a csv file and extracts x, y, z, vx, vy, vz, m in the same order & discards an id field defined in the input file.
     */
    FILE* stream;
    int idx = 0;
    stream = fopen(INPUT, "r");
    for(idx = 0; fscanf(stream,"%f,%f,%f,%f,%f,%f,%f,%*f\n",
                        &phy_attributes[idx].x,
                        &phy_attributes[idx].y,
                        &phy_attributes[idx].z,
                        &velocities[idx].x,
                        &velocities[idx].y,
                        &velocities[idx].z,
                        &phy_attributes[idx].w) != EOF && idx < N_SAMPLES; idx++);
    return;
}
void writeCSV(float4 *phy_attributes, float3 *velocities, int step){
    /*
     * This subroutine writes data to an output file.
     */
    FILE* stream;
    int idx = 0;
    char output[128];
    sprintf(output, "output_%d.csv", step);
    stream = fopen(output, "w");
    for(idx = 0; fprintf(stream,"%f,%f,%f,%f,%f,%f,%f\n",
                            phy_attributes[idx].x,
                            phy_attributes[idx].y,
                            phy_attributes[idx].z,
                            velocities[idx].x,
                            velocities[idx].y,
                            velocities[idx].z,
                            phy_attributes[idx].w) != EOF && idx < N_SAMPLES; idx++);
    return;
}
void generate3DData(float4 *phy_attributes, float3 *velocities){
    /*
     * This subroutine generates 3-dimensional data
     */
    srand(time(NULL));
    float seed = 10.0f;
    int idx = 0;
    for(idx = 0; idx < N_SAMPLES; idx++){
        phy_attributes[idx].x = ((float)rand()/(float)(RAND_MAX)) * seed;
	phy_attributes[idx].y = ((float)rand()/(float)(RAND_MAX)) * seed;
	phy_attributes[idx].z = ((float)rand()/(float)(RAND_MAX)) * seed;
	phy_attributes[idx].w = 1.0f/N_SAMPLES;
    }
    return;
}
/*
 *  Usage example:
 *
 *  int main(){
 *      float4 phy_attributes[N_SAMPLES];
 *      float3 velocities[N_SAMPLES];
 *      parseCSVData((float4 *)phy_attributes,(float3 *)velocities);
 *      // REST OF THE CODE
 *      return 0;
 *  }
 *
 */

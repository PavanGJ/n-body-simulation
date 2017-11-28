#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "global.h"

//  This file defines functions to generate 3-dimensional and 2-dimensional data

//  Defining global seeds
float seed = 10.0f
float velocity_seed = 0.001f;

void generate3DData(float4 *phy_attributes, float3 *velocities){
    /*
     * This subroutine generates 3-dimensional data
     */
    srand(time(NULL));
    int idx = 0;
    for(idx = 0; idx < N_SAMPLES; idx++){
        phy_attributes[idx].x = generateValue(seed);
        phy_attributes[idx].y = generateValue(seed);
        phy_attributes[idx].z = generateValue(seed);
        phy_attributes[idx].w = 1.0f/N_SAMPLES;
        velocities[idx].x = generateValue(velocity_seed);
        velocities[idx].y = generateValue(velocity_seed);
        velocities[idx].z = generateValue(velocity_seed);
        
    }
    return;
}
void generate2DData(float4 *phy_attributes, float3 *velocities){
    /*
     * This subroutine generates 2-dimensional data
     */
    srand(time(NULL));
    int idx = 0;
    for(idx = 0; idx < N_SAMPLES; idx++){
        phy_attributes[idx].x = generateValue(seed);
        phy_attributes[idx].y = generateValue(seed);
        phy_attributes[idx].z = 0.0f
        phy_attributes[idx].w = 1.0f/N_SAMPLES;
        velocities[idx].x = generateValue(velocity_seed);
        velocities[idx].y = generateValue(velocity_seed);
        velocities[idx].z = 0.0f
    }
    return;
}
float generateValue(float seed){
    (((float)rand()/(float)(RAND_MAX)) * seed) - (seed/2);
}

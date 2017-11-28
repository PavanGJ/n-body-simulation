#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "global.h"
#include "csvparser.c"

//  This file defines functions to generate 3-dimensional and 2-dimensional data

//  Defining global seeds
float seed = 10.0f;
float velocity_seed = 0.001f;

float generateValue(float);

void generate3DData(){
    /*
     * This subroutine generates 3-dimensional data
     */
    FILE* stream;
    float4 phy_attributes;
    float3 velocities;
    stream = fopen(INPUT_PATH, "w+");
    srand(time(NULL));
    int idx = 0;
    for(idx = 0; idx < N_SAMPLES; idx++){
        phy_attributes.x = generateValue(seed);
        phy_attributes.y = generateValue(seed);
        phy_attributes.z = generateValue(seed);
        phy_attributes.w = 1.0f/N_SAMPLES;
        velocities.x = generateValue(velocity_seed);
        velocities.y = generateValue(velocity_seed);
        velocities.z = generateValue(velocity_seed);
        fprintf(stream,"%f,%f,%f,%f,%f,%f,%f\n",
                phy_attributes.x,
                phy_attributes.y,
                phy_attributes.z,
                velocities.x,
                velocities.y,
                velocities.z,
                phy_attributes.w);
        
    }
    return;
}
void generate2DData(){
    /*
     * This subroutine generates 2-dimensional data
     */
    FILE* stream;
    float4 phy_attributes;
    float3 velocities;
    stream = fopen(INPUT_PATH, "w+");
    srand(time(NULL));
    int idx = 0;
    for(idx = 0; idx < N_SAMPLES; idx++){
        phy_attributes.x = generateValue(seed);
        phy_attributes.y = generateValue(seed);
        phy_attributes.z = 0.0f;
        phy_attributes.w = 1.0f/N_SAMPLES;
        velocities.x = generateValue(velocity_seed);
        velocities.y = generateValue(velocity_seed);
        velocities.z = 0.0f;
        fprintf(stream,"%f,%f,%f,%f,%f,%f,%f\n",
                phy_attributes.x,
                phy_attributes.y,
                phy_attributes.z,
                velocities.x,
                velocities.y,
                velocities.z,
                phy_attributes.w);
    }
    return;
}
float generateValue(float seed){
    return((((float)rand()/(float)(RAND_MAX)) * seed) - (seed/2));
}

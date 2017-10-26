#include<stdio.h>
#include "global.h"
#include "dataparser.cu"
#include "npairs.cu"

// This file is the entry point for the simulation

int main(){
    /*
     *  Storing physical attributes such as 3-dimensional coordinates and mass of bodies in a
     *  float4 type array.
     *  Storing velocities in x, y & z directions in a float3 type array.
     */
    float4 phy_attributes[N_SAMPLES];
    float3 velocities[N_SAMPLES];
    /*
     *  Populating data.
     */
    parseCSVData(&phy_attributes, &velocities);
    /*  Storing input data in device global memory to be accessed by all blocks.    */
    cudaMemCpyToSymbol(attr, phy_attributes, sizeof(phy_attributes));
    cudaMemCpyToSymbol(velocity, velocities, sizeof(velocities));
    
    // REST OF THE CODE
    
    return 0;
}

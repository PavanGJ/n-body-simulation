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
    int iter;
    //  Populating data.
    parseCSVData((float4 *)phy_attributes, (float3 *)velocities);
    //  Storing input data in device global memory to be accessed by all blocks.
    cudaMemcpyToSymbol(attr, phy_attributes, sizeof(phy_attributes));
    cudaMemcpyToSymbol(vel, velocities, sizeof(velocities));
    for(iter = 0; iter < ITERATIONS; iter++){
        //  Call the kernels to execute over GPU.
        computeUpdates<<<N_BLOCKS, THREADS_PER_BLOCK>>>();
        updateValues<<<N_BLOCKS, THREADS_PER_BLOCK>>>();
        //  Write intermediate results every 100 iterations.
        if(iter % 100 == 0){
            cudaMemcpyFromSymbol(phy_attributes, attr, sizeof(phy_attributes));
            cudaMemcpyFromSymbol(velocities, vel, sizeof(velocities));
            writeCSV((float4 *)phy_attributes, (float3 *)velocities, iter);
        }
    }
    cudaMemcpyFromSymbol(phy_attributes, attr, sizeof(phy_attributes));
    cudaMemcpyFromSymbol(velocities, vel, sizeof(velocities));
    writeCSV((float4 *)phy_attributes, (float3 *)velocities, ITERATIONS);
    
    return 0;
}

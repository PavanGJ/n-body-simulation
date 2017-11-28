#include <stdio.h>
#include <time.h>

#include "../global.h"
#include "../csvparser.c"
#include "../filewriter.c"
#include "npairs.h"
#include "forces.cu"
#include "update.cu"

// This file is the entry point for the simulation

void simulateOnGPU(){
    /*
     *  Storing physical attributes such as 3-dimensional coordinates and mass of bodies in a
     *  float4 type array.
     *  Storing velocities in x, y & z directions in a float3 type array.
     */
    float4 phy_attributes[N_SAMPLES];
    float3 velocities[N_SAMPLES];
    cudaEvent_t start, end, execStart, execEnd;
    float totalTime = 0, execTime = 0;
    int iter;
    
    //  Populating data from CSV file.
    readFromCSV((float4 *)phy_attributes, (float3 *)velocities);
    
    //	Create and register cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventCreate(execStart);
    cudaEventCreate(&execEnd);

    cudaEventRecord(start);

    //  Copying input data to device global memory.
    cudaMemcpyToSymbol(attr, phy_attributes, sizeof(phy_attributes));
    cudaMemcpyToSymbol(vel, velocities, sizeof(velocities));

    cudaEventRecord(execStart);

    for(iter = 0; iter < ITERATIONS; iter++){
        /*
         *  Uncomment to get intermediate results
         *
         //  Write intermediate results every 100 iterations.
        if(iter % 100 == 0){
            cudaMemcpyFromSymbol(phy_attributes, attr, sizeof(phy_attributes));
            cudaMemcpyFromSymbol(velocities, vel, sizeof(velocities));
            writeToCSV((float4 *)phy_attributes, (float3 *)velocities, iter);
        }
         *
         */
        //  Call the kernels to execute over GPU.
        computeUpdates<<<N_BLOCKS, THREADS_PER_BLOCK>>>();
        updateValues<<<N_BLOCKS, THREADS_PER_BLOCK>>>();
    }

    cudaEventRecord(execEnd);

    cudaMemcpyFromSymbol(phy_attributes, attr, sizeof(phy_attributes));
    cudaMemcpyFromSymbol(velocities, vel, sizeof(velocities));

    //	Register end event
    cudaEventRecord(end);

    cudaEventSynchronize(end);
    cudaEventElapsedTime(&totalTime, start, end);
    cudaEventElapsedTime(&execTime, execStart, execEnd);
    //	Write results
    writeMeasureToFile("Nvidia GPU", "Linear", ""Total Time", totalTime);
    writeMeasureToFile("Nvidia GPU", "Linear", "Exec Time", execTime);
    writeToCSV((float4 *)phy_attributes, (float3 *)velocities, ITERATIONS);

    //	Destroy events
    cudaEventDestroy(&start);
    cudaEventDestroy(&end);
    cudaEventDestroy(&execStart);
    cudaEventDestroy(&execEnd);
}

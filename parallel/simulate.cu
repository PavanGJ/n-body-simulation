#include <stdio.h>
#include <time.h>

#include "../global.h"
#include "../csvparser.c"
#include "npairs.h"
#include "forces.cu"
#include "update.cu"

// This file is the entry point for the simulation

void writeTimeToFile(time_t begin, time_t read, time_t end){
    /*
     *  This subroutine appends time computational time for further interpretation.
     */
    FILE* outputStream;
    char output[256] = OUTPUT_DIR;
    char fileName[10] = "timeGPU.csv";
    
    strcat(output, fileName);
    outputStream = fopen(output, "a+");
    
    fprintf(outputStream, "%d, %ld, %ld, %ld\n", N_SAMPLES, begin, read, end);
    
    return;
}

void simulateOnGPU(){
    /*
     *  Storing physical attributes such as 3-dimensional coordinates and mass of bodies in a
     *  float4 type array.
     *  Storing velocities in x, y & z directions in a float3 type array.
     */
    float4 phy_attributes[N_SAMPLES];
    float3 velocities[N_SAMPLES];
    time_t time_begin, time_read, time_end;
    int iter;
    
    //  Recording the beginning time
    time_begin = time(NULL);
    
    //  Populating data from CSV file.
    readFromCSV((float4 *)phy_attributes, (float3 *)velocities);
    
    //  Recording time after the data is parsed from the CSV file.
    time_read = time(NULL);

    //  Copying input data to device global memory.
    cudaMemcpyToSymbol(attr, phy_attributes, sizeof(phy_attributes));
    cudaMemcpyToSymbol(vel, velocities, sizeof(velocities));

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
    cudaMemcpyFromSymbol(phy_attributes, attr, sizeof(phy_attributes));
    cudaMemcpyFromSymbol(velocities, vel, sizeof(velocities));
    
    //  Recording the finished time
    time_end = time(NULL);
    writeTimeToFile(time_begin, time_read, time_end);
    writeToCSV((float4 *)phy_attributes, (float3 *)velocities, ITERATIONS);

}

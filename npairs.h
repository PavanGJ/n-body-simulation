#include "global.h"

//Define # of Blocks in the Device and the # of threads in each block.
#ifndef N_BLOCKS
#define N_BLOCKS 125
#define THREADS_PER_BLOCK 512
#endif

#ifndef GRAVI_CONST
#define GRAVI_CONST 6.67408/100000000000
#endif

//Declaring global memory
__device__ float4 attr[N_SAMPLES];
__device__ float3 vel[N_SAMPLES];
__device__ float3 acc[N_SAMPLES];

//Declaring shared memory
__shared__ float4 sharedObj[THREADS_PER_BLOCK];

//Declaring simulation kernel.
//Takes # of iterations as an input.
__global__ void computeUpdates();

__global__ void updateValues();

__device__ float3 computeForce(float4, float4);

__device__ float3 computeTileUpdates(float4);

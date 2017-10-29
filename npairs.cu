#import "npairs.h"

//This file defines the functions and kernels to be run on the CUDA device.

__global__ void computeUpdates(){
    /*
     *  This kernel computes the acceleration on a body caused as a result of forces acting on it.
     */
    int idx, div, objIdx, tile;
    int N = N_SAMPLES / (gridDim.y + 1);
    float4 obj;
    float3 tileAcc, acceleration = {0.0f, 0.0f, 0.0f};
    extern __shared__ sharedObj[];
    /*
     *  Compute object index as a function of block dimension, block index and thread index.
     *  This computation need to be changed if the programming model is changed from one object per thread.
     */
    objIdx = blockDim.x * blockIdx.x + threadIdx.x;
    obj = attr[objIdx];
    for(div = 0, tile = 0; div < N; div += blockDim.x, tile++){
        //  Update sharedObj values one tile at a time.
        idx = (tile + blockIdx.y) * blockDim.x + threadIdx.x;
        sharedObj[threadIdx.x] = attr[idx];
        //  Sync threads to ensure all threads write its value before proceeding
        __syncthreads();
        tileAcc = computeTileUpdates(obj);
        acceleration.x += tileAcc.x;
        acceleration.y += tileAcc.y;
        acceleration.z += tileAcc.z;
    }
    /*
     *  Update global values of acceleration thus setting it up for updation of spatial coordinates and
     *  velocities for the time step.
     */
    acc[objIdx].x += acceleration.x;
    acc[objIdx].y += acceleration.y;
    acc[objIdx].y += acceleration.y;
}

__global__ void updateValues(){
    /*
     *  This kernel function updates values of spatial coordinates and velocity as computed using equations of
     *  motion. Acceleration values are reset for next computation.
     */
    int idx;
    float4 obj;
    float3 v, a;
    /*
     *  Compute object index as a function of block dimension, block index and thread index.
     *  This computation need to be changed if the programming model is changed from one object per thread.
     */
    idx = blockDim.x * blockIdx.x + threadIdx.x;
    //  Extracting a copy of previous values in order to update the values
    obj = attr[idx];
    v = vel[idx];
    a = acc[idx];
    /*
     *  Update new coordinates for the object in the 3 - dimensional plane.
     *  Updates are calculated as, S = ut + (1/2) * at^2
     *  where   S           -   distance travelled in time t.
     *          u           -   initial velocity
     *          t           -   time
     *          a           -   constant acceleration experience in time t.
     *  Since t = 1 unit, S = u + (1/2) * a
     *  Also, since S defines the relative distance travelled as a result of forces, it is added to the initial
     *  spatial coordinates to get the resultant coordinates.
     */
    obj.x += (v.x + a.x / 2);
    obj.y += (v.y + a.y / 2);
    obj.z += (v.z + a.z / 2);
    /*
     *  Update velocities of the object in 3 - dimensional plane.
     *  Updates are calculated as v = u + at
     *  where   v           -   updated velocity
     *          u           -   initial velocity
     *          t           -   time
     *          a           -   constant acceleration experience in time t.
     *  Again, since t = 1 unit, the formula takes the form v = u + a
     */
    v.x += a.x;
    v.y += a.y;
    v.z += a.z;
    /*
     *  Update global memory to reflect the new values of spatial coordinates and velocity.
     *  Also reset acceleration to be computed for the next time step.
     */
    attr[idx] = obj;
    vel[idx] = v;
    acc[idx] = {0.0f, 0.0f, 0.0f};
}
__device__ float3 computeForce(float4 attr_obj_1, float4 attr_obj_2){
    /*
     *  Computes accelerations acting on obj defined by attr_obj_1, lets call it obj_1, because of obj defined
     *  by attr_obj_2 - obj_2 in 3-dimensions.
     *  Since the mass of the body remains constant for a single time step, force is directly proportional to
     *  acceleration acting on the body. Hence, force acting on a body is represented using acceleration for
     *  each time step.
     *  Equating Newton's law of universal gravitation and Newton's second law of motion,
     *  we compute the acceleration on obj_1 as a result of force acting on it by obj_2.
     */
    float3 r;
    float sq_dist;
    float intermediate;
    float3 acceleration;
    //  Compute distance between the objects in each of the dimensions in the 3-dimensional plane.
    r.x = attr_obj_2.x - attr_obj_1.x;
    r.y = attr_obj_2.y - attr_obj_1.y;
    r.z = attr_obj_2.z - attr_obj_1.z;
    //  Compute Euclidean distance between the objects/
    sq_dist = r.x * r.x + r.y * r.y + r.z * r.z;
    /*
     *  Compute intermediate result for reuse during computation of acceleration.
     *  The unit vector decomposed to direction vector and magnitude and the magnitude is included in
     *  the intermediate result.
     */
    intermediate = GRAVI_CONST * attr_obj_2.w / (sq_dist * sqrtf(sq_dist));
    //  Compute accelerations in each direction by multiplying direction vector with intermediate result.
    acceleration.x = r.x * intermediate;
    acceleration.y = r.y * intermediate;
    acceleration.z = r.z * intermediate;
    
    return acceleration;
}

__device__ float3 computeTileUpdates(float4 obj){
    /*
     *  Computes the relative change in acceleration as a result of the force acting on the body exerted by the
     *  bodies defined within the tile.
     */
    extern __shared__ float4[] sharedObj;                               //Declare sharedObjects for the tile.
    int idx;
    float3 accUpdates, relAcc = {0.0f, 0.0f, 0.0f};
    for(idx = 0; idx < blockDim.x; idx++){
        accUpdates = computeForce(obj, sharedObj[i])
        relAcc.x += accUpdates.x;
        relAcc.y += accUpdates.y;
        relAcc.z += accUpdates.z;
    }
    return relAcc;
}

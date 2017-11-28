//  This file defines the subroutines to compute the forces acting on a body as a result of all other bodies
//  and identifies the resultant forces.

__global__ void computeUpdates(){
    /*
     *  This kernel computes the acceleration on a body caused as a result of forces acting on it.
     */
    int idx, div, objIdx, tile;
    
    //  N defines the number of shared objects the thread would be working on.
    int N = N_SAMPLES / gridDim.y;
    float4 obj;
    float3 tileAcc, acceleration = {0.0f, 0.0f, 0.0f};
    extern __shared__ float4 sharedObj[];
    
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
        //  Sync threads to ensure all threads write its value before proceeding.
        __syncthreads();
        tileAcc = computeTileUpdates(obj);
        acceleration.x += tileAcc.x;
        acceleration.y += tileAcc.y;
        acceleration.z += tileAcc.z;
        // Sync threads to ensure all threads have completed the iteration.
        __syncthreads();
    }
    
    /*
     *  Update global values of acceleration thus setting it up for updation of spatial coordinates and
     *  velocities for the time step.
     */
    acc[objIdx].x += acceleration.x;
    acc[objIdx].y += acceleration.y;
    acc[objIdx].z += acceleration.z;
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
    
    //  Compute Euclidean distance between the objects with an additional softening factor.
    sq_dist = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING_FACTOR * SOFTENING_FACTOR;
    
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
    extern __shared__ float4 sharedObj[];                               //Declare sharedObjects for the tile.
    int idx;
    float3 accUpdates, relAcc = {0.0f, 0.0f, 0.0f};
    float4 shObj;
    for(idx = 0; idx < blockDim.x; idx++){
        shObj = sharedObj[idx];
        
        //  Skip the iteration if the two objects/bodies are the same.
        if(shObj.x == obj.x && shObj.y == obj.y && shObj.z == obj.z && shObj.w == obj.w)
            continue;
        
        accUpdates = computeForce(obj, shObj);
        relAcc.x += accUpdates.x;
        relAcc.y += accUpdates.y;
        relAcc.z += accUpdates.z;
    }
    return relAcc;
}

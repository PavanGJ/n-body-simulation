//  Defines the subroutine to update values for an iteration

__global__ void updateValues(){
    /*
     *  This kernel function updates values of spatial coordinates and velocity as computed using equations of
     *  motion. Acceleration values are reset for next computation.
     */
    
    int idx;
    float4 obj;
    float3 v, a, resetA = {0.0f, 0.0f, 0.0f};
    
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
     *  Update velocities of the object in 3 - dimensional plane.
     *  Updates are calculated as v = u + at
     *  where   v           -   updated velocity
     *          u           -   initial velocity
     *          t           -   time
     *          a           -   constant acceleration experience in time t.
     */
    v.x += a.x * dT;
    v.y += a.y * dT;
    v.z += a.z * dT;
    
    // Update the spatial coordinates
    obj.x += v.x;
    obj.y += v.y;
    obj.z += v.z;
    
    /*
     *  Update global memory to reflect the new values of spatial coordinates and velocity.
     *  Also reset acceleration to be computed for the next time step.
     */
    attr[idx] = obj;
    vel[idx] = v;
    acc[idx] = resetA;
}

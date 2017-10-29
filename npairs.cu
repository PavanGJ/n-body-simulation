#import "npairs.h"

//This file defines the functions and kernels to be run on the CUDA device.

__global__ void simulateNPairs(int iter){
    /*
     *  This function is the kernel called from the host to begin simulation.
     *  Takes an input `iter` - the number of iterations the n-body problem is to be simulated.
     */
    int idx;
    for(idx = 0; idx < iter; idx++){
        break;
    }
}

__device__ float3 computeForce(float4 attr_obj_1, float4 attr_obj_2){
    /*
     *  Computes accelerations acting on obj defined by attr_obj_1, lets call it obj_1, because of obj defined
     *  by attr_obj_2 - obj_2 in 3-dimensions.
     *  Since the mass of the body remains constant for a single time step, force is directly proportional to
     *  acceleration acting on the body. Hence, force acting on a body is represented using acceleration for
     *  each time step.
     *  Force computations follow Newtons law of universal gravitation.
     *              F = (G * m1 * m2 * vec_r12)/(|r12|^2)
     *  where   F           -   Force exerted by obj_2 on obj_1
     *          G           -   Gravitational constant
     *          m1 & m2     -   Mass of obj_1 & obj_2 respectively
     *          vec_r12     -   Unit vector from obj_1 to obj_2
     *          |r12|       -   |r2 - r1|, distance between obj_1 and obj_2
     *
     *  It follows for Newtons second law of motion than Force exerted on a body is equal to the product of mass
     *  of the body and its acceleration.
     *              F = m1 * a12
     *  where   F           -   Force exerted by obj_2 on obj_1
     *          m1          -   mass of obj_1
     *          a12         -   acceleration in obj_1 as a result of force exerted by obj_1
     *  Equating these two formulas we get the equation to compute acceleration.
     *              a12 = (G * m2 * vec_r12)/(|r12|^2)
     *  We compute the acceleration in obj_1 in each direction
     */
    float3 r;
    float sq_dist;
    float intermediate;
    float3 acceleration;
    /*
     *  Compute distance between the objects in each of the dimensions in the 3-dimensional plane.
     */
    r.x = attr_obj_2.x - attr_obj_1.x;
    r.y = attr_obj_2.y - attr_obj_1.y;
    r.z = attr_obj_2.z - attr_obj_1.z;
    /*
     *  Compute Euclidean distance between the objects/
     */
    sq_dist = r.x * r.x + r.y * r.y + r.z * r.z;
    /*
     *  Compute intermediate result for reuse during computation of acceleration.
     *  The unit vector decomposed to direction vector and magnitude and the magnitude is included in
     *  the intermediate result.
     */
    intermediate = GRAVI_CONST * attr_obj_2.w / (sq_dist * sqrtf(sq_dist));
    /*
     *  Computing acceleration in each direction by multiplying direction vector with intermediate result.
     */
    acceleration.x = r.x * intermediate;
    acceleration.y = r.y * intermediate;
    acceleration.z = r.z * intermediate;
    
    return acceleration;
}

__device__ float3 computeTileUpdates(float4 obj, float3 objAcc){
    /*
     *  Updates the attributes such as velocity and spatial coordinates for obj by computing forces acting
     *  on the body as a resultant of individual forces acted upon obj by all other bodies.
     *  This function updates the acceleration experience by obj as a cause of force exerted by objects
     *  defined in sharedObj.
     */
    extern __shared__ float4[] sharedObj;                               //Declare sharedObjects for the tile.
    int idx;
    /*
     *  Update acceleration caused as a result of force acting on the body.
     *  Computes partial sum of accelerations as a result of the force acting on the body exerted by the
     *  bodies defined within the tile.
     */
    for(idx = 0; idx < blockDim.x; idx++){
        float3 accUpdates = computeFloat(obj, sharedObj[i])
        objAcc.x += accUpdates.x;
        objAcc.y += accUpdates.y;
        objAcc.z += accUpdates.z;
    }
    return objAcc;
}

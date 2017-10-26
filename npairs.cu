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
     *  Computes forces acting on obj defined by attr_obj_1, lets call it obj_1, because of obj defined by
     *  attr_obj_2 - obj_2.
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
    r.x = attr_obj_2.x - attr_obj_1.x;
    r.y = attr_obj_2.y - attr_obj_1.y;
    r.z = attr_obj_2.z - attr_obj_1.z;
    /*
     *  Compute squared distance between the objects/
     */
    float sq_dist = r.x * r.x + r.y * r.y + r.z * r.z;
    /*
     *  Compute intermediate result for reuse during computation of acceleration.
     *  The unit vector decomposed to direction vector and magnitude and the magnitude is included in
     *  the intermediate result.
     */
    float intermediate = GRAVI_CONST * attr_obj_2.w / (sq_dist * sqrtf(sq_dist));
    /*
     *  Computing acceleration in each direction by multiplying direction vector with intermediate result.
     */
    float3 acceleration;
    acceleration.x = r.x * intermediate;
    acceleration.y = r.y * intermediate;
    acceleration.z = r.z * intermediate;
    
    return acceleration;
}

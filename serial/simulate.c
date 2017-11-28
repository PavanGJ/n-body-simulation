#include <stdio.h>
#include <time.h>
#include <math.h>

#include "../global.h"
#include "../csvparser.c"
#include "../filewriter.c"
#include "npairs.h"

// This file is the entry point for the simulation

float4 phy_attributes[N_SAMPLES];
float3 velocities[N_SAMPLES];

float3 computeForce(float4 attr_obj_1, float4 attr_obj_2){
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

void updates(){
    /*
     *  Compute updates for each body as a resultant of the forces acting on it.
     *  For each body, compute the force acting on it because of every other body and update
     *  the coordinates.
     */
    int idx_i, idx_j;
    float3 acceleration, updates, vel;
    float4 obj;
    for(idx_i = 0; idx_i < N_SAMPLES; idx_i++){
        acceleration = {0.0f, 0.0f, 0.0f};
        obj = phy_attributes[idx_i];
        for(idx_j = 0; idx_j < N_SAMPLES; idx_j++){
            if(idx_i == idx_j)
                continue;
            updates = computeForce(obj, phy_attributes[idx_j]);
            acceleration.x += updates.x;
            acceleration.y += updates.y;
            acceleration.z += updates.z;
        }
        vel = velocities[idx_i];
        
        vel.x += acceleration.x * dT;
        vel.y += acceleration.y * dT;
        vel.z += acceleration.z * dT;
        
        obj.x += vel.x;
        obj.y += vel.y;
        obj.z += vel.z;
        
        phy_attributes[idx_i] = obj;
        velocities[idx_i] = vel;
    }
}

void simulate(){
    /*
     *  Storing physical attributes such as 3-dimensional coordinates and mass of bodies in a
     *  float4 type array.
     *  Storing velocities in x, y & z directions in a float3 type array.
     */
    time_t start, end, execStart, execEnd;
    int iter;
    
    //  Recording the beginning time
    start = time(NULL);
    
    //  Populating data from CSV file.
    readFromCSV((float4 *)phy_attributes, (float3 *)velocities);
    
    //  Recording time after the data is parsed from the CSV file.
    
    execStart = time(NULL);
    for(iter = 0; iter < ITERATIONS; iter++){
        /*
         *  Uncomment to get intermediate results
         *
         //  Write intermediate results every 100 iterations.
         if(iter % 100 == 0){
         writeCSV((float4 *)phy_attributes, (float3 *)velocities, iter);
         }
         *
         */
        update();
    }
    execEnd = time(NULL);
    //  Recording the finished time
    end = time(NULL);
    writeMeasureToFile("CPU", "Linear", "Total Time", end - start);
    writeMeasureToFile("CPU", "Linear", "Exec Time", execEnd - execStart);
    writeToCSV((float4 *)phy_attributes, (float3 *)velocities, ITERATIONS);
    
}

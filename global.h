#include<stdio.h>
#include<stdlib.h>

#ifndef INPUT
#define INPUT "c_0000.csv"
#endif

#ifndef N
#define N 64000
#endif

#ifndef FEATURES
#define FEATURES 7
#define INDEX_X 0
#define INDEX_Y 1
#define INDEX_Z 2
#define INDEX_VX 3
#define INDEX_VY 4
#define INDEX_VZ 5
#define INDEX_M 6
#define INDEX_ID 7 
#endif

typedef struct {
    double values[N][FEATURES];
} Input;

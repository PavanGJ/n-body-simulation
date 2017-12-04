#include <stdio.h>
#include <string.h>

#include "generator.c"
#include "serial/simulate.c"
#include "parallel/simulate.cu"

#ifndef TRUE
#define TRUE 0
#define FALSE -1
#endif

int main(int args, char *argv[]){
    /*
     *  Possible arguments:
     *      --generate
     *      --cpu
     */
    int generate = FALSE, gpu = TRUE, threeDim = TRUE, idx;
    
    for(idx = 1; idx < args; idx++){
        if(strcmp(argv[idx], "--generate=3d") == TRUE){
            generate = TRUE;
        }
        else if(strcmp(argv[idx], "--generate=2d") == TRUE){
            generate = TRUE;
            threeDim = FALSE;
        }
        else if(strcmp(argv[idx], "--cpu") == TRUE){
            gpu = FALSE;
        }
        else{
            printf("ERROR. Incorrect arguments.\n");
            printf("Arguments Allowed:\n\t--generate=[3d/2d]\t:\tUse to generate input data\n\t--cpu\t\t:\tUse to run on CPU\n");
            return 1;
	}
    }
    if(generate == TRUE){
        switch (threeDim) {
            case TRUE:
            generate3DData();
            break;
            
            case FALSE:
            generate2DData();
            break;
            
            default:
            break;
        }
    }
    if(gpu == FALSE){
        simulate();
    }
    else{
        simulateOnGPU();
    }
    
    return 0;
}

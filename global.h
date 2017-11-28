// Defining input file path
#ifndef INPUT_PATH
#define INPUT_PATH "input.csv"
#endif

//  Defining output directory path
#ifndef OUTPUT_DIR
#define OUTPUT_DIR "data/output/"
#endif

//  Defining number of samples in the input data
#ifndef N_SAMPLES
#define N_SAMPLES 3
#endif

//  Defining gravitational constant for the simulation
#ifndef GRAVI_CONST
#define GRAVI_CONST 1.0
#endif

//  Defining softening factor to overcome the singularity
//  when two bodies are in close promity with each other.
#ifndef SOFTENING_FACTOR
#define SOFTENING_FACTOR 1.5f
#define dT 0.1
#endif

//  Defining the number of iteration of the simulation to be run.
#ifndef ITERATIONS
#define ITERATIONS 2000
#endif

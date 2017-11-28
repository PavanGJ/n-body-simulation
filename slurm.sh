#!/bin/bash
#SBATCH --partition=debug
#SBATCH --qos=gpu
#SBATCH --job-name="n-body-simulation"
#SBATCH --error=error.out
#SBATCH --output=output.out
#SBATCH --gres=gpu:2
#SBATCH --nodes=1

nvcc main.cu -o n-body.out
srun ./n-body.out

echo "Simulation Completed."

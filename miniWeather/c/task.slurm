#!/bin/bash

#SBATCH -o ./output/%j.out
#SBATCH --job-name=jeremy
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

### execute the code 
mpirun ./build/mycode
#!/bin/bash

#SBATCH -o ./output/%j.out
#SBATCH --job-name=jeremy_G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

### execute the code 
mpirun ./serial_cuda
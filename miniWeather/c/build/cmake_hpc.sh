#!/bin/bash

export TEST_MPI_COMMAND="mpirun -n 1"

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpicxx                                                     \
      -DCXXFLAGS="-Ofast -march=native -ffast-math -std=c++11 -I$(spack location -i parallel-netcdf)/include"   \
      -DLDFLAGS="-L$(spack location -i parallel-netcdf)/lib -lpnetcdf"                        \
      -DOPENACC_FLAGS="-fopenacc -foffload=\"-lm -latomic\""                          \
      -DOPENMP_FLAGS="-fopenmp"                                                       \
      -DNX=3200                                                                        \
      -DNZ=1600                                                                       \
      -DDATA_SPEC="DATA_SPEC_COLLISION"                                           \
      -DSIM_TIME=300                                                                 \
      -DOUT_FREQ=50                                                                  \
      ..


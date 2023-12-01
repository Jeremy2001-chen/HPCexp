
DATA_SPEC_COLLISION=1
DATA_SPEC_THERMAL=2
DATA_SPEC_GRAVITY_WAVES=3
DATA_SPEC_DENSITY_CURRENT=5
DATA_SPEC_INJECTION=6

# CUDA compiler
NVCC = nvcc

# 正确性测试用例
TESTCUDAFLAGS =  -O3 -std=c++11 -DSINGLE_PREC -D_NX=100   -D_NZ=50 -D_SIM_TIME=400 -D_OUT_FREQ=400 -D_DATA_SPEC=${DATA_SPEC_THERMAL}

# 性能测试用例
PERFCUDAFLAGS =  -O3 -std=c++11 -DSINGLE_PREC -D_NX=3200 -D_NZ=1600 -D_SIM_TIME=10 -D_OUT_FREQ=50 -D_DATA_SPEC=${DATA_SPEC_COLLISION}

PNETCDF_INCLUDE = -I$(shell spack location -i parallel-netcdf)/include
PNETCDF_LIB = -L$(shell spack location -i parallel-netcdf)/lib -lpnetcdf
MPI_LIB = -lmpi

SRCS = miniWeather_serial.cu

EXEC = serial_cuda

all: $(EXEC) test

test: $(SRCS)
	$(NVCC) $(TESTCUDAFLAGS) $(PNETCDF_INCLUDE) $(PNETCDF_LIB) $(MPI_LIB) $^ -o $@

$(EXEC): $(SRCS)
	$(NVCC) $(PERFCUDAFLAGS) $(PNETCDF_INCLUDE) $(PNETCDF_LIB) $(MPI_LIB) $^ -o $@

clean:
	rm -f $(EXEC)

.PHONY: all clean

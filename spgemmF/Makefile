CUDAC = nvcc
CUDAOPT = -std=c++17 -g
CUDAFLAGS = $(CUDAOPT)

CXX = mpiicpc
CFLAGS = -g

CUDAPATH = $(CUDA_HOME)


LDFLAGS = -lm -lstdc++ -ldl
LDLIBS = $(LDFLAGS) -L$(CUDAPATH)/lib64 -lcudart

targets = benchmark-cusparse benchmark-optimized 
commonobj = utils.o main.o baseline.o
objects = $(commonobj) spgemm-cusparse.o spgemm-optimized.o

.PHONY : default
default : all

.PHONY : all
all : $(targets)

utils.o : utils.cu common.h
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@

spgemm-cusparse.o : spgemm-cusparse.cu common.h
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@

spgemm-optimized.o : spgemm-optimized.cu common.h
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@

baseline.o: baseline.cpp
	$(CXX) -c -I$(CUDA_HOME)/include $(CFLAGS) $< -o $@

main.o: main.cpp
	$(CXX) -c -I$(CUDA_HOME)/include $(CFLAGS) $< -o $@

benchmark-cusparse : baseline.o spgemm-cusparse.o utils.o
	$(CXX) $^ $(LDLIBS) -lcusparse -o $@

benchmark-optimized: main.o spgemm-optimized.o utils.o
	$(CXX) $^ $(LDLIBS) -lcusparse -o $@

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)

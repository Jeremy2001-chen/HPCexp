# 大气模拟器实验

主要包含了CPU集群优化和单GPU节点优化

## 代码运行方式

多CPU集群：

* 在c目录下有一个提交脚本submit.sh，其将按照task.slurm的任务配置向集群提交任务，任务结果将按照任务编号写在c/output目录下面。
* 修改代码为miniWeather_mpi_mycode.cpp.

单GPU节点：

* 在c_gpu目录下有一个提交脚本submit.sh，其将按照gpu.slurm的任务配置向集群提交任务，任务结果将按照任务编号写在c_gpu/output目录下面。
* 直接在miniWeather_serial.cu中进行的修改。

## 性能优化

### 多CPU集群

这里针对多CPU的优化比较难（毕竟前面openMP和mpi）的优化都比较多，~~需要一些奇怪的技术~~。

#### 编译选项优化

代码的Makefile增加参数 `-march=native -ffast-math`，可以让编译器自行安装体系结构进行优化。

#### 任务配置优化

为了高效利用mpi并行库，我们需要拉满任务的配置，这里是提交任务的配置文件：

```bash
#!/bin/bash

#SBATCH -o ./output/%j.out
#SBATCH --job-name=jeremy
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

### execute the code 
mpirun ./build/mycode
```

在这个配置文件中，使用了所有NUMA节点的所有48核CPU，使得并行程度达到最高的水平。

#### 算子融合

这个是在代码中进行的唯一进行的优化（其他优化确实没有太多的效果），针对函数 `compute_tendencies_x`，我们可以发现其在计算过程中运用了tend和flux两个数组，但其本质来说可以仅用tend一个数组即可计算出答案（这样能减少数据存取的次数），下面是针对代码的修改：

```cpp
#pragma omp parallel for private(inds,stencil,vals,d3_vals,r,u,w,t,p,ll,s,val0,val1,val2,val3) collapse(1)
    for (k = 0; k < nz; k++) {
        for (i = 0; i < nx + 1; i++) {
            //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
            ....

            //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
            ....

            val0 = (r * u - hv_coef * d3_vals[ID_DENS]) / dx;
            val1 = (r * u * u + p - hv_coef * d3_vals[ID_UMOM]) / dx;
            val2 = (r * u * w - hv_coef * d3_vals[ID_WMOM]) / dx;
            val3 = (r * u * t - hv_coef * d3_vals[ID_RHOT]) / dx;
            if (i < nx) {
                tend[ID_DENS * nz * nx + k * nx + i] = val0;
                tend[ID_UMOM * nz * nx + k * nx + i] = val1;
                tend[ID_WMOM * nz * nx + k * nx + i] = val2;
                tend[ID_RHOT * nz * nx + k * nx + i] = val3;
            }

            if (i > 0) {
                tend[ID_DENS * nz * nx + k * nx + i - 1] -= val0;
                tend[ID_UMOM * nz * nx + k * nx + i - 1] -= val1;
                tend[ID_WMOM * nz * nx + k * nx + i - 1] -= val2;
                tend[ID_RHOT * nz * nx + k * nx + i - 1] -= val3;
            }
        }
    }
```

这样可以将两个循环合并成一个（注意数组的循环展开次数也从2维变成1维，因为对i存在数据依赖），在性能上有小部分的提升，从最开始的针对1s的方针CPU时间从10.3s变为了9.6s.

### 单GPU节点

针对串行代码进行的单节点GPU优化就可以进行的比较多了。

#### 任务配置优化

我们在不使用GPU的位置，应该让CPU性能打满，任务配置如下所示：

```cpp
#!/bin/bash

#SBATCH -o ./output/%j.out
#SBATCH --job-name=jeremy_G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

### execute the code 
mpirun ./serial_cuda
```

仅仅采用单节点和GPU进行仿真。

#### CUDA核优化

对比于c和cpp版本的代码，可以发现在一些关键函数上openmp都会采用编译宏展开的方式对循环进行分布处理，这里的话我们可以类似于这种方式采用cuda kenrel的方式对循环进行展开。

针对x方向的tend数组kernel如下所示：

```cpp
__global__ void compute_x_tend_kernel(real* tend, real* flux, int nz, int nx, real dx) {
    int index = blockIdx.x * 128 + threadIdx.x;
    int k = (index / 4) / nx;
    int i = (index / 4) % nx;
    int ll = index % 4;
    if (k < nz) {
        int indf1, indf2, indt;
        indt = ll * nz * nx + k * nx + i;
        indf1 = ll * (nz + 1) * (nx + 1) + k * (nx + 1) + i;
        indf2 = ll * (nz + 1) * (nx + 1) + k * (nx + 1) + i + 1;
        tend[indt] = -(flux[indf2] - flux[indf1]) / dx;
    }
}
```

在调用这个kernel时我们采用下面的方式：

```cpp
    dim3 dimGrid_1(nz * nx / 32);
    dim3 dimBlock_1(128);
    compute_x_tend_kernel << <dimGrid_1, dimBlock_1 >> > (dev_tend, dev_flux, nz, nx, dx);

    cudaDeviceSynchronize();
```

通过设置Grid数和Block数，可以高效调用cuda的各种kernel。在最开始时我设置Block数量为1，实验结果1s大气仿真使用的CPU时间为150s，考虑到GPU同时处理的单元为32的倍数，因此重新设置了Block数量以提升并发程度，实验结果发现CPU时间变为了69s。

#### 显存优化

在原始计算梯度时，所有的数据均在主机端，我们应该想办法把数据移动到显存中以提升速度。

```cpp
    state = (real*)malloc((nx + 2 * hs) * (nz + 2 * hs) * NUM_VARS * sizeof(real));
    state_tmp = (real*)malloc((nx + 2 * hs) * (nz + 2 * hs) * NUM_VARS * sizeof(real));
    flux = (real*)malloc((nx + 1) * (nz + 1) * NUM_VARS * sizeof(real));
    tend = (real*)malloc(nx * nz * NUM_VARS * sizeof(real));
    hy_dens_cell = (real*)malloc((nz + 2 * hs) * sizeof(real));
    hy_dens_theta_cell = (real*)malloc((nz + 2 * hs) * sizeof(real));
    hy_dens_int = (real*)malloc((nz + 1) * sizeof(real));
    hy_dens_theta_int = (real*)malloc((nz + 1) * sizeof(real));
    hy_pressure_int = (real*)malloc((nz + 1) * sizeof(real));
    cudaMalloc((void**)&dev_state, (nx + 2 * hs) * (nz + 2 * hs) * NUM_VARS * sizeof(real));
    cudaMalloc((void**)&dev_state_tmp, (nx + 2 * hs) * (nz + 2 * hs) * NUM_VARS * sizeof(real));
    cudaMalloc((void**)&dev_flux, (nx + 1) * (nz + 1) * NUM_VARS * sizeof(real));
    cudaMalloc((void**)&dev_hy_dens_cell, (nz + 2 * hs) * sizeof(real));
    cudaMalloc((void**)&dev_hy_dens_theta_cell, (nz + 2 * hs) * sizeof(real));
    cudaMalloc((void**)&dev_tend, nx * nz * NUM_VARS * sizeof(real));
    cudaMalloc((void**)&dev_hy_dens_int, (nz + 1) * sizeof(real));
    cudaMalloc((void**)&dev_hy_dens_theta_int, (nz + 1) * sizeof(real));
    cudaMalloc((void**)&dev_hy_pressure_int, (nz + 1) * sizeof(real));
```

最终的设计是所有参与计算的数组在开始时均通过cudaMalloc函数申请对应的显存空间，在计算过程中全程不通过cudaMemcpy进行数据拷贝，仅在最后得出结果时从显存中把数据拷贝回来。实验结果发现在将所有中间结果的cudaMemcpy去掉后性能1s大气仿真时间从69s变成了0.3s，性能有了非常高的提升。

## 实验结果及正确性评估

### 多CPU集群

* 实验结果如output中输出文件30514.out，CPU时间为274.175s
* 实验正确性来看，te小于规定的4.5e-5，可以通过Makefile中的make test脚本

### 单GPU节点

* 实验结果如output中输出文件30608.out，CPU时间总过为38.584s
* 实验结果正确性来看，可以通过测试脚本correctness_test.sh，计算结果正确

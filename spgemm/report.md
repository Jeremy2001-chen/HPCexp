# 稀疏矩阵乘法

## 任务描述

本任务为在单个GPU上优化稀疏矩阵乘法。

优化代码为spgemm-optimized.cu.

## 基本算法

在CPU上，稀疏矩阵乘法的基本算法为：

* 首先估计最后非零元个数
* 在matA矩阵上按行枚举i，按照CSR格式枚举非零元j，在matB矩阵上按行枚举对应的非零元k
* 在行内统计非零元k个数
* 接着计算真正矩阵值
* 在matA矩阵上按行枚举i，按照CSR格式枚举非零元j，在matB矩阵上按行枚举对应的非零元k，算出对应(i, k)值大小

在GPU上，总体算法流程是类似的，首先通过 `calculate_nzz`对估算出最终非零元个数，最后在通过 `calculate_values`计算矩阵值。

## 优化算法

### 哈希表统计

在CPU版本的稀疏矩阵乘法中，我们可以利用时间戳的方式复用空间，空间复杂度可以降低到O(N)。

但在GPU上则无法并行这样做，因此为了在同一行上哪些元素已经出现，我们采用哈希表的方式来辅助统计。

```c
__shared__ int hash_idx[HASH_COUNT];
for (int i = tid; i < HASH_COUNT; i += 32) {
    hash_idx[i] = -1;
}
```

哈希函数采用基本的取模操作，比较简单：

```c
int pk = matB->gpu_c_idx[j];
int hash = (pk & (HASH_COUNT - 1));
```

当出现哈希冲突时，我们将继续往后面遍历直到遇到第一个哈希槽插进去：

```c
while (atomicCAS(&hash_idx[hash], -1, pk) != -1) {
    if (hash_idx[hash] == pk) {
        row_nzz--;
        break;
    }
    hash = ((hash + 1) & (HASH_COUNT - 1));
}
```

这里用 `atmoicCAS`是为了防止多个thread对同一个哈希表槽进行操作。

### 多线程并行

考虑到按照行划分任务会出现负载均衡的问题，这里我们按照在同一行枚举的列对任务进行划分。

```c
int rid = blockIdx.x;
int tid = threadIdx.x;
```

变量 `tid`表示线程编号（范围为0～31）。

哈希表初始化时，每个线程初始化一部分内容：

```c
__shared__ int hash_idx[HASH_COUNT];
__shared__ data_t hash_values[HASH_COUNT];
for (int i = tid; i < HASH_COUNT; i += 32) {
    hash_idx[i] = -1;
    hash_values[i] = 0;
}
```

通过 `__syncthreads`函数同步所有线程。

在计算时，每个线程负责自己的部分：

```c
for (int i = matA->gpu_r_pos[rid] + tid; i < matA->gpu_r_pos[rid + 1]; i += 32) {
    int pj = matA->gpu_c_idx[i];
    data_t vj = matA->gpu_values[i];

    for (int j = matB->gpu_r_pos[pj]; j < matB->gpu_r_pos[pj + 1]; j++) {
        int pk = matB->gpu_c_idx[j];
        int hash = (pk & (HASH_COUNT - 1));
        while (atomicCAS(&hash_idx[hash], -1, pk) != -1) {
            if (hash_idx[hash] == pk) {
                break;
            }
            hash = ((hash + 1) & (HASH_COUNT - 1));
        }
        atomicAdd(&hash_values[hash], matB->gpu_values[j] * vj);
    }
}
```

在估算nzz数量时，通过reduction方法规约得到总的行数量：

```c
__syncthreads();
row_nzz += __shfl_xor_sync(int(-1), row_nzz, 16);
row_nzz += __shfl_xor_sync(int(-1), row_nzz, 8);
row_nzz += __shfl_xor_sync(int(-1), row_nzz, 4);
row_nzz += __shfl_xor_sync(int(-1), row_nzz, 2);
row_nzz += __shfl_xor_sync(int(-1), row_nzz, 1);
```

在计算矩阵真正内容时，遍历哈希表中的非零元变为政正常数组：
```c
int start = matC->gpu_r_pos[rid];
for (int i = tid; i < HASH_COUNT; i += 32) {
    if (hash_idx[i] != -1) {
        int pos = atomicAdd(&nzz, 1) + start;
        matC->gpu_c_idx[pos] = hash_idx[i];
        matC->gpu_values[pos] = hash_values[i];
    }
}
```

## 性能对比结果

### 预处理时间


### 计算时间
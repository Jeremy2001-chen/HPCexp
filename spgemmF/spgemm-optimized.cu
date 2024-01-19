#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda.h>
#include "common.h"
#include "utils.h"
#include <cub/cub.cuh>
#include <algorithm>

#define CUDA_CHECK_ERROR(call)\
do{\
	cudaError_t _error = (cudaError_t)(call);\
	if(_error != cudaSuccess)\
	{\
		printf("*** CUDA Error *** at [%s:%d] error=%d, reason:%s \n",\
			__FILE__, __LINE__, _error, cudaGetErrorString(_error));\
	}\
}while(0)

const char* version_name = "optimized version";\
typedef struct {
    dist_matrix_t* matA_gpu = NULL, * matB_gpu = NULL, * matC_gpu = NULL;
    int* cntC_gpu = NULL, * limit_pos_gpu, * max_length_gpu;
    data_t* values_gpu = NULL;
    // int* rk, * rk_gpu;
} additional_info_t;

typedef struct {
    int row;
    int* c_idx;
    data_t* values;
} cpu_cal_t;

#define HASH_COUNT (4096)

typedef additional_info_t* info_ptr_t;

int* calA;

void preprocess(dist_matrix_t* matA, dist_matrix_t* matB) {
    cudaMalloc((void**)&matA->gpu_r_pos, (matA->dist_m + 1) * sizeof(index_t));
    cudaMalloc((void**)&matA->gpu_c_idx, matA->global_nnz * sizeof(index_t));
    cudaMalloc((void**)&matA->gpu_values, matA->global_nnz * sizeof(data_t));
    cudaMemcpy(matA->gpu_r_pos, matA->r_pos, (matA->dist_m + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(matA->gpu_c_idx, matA->c_idx, (matA->global_nnz) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(matA->gpu_values, matA->values, (matA->global_nnz) * sizeof(data_t), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&matB->gpu_r_pos, (matB->global_m + 1) * sizeof(index_t));
    cudaMalloc((void**)&matB->gpu_c_idx, matB->global_nnz * sizeof(index_t));
    cudaMalloc((void**)&matB->gpu_values, matB->global_nnz * sizeof(data_t));
    cudaMemcpy(matB->gpu_r_pos, matB->r_pos, (matB->global_m + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(matB->gpu_c_idx, matB->c_idx, (matB->global_nnz) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(matB->gpu_values, matB->values, (matB->global_nnz) * sizeof(data_t), cudaMemcpyHostToDevice);
}

void destroy_additional_info(void* additional_info) {
    info_ptr_t p = (info_ptr_t)additional_info;
    // CUDA_CHECK_ERROR(cudaFree(p->matA_gpu));
    // CUDA_CHECK_ERROR(cudaFree(p->matB_gpu));
    // CUDA_CHECK_ERROR(cudaFree(p->matC_gpu));
    // CUDA_CHECK_ERROR(cudaFree(p->cntC_gpu));
    // CUDA_CHECK_ERROR(cudaFree(p->limit_pos_gpu));
    // CUDA_CHECK_ERROR(cudaFree(p->max_length_gpu));
    // CUDA_CHECK_ERROR(cudaFree(p->values_gpu));
    free(p);
}

inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}

#define HASH_COUNT_MAX (16384)

__global__ void calculate_nzz(dist_matrix_t* matA, dist_matrix_t* matB, int* nzzC/*, int* calA*/, int* limit, int* limit_pos) {
    int rid = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ int h_int[];
    // __shared__ int hash_idx[HASH_COUNT_MAX];
    int* hash_idx = (int*)h_int;
    for (int i = tid; i < HASH_COUNT_MAX; i += 32) {
        hash_idx[i] = -1;
    }
    __syncthreads();

    int row_nzz = 0, cal = 0;
    for (int i = matA->gpu_r_pos[rid] + tid; i < matA->gpu_r_pos[rid + 1]; i += 32) {
        int pj = matA->gpu_c_idx[i];

        for (int j = matB->gpu_r_pos[pj]; j < matB->gpu_r_pos[pj + 1]; j++) {
            int pk = matB->gpu_c_idx[j];
            int hash = (pk & (HASH_COUNT_MAX - 1));
            // int hash = pk % HASH_COUNT;
            while (atomicCAS(&hash_idx[hash], -1, pk) != -1) {
                if (hash_idx[hash] == pk) {
                    row_nzz--;
                    break;
                }
                hash = ((hash + 1) & (HASH_COUNT_MAX - 1));
                // hash++;
                // if (hash == HASH_COUNT_MAX) hash = 0;
            }
            row_nzz++;
        }
        cal += matB->gpu_r_pos[pj + 1] - matB->gpu_r_pos[pj];
    }

    __syncthreads();
    row_nzz += __shfl_xor_sync(int(-1), row_nzz, 16);
    row_nzz += __shfl_xor_sync(int(-1), row_nzz, 8);
    row_nzz += __shfl_xor_sync(int(-1), row_nzz, 4);
    row_nzz += __shfl_xor_sync(int(-1), row_nzz, 2);
    row_nzz += __shfl_xor_sync(int(-1), row_nzz, 1);
    // cal += __shfl_xor_sync(int(-1), cal, 16);
    // cal += __shfl_xor_sync(int(-1), cal, 8);
    // cal += __shfl_xor_sync(int(-1), cal, 4);
    // cal += __shfl_xor_sync(int(-1), cal, 2);
    // cal += __shfl_xor_sync(int(-1), cal, 1);

    __syncthreads();
    if (tid == 0) {
        nzzC[rid] = row_nzz;
        if (row_nzz > HASH_COUNT) {
            int pos = atomicAdd(&limit[0], 1);
            limit_pos[pos] = rid;
        }
        else {
            atomicMax(&limit[1], row_nzz);
        }
        // calA[rid] = cal;
    }
}

__global__ void calculate_values_large(
    dist_matrix_t* matA, dist_matrix_t* matB, dist_matrix_t* matC,
    int* limit_pos, data_t* valuess) {
    int rid = limit_pos[blockIdx.x];
    int tid = threadIdx.x;

    extern __shared__ int h_int[];
    int* hash_idx = (int*)h_int;
    data_t* hash_values = valuess + blockIdx.x * HASH_COUNT_MAX;

    for (int i = tid; i < HASH_COUNT_MAX; i += 32) {
        hash_idx[i] = -1;
        hash_values[i] = 0;
    }
    __syncthreads();

    for (int i = matA->gpu_r_pos[rid] + tid; i < matA->gpu_r_pos[rid + 1]; i += 32) {
        int pj = matA->gpu_c_idx[i];
        data_t vj = matA->gpu_values[i];

        for (int j = matB->gpu_r_pos[pj]; j < matB->gpu_r_pos[pj + 1]; j++) {
            int pk = matB->gpu_c_idx[j];
            int hash = (pk & (HASH_COUNT_MAX - 1));
            while (atomicCAS(&hash_idx[hash], -1, pk) != -1) {
                if (hash_idx[hash] == pk) {
                    break;
                }
                hash = ((hash + 1) & (HASH_COUNT_MAX - 1));
            }
            atomicAdd(&hash_values[hash], matB->gpu_values[j] * vj);
        }
    }

    __syncthreads();
    __shared__ int nzz;
    if (tid == 0) {
        nzz = 0;
    }

    __syncthreads();
    int start = matC->gpu_r_pos[rid];
    for (int i = tid; i < HASH_COUNT_MAX; i += 32) {
        if (hash_idx[i] != -1) {
            int pos = atomicAdd(&nzz, 1) + start;
            matC->gpu_c_idx[pos] = hash_idx[i];
            matC->gpu_values[pos] = hash_values[i];
        }
    }
}

__global__ void calculate_values(dist_matrix_t* matA, dist_matrix_t* matB, dist_matrix_t* matC, int* hash_count_max/*, int* rk*/) {
    // int rid = rk[blockIdx.x];
    int rid = blockIdx.x;
    int tid = threadIdx.x;

    if (matC->gpu_r_pos[rid + 1] - matC->gpu_r_pos[rid] > HASH_COUNT) {
        return;
    }

    int HASH_SIZE = (*hash_count_max);
    extern __shared__ int h_int[];
    int* hash_idx = (int*)h_int;
    data_t* hash_values = (data_t*)(h_int + HASH_SIZE);
    // __shared__ int hash_idx[HASH_COUNT];
    // __shared__ data_t hash_values[HASH_COUNT];
    for (int i = tid; i < HASH_SIZE; i += 32) {
        hash_idx[i] = -1;
        hash_values[i] = 0;
    }
    __syncthreads();

    for (int i = matA->gpu_r_pos[rid] + tid; i < matA->gpu_r_pos[rid + 1]; i += 32) {
        int pj = matA->gpu_c_idx[i];
        data_t vj = matA->gpu_values[i];

        for (int j = matB->gpu_r_pos[pj]; j < matB->gpu_r_pos[pj + 1]; j++) {
            int pk = matB->gpu_c_idx[j];
            int hash = (pk & (HASH_SIZE - 1));
            while (atomicCAS(&hash_idx[hash], -1, pk) != -1) {
                if (hash_idx[hash] == pk) {
                    break;
                }
                hash = ((hash + 1) & (HASH_SIZE - 1));
            }
            atomicAdd(&hash_values[hash], matB->gpu_values[j] * vj);
        }
    }

    __syncthreads();
    __shared__ int nzz;
    if (tid == 0) {
        nzz = 0;
    }

    __syncthreads();
    int start = matC->gpu_r_pos[rid];
    for (int i = tid; i < HASH_SIZE; i += 32) {
        if (hash_idx[i] != -1) {
            int pos = atomicAdd(&nzz, 1) + start;
            matC->gpu_c_idx[pos] = hash_idx[i];
            matC->gpu_values[pos] = hash_values[i];
        }
    }
}

void calculate_values_cpu_version(dist_matrix_t* matA, dist_matrix_t* matB, cpu_cal_t* cpu_cals, int rid) {
    int hash_idx[HASH_COUNT_MAX];
    data_t hash_values[HASH_COUNT_MAX];
    for (int i = 0; i < HASH_COUNT_MAX; i++) {
        hash_idx[i] = -1;
    }

    for (int i = matA->r_pos[rid]; i < matA->r_pos[rid + 1]; i++) {
        int pj = matA->c_idx[i];
        data_t vj = matA->values[i];

        for (int j = matB->r_pos[pj]; j < matB->r_pos[pj + 1]; j++) {
            int pk = matB->c_idx[j];
            int hash = (pk & (HASH_COUNT_MAX - 1));
            while (hash_idx[hash] != -1) {
                if (hash_idx[hash] == pk) {
                    break;
                }
                hash = ((hash + 1) & (HASH_COUNT_MAX - 1));
            }
            if (hash_idx[hash] == -1) {
                hash_values[hash] = 0;
                hash_idx[hash] = pk;
            }
            hash_values[hash] += matB->values[j] * vj;
        }
    }

    int cnt = 0;
    for (int i = 0; i < HASH_COUNT_MAX; i++) {
        if (hash_idx[i] != -1) {
            cpu_cals->c_idx[cnt] = hash_idx[i];
            cpu_cals->values[cnt] = hash_values[i];
            cnt++;
        }
    }
}

// int cmp(const void* a, const void* b) {
//     int* aa = (int*)a, * bb = (int*)b;
//     return calA[(*aa)] > calA[(*bb)];
// }

void spgemm(dist_matrix_t* matA, dist_matrix_t* matB, dist_matrix_t* matC) {
    int n_row = matA->dist_m, n_col = matA->global_m;

    matC->global_m = n_col;
    matC->dist_m = n_row;
    dist_matrix_t* matA_gpu, * matB_gpu, * matC_gpu;
    int* cntC_gpu, * calA_gpu;
    // int* rk, * rk_gpu;
    // rk = (int*)malloc(sizeof(int) * n_row);
    // calA = (int*)malloc(sizeof(int) * n_row);

    // sort(rk + 1, rk + n_row);

    CUDA_CHECK_ERROR(cudaMalloc((void**)&matA_gpu, sizeof(dist_matrix_t)));
    CUDA_CHECK_ERROR(cudaMemcpy(matA_gpu, matA, sizeof(dist_matrix_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&matB_gpu, sizeof(dist_matrix_t)));
    CUDA_CHECK_ERROR(cudaMemcpy(matB_gpu, matB, sizeof(dist_matrix_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&cntC_gpu, sizeof(int) * (n_row + 1)));

    // cudaMemset(cntC_gpu, 0, sizeof(int) * (n_row + 1));
    // cudaMalloc((void**)&calA_gpu, sizeof(int) * (n_row + 1));

    int cnt = 0;

    int* limit_gpu, * limit_pos_gpu;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&limit_gpu, 2 * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemset(limit_gpu, 0, 2 * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&limit_pos_gpu, n_row * sizeof(int)));

    dim3 dimGrid1(n_row);
    dim3 dimBlock1(32);

    cudaFuncSetAttribute(calculate_nzz, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    calculate_nzz << <dimGrid1, dimBlock1, HASH_COUNT_MAX * sizeof(int) >> > (matA_gpu, matB_gpu, cntC_gpu/*, calA_gpu*/, limit_gpu, limit_pos_gpu);
    // cpu_cal_t* cpu_cals;
    // if (limit > 0) {
    //     cpu_cals = (cpu_cal_t*)malloc(limit * sizeof(cpu_cal_t));
    // }

    // CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    CUDA_CHECK_ERROR(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, cntC_gpu, cntC_gpu, n_row));
    // Allocate temporary storage
    CUDA_CHECK_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run exclusive prefix sum
    CUDA_CHECK_ERROR(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, cntC_gpu, cntC_gpu, n_row));

    matC->r_pos = (int*)malloc(sizeof(int) * (n_row + 1));
    matC->r_pos[0] = 0;
    CUDA_CHECK_ERROR(cudaMemcpy(matC->r_pos + 1, cntC_gpu, sizeof(int) * n_row, cudaMemcpyDeviceToHost));

    cnt = matC->r_pos[n_row];

    CUDA_CHECK_ERROR(cudaMalloc((void**)&matC->gpu_r_pos, (n_row + 1) * sizeof(data_t)));
    CUDA_CHECK_ERROR(cudaMemcpy(matC->gpu_r_pos, matC->r_pos, (n_row + 1) * sizeof(data_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&matC->gpu_c_idx, cnt * sizeof(index_t)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&matC->gpu_values, cnt * sizeof(data_t)));

    CUDA_CHECK_ERROR(cudaMalloc((void**)&matC_gpu, sizeof(dist_matrix_t)));
    CUDA_CHECK_ERROR(cudaMemcpy(matC_gpu, matC, sizeof(dist_matrix_t), cudaMemcpyHostToDevice));

    int limit[2] = { 0, 0 };
    CUDA_CHECK_ERROR(cudaMemcpy(limit, limit_gpu, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    // if (pid == 0)
    //     printf("Total = %d, Max_line = %d\n", limit[0], limit[1]);

    int max_length = 1;
    while (max_length < limit[1]) {
        max_length <<= 1;
    }
    int* max_length_gpu;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&max_length_gpu, sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemcpy(max_length_gpu, &max_length, sizeof(int), cudaMemcpyHostToDevice));

    calculate_values << <dimGrid1, dimBlock1, max_length* (sizeof(int) + sizeof(data_t)) >> > (matA_gpu, matB_gpu, matC_gpu/*, rk_gpu*/, max_length_gpu);

    data_t* values_gpu = NULL;

    if (limit[0] > 0) {
        dim3 dimGrid2(limit[0]);
        dim3 dimBlock2(32);
        int* limit_pos;
        // limit_pos = (int*)malloc(limit * sizeof(int));
        // CUDA_CHECK_ERROR(cudaMemcpy(limit_pos, limit_pos_gpu, limit[0] * sizeof(int), cudaMemcpyDeviceToHost));

        // int* hash_idx_gpu;
        // CUDA_CHECK_ERROR(cudaMalloc((void**)&hash_idx_gpu, sizeof(int) * HASH_COUNT_MAX * limit[0]));
        CUDA_CHECK_ERROR(cudaMalloc((void**)&values_gpu, sizeof(data_t) * HASH_COUNT_MAX * limit[0]));

        cudaFuncSetAttribute(calculate_values_large, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
        calculate_values_large << < dimGrid2, dimBlock2, HASH_COUNT_MAX * sizeof(int) >> > (matA_gpu, matB_gpu, matC_gpu, limit_pos_gpu, values_gpu);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    matC->global_nnz = cnt;
    matC->c_idx = (int*)malloc(sizeof(int) * cnt);
    matC->values = (data_t*)malloc(sizeof(data_t) * cnt);
    CUDA_CHECK_ERROR(cudaMemcpy(matC->c_idx, matC->gpu_c_idx, cnt * sizeof(index_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(matC->values, matC->gpu_values, cnt * sizeof(data_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // if (pid == 0) {
    //     for (int i = 0; i < matC->r_pos[1]; i++) {
    //         printf("%d %f\n", matC->c_idx[i], matC->values[i]);
    //     }
    //     printf("=======########=========\n");
    // }
    // if (limit[0] > 0) {
//     CUDA_CHECK_ERROR(cudaFree(values_gpu));
// }
// info_ptr_t p = (info_ptr_t)malloc(sizeof(additional_info_t));
// matC->additional_info = p;
// p->matA_gpu = matA_gpu;
// p->matB_gpu = matB_gpu;
// p->matC_gpu = matC_gpu;
// p->cntC_gpu = cntC_gpu;
// p->limit_pos_gpu = limit_pos_gpu;
// p->max_length_gpu = max_length_gpu;
// p->values_gpu = values_gpu;
    // CUDA_CHECK_ERROR(cudaFree(matA_gpu));
    // CUDA_CHECK_ERROR(cudaFree(matB_gpu));
    // CUDA_CHECK_ERROR(cudaFree(matC_gpu));
    // CUDA_CHECK_ERROR(cudaFree(limit_pos_gpu));
    // CUDA_CHECK_ERROR(cudaFree(limit_gpu));
    // CUDA_CHECK_ERROR(cudaFree(values_gpu));
    // CUDA_CHECK_ERROR(cudaFree(cntC_gpu));
    // CUDA_CHECK_ERROR(cudaFree(max_length_gpu));
}

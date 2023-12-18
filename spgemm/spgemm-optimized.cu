#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda.h>
#include "common.h"
#include "utils.h"
#include <cub/cub.cuh>
#include <algorithm>
const char* version_name = "optimized version";\
typedef struct {
    dist_matrix_t* matA_gpu, * matB_gpu, * matC_gpu;
    int* cntC_gpu;
    int* rk, * rk_gpu;
} additional_info_t;

typedef additional_info_t* info_ptr_t;

int* calA;

void preprocess(dist_matrix_t* matA, dist_matrix_t* matB) {
    cudaMalloc((void**)&matA->gpu_r_pos, (matA->global_m + 1) * sizeof(index_t));
    cudaMalloc((void**)&matA->gpu_c_idx, matA->global_nnz * sizeof(index_t));
    cudaMalloc((void**)&matA->gpu_values, matA->global_nnz * sizeof(data_t));
    cudaMemcpy(matA->gpu_r_pos, matA->r_pos, (matA->global_m + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(matA->gpu_c_idx, matA->c_idx, (matA->global_nnz) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(matA->gpu_values, matA->values, (matA->global_nnz) * sizeof(data_t), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&matB->gpu_r_pos, (matB->global_m + 1) * sizeof(index_t));
    cudaMalloc((void**)&matB->gpu_c_idx, matB->global_nnz * sizeof(index_t));
    cudaMalloc((void**)&matB->gpu_values, matB->global_nnz * sizeof(data_t));
    cudaMemcpy(matB->gpu_r_pos, matB->r_pos, (matB->global_m + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(matB->gpu_c_idx, matB->c_idx, (matB->global_nnz) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(matB->gpu_values, matB->values, (matB->global_nnz) * sizeof(data_t), cudaMemcpyHostToDevice);

    // info_ptr_t p = (info_ptr_t)malloc(sizeof(additional_info_t));
    // cudaMalloc((void**)&p->cntC, (matA->global_m + 1) * sizeof(index_t));
    // cudaMemset(p->cntC, 0, (matA->global_m + 1) * sizeof(index_t));
    // matA->additional_info = p;
}

void destroy_additional_info(void* additional_info) {
    // info_ptr_t p = (info_ptr_t)additional_info;
    // cudaFree(p->cntC);
    // free(p);
}

inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}

#define HASH_COUNT (2048)

__global__ void calculate_nzz(dist_matrix_t* matA, dist_matrix_t* matB, int* nzzC/*, int* calA*/) {
    int rid = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ int hash_idx[HASH_COUNT];
    for (int i = tid; i < HASH_COUNT; i += 32) {
        hash_idx[i] = -1;
    }
    __syncthreads();

    int row_nzz = 0, cal = 0;
    for (int i = matA->gpu_r_pos[rid] + tid; i < matA->gpu_r_pos[rid + 1]; i++) {
        int pj = matA->gpu_c_idx[i];

        for (int j = matB->gpu_r_pos[pj]; j < matB->gpu_r_pos[pj + 1]; j++) {
            int pk = matB->gpu_c_idx[j];
            int hash = (pk & (HASH_COUNT - 1));
            while (atomicCAS(&hash_idx[hash], -1, pk) != -1) {
                if (hash_idx[hash] == pk) {
                    row_nzz--;
                    break;
                }
                hash = ((hash + 1) & (HASH_COUNT - 1));
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
        // calA[rid] = cal;
    }
}

__global__ void calculate_values(dist_matrix_t* matA, dist_matrix_t* matB, dist_matrix_t* matC/*, int* rk*/) {
    // int rid = rk[blockIdx.x];
    int rid = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ int hash_idx[HASH_COUNT];
    __shared__ data_t hash_values[HASH_COUNT];
    for (int i = tid; i < HASH_COUNT; i += 32) {
        hash_idx[i] = -1;
        hash_values[i] = 0;
    }
    __syncthreads();

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

    __syncthreads();
    __shared__ int nzz;
    if (tid == 0) {
        nzz = 0;
    }

    __syncthreads();
    int start = matC->gpu_r_pos[rid];
    for (int i = tid; i < HASH_COUNT; i += 32) {
        if (hash_idx[i] != -1) {
            int pos = atomicAdd(&nzz, 1) + start;
            matC->gpu_c_idx[pos] = hash_idx[i];
            matC->gpu_values[pos] = hash_values[i];
        }
    }
}

#define START_HEAD (-2)

void matrix_value_calculate(dist_matrix_t* matA, dist_matrix_t* matB, dist_matrix_t* matC) {
    int n_row = matA->global_m;
    int flag = 0;
    int next[n_row];
    data_t sum[n_row];
    for (int i = 0; i < n_row; i++) {
        next[i] = -1;
    }
    for (int i = 0; i < n_row; i++) {
        sum[i] = 0.0;
    }
    int offset = 0;

    for (int i = 0; i < n_row; i++) {
        matC->r_pos[i] = offset;
        int row_nzz = 0;
        int head = START_HEAD;
        for (int j = matA->r_pos[i]; j < matA->r_pos[i + 1]; j++) {
            int pj = matA->c_idx[j];
            data_t vj = matA->values[j];
            for (int k = matB->r_pos[pj]; k < matB->r_pos[pj + 1]; k++) {
                int pk = matB->c_idx[k];
                sum[pk] += matB->values[k] * vj;
                if (next[pk] == -1) {
                    next[pk] = head;
                    head = pk;
                    row_nzz++;
                }
            }
        }
        for (int j = 0; j < row_nzz; j++) {
            if (sum[head] != 0) {
                matC->c_idx[offset] = head;
                matC->values[offset] = sum[head];
                offset++;
            }
            int front = next[head];
            next[head] = -1;
            sum[head] = 0.0;
            head = front;
        }
    }
    matC->r_pos[n_row] = offset;
    // printf("offset = %d\n", offset);
}

// int cmp(const void* a, const void* b) {
//     int* aa = (int*)a, * bb = (int*)b;
//     return calA[(*aa)] > calA[(*bb)];
// }

void spgemm(dist_matrix_t* matA, dist_matrix_t* matB, dist_matrix_t* matC) {
    int n_row = matA->global_m;
    matC->global_m = n_row;
    dist_matrix_t* matA_gpu, * matB_gpu, * matC_gpu;
    int* cntC_gpu, * calA_gpu;
    // int* rk, * rk_gpu;
    // rk = (int*)malloc(sizeof(int) * n_row);
    // calA = (int*)malloc(sizeof(int) * n_row);

    // sort(rk + 1, rk + n_row);


    cudaMalloc((void**)&matA_gpu, sizeof(dist_matrix_t));
    cudaMemcpy(matA_gpu, matA, sizeof(dist_matrix_t), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&matB_gpu, sizeof(dist_matrix_t));
    cudaMemcpy(matB_gpu, matB, sizeof(dist_matrix_t), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&cntC_gpu, sizeof(int) * (n_row + 1));
    // cudaMemset(cntC_gpu, 0, sizeof(int) * (n_row + 1));
    // cudaMalloc((void**)&calA_gpu, sizeof(int) * (n_row + 1));

    int cnt = 0;
    // int* total_gpu;
    // cudaMalloc((void**)&total_gpu, sizeof(int));
    // cudaMemset(total_gpu, 0, sizeof(int));
    dim3 dimGrid1(n_row);
    dim3 dimBlock1(32);
    calculate_nzz << <dimGrid1, dimBlock1 >> > (matA_gpu, matB_gpu, cntC_gpu/*, calA_gpu*/);
    // cudaMemcpy(calA, calA_gpu, n_row * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < n_row; i++) {
    //     rk[i] = i;
    // }
    // qsort(rk, n_row, sizeof(int), cmp);
    // for (int i = 0; i < n_row; i++) {
    //     printf("rk[%d] = %d, times[%d] = %d\n", i, rk[i], i, calA[rk[i]]);
    // }

    // cudaMalloc((void**)&rk_gpu, sizeof(int) * n_row);
    // cudaMemcpy(rk_gpu, rk, sizeof(int) * n_row, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // cudaMemcpy(&cnt, total_gpu, sizeof(int), cudaMemcpyDeviceToHost);

    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, cntC_gpu, cntC_gpu, n_row);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, cntC_gpu, cntC_gpu, n_row);
    // printf("cnt = %d\n", cnt);

    matC->r_pos = (int*)malloc(sizeof(int) * (matC->global_m + 1));
    matC->r_pos[0] = 0;
    cudaMemcpy(matC->r_pos + 1, cntC_gpu, sizeof(int) * matC->global_m, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < 10; i++) {
    //     printf("%d ", matC->r_pos[i]);
    // }
    // printf("\n");

    cnt = matC->r_pos[n_row];
    // printf("cnt = %d\n", cnt);

    cudaMalloc((void**)&matC->gpu_r_pos, (n_row + 1) * sizeof(data_t));
    cudaMemcpy(matC->gpu_r_pos, matC->r_pos, (n_row + 1) * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&matC->gpu_c_idx, cnt * sizeof(index_t));
    cudaMalloc((void**)&matC->gpu_values, cnt * sizeof(data_t));

    cudaMalloc((void**)&matC_gpu, sizeof(dist_matrix_t));
    cudaMemcpy(matC_gpu, matC, sizeof(dist_matrix_t), cudaMemcpyHostToDevice);
    calculate_values << <dimGrid1, dimBlock1 >> > (matA_gpu, matB_gpu, matC_gpu/*, rk_gpu*/);


    // // // info_ptr_t p = (info_ptr_t)matA->additional_info;
    // // // int cnt = matrix_max_nzz_calculate(matA, matB, p->cntC);
    matC->c_idx = (int*)malloc(sizeof(int) * cnt);
    matC->values = (data_t*)malloc(sizeof(data_t) * cnt);
    cudaMemcpy(matC->c_idx, matC->gpu_c_idx, cnt * sizeof(index_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(matC->values, matC->gpu_values, cnt * sizeof(data_t), cudaMemcpyDeviceToHost);

    // matrix_value_calculate(matA, matB, matC);
}

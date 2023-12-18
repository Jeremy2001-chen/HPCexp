#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda.h>
#include "common.h"
#include "utils.h"
#include <cuda_runtime.h>
const char* version_name = "optimized version";\
typedef struct {
    index_t* pos_C, * gpu_cnt_C;
    data_t* gpu_temp_values, * temp_values;
    index_t* gpu_temp_index, * temp_index;
    index_t* gpu_key_pos, * gpu_key_cnt;
} additional_info_t;

typedef additional_info_t* info_ptr_t;

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

    info_ptr_t p = (info_ptr_t)malloc(sizeof(additional_info_t));
    p->pos_C = (index_t*)malloc((matA->global_m + 1) * sizeof(index_t));
    cudaMalloc((void**)&p->gpu_cnt_C, (matA->global_m + 1) * sizeof(index_t));
    cudaMemset(p->gpu_cnt_C, 0, (matA->global_m + 1) * sizeof(index_t));
    cudaMalloc((void**)&p->gpu_key_pos, matA->global_nnz * sizeof(index_t));
    cudaMemset(p->gpu_key_pos, 0, (matA->global_nnz) * sizeof(index_t));
    cudaMalloc((void**)&p->gpu_key_cnt, matA->global_nnz * sizeof(index_t));
    cudaMemset(p->gpu_key_cnt, 0, (matA->global_nnz) * sizeof(index_t));

    matA->additional_info = p;
}

void destroy_additional_info(void* additional_info) {
    info_ptr_t p = (info_ptr_t)additional_info;
    cudaFree(p->gpu_cnt_C);
    cudaFree(p->gpu_key_pos);
    free(p);
}

inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}

__global__ void calculate_nzz(index_t* A_gpu_r_pos,
    index_t* A_gpu_c_idx,
    index_t* A_gpu_key_pos,
    index_t* A_gpu_key_cnt,
    index_t* B_gpu_r_pos,
    index_t* C_gpu_r_cnt,
    int row,
    int* total) {
    int row_id = blockIdx.x * 32 + threadIdx.x;
    if (row_id >= row) return;
    int row_nzz = 0;
    for (int j = A_gpu_r_pos[row_id]; j < A_gpu_r_pos[row_id + 1]; j++) {
        int pj = A_gpu_c_idx[j], cnt = B_gpu_r_pos[pj + 1] - B_gpu_r_pos[pj];
        A_gpu_key_pos[j] = row_nzz;
        A_gpu_key_cnt[j] = cnt;
        row_nzz += cnt;
    }
    C_gpu_r_cnt[row_id + 1] = row_nzz;
    atomicAdd(total, row_nzz);
}

__global__ void calculate_init_pos(index_t* A_gpu_r_pos,
    index_t* A_gpu_key_pos,
    index_t* C_gpu_r_cnt,
    int row) {
    int row_id = blockIdx.x * 32 + threadIdx.x;
    if (row_id >= row) return;
    int base = C_gpu_r_cnt[row_id];
    for (int j = A_gpu_r_pos[row_id]; j < A_gpu_r_pos[row_id + 1]; j++) {
        A_gpu_key_pos[j] += base;
    }
}

int matrix_max_nzz_calculate(dist_matrix_t* matA, dist_matrix_t* matB, index_t* pos_C, index_t* gpu_cnt_C, index_t* gpu_key_pos, index_t* gpu_key_cnt) {
    int n_row = matA->global_m;
    dim3 dimGrid(ceiling(n_row, 32));
    dim3 dimBlock(32);
    int row_nzz;
    int* gpu_row_nzz;
    cudaMalloc((void**)&gpu_row_nzz, sizeof(int));
    cudaMemset(gpu_row_nzz, 0, sizeof(int));

    calculate_nzz << <dimGrid, dimBlock >> > (matA->gpu_r_pos, matA->gpu_c_idx, gpu_key_pos, gpu_key_cnt, matB->gpu_r_pos, gpu_cnt_C, n_row, gpu_row_nzz);

    cudaDeviceSynchronize();

    cudaMemcpy(&row_nzz, gpu_row_nzz, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pos_C, gpu_cnt_C, (n_row + 1) * sizeof(index_t), cudaMemcpyDeviceToHost);

    pos_C[0] = 0;
    for (int i = 1; i <= n_row; i++) {
        pos_C[i] += pos_C[i - 1];
    }

    cudaMemcpy(gpu_cnt_C, pos_C, (n_row + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    calculate_init_pos << <dimGrid, dimBlock >> > (matA->gpu_r_pos, gpu_key_pos, gpu_cnt_C, n_row);

    return row_nzz;
}

__global__ void calculate_value(index_t* A_gpu_r_pos,
    // index_t* A_gpu_c_idx,
    // data_t* A_gpu_values,
    // index_t* A_gpu_key_pos,
    index_t* A_gpu_key_cnt,
    // index_t* B_gpu_r_pos,
    // index_t* B_gpu_c_idx,
    // data_t* B_gpu_values,
    index_t* C_gpu_r_pos,
    // index_t* C_gpu_c_idx,
    // data_t* C_gpu_values,
    // index_t* temp_c_idx,
    // data_t* temp_c_values,
    int n_row) {
    int row_id = blockIdx.x * 32 + threadIdx.x;
    if (row_id >= n_row) return;
    // if (row_id == 0) {
    //     for (int i = 0; i < 10; i++) {
    //         printf("%d ", A_gpu_key_pos[i]);
    //     }
    //     printf("\n");
    // }
    // printf("%d\n", row_id);

    if (row_id <= 10) {
        int pos_start = A_gpu_r_pos[row_id], pos_end = A_gpu_r_pos[row_id + 1];
        printf("[Row ID %d]%d %d\n", row_id, pos_start, pos_end);

        int total = A_gpu_key_cnt[pos_start];
        printf("%d\n", total);
        // for (int j = pos_start; j < pos_end; j++) {
        //     int pj = A_gpu_c_idx[j];
        //     data_t vj = A_gpu_values[j];
        //     int col_start = B_gpu_r_pos[pj], col_end = B_gpu_r_pos[pj + 1], pos = A_gpu_key_pos[j];
        //     for (int k = col_start; k < col_end; k++) {
        //         // C_gpu_c_idx[pos] = B_gpu_c_idx[k];
        //         // C_gpu_values[pos] = B_gpu_values[k] * vj;
        //         printf("%d %d %d %.12f %.12f %.12f %.12f\n", row_id, pos, C_gpu_c_idx[pos], C_gpu_values[pos], B_gpu_values[k], vj, B_gpu_values[k] * vj);
        //         pos++;
        //     }
        // }

        // for (int i = 1; i < pos_end - pos_start; i <<= 1) {
        //     printf("%d %d\n", pos_start, pos_end);
        //     for (int j = pos_start; j + i < pos_end; j += (i << 1)) {
        //         int left_idx = A_gpu_key_pos[j], left_fin = left_idx + A_gpu_key_cnt[j];
        //         int right_idx = A_gpu_key_pos[j + i], right_fin = right_idx + A_gpu_key_cnt[j + i];
        //         int idx = left_idx;
        //         printf("%d %d %d %d\n", left_idx, right_idx, left_fin, right_fin);
        //         while (left_idx < left_fin && right_idx < right_fin) {
        //             if (C_gpu_c_idx[left_idx] < C_gpu_c_idx[right_idx]) {
        //                 temp_c_idx[idx] = C_gpu_c_idx[left_idx];
        //                 temp_c_values[idx] = C_gpu_values[left_idx];
        //                 idx++; left_idx++;
        //             }
        //             else if (C_gpu_c_idx[left_idx] == C_gpu_c_idx[right_idx]) {
        //                 temp_c_idx[idx] = C_gpu_c_idx[left_idx];
        //                 temp_c_values[idx] = C_gpu_values[left_idx] + C_gpu_values[right_idx];
        //                 idx++; left_idx++; right_idx++;
        //             }
        //             else {
        //                 temp_c_idx[idx] = C_gpu_c_idx[right_idx];
        //                 temp_c_values[idx] = C_gpu_values[right_idx];
        //                 idx++; right_idx++;
        //             }
        //         }
        //         while (left_idx < left_fin) {
        //             temp_c_idx[idx] = C_gpu_c_idx[left_idx];
        //             temp_c_values[idx] = C_gpu_values[left_idx];
        //             idx++; left_idx++;
        //         }
        //         while (right_idx < right_fin) {
        //             temp_c_idx[idx] = C_gpu_c_idx[right_idx];
        //             temp_c_values[idx] = C_gpu_values[right_idx];
        //             idx++; right_idx++;
        //         }
        //         printf("idx = %d\n", idx);
        //         A_gpu_key_cnt[j] = idx - A_gpu_key_pos[j];
        //         for (int k = A_gpu_key_pos[j]; k < idx; k++) {
        //             C_gpu_c_idx[k] = temp_c_idx[k];
        //             C_gpu_values[k] = temp_c_values[k];
        //         }
        //     }
        // }

        // if (pos_start + A_gpu_key_cnt[pos_start] < pos_end) {
        //     temp_c_idx[pos_start + A_gpu_key_cnt[pos_start]] = -1;
        // }
        // C_gpu_r_pos[row_id] = A_gpu_key_cnt[pos_start];
        // C_gpu_r_pos[row_id] = A_gpu_key_pos[pos_start];
        C_gpu_r_pos[row_id] = total;
        // printf("[%d]%d %d %d\n", row_id, pos_start, C_gpu_r_pos[row_id], pos_end);
        // for (int k = pos_start; k < pos_start + A_gpu_key_cnt[pos_start]; k++) {
        //     printf("[%d](%d, %.12f)\n", row_id, C_gpu_c_idx[k], C_gpu_values[k]);
        // }
        // printf("A: row_id = %d, r_pos = %d\n", row_id, C_gpu_r_pos[row_id]);
    }
}

inline void matrix_value_calculate(dist_matrix_t* matA, dist_matrix_t* matB, dist_matrix_t* matC,
    index_t* gpu_key_pos, index_t* gpu_key_cnt, index_t* gpu_temp_idx, data_t* gpu_temp_values,
    index_t* temp_idx, data_t* temp_values,
    index_t* pos_C, int est) {
    int n_row = matA->global_m;
    dim3 dimGrid(ceiling(n_row, 32));
    dim3 dimBlock(32);

    printf("hi~\n");
    calculate_value << <dimGrid, dimBlock >> > (matA->gpu_r_pos,
        // matA->gpu_c_idx,
        // matA->gpu_values,
        // gpu_key_pos,
        gpu_key_cnt,
        // matB->gpu_r_pos,
        // matB->gpu_c_idx,
        // matB->gpu_values,
        matC->gpu_r_pos,
        // matC->gpu_c_idx,
        // matC->gpu_values,
        // gpu_temp_idx,
        // gpu_temp_values,
        n_row);

    // cudaError err = cudaGetLastError();
    cudaDeviceSynchronize();
    // printf("%d\n", err);

    cudaMemcpy(matC->r_pos, matC->gpu_r_pos, (n_row + 1) * sizeof(index_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(temp_idx, gpu_temp_idx, est * sizeof(index_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(temp_values, gpu_temp_values, est * sizeof(data_t), cudaMemcpyDeviceToHost);
    // err = cudaGetLastError();

    // printf("%d\n", err);

    cudaDeviceSynchronize();

    // printf("%d\n", err);


    int cnt = 1;
    // for (int i = 0; i < 0; i++) {
    //     cnt += matC->r_pos[i];
    //     matC->r_pos[i] = cnt - matC->r_pos[i];
    // }

// matC->r_pos[n_row] = cnt;

    // printf("GG\n");
    // printf("cnt = %d, n_row = %d\n", cnt, n_row);
    for (int i = 0; i < 1; i++) {
        printf("%d\n", matC->r_pos[i]);
    }

    // matC->c_idx = (int*)malloc(sizeof(index_t) * cnt);
    // matC->values = (data_t*)malloc(sizeof(data_t) * cnt);

    // printf("GG!\n");

    // int idx = 0;
    // for (int i = 0; i < n_row; i++) {
    //     for (int j = pos_C[i]; j < pos_C[i + 1]; j++) {
    //         if (temp_idx[j] == -1) { break; }
    //         matC->c_idx[idx] = temp_idx[j];
    //         matC->values[idx] = temp_values[j];
    //         idx++;
    //     }
    // }
}

// #define START_HEAD (-2)

// void matrix_value_calculate(dist_matrix_t* matA, dist_matrix_t* matB, dist_matrix_t* matC) {
//     int n_row = matA->global_m;
//     int flag = 0;
//     int next[n_row];
//     data_t sum[n_row];
//     for (int i = 0; i < n_row; i++) {
//         next[i] = -1;
//     }
//     for (int i = 0; i < n_row; i++) {
//         sum[i] = 0.0;
//     }
//     int offset = 0;

//     for (int i = 0; i < n_row; i++) {
//         matC->r_pos[i] = offset;
//         int row_nzz = 0;
//         int head = START_HEAD;
//         for (int j = matA->r_pos[i]; j < matA->r_pos[i + 1]; j++) {
//             int pj = matA->c_idx[j];
//             data_t vj = matA->values[j];
//             for (int k = matB->r_pos[pj]; k < matB->r_pos[pj + 1]; k++) {
//                 int pk = matB->c_idx[k];
//                 sum[pk] += matB->values[k] * vj;
//                 if (next[pk] == -1) {
//                     next[pk] = head;
//                     head = pk;
//                     row_nzz++;
//                 }
//             }
//         }
//         for (int j = 0; j < row_nzz; j++) {
//             if (sum[head] != 0) {
//                 matC->c_idx[offset] = head;
//                 matC->values[offset] = sum[head];
//                 offset++;
//             }
//             int front = next[head];
//             next[head] = -1;
//             sum[head] = 0.0;
//             head = front;
//         }
//     }
//     matC->r_pos[n_row] = offset;
// }

__global__ void test(index_t* array, int n_row) {
    int row_id = blockIdx.x * 32 + threadIdx.x;
    if (row_id >= n_row) return;
    array[row_id] = n_row;
}

void spgemm(dist_matrix_t* matA, dist_matrix_t* matB, dist_matrix_t* matC) {
    matC->global_m = matA->global_m;

    matC->r_pos = (index_t*)malloc(sizeof(index_t) * (matC->global_m + 1));
    memset(matC->r_pos, 0, sizeof(index_t) * (matC->global_m + 1));
    // please put your result back in matC->r_pos / c_idx / values in CSR format
    info_ptr_t p = (info_ptr_t)matA->additional_info;
    int cnt = matrix_max_nzz_calculate(matA, matB, p->pos_C, p->gpu_cnt_C, p->gpu_key_pos, p->gpu_key_cnt);
    printf("cnt = %d, m = %d\n, nzz = %d\n", cnt, matC->global_m + 1, matA->global_nnz);
    cnt = 0;

    // index_t* gpu_r_pos;
    // cudaMalloc((void**)&gpu_r_pos, (matC->global_m + 1) * sizeof(index_t));
    cudaMalloc((void**)&matC->gpu_r_pos, (matC->global_m + 1) * sizeof(index_t));
    cudaMemset(matC->gpu_r_pos, 0, (matC->global_m + 1) * sizeof(index_t));
    cudaMalloc((void**)&matC->gpu_c_idx, cnt * sizeof(index_t));
    cudaMemset(matC->gpu_c_idx, 0, cnt * sizeof(index_t));
    cudaMalloc((void**)&matC->gpu_values, cnt * sizeof(data_t));
    cudaMemset(matC->gpu_values, 0, cnt * sizeof(data_t));
    // printf("hello world!\n");
    // cudaMemcpy(gpu_r_pos, matC->r_pos, (matC->global_m + 1) * sizeof(index_t), cudaMemcpyHostToDevice);

    p->temp_index = (index_t*)malloc(sizeof(index_t) * cnt);
    p->temp_values = (data_t*)malloc(sizeof(data_t) * cnt);

    cudaMalloc((void**)&p->gpu_temp_index, cnt * sizeof(index_t));
    // cudaMemset(p->gpu_temp_index, 0, cnt * sizeof(index_t));
    cudaMalloc((void**)&p->gpu_temp_values, cnt * sizeof(data_t));
    // cudaMemset(p->gpu_temp_values, 0, cnt * sizeof(data_t));

    matrix_value_calculate(matA, matB, matC, p->gpu_key_pos, p->gpu_key_cnt, p->gpu_temp_index, p->gpu_temp_values, p->temp_index, p->temp_values, p->pos_C, cnt);

    // dim3 dimGrid(ceiling(matC->global_m, 32));
    // dim3 dimBlock(32);
    // // // cudaMemcpy(matC->r_pos, matC->gpu_r_pos, (matC->global_m + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    // test << <dimGrid, dimBlock >> > (matC->gpu_r_pos, matC->global_m);
    // cudaMemcpy(matC->r_pos, matC->gpu_r_pos, (matC->global_m + 1) * sizeof(index_t), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 10; i++) {
    //     printf("%d ", matC->r_pos[i]);
    // }
    // printf("\n");
}

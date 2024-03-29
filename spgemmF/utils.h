#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED 1

#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK(err, err_code) if(err) { return err_code; }
#define CHECK_ERROR(ret, err_code) CHECK(ret != 0, err_code)
#define CHECK_NULL(ret, err_code) CHECK(ret == NULL, err_code)

#ifdef __cplusplus
extern "C" {
#endif

int fatal_error(int code);

int read_matrix_default(dist_matrix_t *mat, const char* filename);
void destroy_dist_matrix(dist_matrix_t *mat);

int read_vector(dist_matrix_t *mat, const char* filename, const char* suffix, int n, data_t* x);
int check_answer(dist_matrix_t *mat, const char* filename);
void init_one_gpu_per_process(int my_pid);

#ifdef __cplusplus
}
#endif

#define SUCCESS cudaSuccess
#define NO_MEM cudaErrorMemoryAllocation
#define IO_ERR 3

#endif
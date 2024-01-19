#include "mpi.h"
#include <sys/time.h>
#include "common.h"
#include "utils.h"

extern const char* version_name;

int parse_args(int* reps, int p_id, int argc, char** argv);
int my_abort(int line, int code);
void main_gpufree(void* p);

#define MY_ABORT(ret) my_abort(__LINE__, ret)
#define ABORT_IF_ERROR(ret) CHECK_ERROR(ret, MY_ABORT(ret))
#define ABORT_IF_NULL(ret) CHECK_NULL(ret, MY_ABORT(NO_MEM))
#define INDENT "    "
#define TIME_DIFF(start, stop) 1.0 * (stop.tv_sec - start.tv_sec) + 1e-6 * (stop.tv_usec - start.tv_usec)


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int my_pid, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // TODO: This function allocates one GPU per process.
    //       You could change the way of device assignment 
    init_one_gpu_per_process(my_pid);

    int warm_up = 5, reps, i, ret;
    dist_matrix_t mat;
    dist_matrix_t matB;
    dist_matrix_t matC;
    double compute_time = 0.0, pre_time = 0.0;
    struct timeval tick0, tick1;

    ret = parse_args(&reps, 0, argc, argv);
    ABORT_IF_ERROR(ret)
        // TODO: if you want to read the matrix in a distributed way, 
        //       or redistribute the matrix after reading it on a particular process,
        //       you could modify here.
        ret = read_matrix_default(&mat, argv[2]);
    ABORT_IF_ERROR(ret)
        ret = read_matrix_default(&matB, argv[2]);
    ABORT_IF_ERROR(ret)

        matC.additional_info = NULL;
    matC.CPU_free = free;
    matC.GPU_free = main_gpufree;

    if (my_pid == 0) {
        printf("Benchmarking %s on %s.\n", version_name, argv[2]);
        printf(INDENT"%d x %d, %d non-zeros, %d run(s)\n", \
            mat.global_m, mat.global_m, mat.global_nnz, reps);
    }

    // This is a very naive implementation that only rank 0 does the work, and the others keep idle
    // Thus, only one GPU is utilized here.
    // You should replace it with your own optimized multi-GPU implementation.
    // Load balance across different devices may be important.

    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&tick0, NULL);

    // TODO: You should replace it in your distributed way of preprocess.
    if (my_pid == 0) {// only rank 0 does the work
        preprocess(&mat, &matB);
    }
    else {
        // idle
    }

    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&tick1, NULL);
    pre_time += TIME_DIFF(tick0, tick1);

    for (int test = 0; test < reps + warm_up; test++) {

        // TODO: Clear the working space and the result matrix for multiple tests
        //       You should modify it in your distributed way.
        if (my_pid == 0) {// only rank 0 has the data
            destroy_dist_matrix(&matC);
        }
        else {
            // idle
        }

        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&tick0, NULL);

        // TODO: You should modify it in your distributed way.
        if (my_pid == 0) {// only rank 0 does the work
            spgemm(&mat, &matB, &matC);
        }
        else {
            // idle
        }

        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&tick1, NULL);
        if (test >= warm_up) compute_time += TIME_DIFF(tick0, tick1);
    }

    // TODO: Here collect the distributed result to rank 0 for correctness check.
    if (my_pid == 0) {
        // recv the local results from other ranks if needed

        // Rank 0 check correctness
        printf(INDENT"Checking.\n");
        ret = check_answer(&matC, argv[3]);
        if (ret == 0) {
            printf("\e[1;32m"INDENT"Result validated.\e[0m\n");
        }
        else {
            fprintf(stderr, "\e[1;31m"INDENT"Result NOT validated.\e[0m\n");
            MY_ABORT(ret);
        }
        printf(INDENT INDENT"preprocess time = %lf s\n", pre_time);
        printf("\e[1;34m"INDENT INDENT"compute time = %lf s\e[0m\n", compute_time / reps);
    }
    else {
        // send the local result to rank 0 if needed

    }

    destroy_dist_matrix(&mat);
    destroy_dist_matrix(&matB);
    destroy_dist_matrix(&matC);

    MPI_Finalize();
    return 0;
}

void main_gpufree(void* p) {
    cudaFree(p);
}

void print_help(const char* argv0, int p_id) {
    if (p_id == 0) {
        printf("\e[1;31mUSAGE: %s <repetitions> <input-file>\e[0m\n", argv0);
    }
}

int parse_args(int* reps, int p_id, int argc, char** argv) {
    int r;
    if (argc < 3) {
        print_help(argv[0], p_id);
        return 1;
    }
    r = atoi(argv[1]);
    if (r <= 0) {
        print_help(argv[0], p_id);
        return 1;
    }
    *reps = r;
    return SUCCESS;
}

int my_abort(int line, int code) {
    fprintf(stderr, "\e[1;33merror at line %d, error code = %d\e[0m\n", line, code);
    return fatal_error(code);
}

int fatal_error(int code) {
    exit(code);
    return code;
}

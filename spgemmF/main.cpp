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

#define PROCESS_COUNT (8)

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int my_pid, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_pid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // TODO: This function allocates one GPU per process.
    //       You could change the way of device assignment 
    init_one_gpu_per_process(my_pid);

    // char content[1000];
    // FILE* file = fopen("/etc/hostname", "r");
    // if (file != NULL) {
    //     printf("Read Success!\n");
    //     while (fgets(content, sizeof(content), file) != NULL) {
    //         printf("%s", content);
    //     }
    //     printf("\n");
    // }

    int warm_up = 5, reps, i, ret;
    dist_matrix_t mat;
    dist_matrix_t matB;
    dist_matrix_t matC;
    dist_matrix_t matA_d[PROCESS_COUNT], matC_d[PROCESS_COUNT];

    double compute_time = 0.0, pre_time = 0.0;
    struct timeval tick0, tick1;

    ret = parse_args(&reps, 0, argc, argv);
    ABORT_IF_ERROR(ret);

    ret = read_matrix_default(&mat, argv[2]);
    ABORT_IF_ERROR(ret);
    ret = read_matrix_default(&matB, argv[2]);
    ABORT_IF_ERROR(ret);

    for (int i = 0; i < PROCESS_COUNT; i++) {
        matA_d[i].global_m = mat.global_m;
    }
    for (int i = 0; i < mat.global_m; i++) {
        int idx = i % PROCESS_COUNT;
        matA_d[idx].global_nnz += mat.r_pos[i + 1] - mat.r_pos[i];
        matA_d[idx].dist_m++;
    }
    for (int i = 0; i < PROCESS_COUNT; i++) {
        matA_d[i].r_pos = (index_t*)malloc((matA_d[i].dist_m + 1) * sizeof(index_t));
        matA_d[i].r_pos[0] = 0;
        matA_d[i].c_idx = (index_t*)malloc((matA_d[i].global_nnz) * sizeof(index_t));
        matA_d[i].values = (data_t*)malloc((matA_d[i].global_nnz) * sizeof(data_t));
    }
    for (int i = 0; i < mat.global_m; i++) {
        int m_idx = i % PROCESS_COUNT, p_idx = i / PROCESS_COUNT;
        int start = matA_d[m_idx].r_pos[p_idx];
        for (int j = mat.r_pos[i]; j < mat.r_pos[i + 1]; j++) {
            matA_d[m_idx].c_idx[start] = mat.c_idx[j];
            matA_d[m_idx].values[start] = mat.values[j];
            start++;
        }
        matA_d[m_idx].r_pos[p_idx + 1] = start;
    }

    matC.additional_info = NULL;
    matC.CPU_free = free;
    matC.GPU_free = main_gpufree;

    for (int i = 0; i < PROCESS_COUNT; i++) {
        matC_d[i].additional_info = NULL;
        matC_d[i].CPU_free = free;
        matC_d[i].GPU_free = main_gpufree;
    }

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

    preprocess(&matA_d[my_pid], &matB);

    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&tick1, NULL);
    pre_time += TIME_DIFF(tick0, tick1);

    // printf("hello world at %d\n", my_pid);

    for (int test = 0; test < reps + warm_up; test++) {
        destroy_dist_matrix(&matC_d[my_pid]);

        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&tick0, NULL);
        // printf("[%d]Start at %d\n", test, my_pid);

        spgemm(&matA_d[my_pid], &matB, &matC_d[my_pid]);

        // printf("[%d]Finish at %d\n", test, my_pid);

        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&tick1, NULL);
        if (test >= warm_up) compute_time += TIME_DIFF(tick0, tick1);
    }

    MPI_Request request[(PROCESS_COUNT - 1) * 4];
    if (my_pid == 0) {
        // recv the local results from other ranks

        // printf("[Finish]%d main core!\n", my_pid);

        for (int i = 1; i < PROCESS_COUNT; i++) {
            MPI_Irecv(&(matC_d[i].global_nnz), 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request[(i - 1) * 4]);
            MPI_Wait(&request[(i - 1) * 4], MPI_STATUS_IGNORE);
            matC_d[i].r_pos = (index_t*)malloc((matA_d[i].dist_m + 1) * sizeof(index_t));
            matC_d[i].c_idx = (index_t*)malloc((matC_d[i].global_nnz) * sizeof(index_t));
            matC_d[i].values = (data_t*)malloc((matC_d[i].global_nnz) * sizeof(data_t));

            MPI_Irecv(matC_d[i].r_pos, matA_d[i].dist_m + 1, MPI_INT, i, 1, MPI_COMM_WORLD, &request[(i - 1) * 4 + 1]);
            MPI_Irecv(matC_d[i].c_idx, matC_d[i].global_nnz, MPI_INT, i, 2, MPI_COMM_WORLD, &request[(i - 1) * 4 + 2]);
            MPI_Irecv(matC_d[i].values, matC_d[i].global_nnz, MPI_FLOAT, i, 3, MPI_COMM_WORLD, &request[(i - 1) * 4 + 3]);
            MPI_Waitall(3, request + (i - 1) * 4 + 1, MPI_STATUS_IGNORE);
        }

        matC.global_m = matC_d[0].global_m;
        for (int i = 0; i < PROCESS_COUNT; i++) {
            matC.global_nnz += matC_d[i].global_nnz;
            // printf("%d %d %d\n", i, matC_d[i].global_nnz, matC_d[i].r_pos[matA_d[i].dist_m]);
        }
        matC.r_pos = (index_t*)malloc((matC.global_m + 1) * sizeof(index_t));
        matC.c_idx = (index_t*)malloc(matC.global_nnz * sizeof(index_t));
        matC.values = (data_t*)malloc(matC.global_nnz * sizeof(data_t));
        int cnt = 0;
        matC.r_pos[0] = 0;
        for (int i = 0; i < matC.global_m; i++) {
            int m_idx = i % PROCESS_COUNT, p_idx = i / PROCESS_COUNT;
            for (int j = matC_d[m_idx].r_pos[p_idx]; j < matC_d[m_idx].r_pos[p_idx + 1]; j++) {
                // printf("%d %d %d\n", cnt, m_idx, j);
                matC.c_idx[cnt] = matC_d[m_idx].c_idx[j];
                matC.values[cnt] = matC_d[m_idx].values[j];
                cnt++;
            }
            matC.r_pos[i + 1] = cnt;
        }

        // printf("%d %d\n", matC.global_nnz, cnt);
        // // Rank 0 check correctness
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
        // send the local result to rank 0
        // printf("Send: %d %d %d\n", my_pid, matC_d[my_pid].global_nnz, matC_d[my_pid].r_pos[matA_d[my_pid].dist_m]);
        MPI_Isend(&(matC_d[my_pid].global_nnz), 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request[(my_pid - 1) * 4]);
        MPI_Isend(matC_d[my_pid].r_pos, matA_d[my_pid].dist_m + 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &request[(my_pid - 1) * 4 + 1]);
        MPI_Isend(matC_d[my_pid].c_idx, matC_d[my_pid].global_nnz, MPI_INT, 0, 2, MPI_COMM_WORLD, &request[(my_pid - 1) * 4 + 2]);
        MPI_Isend(matC_d[my_pid].values, matC_d[my_pid].global_nnz, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, &request[(my_pid - 1) * 4 + 3]);

        // printf("[Finish]%d!\n", my_pid);
        MPI_Waitall(4, request + ((my_pid - 1) * 4), MPI_STATUS_IGNORE);
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

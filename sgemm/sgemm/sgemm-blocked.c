#include "header.h"
const char* sgemm_desc = "Simple blocked sgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 192
#endif

#define index(a,b) ((a) + (b) * lda)
#define index_T(a, b) ((a) * lda + (b))
#define min(a,b) (((a)<(b))?(a):(b))
#define prefetch_T0(x) _mm_prefetch((x), _MM_HINT_T0)

void print_buffer(const char* str, float* A, int len) {
    printf("%s\n", str);
    for (int i = 0; i < len; i++) {
        printf("%.2f ", A[i]);
    }
    printf("\n");
}

/* This auxiliary subroutine performs a smaller sgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block(int lda, int M, int N, int K, float* A, float* B, float* C)
{
    /* For each row i of A */
    for (int i = 0; i < M; ++i)
        /* For each column j of B */
        for (int j = 0; j < N; ++j)
        {
            /* Compute C(i,j) */
            float cij = C[index(i, j)];
            for (int k = 0; k < K; ++k)
                cij += A[index(i, k)] * B[index(k, j)];
            C[index(i, j)] = cij;
        }
}

static void inline assert(int exp) {
    if (!exp) {
        abort();
    }
}

static inline void packing_matrix_col(float* src, float* dst, int lda, int M, int K) {
    int M_32 = M & -32, M_16 = M & -16, M_8 = M & -8;
    int cnt = 0;
    float* cur_src = src, * cur_dst = dst;
    for (int i = 0; i < M_32; i += 32) {
        cur_src = src + i;
        for (int j = 0; j < K; ++j) {
            _mm512_store_ps(cur_dst, _mm512_loadu_ps(cur_src));
            _mm512_store_ps(cur_dst + 16, _mm512_loadu_ps(cur_src + 16));
            cur_src += lda;
            cur_dst += 32;
        }
    }
    if (M - M_32 == 31) {
        cur_src = src + M_32;
        for (int j = 0; j < K; ++j) {
            _mm512_store_ps(cur_dst, _mm512_loadu_ps(cur_src));
            _mm512_mask_storeu_ps(cur_dst + 16, 0x7FFF, _mm512_loadu_ps(cur_src + 16));
            cur_dst[31] = 0;
            cur_src += lda;
            cur_dst += 32;
        }
    }
}

static inline void packing_matrix_row_together(float* src, float* dst, int lda, int K, int N) {
    int N_8 = N & -8;
    float* s;
    for (int j = 0; j < N_8; j += 8) {
        for (int i = 0; i < K; i++) {
            s = src + index(i, j);
            *dst++ = *s; s += lda;
            *dst++ = *s; s += lda;
            *dst++ = *s; s += lda;
            *dst++ = *s; s += lda;
            *dst++ = *s; s += lda;
            *dst++ = *s; s += lda;
            *dst++ = *s; s += lda;
            *dst++ = *s; s += lda;
        }
    }
    for (int i = 0; i < K; i++) {
        s = src + index(i, N_8);
        *dst++ = *s; s += lda;
        *dst++ = *s; s += lda;
        *dst++ = *s; s += lda;
        *dst++ = *s; s += lda;
        *dst++ = *s; s += lda;
        *dst++ = *s; s += lda;
        *dst++ = *s; s += lda;
        *dst++ = *s; s += lda;
    }
}

static inline void packing_matrix_row(float* src, float* dst, int lda, int K, int N) {
    int N_8 = N & -8;
    float* s0;
    for (int j = 0; j < N_8; j += 8) {
        for (int i = 0; i < K; i++) {
            s0 = src + index(i, j);
            *dst++ = *s0;
            s0 += lda;
            *dst++ = *s0;
            s0 += lda;
            *dst++ = *s0;
            s0 += lda;
            *dst++ = *s0;
            s0 += lda;
            *dst++ = *s0;
            s0 += lda;
            *dst++ = *s0;
            s0 += lda;
            *dst++ = *s0;
            s0 += lda;
            *dst++ = *s0;
            s0 += lda;
        }
    }
    for (int i = 0; i < K; i++) {
        s0 = src + index(i, N_8);
        for (int j = N_8; j < N; j++) {
            *dst++ = *s0;
            s0 += lda;
        }
    }
}


#define _m256_y_add_to_x(x, y) _mm256_storeu_ps((x), _mm256_add_ps((y), _mm256_loadu_ps((x))))
#define _m512_y_add_to_x(x, y) _mm512_storeu_ps((x), _mm512_add_ps((y), _mm512_loadu_ps((x))))
#define _m512_y_add_to_x_15(x, y) _mm512_mask_storeu_ps((x), 0x7FFF, _mm512_add_ps((y), _mm512_loadu_ps((x))))

// #define _m256_y_add_to_x(x, y) _mm256_storeu_ps((x), (y))

static inline void kernel_16xKx8_packing(int lda, int i, int j, int K, float* packing_A, float* packing_B, float* C) {
    __m256 c00 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps();
    __m256 c60 = _mm256_setzero_ps();
    __m256 c70 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c41 = _mm256_setzero_ps();
    __m256 c51 = _mm256_setzero_ps();
    __m256 c61 = _mm256_setzero_ps();
    __m256 c71 = _mm256_setzero_ps();
    prefetch_T0(C + index(i, j));
    prefetch_T0(C + index(i, j + 1));
    prefetch_T0(C + index(i, j + 2));
    prefetch_T0(C + index(i, j + 3));
    prefetch_T0(C + index(i, j + 4));
    prefetch_T0(C + index(i, j + 5));
    prefetch_T0(C + index(i, j + 6));
    prefetch_T0(C + index(i, j + 7));
    __m256 a0, a1, b0, b1, b2, b3, b4, b5, b6, b7;
    int k = 0;
    for (k = 0; k < K; k++) {
        a0 = _mm256_loadu_ps(packing_A);
        a1 = _mm256_loadu_ps(packing_A + 8);
        b0 = _mm256_broadcast_ss(packing_B);
        b1 = _mm256_broadcast_ss(packing_B + 1);
        b2 = _mm256_broadcast_ss(packing_B + 2);
        b3 = _mm256_broadcast_ss(packing_B + 3);
        b4 = _mm256_broadcast_ss(packing_B + 4);
        b5 = _mm256_broadcast_ss(packing_B + 5);
        b6 = _mm256_broadcast_ss(packing_B + 6);
        b7 = _mm256_broadcast_ss(packing_B + 7);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c10 = _mm256_fmadd_ps(a0, b1, c10);
        c20 = _mm256_fmadd_ps(a0, b2, c20);
        c30 = _mm256_fmadd_ps(a0, b3, c30);
        c40 = _mm256_fmadd_ps(a0, b4, c40);
        c50 = _mm256_fmadd_ps(a0, b5, c50);
        c60 = _mm256_fmadd_ps(a0, b6, c60);
        c70 = _mm256_fmadd_ps(a0, b7, c70);
        c01 = _mm256_fmadd_ps(a1, b0, c01);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c21 = _mm256_fmadd_ps(a1, b2, c21);
        c31 = _mm256_fmadd_ps(a1, b3, c31);
        c41 = _mm256_fmadd_ps(a1, b4, c41);
        c51 = _mm256_fmadd_ps(a1, b5, c51);
        c61 = _mm256_fmadd_ps(a1, b6, c61);
        c71 = _mm256_fmadd_ps(a1, b7, c71);
        packing_A += 16; packing_B += 8;
    }
    _m256_y_add_to_x(C + index(i, j), c00);
    _m256_y_add_to_x(C + index(i + 8, j), c01);
    _m256_y_add_to_x(C + index(i, j + 1), c10);
    _m256_y_add_to_x(C + index(i + 8, j + 1), c11);
    _m256_y_add_to_x(C + index(i, j + 2), c20);
    _m256_y_add_to_x(C + index(i + 8, j + 2), c21);
    _m256_y_add_to_x(C + index(i, j + 3), c30);
    _m256_y_add_to_x(C + index(i + 8, j + 3), c31);
    _m256_y_add_to_x(C + index(i, j + 4), c40);
    _m256_y_add_to_x(C + index(i + 8, j + 4), c41);
    _m256_y_add_to_x(C + index(i, j + 5), c50);
    _m256_y_add_to_x(C + index(i + 8, j + 5), c51);
    _m256_y_add_to_x(C + index(i, j + 6), c60);
    _m256_y_add_to_x(C + index(i + 8, j + 6), c61);
    _m256_y_add_to_x(C + index(i, j + 7), c70);
    _m256_y_add_to_x(C + index(i + 8, j + 7), c71);
}

static inline void kernel_16xKx8_512_packing(int lda, int i, int j, int K, float* packing_A, float* packing_B, float* C) {
    __m512 c_0 = _mm512_setzero_ps();
    __m512 c_1 = _mm512_setzero_ps();
    __m512 c_2 = _mm512_setzero_ps();
    __m512 c_3 = _mm512_setzero_ps();
    __m512 c_4 = _mm512_setzero_ps();
    __m512 c_5 = _mm512_setzero_ps();
    __m512 c_6 = _mm512_setzero_ps();
    __m512 c_7 = _mm512_setzero_ps();
    prefetch_T0(C + index(i, j));
    prefetch_T0(C + index(i, j + 1));
    prefetch_T0(C + index(i, j + 2));
    prefetch_T0(C + index(i, j + 3));
    prefetch_T0(C + index(i, j + 4));
    prefetch_T0(C + index(i, j + 5));
    prefetch_T0(C + index(i, j + 6));
    prefetch_T0(C + index(i, j + 7));
    __m512 a_0;
    __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7;
    int k = 0;
    for (k = 0; k < K; k++) {
        a_0 = _mm512_loadu_ps(packing_A);
        b_0 = _mm512_set1_ps(*packing_B);
        b_1 = _mm512_set1_ps(*(packing_B + 1));
        b_2 = _mm512_set1_ps(*(packing_B + 2));
        b_3 = _mm512_set1_ps(*(packing_B + 3));
        b_4 = _mm512_set1_ps(*(packing_B + 4));
        b_5 = _mm512_set1_ps(*(packing_B + 5));
        b_6 = _mm512_set1_ps(*(packing_B + 6));
        b_7 = _mm512_set1_ps(*(packing_B + 7));
        c_0 = _mm512_fmadd_ps(a_0, b_0, c_0);
        c_1 = _mm512_fmadd_ps(a_0, b_1, c_1);
        c_2 = _mm512_fmadd_ps(a_0, b_2, c_2);
        c_3 = _mm512_fmadd_ps(a_0, b_3, c_3);
        c_4 = _mm512_fmadd_ps(a_0, b_4, c_4);
        c_5 = _mm512_fmadd_ps(a_0, b_5, c_5);
        c_6 = _mm512_fmadd_ps(a_0, b_6, c_6);
        c_7 = _mm512_fmadd_ps(a_0, b_7, c_7);
        packing_A += 16; packing_B += 8;
    }
    _m512_y_add_to_x(C + index(i, j), c_0);
    _m512_y_add_to_x(C + index(i, j + 1), c_1);
    _m512_y_add_to_x(C + index(i, j + 2), c_2);
    _m512_y_add_to_x(C + index(i, j + 3), c_3);
    _m512_y_add_to_x(C + index(i, j + 4), c_4);
    _m512_y_add_to_x(C + index(i, j + 5), c_5);
    _m512_y_add_to_x(C + index(i, j + 6), c_6);
    _m512_y_add_to_x(C + index(i, j + 7), c_7);
}

static inline void kernel_48xKx8_512(int lda, int i, int j, int K, float* A, float* B, float* C) {
    __m512 c_00 = _mm512_setzero_ps();
    __m512 c_10 = _mm512_setzero_ps();
    __m512 c_20 = _mm512_setzero_ps();
    __m512 c_30 = _mm512_setzero_ps();
    __m512 c_40 = _mm512_setzero_ps();
    __m512 c_50 = _mm512_setzero_ps();
    __m512 c_60 = _mm512_setzero_ps();
    __m512 c_70 = _mm512_setzero_ps();
    __m512 c_01 = _mm512_setzero_ps();
    __m512 c_11 = _mm512_setzero_ps();
    __m512 c_21 = _mm512_setzero_ps();
    __m512 c_31 = _mm512_setzero_ps();
    __m512 c_41 = _mm512_setzero_ps();
    __m512 c_51 = _mm512_setzero_ps();
    __m512 c_61 = _mm512_setzero_ps();
    __m512 c_71 = _mm512_setzero_ps();
    __m512 c_02 = _mm512_setzero_ps();
    __m512 c_12 = _mm512_setzero_ps();
    __m512 c_22 = _mm512_setzero_ps();
    __m512 c_32 = _mm512_setzero_ps();
    __m512 c_42 = _mm512_setzero_ps();
    __m512 c_52 = _mm512_setzero_ps();
    __m512 c_62 = _mm512_setzero_ps();
    __m512 c_72 = _mm512_setzero_ps();
    prefetch_T0(C + index(i, j));
    prefetch_T0(C + index(i, j + 1));
    prefetch_T0(C + index(i, j + 2));
    prefetch_T0(C + index(i, j + 3));
    prefetch_T0(C + index(i, j + 4));
    prefetch_T0(C + index(i, j + 5));
    prefetch_T0(C + index(i, j + 6));
    prefetch_T0(C + index(i, j + 7));
    __m512 a_0, a_1, a_2;
    // __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9, b_10, b_11, b_12, b_13, b_14, b_15;
    __m512 b_0, b_1;
    int k = 0;
    for (k = 0; k < K; k++) {
        a_0 = _mm512_loadu_ps(A + index(i, k));
        a_1 = _mm512_loadu_ps(A + index(i + 16, k));
        a_2 = _mm512_loadu_ps(A + index(i + 32, k));
        b_0 = _mm512_set1_ps(B[index(k, j)]);
        b_1 = _mm512_set1_ps(B[index(k, j + 1)]);
        c_00 = _mm512_fmadd_ps(a_0, b_0, c_00);
        c_01 = _mm512_fmadd_ps(a_1, b_0, c_01);
        c_02 = _mm512_fmadd_ps(a_2, b_0, c_02);
        c_10 = _mm512_fmadd_ps(a_0, b_1, c_10);
        c_11 = _mm512_fmadd_ps(a_1, b_1, c_11);
        c_12 = _mm512_fmadd_ps(a_2, b_1, c_12);
        b_0 = _mm512_set1_ps(B[index(k, j + 2)]);
        b_1 = _mm512_set1_ps(B[index(k, j + 3)]);
        c_20 = _mm512_fmadd_ps(a_0, b_0, c_20);
        c_21 = _mm512_fmadd_ps(a_1, b_0, c_21);
        c_22 = _mm512_fmadd_ps(a_2, b_0, c_22);
        c_30 = _mm512_fmadd_ps(a_0, b_1, c_30);
        c_31 = _mm512_fmadd_ps(a_1, b_1, c_31);
        c_32 = _mm512_fmadd_ps(a_2, b_1, c_32);
        b_0 = _mm512_set1_ps(B[index(k, j + 4)]);
        b_1 = _mm512_set1_ps(B[index(k, j + 5)]);
        c_40 = _mm512_fmadd_ps(a_0, b_0, c_40);
        c_41 = _mm512_fmadd_ps(a_1, b_0, c_41);
        c_42 = _mm512_fmadd_ps(a_2, b_0, c_42);
        c_50 = _mm512_fmadd_ps(a_0, b_1, c_50);
        c_51 = _mm512_fmadd_ps(a_1, b_1, c_51);
        c_52 = _mm512_fmadd_ps(a_2, b_1, c_52);
        b_0 = _mm512_set1_ps(B[index(k, j + 6)]);
        b_1 = _mm512_set1_ps(B[index(k, j + 7)]);
        c_60 = _mm512_fmadd_ps(a_0, b_0, c_60);
        c_70 = _mm512_fmadd_ps(a_0, b_1, c_70);
        c_61 = _mm512_fmadd_ps(a_1, b_0, c_61);
        c_71 = _mm512_fmadd_ps(a_1, b_1, c_71);
        c_62 = _mm512_fmadd_ps(a_2, b_0, c_62);
        c_72 = _mm512_fmadd_ps(a_2, b_1, c_72);
    }
    _m512_y_add_to_x(C + index(i, j), c_00);
    _m512_y_add_to_x(C + index(i + 16, j), c_01);
    _m512_y_add_to_x(C + index(i + 32, j), c_02);
    _m512_y_add_to_x(C + index(i, j + 1), c_10);
    _m512_y_add_to_x(C + index(i + 16, j + 1), c_11);
    _m512_y_add_to_x(C + index(i + 32, j + 1), c_12);
    _m512_y_add_to_x(C + index(i, j + 2), c_20);
    _m512_y_add_to_x(C + index(i + 16, j + 2), c_21);
    _m512_y_add_to_x(C + index(i + 32, j + 2), c_22);
    _m512_y_add_to_x(C + index(i, j + 3), c_30);
    _m512_y_add_to_x(C + index(i + 16, j + 3), c_31);
    _m512_y_add_to_x(C + index(i + 32, j + 3), c_32);
    _m512_y_add_to_x(C + index(i, j + 4), c_40);
    _m512_y_add_to_x(C + index(i + 16, j + 4), c_41);
    _m512_y_add_to_x(C + index(i + 32, j + 4), c_42);
    _m512_y_add_to_x(C + index(i, j + 5), c_50);
    _m512_y_add_to_x(C + index(i + 16, j + 5), c_51);
    _m512_y_add_to_x(C + index(i + 32, j + 5), c_52);
    _m512_y_add_to_x(C + index(i, j + 6), c_60);
    _m512_y_add_to_x(C + index(i + 16, j + 6), c_61);
    _m512_y_add_to_x(C + index(i + 32, j + 6), c_62);
    _m512_y_add_to_x(C + index(i, j + 7), c_70);
    _m512_y_add_to_x(C + index(i + 16, j + 7), c_71);
    _m512_y_add_to_x(C + index(i + 32, j + 7), c_72);
}

static inline void kernel_32xKx8_512(int lda, int i, int j, int K, float* A, float* B, float* C) {
    __m512 c_00 = _mm512_setzero_ps();
    __m512 c_10 = _mm512_setzero_ps();
    __m512 c_20 = _mm512_setzero_ps();
    __m512 c_30 = _mm512_setzero_ps();
    __m512 c_40 = _mm512_setzero_ps();
    __m512 c_50 = _mm512_setzero_ps();
    __m512 c_60 = _mm512_setzero_ps();
    __m512 c_70 = _mm512_setzero_ps();
    __m512 c_01 = _mm512_setzero_ps();
    __m512 c_11 = _mm512_setzero_ps();
    __m512 c_21 = _mm512_setzero_ps();
    __m512 c_31 = _mm512_setzero_ps();
    __m512 c_41 = _mm512_setzero_ps();
    __m512 c_51 = _mm512_setzero_ps();
    __m512 c_61 = _mm512_setzero_ps();
    __m512 c_71 = _mm512_setzero_ps();
    prefetch_T0(C + index(i, j));
    prefetch_T0(C + index(i, j + 1));
    prefetch_T0(C + index(i, j + 2));
    prefetch_T0(C + index(i, j + 3));
    prefetch_T0(C + index(i, j + 4));
    prefetch_T0(C + index(i, j + 5));
    prefetch_T0(C + index(i, j + 6));
    prefetch_T0(C + index(i, j + 7));
    __m512 a_0, a_1;
    // __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9, b_10, b_11, b_12, b_13, b_14, b_15;
    __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7;
    int k = 0;
    for (k = 0; k < K; k++) {
        a_0 = _mm512_loadu_ps(A + index(i, k));
        a_1 = _mm512_loadu_ps(A + index(i + 16, k));
        b_0 = _mm512_set1_ps(B[index(k, j)]);
        b_1 = _mm512_set1_ps(B[index(k, j + 1)]);
        b_2 = _mm512_set1_ps(B[index(k, j + 2)]);
        b_3 = _mm512_set1_ps(B[index(k, j + 3)]);
        b_4 = _mm512_set1_ps(B[index(k, j + 4)]);
        b_5 = _mm512_set1_ps(B[index(k, j + 5)]);
        b_6 = _mm512_set1_ps(B[index(k, j + 6)]);
        b_7 = _mm512_set1_ps(B[index(k, j + 7)]);
        c_00 = _mm512_fmadd_ps(a_0, b_0, c_00);
        c_10 = _mm512_fmadd_ps(a_0, b_1, c_10);
        c_20 = _mm512_fmadd_ps(a_0, b_2, c_20);
        c_30 = _mm512_fmadd_ps(a_0, b_3, c_30);
        c_40 = _mm512_fmadd_ps(a_0, b_4, c_40);
        c_50 = _mm512_fmadd_ps(a_0, b_5, c_50);
        c_60 = _mm512_fmadd_ps(a_0, b_6, c_60);
        c_70 = _mm512_fmadd_ps(a_0, b_7, c_70);
        c_01 = _mm512_fmadd_ps(a_1, b_0, c_01);
        c_11 = _mm512_fmadd_ps(a_1, b_1, c_11);
        c_21 = _mm512_fmadd_ps(a_1, b_2, c_21);
        c_31 = _mm512_fmadd_ps(a_1, b_3, c_31);
        c_41 = _mm512_fmadd_ps(a_1, b_4, c_41);
        c_51 = _mm512_fmadd_ps(a_1, b_5, c_51);
        c_61 = _mm512_fmadd_ps(a_1, b_6, c_61);
        c_71 = _mm512_fmadd_ps(a_1, b_7, c_71);
    }
    _m512_y_add_to_x(C + index(i, j), c_00);
    _m512_y_add_to_x(C + index(i + 16, j), c_01);
    _m512_y_add_to_x(C + index(i, j + 1), c_10);
    _m512_y_add_to_x(C + index(i + 16, j + 1), c_11);
    _m512_y_add_to_x(C + index(i, j + 2), c_20);
    _m512_y_add_to_x(C + index(i + 16, j + 2), c_21);
    _m512_y_add_to_x(C + index(i, j + 3), c_30);
    _m512_y_add_to_x(C + index(i + 16, j + 3), c_31);
    _m512_y_add_to_x(C + index(i, j + 4), c_40);
    _m512_y_add_to_x(C + index(i + 16, j + 4), c_41);
    _m512_y_add_to_x(C + index(i, j + 5), c_50);
    _m512_y_add_to_x(C + index(i + 16, j + 5), c_51);
    _m512_y_add_to_x(C + index(i, j + 6), c_60);
    _m512_y_add_to_x(C + index(i + 16, j + 6), c_61);
    _m512_y_add_to_x(C + index(i, j + 7), c_70);
    _m512_y_add_to_x(C + index(i + 16, j + 7), c_71);
}

static inline void kernel_32xKx7_512_packing(int lda, int K, float* packing_A, float* packing_B, float* C) {
    __m512 c_00 = _mm512_setzero_ps();
    __m512 c_10 = _mm512_setzero_ps();
    __m512 c_20 = _mm512_setzero_ps();
    __m512 c_30 = _mm512_setzero_ps();
    __m512 c_40 = _mm512_setzero_ps();
    __m512 c_50 = _mm512_setzero_ps();
    __m512 c_60 = _mm512_setzero_ps();
    __m512 c_01 = _mm512_setzero_ps();
    __m512 c_11 = _mm512_setzero_ps();
    __m512 c_21 = _mm512_setzero_ps();
    __m512 c_31 = _mm512_setzero_ps();
    __m512 c_41 = _mm512_setzero_ps();
    __m512 c_51 = _mm512_setzero_ps();
    __m512 c_61 = _mm512_setzero_ps();
    prefetch_T0(C);
    prefetch_T0(C + index(0, 1));
    prefetch_T0(C + index(0, 2));
    prefetch_T0(C + index(0, 3));
    prefetch_T0(C + index(0, 4));
    prefetch_T0(C + index(0, 5));
    prefetch_T0(C + index(0, 6));
    __m512 a_0, a_1;
    // __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9, b_10, b_11, b_12, b_13, b_14, b_15;
    __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6;
    int k = 0;
    for (k = 0; k < K; k++) {
        a_0 = _mm512_loadu_ps(packing_A + k * 32);
        a_1 = _mm512_loadu_ps(packing_A + k * 32 + 16);
        b_0 = _mm512_set1_ps(*(packing_B + 0));
        c_00 = _mm512_fmadd_ps(a_0, b_0, c_00);
        c_01 = _mm512_fmadd_ps(a_1, b_0, c_01);
        b_1 = _mm512_set1_ps(*(packing_B + 1));
        c_10 = _mm512_fmadd_ps(a_0, b_1, c_10);
        c_11 = _mm512_fmadd_ps(a_1, b_1, c_11);
        b_2 = _mm512_set1_ps(*(packing_B + 2));
        c_20 = _mm512_fmadd_ps(a_0, b_2, c_20);
        c_21 = _mm512_fmadd_ps(a_1, b_2, c_21);
        b_3 = _mm512_set1_ps(*(packing_B + 3));
        c_30 = _mm512_fmadd_ps(a_0, b_3, c_30);
        c_31 = _mm512_fmadd_ps(a_1, b_3, c_31);
        b_4 = _mm512_set1_ps(*(packing_B + 4));
        c_40 = _mm512_fmadd_ps(a_0, b_4, c_40);
        c_41 = _mm512_fmadd_ps(a_1, b_4, c_41);
        b_5 = _mm512_set1_ps(*(packing_B + 5));
        c_50 = _mm512_fmadd_ps(a_0, b_5, c_50);
        c_51 = _mm512_fmadd_ps(a_1, b_5, c_51);
        b_6 = _mm512_set1_ps(*(packing_B + 6));
        c_60 = _mm512_fmadd_ps(a_0, b_6, c_60);
        c_61 = _mm512_fmadd_ps(a_1, b_6, c_61);
        packing_B += 7;
    }
    _m512_y_add_to_x(C, c_00);
    _m512_y_add_to_x(C + 16, c_01);
    _m512_y_add_to_x(C + index(0, 1), c_10);
    _m512_y_add_to_x(C + index(0, 1) + 16, c_11);
    _m512_y_add_to_x(C + index(0, 2), c_20);
    _m512_y_add_to_x(C + index(0, 2) + 16, c_21);
    _m512_y_add_to_x(C + index(0, 3), c_30);
    _m512_y_add_to_x(C + index(0, 3) + 16, c_31);
    _m512_y_add_to_x(C + index(0, 4), c_40);
    _m512_y_add_to_x(C + index(0, 4) + 16, c_41);
    _m512_y_add_to_x(C + index(0, 5), c_50);
    _m512_y_add_to_x(C + index(0, 5) + 16, c_51);
    _m512_y_add_to_x(C + index(0, 6), c_60);
    _m512_y_add_to_x(C + index(0, 6) + 16, c_61);
}

static inline void kernel_32xKx1_512_packing(int lda, int K, float* packing_A, float* packing_B, float* C) {
    __m512 c_00 = _mm512_setzero_ps();
    __m512 c_01 = _mm512_setzero_ps();
    prefetch_T0(C);
    __m512 a_0, a_1;
    // __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9, b_10, b_11, b_12, b_13, b_14, b_15;
    __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6;
    int k = 0;
    for (k = 0; k < K; k++) {
        a_0 = _mm512_loadu_ps(packing_A + k * 32);
        a_1 = _mm512_loadu_ps(packing_A + k * 32 + 16);
        b_0 = _mm512_set1_ps(*(packing_B + 0));
        c_00 = _mm512_fmadd_ps(a_0, b_0, c_00);
        c_01 = _mm512_fmadd_ps(a_1, b_0, c_01);
        packing_B++;
    }
    _m512_y_add_to_x(C, c_00);
    _m512_y_add_to_x(C + 16, c_01);
}


static inline void kernel_31xKx8_512_packing(int lda, int K, float* packing_A, float* packing_B, float* C) {
    __m512 c_00 = _mm512_setzero_ps();
    __m512 c_10 = _mm512_setzero_ps();
    __m512 c_20 = _mm512_setzero_ps();
    __m512 c_30 = _mm512_setzero_ps();
    __m512 c_40 = _mm512_setzero_ps();
    __m512 c_50 = _mm512_setzero_ps();
    __m512 c_60 = _mm512_setzero_ps();
    __m512 c_70 = _mm512_setzero_ps();
    __m512 c_01 = _mm512_setzero_ps();
    __m512 c_11 = _mm512_setzero_ps();
    __m512 c_21 = _mm512_setzero_ps();
    __m512 c_31 = _mm512_setzero_ps();
    __m512 c_41 = _mm512_setzero_ps();
    __m512 c_51 = _mm512_setzero_ps();
    __m512 c_61 = _mm512_setzero_ps();
    __m512 c_71 = _mm512_setzero_ps();
    prefetch_T0(C);
    prefetch_T0(C + index(0, 1));
    prefetch_T0(C + index(0, 2));
    prefetch_T0(C + index(0, 3));
    prefetch_T0(C + index(0, 4));
    prefetch_T0(C + index(0, 5));
    prefetch_T0(C + index(0, 6));
    prefetch_T0(C + index(0, 7));
    __m512 a_0, a_1;
    // __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9, b_10, b_11, b_12, b_13, b_14, b_15;
    __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7;
    int k = 0;
    for (k = 0; k < K; k++) {
        a_0 = _mm512_loadu_ps(packing_A + k * 32);
        a_1 = _mm512_loadu_ps(packing_A + k * 32 + 16);
        b_0 = _mm512_set1_ps(*(packing_B + 0));
        c_00 = _mm512_fmadd_ps(a_0, b_0, c_00);
        c_01 = _mm512_fmadd_ps(a_1, b_0, c_01);
        b_1 = _mm512_set1_ps(*(packing_B + 1));
        c_10 = _mm512_fmadd_ps(a_0, b_1, c_10);
        c_11 = _mm512_fmadd_ps(a_1, b_1, c_11);
        b_2 = _mm512_set1_ps(*(packing_B + 2));
        c_20 = _mm512_fmadd_ps(a_0, b_2, c_20);
        c_21 = _mm512_fmadd_ps(a_1, b_2, c_21);
        b_3 = _mm512_set1_ps(*(packing_B + 3));
        c_30 = _mm512_fmadd_ps(a_0, b_3, c_30);
        c_31 = _mm512_fmadd_ps(a_1, b_3, c_31);
        b_4 = _mm512_set1_ps(*(packing_B + 4));
        c_40 = _mm512_fmadd_ps(a_0, b_4, c_40);
        c_41 = _mm512_fmadd_ps(a_1, b_4, c_41);
        b_5 = _mm512_set1_ps(*(packing_B + 5));
        c_50 = _mm512_fmadd_ps(a_0, b_5, c_50);
        c_51 = _mm512_fmadd_ps(a_1, b_5, c_51);
        b_6 = _mm512_set1_ps(*(packing_B + 6));
        c_60 = _mm512_fmadd_ps(a_0, b_6, c_60);
        c_61 = _mm512_fmadd_ps(a_1, b_6, c_61);
        b_7 = _mm512_set1_ps(*(packing_B + 7));
        c_70 = _mm512_fmadd_ps(a_0, b_7, c_70);
        c_71 = _mm512_fmadd_ps(a_1, b_7, c_71);
        packing_B += 8;
    }
    _m512_y_add_to_x(C, c_00);
    _m512_y_add_to_x_15(C + 16, c_01);
    _m512_y_add_to_x(C + index(0, 1), c_10);
    _m512_y_add_to_x_15(C + index(0, 1) + 16, c_11);
    _m512_y_add_to_x(C + index(0, 2), c_20);
    _m512_y_add_to_x_15(C + index(0, 2) + 16, c_21);
    _m512_y_add_to_x(C + index(0, 3), c_30);
    _m512_y_add_to_x_15(C + index(0, 3) + 16, c_31);
    _m512_y_add_to_x(C + index(0, 4), c_40);
    _m512_y_add_to_x_15(C + index(0, 4) + 16, c_41);
    _m512_y_add_to_x(C + index(0, 5), c_50);
    _m512_y_add_to_x_15(C + index(0, 5) + 16, c_51);
    _m512_y_add_to_x(C + index(0, 6), c_60);
    _m512_y_add_to_x_15(C + index(0, 6) + 16, c_61);
    _m512_y_add_to_x(C + index(0, 7), c_70);
    _m512_y_add_to_x_15(C + index(0, 7) + 16, c_71);
}

static inline void kernel_31xKx7_512_packing(int lda, int K, float* packing_A, float* packing_B, float* C) {
    __m512 c_00 = _mm512_setzero_ps();
    __m512 c_10 = _mm512_setzero_ps();
    __m512 c_20 = _mm512_setzero_ps();
    __m512 c_30 = _mm512_setzero_ps();
    __m512 c_40 = _mm512_setzero_ps();
    __m512 c_50 = _mm512_setzero_ps();
    __m512 c_60 = _mm512_setzero_ps();
    __m512 c_01 = _mm512_setzero_ps();
    __m512 c_11 = _mm512_setzero_ps();
    __m512 c_21 = _mm512_setzero_ps();
    __m512 c_31 = _mm512_setzero_ps();
    __m512 c_41 = _mm512_setzero_ps();
    __m512 c_51 = _mm512_setzero_ps();
    __m512 c_61 = _mm512_setzero_ps();
    prefetch_T0(C);
    prefetch_T0(C + index(0, 1));
    prefetch_T0(C + index(0, 2));
    prefetch_T0(C + index(0, 3));
    prefetch_T0(C + index(0, 4));
    prefetch_T0(C + index(0, 5));
    prefetch_T0(C + index(0, 6));
    __m512 a_0, a_1;
    // __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9, b_10, b_11, b_12, b_13, b_14, b_15;
    __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6;
    int k = 0;
    for (k = 0; k < K; k++) {
        a_0 = _mm512_loadu_ps(packing_A + k * 32);
        a_1 = _mm512_loadu_ps(packing_A + k * 32 + 16);
        b_0 = _mm512_set1_ps(*(packing_B + 0));
        c_00 = _mm512_fmadd_ps(a_0, b_0, c_00);
        c_01 = _mm512_fmadd_ps(a_1, b_0, c_01);
        b_1 = _mm512_set1_ps(*(packing_B + 1));
        c_10 = _mm512_fmadd_ps(a_0, b_1, c_10);
        c_11 = _mm512_fmadd_ps(a_1, b_1, c_11);
        b_2 = _mm512_set1_ps(*(packing_B + 2));
        c_20 = _mm512_fmadd_ps(a_0, b_2, c_20);
        c_21 = _mm512_fmadd_ps(a_1, b_2, c_21);
        b_3 = _mm512_set1_ps(*(packing_B + 3));
        c_30 = _mm512_fmadd_ps(a_0, b_3, c_30);
        c_31 = _mm512_fmadd_ps(a_1, b_3, c_31);
        b_4 = _mm512_set1_ps(*(packing_B + 4));
        c_40 = _mm512_fmadd_ps(a_0, b_4, c_40);
        c_41 = _mm512_fmadd_ps(a_1, b_4, c_41);
        b_5 = _mm512_set1_ps(*(packing_B + 5));
        c_50 = _mm512_fmadd_ps(a_0, b_5, c_50);
        c_51 = _mm512_fmadd_ps(a_1, b_5, c_51);
        b_6 = _mm512_set1_ps(*(packing_B + 6));
        c_60 = _mm512_fmadd_ps(a_0, b_6, c_60);
        c_61 = _mm512_fmadd_ps(a_1, b_6, c_61);
        packing_B += 7;
    }
    _m512_y_add_to_x(C, c_00);
    _m512_y_add_to_x_15(C + 16, c_01);
    _m512_y_add_to_x(C + index(0, 1), c_10);
    _m512_y_add_to_x_15(C + index(0, 1) + 16, c_11);
    _m512_y_add_to_x(C + index(0, 2), c_20);
    _m512_y_add_to_x_15(C + index(0, 2) + 16, c_21);
    _m512_y_add_to_x(C + index(0, 3), c_30);
    _m512_y_add_to_x_15(C + index(0, 3) + 16, c_31);
    _m512_y_add_to_x(C + index(0, 4), c_40);
    _m512_y_add_to_x_15(C + index(0, 4) + 16, c_41);
    _m512_y_add_to_x(C + index(0, 5), c_50);
    _m512_y_add_to_x_15(C + index(0, 5) + 16, c_51);
    _m512_y_add_to_x(C + index(0, 6), c_60);
    _m512_y_add_to_x_15(C + index(0, 6) + 16, c_61);
}

static inline void kernel_31xKx1_512_packing(int lda, int K, float* packing_A, float* packing_B, float* C) {
    __m512 c_00 = _mm512_setzero_ps();
    __m512 c_01 = _mm512_setzero_ps();
    prefetch_T0(C);
    __m512 a_0, a_1;
    // __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9, b_10, b_11, b_12, b_13, b_14, b_15;
    __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6;
    int k = 0;
    for (k = 0; k < K; k++) {
        a_0 = _mm512_loadu_ps(packing_A + k * 32);
        a_1 = _mm512_loadu_ps(packing_A + k * 32 + 16);
        b_0 = _mm512_set1_ps(*(packing_B + 0));
        c_00 = _mm512_fmadd_ps(a_0, b_0, c_00);
        c_01 = _mm512_fmadd_ps(a_1, b_0, c_01);
        packing_B++;
    }
    _m512_y_add_to_x(C, c_00);
    _m512_y_add_to_x_15(C + 16, c_01);
}

static inline void kernel_32xKx8_512_packing(int lda, int K, float* packing_A, float* packing_B, float* C) {
    __m512 c_00 = _mm512_setzero_ps();
    __m512 c_10 = _mm512_setzero_ps();
    __m512 c_20 = _mm512_setzero_ps();
    __m512 c_30 = _mm512_setzero_ps();
    __m512 c_40 = _mm512_setzero_ps();
    __m512 c_50 = _mm512_setzero_ps();
    __m512 c_60 = _mm512_setzero_ps();
    __m512 c_70 = _mm512_setzero_ps();
    __m512 c_01 = _mm512_setzero_ps();
    __m512 c_11 = _mm512_setzero_ps();
    __m512 c_21 = _mm512_setzero_ps();
    __m512 c_31 = _mm512_setzero_ps();
    __m512 c_41 = _mm512_setzero_ps();
    __m512 c_51 = _mm512_setzero_ps();
    __m512 c_61 = _mm512_setzero_ps();
    __m512 c_71 = _mm512_setzero_ps();
    prefetch_T0(C);
    prefetch_T0(C + index(0, 1));
    prefetch_T0(C + index(0, 2));
    prefetch_T0(C + index(0, 3));
    prefetch_T0(C + index(0, 4));
    prefetch_T0(C + index(0, 5));
    prefetch_T0(C + index(0, 6));
    prefetch_T0(C + index(0, 7));
    __m512 a_0, a_1;
    // __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9, b_10, b_11, b_12, b_13, b_14, b_15;
    __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7;
    int k = 0;
    for (k = 0; k < K; k++) {
        a_0 = _mm512_loadu_ps(packing_A + k * 32);
        a_1 = _mm512_loadu_ps(packing_A + k * 32 + 16);
        b_0 = _mm512_set1_ps(*(packing_B + 0));
        c_00 = _mm512_fmadd_ps(a_0, b_0, c_00);
        c_01 = _mm512_fmadd_ps(a_1, b_0, c_01);
        b_1 = _mm512_set1_ps(*(packing_B + 1));
        c_10 = _mm512_fmadd_ps(a_0, b_1, c_10);
        c_11 = _mm512_fmadd_ps(a_1, b_1, c_11);
        b_2 = _mm512_set1_ps(*(packing_B + 2));
        c_20 = _mm512_fmadd_ps(a_0, b_2, c_20);
        c_21 = _mm512_fmadd_ps(a_1, b_2, c_21);
        b_3 = _mm512_set1_ps(*(packing_B + 3));
        c_30 = _mm512_fmadd_ps(a_0, b_3, c_30);
        c_31 = _mm512_fmadd_ps(a_1, b_3, c_31);
        b_4 = _mm512_set1_ps(*(packing_B + 4));
        c_40 = _mm512_fmadd_ps(a_0, b_4, c_40);
        c_41 = _mm512_fmadd_ps(a_1, b_4, c_41);
        b_5 = _mm512_set1_ps(*(packing_B + 5));
        c_50 = _mm512_fmadd_ps(a_0, b_5, c_50);
        c_51 = _mm512_fmadd_ps(a_1, b_5, c_51);
        b_6 = _mm512_set1_ps(*(packing_B + 6));
        c_60 = _mm512_fmadd_ps(a_0, b_6, c_60);
        c_61 = _mm512_fmadd_ps(a_1, b_6, c_61);
        b_7 = _mm512_set1_ps(*(packing_B + 7));
        c_70 = _mm512_fmadd_ps(a_0, b_7, c_70);
        c_71 = _mm512_fmadd_ps(a_1, b_7, c_71);
        packing_B += 8;
    }
    _m512_y_add_to_x(C, c_00);
    _m512_y_add_to_x(C + 16, c_01);
    _m512_y_add_to_x(C + index(0, 1), c_10);
    _m512_y_add_to_x(C + index(0, 1) + 16, c_11);
    _m512_y_add_to_x(C + index(0, 2), c_20);
    _m512_y_add_to_x(C + index(0, 2) + 16, c_21);
    _m512_y_add_to_x(C + index(0, 3), c_30);
    _m512_y_add_to_x(C + index(0, 3) + 16, c_31);
    _m512_y_add_to_x(C + index(0, 4), c_40);
    _m512_y_add_to_x(C + index(0, 4) + 16, c_41);
    _m512_y_add_to_x(C + index(0, 5), c_50);
    _m512_y_add_to_x(C + index(0, 5) + 16, c_51);
    _m512_y_add_to_x(C + index(0, 6), c_60);
    _m512_y_add_to_x(C + index(0, 6) + 16, c_61);
    _m512_y_add_to_x(C + index(0, 7), c_70);
    _m512_y_add_to_x(C + index(0, 7) + 16, c_71);
}

static inline void kernel_16xKx8_512(int lda, int i, int j, int K, float* A, float* B, float* C) {
    __m512 c_0 = _mm512_setzero_ps();
    __m512 c_1 = _mm512_setzero_ps();
    __m512 c_2 = _mm512_setzero_ps();
    __m512 c_3 = _mm512_setzero_ps();
    __m512 c_4 = _mm512_setzero_ps();
    __m512 c_5 = _mm512_setzero_ps();
    __m512 c_6 = _mm512_setzero_ps();
    __m512 c_7 = _mm512_setzero_ps();
    // __m512 c_8 = _mm512_setzero_ps();
    // __m512 c_9 = _mm256_setzero_ps();
    // __m512 c_10 = _mm256_setzero_ps();
    // __m512 c_11 = _mm256_setzero_ps();
    // __m512 c_12 = _mm256_setzero_ps();
    // __m512 c_13 = _mm256_setzero_ps();
    // __m512 c_14 = _mm256_setzero_ps();
    // __m512 c_15 = _mm256_setzero_ps();
    __m512 a_0;
    // __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9, b_10, b_11, b_12, b_13, b_14, b_15;
    __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7;
    int k = 0;
    for (k = 0; k < K; k++) {
        a_0 = _mm512_loadu_ps(A + index(i, k));
        b_0 = _mm512_set1_ps(B[index(k, j)]);
        b_1 = _mm512_set1_ps(B[index(k, j + 1)]);
        b_2 = _mm512_set1_ps(B[index(k, j + 2)]);
        b_3 = _mm512_set1_ps(B[index(k, j + 3)]);
        b_4 = _mm512_set1_ps(B[index(k, j + 4)]);
        b_5 = _mm512_set1_ps(B[index(k, j + 5)]);
        b_6 = _mm512_set1_ps(B[index(k, j + 6)]);
        b_7 = _mm512_set1_ps(B[index(k, j + 7)]);
        c_0 = _mm512_fmadd_ps(a_0, b_0, c_0);
        c_1 = _mm512_fmadd_ps(a_0, b_1, c_1);
        c_2 = _mm512_fmadd_ps(a_0, b_2, c_2);
        c_3 = _mm512_fmadd_ps(a_0, b_3, c_3);
        c_4 = _mm512_fmadd_ps(a_0, b_4, c_4);
        c_5 = _mm512_fmadd_ps(a_0, b_5, c_5);
        c_6 = _mm512_fmadd_ps(a_0, b_6, c_6);
        c_7 = _mm512_fmadd_ps(a_0, b_7, c_7);
    }
    _m512_y_add_to_x(C + index(i, j), c_0);
    _m512_y_add_to_x(C + index(i, j + 1), c_1);
    _m512_y_add_to_x(C + index(i, j + 2), c_2);
    _m512_y_add_to_x(C + index(i, j + 3), c_3);
    _m512_y_add_to_x(C + index(i, j + 4), c_4);
    _m512_y_add_to_x(C + index(i, j + 5), c_5);
    _m512_y_add_to_x(C + index(i, j + 6), c_6);
    _m512_y_add_to_x(C + index(i, j + 7), c_7);
}

static inline void kernel_16xKx32_512(int lda, int i, int j, int K, float* A, float* B, float* C) {
    __m512 c_0 = _mm512_setzero_ps();
    __m512 c_1 = _mm512_setzero_ps();
    __m512 c_2 = _mm512_setzero_ps();
    __m512 c_3 = _mm512_setzero_ps();
    __m512 c_4 = _mm512_setzero_ps();
    __m512 c_5 = _mm512_setzero_ps();
    __m512 c_6 = _mm512_setzero_ps();
    __m512 c_7 = _mm512_setzero_ps();
    __m512 c_8 = _mm512_setzero_ps();
    __m512 c_9 = _mm512_setzero_ps();
    __m512 c_10 = _mm512_setzero_ps();
    __m512 c_11 = _mm512_setzero_ps();
    __m512 c_12 = _mm512_setzero_ps();
    __m512 c_13 = _mm512_setzero_ps();
    __m512 c_14 = _mm512_setzero_ps();
    __m512 c_15 = _mm512_setzero_ps();
    __m512 c_16 = _mm512_setzero_ps();
    __m512 c_17 = _mm512_setzero_ps();
    __m512 c_18 = _mm512_setzero_ps();
    __m512 c_19 = _mm512_setzero_ps();
    __m512 c_20 = _mm512_setzero_ps();
    __m512 c_21 = _mm512_setzero_ps();
    __m512 c_22 = _mm512_setzero_ps();
    __m512 c_23 = _mm512_setzero_ps();
    __m512 c_24 = _mm512_setzero_ps();
    __m512 c_25 = _mm512_setzero_ps();
    __m512 c_26 = _mm512_setzero_ps();
    __m512 c_27 = _mm512_setzero_ps();
    __m512 c_28 = _mm512_setzero_ps();
    __m512 c_29 = _mm512_setzero_ps();
    __m512 c_30 = _mm512_setzero_ps();
    __m512 c_31 = _mm512_setzero_ps();
    __m512 a_0;
    __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9, b_10, b_11, b_12, b_13, b_14, b_15;
    __m512 b_16, b_17, b_18, b_19, b_20, b_21, b_22, b_23, b_24, b_25, b_26, b_27, b_28, b_29, b_30, b_31;

    // __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7;
    int k = 0;
    for (k = 0; k < K; k++) {
        a_0 = _mm512_loadu_ps(A + index(i, k));
        b_0 = _mm512_set1_ps(B[index(k, j)]);
        b_1 = _mm512_set1_ps(B[index(k, j + 1)]);
        b_2 = _mm512_set1_ps(B[index(k, j + 2)]);
        b_3 = _mm512_set1_ps(B[index(k, j + 3)]);
        b_4 = _mm512_set1_ps(B[index(k, j + 4)]);
        b_5 = _mm512_set1_ps(B[index(k, j + 5)]);
        b_6 = _mm512_set1_ps(B[index(k, j + 6)]);
        b_7 = _mm512_set1_ps(B[index(k, j + 7)]);
        b_8 = _mm512_set1_ps(B[index(k, j + 8)]);
        b_9 = _mm512_set1_ps(B[index(k, j + 9)]);
        b_10 = _mm512_set1_ps(B[index(k, j + 10)]);
        b_11 = _mm512_set1_ps(B[index(k, j + 11)]);
        b_12 = _mm512_set1_ps(B[index(k, j + 12)]);
        b_13 = _mm512_set1_ps(B[index(k, j + 13)]);
        b_14 = _mm512_set1_ps(B[index(k, j + 14)]);
        b_15 = _mm512_set1_ps(B[index(k, j + 15)]);
        b_16 = _mm512_set1_ps(B[index(k, j + 16)]);
        b_17 = _mm512_set1_ps(B[index(k, j + 17)]);
        b_18 = _mm512_set1_ps(B[index(k, j + 18)]);
        b_19 = _mm512_set1_ps(B[index(k, j + 19)]);
        b_20 = _mm512_set1_ps(B[index(k, j + 20)]);
        b_21 = _mm512_set1_ps(B[index(k, j + 21)]);
        b_22 = _mm512_set1_ps(B[index(k, j + 22)]);
        b_23 = _mm512_set1_ps(B[index(k, j + 23)]);
        b_24 = _mm512_set1_ps(B[index(k, j + 24)]);
        b_25 = _mm512_set1_ps(B[index(k, j + 25)]);
        b_26 = _mm512_set1_ps(B[index(k, j + 26)]);
        b_27 = _mm512_set1_ps(B[index(k, j + 27)]);
        b_28 = _mm512_set1_ps(B[index(k, j + 28)]);
        b_29 = _mm512_set1_ps(B[index(k, j + 29)]);
        b_30 = _mm512_set1_ps(B[index(k, j + 30)]);
        b_31 = _mm512_set1_ps(B[index(k, j + 31)]);
        c_0 = _mm512_fmadd_ps(a_0, b_0, c_0);
        c_1 = _mm512_fmadd_ps(a_0, b_1, c_1);
        c_2 = _mm512_fmadd_ps(a_0, b_2, c_2);
        c_3 = _mm512_fmadd_ps(a_0, b_3, c_3);
        c_4 = _mm512_fmadd_ps(a_0, b_4, c_4);
        c_5 = _mm512_fmadd_ps(a_0, b_5, c_5);
        c_6 = _mm512_fmadd_ps(a_0, b_6, c_6);
        c_7 = _mm512_fmadd_ps(a_0, b_7, c_7);
        c_8 = _mm512_fmadd_ps(a_0, b_8, c_8);
        c_9 = _mm512_fmadd_ps(a_0, b_9, c_9);
        c_10 = _mm512_fmadd_ps(a_0, b_10, c_10);
        c_11 = _mm512_fmadd_ps(a_0, b_11, c_11);
        c_12 = _mm512_fmadd_ps(a_0, b_12, c_12);
        c_13 = _mm512_fmadd_ps(a_0, b_13, c_13);
        c_14 = _mm512_fmadd_ps(a_0, b_14, c_14);
        c_15 = _mm512_fmadd_ps(a_0, b_15, c_15);
        c_16 = _mm512_fmadd_ps(a_0, b_16, c_16);
        c_17 = _mm512_fmadd_ps(a_0, b_17, c_17);
        c_18 = _mm512_fmadd_ps(a_0, b_18, c_18);
        c_19 = _mm512_fmadd_ps(a_0, b_19, c_19);
        c_20 = _mm512_fmadd_ps(a_0, b_20, c_20);
        c_21 = _mm512_fmadd_ps(a_0, b_21, c_21);
        c_22 = _mm512_fmadd_ps(a_0, b_22, c_22);
        c_23 = _mm512_fmadd_ps(a_0, b_23, c_23);
        c_24 = _mm512_fmadd_ps(a_0, b_24, c_24);
        c_25 = _mm512_fmadd_ps(a_0, b_25, c_25);
        c_26 = _mm512_fmadd_ps(a_0, b_26, c_26);
        c_27 = _mm512_fmadd_ps(a_0, b_27, c_27);
        c_28 = _mm512_fmadd_ps(a_0, b_28, c_28);
        c_29 = _mm512_fmadd_ps(a_0, b_29, c_29);
        c_30 = _mm512_fmadd_ps(a_0, b_30, c_30);
        c_31 = _mm512_fmadd_ps(a_0, b_31, c_31);
    }
    _m512_y_add_to_x(C + index(i, j), c_0);
    _m512_y_add_to_x(C + index(i, j + 1), c_1);
    _m512_y_add_to_x(C + index(i, j + 2), c_2);
    _m512_y_add_to_x(C + index(i, j + 3), c_3);
    _m512_y_add_to_x(C + index(i, j + 4), c_4);
    _m512_y_add_to_x(C + index(i, j + 5), c_5);
    _m512_y_add_to_x(C + index(i, j + 6), c_6);
    _m512_y_add_to_x(C + index(i, j + 7), c_7);
    _m512_y_add_to_x(C + index(i, j + 8), c_8);
    _m512_y_add_to_x(C + index(i, j + 9), c_9);
    _m512_y_add_to_x(C + index(i, j + 10), c_10);
    _m512_y_add_to_x(C + index(i, j + 11), c_11);
    _m512_y_add_to_x(C + index(i, j + 12), c_12);
    _m512_y_add_to_x(C + index(i, j + 13), c_13);
    _m512_y_add_to_x(C + index(i, j + 14), c_14);
    _m512_y_add_to_x(C + index(i, j + 15), c_15);
    _m512_y_add_to_x(C + index(i, j + 16), c_16);
    _m512_y_add_to_x(C + index(i, j + 17), c_17);
    _m512_y_add_to_x(C + index(i, j + 18), c_18);
    _m512_y_add_to_x(C + index(i, j + 19), c_19);
    _m512_y_add_to_x(C + index(i, j + 20), c_20);
    _m512_y_add_to_x(C + index(i, j + 21), c_21);
    _m512_y_add_to_x(C + index(i, j + 22), c_22);
    _m512_y_add_to_x(C + index(i, j + 23), c_23);
    _m512_y_add_to_x(C + index(i, j + 24), c_24);
    _m512_y_add_to_x(C + index(i, j + 25), c_25);
    _m512_y_add_to_x(C + index(i, j + 26), c_26);
    _m512_y_add_to_x(C + index(i, j + 27), c_27);
    _m512_y_add_to_x(C + index(i, j + 28), c_28);
    _m512_y_add_to_x(C + index(i, j + 29), c_29);
    _m512_y_add_to_x(C + index(i, j + 30), c_30);
    _m512_y_add_to_x(C + index(i, j + 31), c_31);
}

static inline void kernel_16xKx16_512(int lda, int i, int j, int K, float* A, float* B, float* C) {
    __m512 c_0 = _mm512_setzero_ps();
    __m512 c_1 = _mm512_setzero_ps();
    __m512 c_2 = _mm512_setzero_ps();
    __m512 c_3 = _mm512_setzero_ps();
    __m512 c_4 = _mm512_setzero_ps();
    __m512 c_5 = _mm512_setzero_ps();
    __m512 c_6 = _mm512_setzero_ps();
    __m512 c_7 = _mm512_setzero_ps();
    __m512 c_8 = _mm512_setzero_ps();
    __m512 c_9 = _mm512_setzero_ps();
    __m512 c_10 = _mm512_setzero_ps();
    __m512 c_11 = _mm512_setzero_ps();
    __m512 c_12 = _mm512_setzero_ps();
    __m512 c_13 = _mm512_setzero_ps();
    __m512 c_14 = _mm512_setzero_ps();
    __m512 c_15 = _mm512_setzero_ps();
    prefetch_T0(C + index(i, j));
    prefetch_T0(C + index(i, j + 1));
    prefetch_T0(C + index(i, j + 2));
    prefetch_T0(C + index(i, j + 3));
    prefetch_T0(C + index(i, j + 4));
    prefetch_T0(C + index(i, j + 5));
    prefetch_T0(C + index(i, j + 6));
    prefetch_T0(C + index(i, j + 7));
    prefetch_T0(C + index(i, j + 8));
    prefetch_T0(C + index(i, j + 9));
    prefetch_T0(C + index(i, j + 10));
    prefetch_T0(C + index(i, j + 11));
    prefetch_T0(C + index(i, j + 12));
    prefetch_T0(C + index(i, j + 13));
    prefetch_T0(C + index(i, j + 14));
    prefetch_T0(C + index(i, j + 15));
    __m512 a_0;
    __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9, b_10, b_11, b_12, b_13, b_14, b_15;
    // __m512 b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7;
    int k = 0;
    for (k = 0; k < K; k++) {
        a_0 = _mm512_loadu_ps(A + index(i, k));
        b_0 = _mm512_set1_ps(B[index(k, j)]);
        c_0 = _mm512_fmadd_ps(a_0, b_0, c_0);
        b_1 = _mm512_set1_ps(B[index(k, j + 1)]);
        c_1 = _mm512_fmadd_ps(a_0, b_1, c_1);
        b_2 = _mm512_set1_ps(B[index(k, j + 2)]);
        c_2 = _mm512_fmadd_ps(a_0, b_2, c_2);
        b_3 = _mm512_set1_ps(B[index(k, j + 3)]);
        c_3 = _mm512_fmadd_ps(a_0, b_3, c_3);
        b_4 = _mm512_set1_ps(B[index(k, j + 4)]);
        c_4 = _mm512_fmadd_ps(a_0, b_4, c_4);
        b_5 = _mm512_set1_ps(B[index(k, j + 5)]);
        c_5 = _mm512_fmadd_ps(a_0, b_5, c_5);
        b_6 = _mm512_set1_ps(B[index(k, j + 6)]);
        c_6 = _mm512_fmadd_ps(a_0, b_6, c_6);
        b_7 = _mm512_set1_ps(B[index(k, j + 7)]);
        c_7 = _mm512_fmadd_ps(a_0, b_7, c_7);
        b_8 = _mm512_set1_ps(B[index(k, j + 8)]);
        c_8 = _mm512_fmadd_ps(a_0, b_8, c_8);
        b_9 = _mm512_set1_ps(B[index(k, j + 9)]);
        c_9 = _mm512_fmadd_ps(a_0, b_9, c_9);
        b_10 = _mm512_set1_ps(B[index(k, j + 10)]);
        c_10 = _mm512_fmadd_ps(a_0, b_10, c_10);
        b_11 = _mm512_set1_ps(B[index(k, j + 11)]);
        c_11 = _mm512_fmadd_ps(a_0, b_11, c_11);
        b_12 = _mm512_set1_ps(B[index(k, j + 12)]);
        c_12 = _mm512_fmadd_ps(a_0, b_12, c_12);
        b_13 = _mm512_set1_ps(B[index(k, j + 13)]);
        c_13 = _mm512_fmadd_ps(a_0, b_13, c_13);
        b_14 = _mm512_set1_ps(B[index(k, j + 14)]);
        c_14 = _mm512_fmadd_ps(a_0, b_14, c_14);
        b_15 = _mm512_set1_ps(B[index(k, j + 15)]);
        c_15 = _mm512_fmadd_ps(a_0, b_15, c_15);
    }
    _m512_y_add_to_x(C + index(i, j), c_0);
    _m512_y_add_to_x(C + index(i, j + 1), c_1);
    _m512_y_add_to_x(C + index(i, j + 2), c_2);
    _m512_y_add_to_x(C + index(i, j + 3), c_3);
    _m512_y_add_to_x(C + index(i, j + 4), c_4);
    _m512_y_add_to_x(C + index(i, j + 5), c_5);
    _m512_y_add_to_x(C + index(i, j + 6), c_6);
    _m512_y_add_to_x(C + index(i, j + 7), c_7);
    _m512_y_add_to_x(C + index(i, j + 8), c_8);
    _m512_y_add_to_x(C + index(i, j + 9), c_9);
    _m512_y_add_to_x(C + index(i, j + 10), c_10);
    _m512_y_add_to_x(C + index(i, j + 11), c_11);
    _m512_y_add_to_x(C + index(i, j + 12), c_12);
    _m512_y_add_to_x(C + index(i, j + 13), c_13);
    _m512_y_add_to_x(C + index(i, j + 14), c_14);
    _m512_y_add_to_x(C + index(i, j + 15), c_15);
}

static inline void kernel_16xKx8(int lda, int i, int j, int K, float* A, float* B, float* C) {
    __m256 c00 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps();
    __m256 c60 = _mm256_setzero_ps();
    __m256 c70 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c41 = _mm256_setzero_ps();
    __m256 c51 = _mm256_setzero_ps();
    __m256 c61 = _mm256_setzero_ps();
    __m256 c71 = _mm256_setzero_ps();
    __m256 a0, a1, b0, b1, b2, b3, b4, b5, b6, b7;
    prefetch_T0(C + index(i, j));
    prefetch_T0(C + index(i, j + 1));
    prefetch_T0(C + index(i, j + 2));
    prefetch_T0(C + index(i, j + 3));
    prefetch_T0(C + index(i, j + 4));
    prefetch_T0(C + index(i, j + 5));
    prefetch_T0(C + index(i, j + 6));
    prefetch_T0(C + index(i, j + 7));
    int k = 0;
    for (k = 0; k < K; k++) {
        a0 = _mm256_loadu_ps(A + index(i, k));
        a1 = _mm256_loadu_ps(A + index(i + 8, k));
        b0 = _mm256_broadcast_ss(B + index(k, j));
        b1 = _mm256_broadcast_ss(B + index(k, j + 1));
        b2 = _mm256_broadcast_ss(B + index(k, j + 2));
        b3 = _mm256_broadcast_ss(B + index(k, j + 3));
        b4 = _mm256_broadcast_ss(B + index(k, j + 4));
        b5 = _mm256_broadcast_ss(B + index(k, j + 5));
        b6 = _mm256_broadcast_ss(B + index(k, j + 6));
        b7 = _mm256_broadcast_ss(B + index(k, j + 7));
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c10 = _mm256_fmadd_ps(a0, b1, c10);
        c20 = _mm256_fmadd_ps(a0, b2, c20);
        c30 = _mm256_fmadd_ps(a0, b3, c30);
        c40 = _mm256_fmadd_ps(a0, b4, c40);
        c50 = _mm256_fmadd_ps(a0, b5, c50);
        c60 = _mm256_fmadd_ps(a0, b6, c60);
        c70 = _mm256_fmadd_ps(a0, b7, c70);
        c01 = _mm256_fmadd_ps(a1, b0, c01);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c21 = _mm256_fmadd_ps(a1, b2, c21);
        c31 = _mm256_fmadd_ps(a1, b3, c31);
        c41 = _mm256_fmadd_ps(a1, b4, c41);
        c51 = _mm256_fmadd_ps(a1, b5, c51);
        c61 = _mm256_fmadd_ps(a1, b6, c61);
        c71 = _mm256_fmadd_ps(a1, b7, c71);
    }
    _m256_y_add_to_x(C + index(i, j), c00);
    _m256_y_add_to_x(C + index(i + 8, j), c01);
    _m256_y_add_to_x(C + index(i, j + 1), c10);
    _m256_y_add_to_x(C + index(i + 8, j + 1), c11);
    _m256_y_add_to_x(C + index(i, j + 2), c20);
    _m256_y_add_to_x(C + index(i + 8, j + 2), c21);
    _m256_y_add_to_x(C + index(i, j + 3), c30);
    _m256_y_add_to_x(C + index(i + 8, j + 3), c31);
    _m256_y_add_to_x(C + index(i, j + 4), c40);
    _m256_y_add_to_x(C + index(i + 8, j + 4), c41);
    _m256_y_add_to_x(C + index(i, j + 5), c50);
    _m256_y_add_to_x(C + index(i + 8, j + 5), c51);
    _m256_y_add_to_x(C + index(i, j + 6), c60);
    _m256_y_add_to_x(C + index(i + 8, j + 6), c61);
    _m256_y_add_to_x(C + index(i, j + 7), c70);
    _m256_y_add_to_x(C + index(i + 8, j + 7), c71);
}

static inline void kernel_8xKx8_packing(int lda, int i, int j, int K, float* packing_A, float* packing_B, float* C) {
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps();
    __m256 c7 = _mm256_setzero_ps();
    __m256 a, b0, b1, b2, b3, b4, b5, b6, b7;
    int k = 0;
    for (k = 0; k < K; k++) {
        a = _mm256_loadu_ps(packing_A);
        b0 = _mm256_broadcast_ss(packing_B);
        b1 = _mm256_broadcast_ss(packing_B + 1);
        b2 = _mm256_broadcast_ss(packing_B + 2);
        b3 = _mm256_broadcast_ss(packing_B + 3);
        b4 = _mm256_broadcast_ss(packing_B + 4);
        b5 = _mm256_broadcast_ss(packing_B + 5);
        b6 = _mm256_broadcast_ss(packing_B + 6);
        b7 = _mm256_broadcast_ss(packing_B + 7);
        c0 = _mm256_fmadd_ps(a, b0, c0);
        c1 = _mm256_fmadd_ps(a, b1, c1);
        c2 = _mm256_fmadd_ps(a, b2, c2);
        c3 = _mm256_fmadd_ps(a, b3, c3);
        c4 = _mm256_fmadd_ps(a, b4, c4);
        c5 = _mm256_fmadd_ps(a, b5, c5);
        c6 = _mm256_fmadd_ps(a, b6, c6);
        c7 = _mm256_fmadd_ps(a, b7, c7);
        packing_A += 8; packing_B += 8;
    }
    _m256_y_add_to_x(C + index(i, j), c0);
    _m256_y_add_to_x(C + index(i, j + 1), c1);
    _m256_y_add_to_x(C + index(i, j + 2), c2);
    _m256_y_add_to_x(C + index(i, j + 3), c3);
    _m256_y_add_to_x(C + index(i, j + 4), c4);
    _m256_y_add_to_x(C + index(i, j + 5), c5);
    _m256_y_add_to_x(C + index(i, j + 6), c6);
    _m256_y_add_to_x(C + index(i, j + 7), c7);
}

static inline void kernel_8xKx8(int lda, int i, int j, int K, float* A, float* B, float* C) {
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps();
    __m256 c7 = _mm256_setzero_ps();
    __m256 a, b0, b1, b2, b3, b4, b5, b6, b7;
    int k = 0;
    for (k = 0; k < K; k++) {
        a = _mm256_loadu_ps(A + index(i, k));
        b0 = _mm256_broadcast_ss(B + index(k, j));
        b1 = _mm256_broadcast_ss(B + index(k, j + 1));
        b2 = _mm256_broadcast_ss(B + index(k, j + 2));
        b3 = _mm256_broadcast_ss(B + index(k, j + 3));
        b4 = _mm256_broadcast_ss(B + index(k, j + 4));
        b5 = _mm256_broadcast_ss(B + index(k, j + 5));
        b6 = _mm256_broadcast_ss(B + index(k, j + 6));
        b7 = _mm256_broadcast_ss(B + index(k, j + 7));
        c0 = _mm256_fmadd_ps(a, b0, c0);
        c1 = _mm256_fmadd_ps(a, b1, c1);
        c2 = _mm256_fmadd_ps(a, b2, c2);
        c3 = _mm256_fmadd_ps(a, b3, c3);
        c4 = _mm256_fmadd_ps(a, b4, c4);
        c5 = _mm256_fmadd_ps(a, b5, c5);
        c6 = _mm256_fmadd_ps(a, b6, c6);
        c7 = _mm256_fmadd_ps(a, b7, c7);
    }
    _m256_y_add_to_x(C + index(i, j), c0);
    _m256_y_add_to_x(C + index(i, j + 1), c1);
    _m256_y_add_to_x(C + index(i, j + 2), c2);
    _m256_y_add_to_x(C + index(i, j + 3), c3);
    _m256_y_add_to_x(C + index(i, j + 4), c4);
    _m256_y_add_to_x(C + index(i, j + 5), c5);
    _m256_y_add_to_x(C + index(i, j + 6), c6);
    _m256_y_add_to_x(C + index(i, j + 7), c7);
}

void output_vec(const char* str, int size, float* A) {
    printf("%s Vec\n", str);
    for (int i = 0; i < size; i++)
        printf("%.1f ", A[i]);
    printf("\n");
}

void output(const char* str, int lda, float* A) {
    printf("%s Matrix\n", str);
    for (int i = 0; i < lda; i++, printf("\n"))
        for (int j = 0; j < lda; j++) {
            printf("%.1f ", A[index(i, j)]);
        }
}

void print_512(const char* str, __m512 vec) {
    float test[16];
    printf("%s:\n", str);
    _mm512_storeu_ps(test, vec);
    for (int k = 0; k < 16; k++) {
        printf("%f ", test[k]);
    }
    printf("\n");
}


void print_256(const char* str, __m256 vec) {
    float test[8];
    printf("%s:\n", str);
    _mm256_storeu_ps(test, vec);
    for (int k = 0; k < 8; k++) {
        printf("%f ", test[k]);
    }
    printf("\n");
}

static inline void do_block_avx512(int lda, int M, int N, int K, float* A, float* B, float* C) {
    int M_32 = M & -32, M_16 = M & -16, M_8 = M & -8, N_8 = N & -8, N_16 = N & -16, N_32 = N & -32;
    // for (int i = 0; i < M_16; i += 16) {
    //     for (int j = 0; j < N_32; j += 32) {
    //         kernel_16xKx32_512(lda, i, j, K, A, B, C);
    //     }
    //     if (N_32 < N_16) {
    //         kernel_16xKx16_512(lda, i, N_32, K, A, B, C);
    //         if (N_16 < N_8) {
    //             kernel_16xKx8_512(lda, i, N_16, K, A, B, C);
    //         }
    //     }
    // }
    // int M_48 = M - M % 48;
    // for (int i = 0; i < M_48; i += 48) {
    //     for (int j = 0; j < N_8; j += 8) {
    //         kernel_48xKx8_512(lda, i, j, K, A, B, C);
    //     }
    // }
    // if (M - M_48 > 32) {
    //     for (int j = 0; j < N_8; j += 8) {
    //         kernel_32xKx16_512(lda, M_48, j, K, A, B, C);
    //     }
    // }
    for (int i = 0; i < M_32; i += 32) {
        for (int j = 0; j < N_8; j += 8) {
            kernel_32xKx8_512(lda, i, j, K, A, B, C);
        }
    }
    // if (M_48 < M_16) {
    //     for (int i = M_48; i < M_16; i += 16) {
    //         for (int j = 0; j < N_8; j += 8) {
    //             kernel_16xKx8_512(lda, i, j, K, A, B, C);
    //         }
    //     }
    // }
    // for (int i = 0; i < M_16; i += 16) {
    //     for (int j = 0; j < N_8; j += 8) {
    //         kernel_16xKx8_512(lda, i, j, K, A, B, C);
    //     }
    // }
    // if (M_16 < M_8) {
    //     for (int j = 0; j < N_8; j += 8) {
    //         kernel_8xKx8(lda, M_16, j, K, A, B, C);
    //     }
    // }
    // if (N_8 == N && M_8 == M) { return; }
    // else {
    //     if (M_8 == M) {
    //         do_block(lda, M, N - N_8, K, A, B + index(0, N_8), C + index(0, N_8));
    //     }
    //     else if (N_8 == N) {
    //         do_block(lda, M - M_8, N, K, A + index(M_8, 0), B, C + index(M_8, 0));
    //     }
    //     else {
    //         do_block(lda, M_8, N - N_8, K, A, B + index(0, N_8), C + index(0, N_8));
    //         do_block(lda, M - M_8, N, K, A + index(M_8, 0), B, C + index(M_8, 0));
    //     }
    // }
}

static inline void avx512_MxKxN(int lda, int M, int N, int K, float* buffer_A, float* A, float* buffer_B, float* B, float* C) {
    // int N_8 = N & -8, M_16 = M & -16, M_8 = M & -8;
    if (M == 32) {
        if (N == 8) {
            kernel_32xKx8_512_packing(lda, K, buffer_A, buffer_B, C);
        }
        else if (N == 7) {
            kernel_32xKx7_512_packing(lda, K, buffer_A, buffer_B, C);
        }
        else if (N == 1) {
            kernel_32xKx1_512_packing(lda, K, buffer_A, buffer_B, C);
        }
    }
    else if (M == 31) {
        if (N == 8) {
            kernel_31xKx8_512_packing(lda, K, buffer_A, buffer_B, C);
        }
        else if (N == 7) {
            kernel_31xKx7_512_packing(lda, K, buffer_A, buffer_B, C);
        }
        else if (N == 1) {
            kernel_31xKx1_512_packing(lda, K, buffer_A, buffer_B, C);
        }
    }
    else {
        for (int j = 0; j < N; ++j)
        {
            float cij = C[index(0, j)];
            for (int k = 0; k < K; ++k)
                cij += A[index(0, k)] * B[index(k, j)];
            C[index(0, j)] = cij;
        }
    }
}

static inline void do_block_avx2_packing(int lda, int M, int N, int K, float* buffer_A, float* A, float* buffer_B, float* B, float* C) {
    int N_8 = N & -8, M_32 = M & -32, M_16 = M & -16, M_8 = M & -8;
    for (int i = 0; i < M_32; i += 32)
        for (int j = 0; j < N_8; j += 8) {
            float* packing_A = buffer_A + i * K;
            float* packing_B = buffer_B + j * K;
            kernel_32xKx8_512_packing(lda, K, packing_A, packing_B, C + index(i, j));
        }
    if (M_32 < M_16) {
        for (int i = M_32; i < M_16; i += 16)
            for (int j = 0; j < N_8; j += 8) {
                float* packing_A = buffer_A + i * K;
                float* packing_B = buffer_B + j * K;
                kernel_16xKx8_512_packing(lda, i, j, K, packing_A, packing_B, C);
            }
    }
    if (M_16 + 8 < M) {
        for (int j = 0; j < N_8; j += 8) {
            float* packing_A = buffer_A + M_16 * K;
            float* packing_B = buffer_B + j * K;
            kernel_8xKx8_packing(lda, M_16, j, K, packing_A, packing_B, C);
        }
    }
    if (N_8 == N && M_8 == M) { return; }
    else {
        if (M_8 == M) {
            do_block(lda, M, N - N_8, K, A, B + index(0, N_8), C + index(0, N_8));
        }
        else if (N_8 == N) {
            do_block(lda, M - M_8, N, K, A + index(M_8, 0), B, C + index(M_8, 0));
        }
        else {
            do_block(lda, M - M_8, N, K, A + index(M_8, 0), B, C + index(M_8, 0));
            do_block(lda, M_8, N - N_8, K, A, B + index(0, N_8), C + index(0, N_8));
        }
    }

    // printf("%d %d %d %d %d\n", M_16, N_8, M, N, K);
    // if (N_8 == N && M_16 == M) { return; }
    // else {
    //     // output("before", lda, C);
    //     if (M_16 == M) {
    //         do_block(lda, M, N - N_8, K, A, B + index(0, N_8), C + index(0, N_8));
    //     }
    //     else if (N_8 == N) {
    //         do_block(lda, M - M_16, N, K, A + index(M_16, 0), B, C + index(M_16, 0));
    //     }
    //     else {
    //         do_block(lda, M - M_16, N, K, A + index(M_16, 0), B, C + index(M_16, 0));
    //         do_block(lda, M_16, N - N_8, K, A, B + index(0, N_8), C + index(0, N_8));
    //     }
    // }
}

static inline void do_block_avx2(int lda, int M, int N, int K, float* A, float* B, float* C) {
    int M_16 = M & -16, M_8 = M & -8, N_8 = N & -8, N_16 = N & -16, N_32 = N & -32;
    for (int i = 0; i < M_16; i += 16) {
        for (int j = 0; j < N_16; j += 16) {
            kernel_16xKx16_512(lda, i, j, K, A, B, C);
        }
        if (N_16 < N_8) {
            kernel_16xKx8_512(lda, i, N_16, K, A, B, C);
        }
    }
    if (M_16 < M_8) {
        for (int j = 0; j < N_8; j += 8) {
            kernel_8xKx8(lda, M_16, j, K, A, B, C);
        }
    }
    if (N_8 == N && M_8 == M) { return; }
    else {
        if (M_8 == M) {
            do_block(lda, M, N - N_8, K, A, B + index(0, N_8), C + index(0, N_8));
        }
        else if (N_8 == N) {
            do_block(lda, M - M_8, N, K, A + index(M_8, 0), B, C + index(M_8, 0));
        }
        else {
            do_block(lda, M_8, N - N_8, K, A, B + index(0, N_8), C + index(0, N_8));
            do_block(lda, M - M_8, N, K, A + index(M_8, 0), B, C + index(M_8, 0));
        }
    }
}

/* This use AVX2 instruction for 4 * 4 matrix
* This add cache block
*/

/* naive block */
void std(int lda, float* A, float* B, float* C)
{
    /* For each block-row of A */
    for (int i = 0; i < lda; i += BLOCK_SIZE)
        for (int k = 0; k < lda; k += BLOCK_SIZE)
            /* For each block-column of B */
            for (int j = 0; j < lda; j += BLOCK_SIZE)
                /* Accumulate block sgemms into block of C */
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);

                /* Perform individual block sgemm */
                do_block(lda, M, N, K, A + index(i, k), B + index(k, j), C + index(i, j));
            }
}

#define M_SIZE (32)
#define K_SIZE (256)
#define N_SIZE (8)

#define BUFF_SIZE (2048 * 2048)
float buffer_A[BUFF_SIZE] __attribute__((aligned(4096)));
float buffer_B[BUFF_SIZE] __attribute__((aligned(4096)));

void square_sgemm(int lda, float* A, float* B, float* C)
{
    assert(M_SIZE == 32);
    if (lda == 33) {
        packing_matrix_row(B, buffer_B, lda, lda, lda);
        packing_matrix_col(A, buffer_A, lda, 32, lda);
        for (int j = 0; j < lda; j += N_SIZE) {
            int N = min(N_SIZE, lda - j);
            avx512_MxKxN(lda, 32, N, lda, buffer_A, A, buffer_B + lda * j, B + index(0, j), C + index(0, j));
        }
        packing_matrix_col(A + 32, buffer_A, lda, 1, lda);
        for (int j = 0; j < lda; j += N_SIZE) {
            int N = min(N_SIZE, lda - j);
            avx512_MxKxN(lda, 1, N, lda, buffer_A, A + 32, buffer_B + lda * j, B + index(0, j), C + index(32, j));
        }
        return;
    }

    for (int k = 0; k < lda; k += K_SIZE) {
        int K = min(K_SIZE, lda - k);
        packing_matrix_row(B + index(k, 0), buffer_B, lda, K, lda);

        for (int i = 0; i < lda; i += M_SIZE) {
            int M = min(M_SIZE, lda - i);
            packing_matrix_col(A + index(i, k), buffer_A, lda, M, K);

            for (int j = 0; j < lda; j += N_SIZE) {
                int N = min(N_SIZE, lda - j);

                // do_block_avx2_packing(lda, M, N, K, buffer_A, A + index(i, k), buffer_B + K * j, B + index(k, j), C + index(i, j));
                avx512_MxKxN(lda, M, N, K, buffer_A, A + index(i, k), buffer_B + K * j, B + index(k, j), C + index(i, j));
                // do_block_avx512(lda, M, N, K, A + index(i, k), B + index(k, j), C + index(i, j));
                // do_block_avx2(lda, M, N, K, A + index(i, k), B + index(k, j), C + index(i, j));
            }
        }
    }
}

/* This use AVX2 instruction for 16 * 8 matrix
void square_sgemm(int lda, float* A, float* B, float* C)
{
    int lda_8 = lda & -8, lda_16 = lda & -16;
    float* A_T = (float*)malloc(sizeof(float) * lda * lda);
    for (int i = 0; i < lda; i++)
        for (int j = 0; j < lda; j++) {
            A_T[index_T(i, j)] = A[index(i, j)];
        }
    // float* B_T = (float*)malloc(sizeof(float) * lda * lda);
    // for (int i = 0; i < lda; i++)
    //     for (int j = 0; j < lda; j++) {
    //         B_T[index_T(i, j)] = B[index(i, j)];
    //     }
    // float* C_T = (float*)malloc(sizeof(float) * lda * lda);
    // memset(C_T, 0, sizeof(float) * lda * lda);
    for (int i = 0; i < lda_16; i += 16)
    for (int j = 0; j < lda_8; j += 8) {
        // kernel_4xldax4(lda, A + index(i, 0), B + index(0, j), C_T + index_T(i, j));
        kernel_16xldax8(lda, i, j, lda, A, B, C);
        // __m256 c0 = _mm256_setzero_ps();
        // __m256 c1 = _mm256_setzero_ps();
        // __m256 c2 = _mm256_setzero_ps();
        // __m256 c3 = _mm256_setzero_ps();
        // __m256 c4 = _mm256_setzero_ps();
        // __m256 c5 = _mm256_setzero_ps();
        // __m256 c6 = _mm256_setzero_ps();
        // __m256 c7 = _mm256_setzero_ps();
        // for (int k = 0; k < lda; k++) {
        //     __m256 a = _mm256_loadu_ps(A + index(i, k));
        //     __m256 b0 = _mm256_broadcast_ss(B + index(k, j));
        //     __m256 b1 = _mm256_broadcast_ss(B + index(k, j + 1));
        //     __m256 b2 = _mm256_broadcast_ss(B + index(k, j + 2));
        //     __m256 b3 = _mm256_broadcast_ss(B + index(k, j + 3));
        //     __m256 b4 = _mm256_broadcast_ss(B + index(k, j + 4));
        //     __m256 b5 = _mm256_broadcast_ss(B + index(k, j + 5));
        //     __m256 b6 = _mm256_broadcast_ss(B + index(k, j + 6));
        //     __m256 b7 = _mm256_broadcast_ss(B + index(k, j + 7));
        //     c0 = _mm256_fmadd_ps(a, b0, c0);
        //     c1 = _mm256_fmadd_ps(a, b1, c1);
        //     c2 = _mm256_fmadd_ps(a, b2, c2);
        //     c3 = _mm256_fmadd_ps(a, b3, c3);
        //     c4 = _mm256_fmadd_ps(a, b4, c4);
        //     c5 = _mm256_fmadd_ps(a, b5, c5);
        //     c6 = _mm256_fmadd_ps(a, b6, c6);
        //     c7 = _mm256_fmadd_ps(a, b7, c7);
        // }
        // _mm256_storeu_ps(C + index(i, j), c0);
        // _mm256_storeu_ps(C + index(i, j + 1), c1);
        // _mm256_storeu_ps(C + index(i, j + 2), c2);
        // _mm256_storeu_ps(C + index(i, j + 3), c3);
        // _mm256_storeu_ps(C + index(i, j + 4), c4);
        // _mm256_storeu_ps(C + index(i, j + 5), c5);
        // _mm256_storeu_ps(C + index(i, j + 6), c6);
        // _mm256_storeu_ps(C + index(i, j + 7), c7);
    }
    // output("A", lda, A);
    // output("B", lda, B);
    // output("my", lda, C);
    // float* ans = (float*)malloc(sizeof(float) * lda * lda);
    // for (int i = 0; i < lda * lda; i++) ans[i] = 0.0;
    // do_block(lda, lda, lda, lda, A, B, ans);
    // output("std", lda, ans);
    // for (int i = 0; i < lda; i++)
    //     for (int j = 0; j < lda; j++) {
    //         C[index(i, j)] = C_T[index(i, j)];
    //     }
    if (lda_16 + 8 < lda) {
        for (int j = 0; j < lda_8; j += 8) {
            kernel_8xldax8(lda, lda_16, lda, j, A, B, C);
        }
    }
    if (lda_8 != lda) {
        do_block(lda, lda - lda_8, lda, lda, A_T + index_T(lda_8, 0), B, C + index(lda_8, 0));
        do_block(lda, lda_8, lda - lda_8, lda, A_T, B + index(0, lda_8), C + index(0, lda_8));
        // do_block(lda, lda - lda_8, lda, lda, A + index(lda_8, 0), B, C + index(lda_8, 0));
        // do_block(lda, lda_8, lda - lda_8, lda, A, B + index(0, lda_8), C + index(0, lda_8));
    }
    free(A_T);
    // free(B_T);
    // free(C_T);
}
*/


/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
 // void square_sgemm(int lda, float* A, float* B, float* C)
 // {
 //     /* matrix transpose of A */
 //     float* A1 = (float*)malloc(sizeof(float) * lda * lda);
 //     for (int i = 0; i < lda; i++)
 //         for (int j = 0; j < lda; j++) {
 //             A1[index_T(i, j)] = A[index(i, j)];
 //         }
 //     /* For each block-row of A */
 //     for (int i = 0; i < lda; i += BLOCK_SIZE)
 //         for (int k = 0; k < lda; k += BLOCK_SIZE)
 //             /* For each block-column of B */
 //             for (int j = 0; j < lda; j += BLOCK_SIZE)
 //                 /* Accumulate block sgemms into block of C */
 //             {
 //                 /* Correct block dimensions if block "goes off edge of" the matrix */
 //                 int M = min(BLOCK_SIZE, lda - i);
 //                 int N = min(BLOCK_SIZE, lda - j);
 //                 int K = min(BLOCK_SIZE, lda - k);

 //                 /* Perform individual block sgemm */
 //                 do_block(lda, M, N, K, A1 + index_T(i, k), B + index(k, j), C + index(i, j));
 //             }
 //     free(A1);
 // }


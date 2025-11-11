// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh", "wgmma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}

#include <cuda.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdio.h>

#include "tma-interface.cuh"
#include "wgmma-interface.cuh"

typedef __nv_bfloat16 bf16;

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Part 0: No Swizzle WGGMA load for M = 64, N = 8, K = 16
////////////////////////////////////////////////////////////////////////////////

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void wgmma_m64n8k16(bf16 *a, bf16 *b, float *c) {

    // <--- your code here --->

    constexpr int core_matrix_cols_bytes = 16;
    constexpr int core_matrix_rows = 8;

    constexpr int core_matrix_cols = core_matrix_cols_bytes / sizeof(bf16);
    constexpr int core_matrix_tot = core_matrix_cols * core_matrix_rows;

    __shared__ bf16 __align__(128) a_shm[TILE_M * TILE_K];
    __shared__ bf16 __align__(128) b_shm[TILE_N * TILE_K];

    constexpr int CORE_M = TILE_M / core_matrix_rows;
    constexpr int CORE_K = TILE_K / core_matrix_cols;

    for (int m_out = 0; m_out * core_matrix_rows < TILE_M; m_out++) {
        for (int k_out = 0; k_out * core_matrix_cols < TILE_K; k_out++) {
            for (int m_in = 0; m_in < core_matrix_rows; ++m_in) {
                for (int k_in = 0; k_in < core_matrix_cols; ++k_in) {

                    int global_idx = (m_out * core_matrix_rows + m_in) * TILE_K +
                        (k_out * core_matrix_cols + k_in);
                    int shared_idx = (m_out * CORE_K + k_out) * core_matrix_tot +
                        m_in * core_matrix_cols + k_in;

                    a_shm[shared_idx] = a[global_idx];
                }
            }
        }
    }

    for (int n_out = 0; n_out * core_matrix_rows < TILE_N; n_out++) {
        for (int k_out = 0; k_out * core_matrix_cols < TILE_K; k_out++) {
            for (int n_in = 0; n_in < core_matrix_rows; ++n_in) {
                for (int k_in = 0; k_in < core_matrix_cols; ++k_in) {

                    int global_idx = (n_out * core_matrix_rows + n_in) * TILE_K +
                        (k_out * core_matrix_cols + k_in);
                    int shared_idx = (n_out * CORE_K + k_out) * core_matrix_tot +
                        n_in * core_matrix_cols + k_in;

                    b_shm[shared_idx] = b[global_idx];
                }
            }
        }
    }

    uint64_t a_des = make_smem_desc<NO_SWIZZLE>(
        a_shm,
        core_matrix_tot * sizeof(bf16),
        core_matrix_tot * CORE_K * sizeof(bf16));
    uint64_t b_des = make_smem_desc<NO_SWIZZLE>(
        b_shm,
        core_matrix_tot * sizeof(bf16),
        core_matrix_tot * CORE_K * sizeof(bf16));

    float d[4];

    async_proxy_fence();
    warpgroup_arrive();
    wgmma_n8<0, 1, 1, 0, 0>(a_des, b_des, d);
    wgmma_commit();
    wgmma_wait<0>();

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    int m0 = 16 * warp + (lane / 4);
    int n0 = 2 * (lane % 4);

    int m1 = 16 * warp + (lane / 4);
    int n1 = 2 * (lane % 4) + 1;

    int m2 = 16 * warp + 8 + (lane / 4);
    int n2 = 2 * (lane % 4);

    int m3 = 16 * warp + 8 + (lane / 4);
    int n3 = 2 * (lane % 4) + 1;

    c[n0 * TILE_M + m0] = d[0];
    c[n1 * TILE_M + m1] = d[1];
    c[n2 * TILE_M + m2] = d[2];
    c[n3 * TILE_M + m3] = d[3];
}

template <int TILE_M, int TILE_N, int TILE_K>
void launch_wgmma_m64n8k16(bf16 *a, bf16 *b, float *c) {

    // <--- your code here --->

    wgmma_m64n8k16<TILE_M, TILE_N, TILE_K><<<1, 4 * 32>>>(a, b, c);
}

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

int main() {
    const int M = 64;
    const int N = 8;
    const int K = 16;

    // Initialize source matrix on host
    bf16 *a = (bf16 *)malloc(M * K * sizeof(bf16));
    bf16 *b = (bf16 *)malloc(N * K * sizeof(bf16));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            a[i * K + j] = i + j;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            b[j * N + i] = i + j;
        }
    }

    float *d_c;
    bf16 *d_a, *d_b;
    cudaMalloc(&d_a, M * K * sizeof(bf16));
    cudaMalloc(&d_b, N * K * sizeof(bf16));
    cudaMalloc(&d_c, M * N * sizeof(float));
    cudaMemcpy(d_a, a, M * K * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * K * sizeof(bf16), cudaMemcpyHostToDevice);

    // Compute CPU reference
    float *cpu_output = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float temp = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_row = (float)a[i * K + k];
                float a_col = (float)b[k + j * K];
                temp += a_row * a_col;
            }
            cpu_output[j * M + i] = temp;
        }
    }

    float *gpu_output = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * N; i++) {
        gpu_output[i] = 0;
    }
    cudaMemcpy(d_c, gpu_output, M * N * sizeof(float), cudaMemcpyHostToDevice);

    printf("\n\nRunning No Swizzle WGMMA M=%d, N=%d, K=%d...\n\n", M, N, K);
    launch_wgmma_m64n8k16<M, N, K>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(gpu_output, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // check results
    bool correct = true;
    for (int idx = 0; idx < M * N; idx++) {
        if (fabs(cpu_output[idx] - gpu_output[idx]) > 0.01f) {
            correct = false;
            int j = idx / M;
            int i = idx % M;
            printf(
                "\nFirst mismatch at (%d, %d): CPU=%.0f, GPU=%.0f\n",
                i,
                j,
                cpu_output[idx],
                gpu_output[idx]);
            break;
        }
    }

    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);

    return 0;
}
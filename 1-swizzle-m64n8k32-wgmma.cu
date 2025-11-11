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
// Part 0: 64B Swizzle WGGMA load for M = 64, N = 8, K = 32
////////////////////////////////////////////////////////////////////////////////

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void swizzle_wgmma_m64n8k32(
    __grid_constant__ const CUtensorMap a_map,
    __grid_constant__ const CUtensorMap b_map,
    float *c) {

    constexpr int core_matrix_cols_bytes = 64;
    constexpr int core_matrix_rows = 8;

    constexpr int core_matrix_cols = core_matrix_cols_bytes / sizeof(bf16);
    constexpr int core_matrix_tot = core_matrix_cols * core_matrix_rows;

    __shared__ bf16 __align__(128) a_shm[TILE_M * TILE_K];
    __shared__ bf16 __align__(128) b_shm[TILE_N * TILE_K];

    constexpr int CORE_K = TILE_K / core_matrix_cols;

    __shared__ alignas(8) uint64_t barrier;

    if (threadIdx.x == 0) {
        init_barrier(&barrier, 1);
        expect_bytes_and_arrive(
            &barrier,
            (TILE_M * TILE_K + TILE_N * TILE_K) * sizeof(bf16));

        async_proxy_fence();

        cp_async_bulk_tensor_2d_global_to_shared(a_shm, &a_map, 0, 0, &barrier);
        cp_async_bulk_tensor_2d_global_to_shared(b_shm, &b_map, 0, 0, &barrier);
        wait(&barrier, 0);
    }
    __syncthreads();

    int second_wgmma_offset = 16;
    int stride_byte_offset = core_matrix_tot * CORE_K * sizeof(bf16);

    uint64_t a_des0 = make_smem_desc<SWIZZLE_64B>(a_shm, 1, stride_byte_offset);
    uint64_t b_des0 = make_smem_desc<SWIZZLE_64B>(b_shm, 1, stride_byte_offset);

    uint64_t a_des1 =
        make_smem_desc<SWIZZLE_64B>(a_shm + second_wgmma_offset, 1, stride_byte_offset);
    uint64_t b_des1 =
        make_smem_desc<SWIZZLE_64B>(b_shm + second_wgmma_offset, 1, stride_byte_offset);

    float d[4];

    async_proxy_fence();
    warpgroup_arrive();
    wgmma_n8<0, 1, 1, 0, 0>(a_des0, b_des0, d);
    wgmma_n8<1, 1, 1, 0, 0>(a_des1, b_des1, d);
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
void launch_swizzle_wgmma_m64n8k32(bf16 *a, bf16 *b, float *c) {

    // <--- your code here --->

    CUtensorMap a_map;
    const uint64_t a_globalDim[] = {TILE_K, TILE_M};
    const uint64_t a_globalStrides[] = {TILE_K * sizeof(bf16)};
    const uint32_t a_boxDim[] = {TILE_K, TILE_M};
    const uint32_t a_elementStrides[] = {1, 1};

    CUDA_CHECK(cuTensorMapEncodeTiled(
        &a_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        a,
        a_globalDim,
        a_globalStrides,
        a_boxDim,
        a_elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CUtensorMap b_map;
    const uint64_t b_globalDim[] = {TILE_K, TILE_N};
    const uint64_t b_globalStrides[] = {TILE_K * sizeof(bf16)};
    const uint32_t b_boxDim[] = {TILE_K, TILE_N};
    const uint32_t b_elementStrides[] = {1, 1};

    CUDA_CHECK(cuTensorMapEncodeTiled(
        &b_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        b,
        b_globalDim,
        b_globalStrides,
        b_boxDim,
        b_elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    swizzle_wgmma_m64n8k32<TILE_M, TILE_N, TILE_K><<<1, 4 * 32>>>(a_map, b_map, c);
}

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

int main() {
    const int M = 64;
    const int N = 8;
    const int K = 32;

    // Initialize source matrix on host
    bf16 *a = (bf16 *)malloc(M * K * sizeof(bf16));
    bf16 *b = (bf16 *)malloc(N * K * sizeof(bf16));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            a[i * K + j] = (i + j) / 10.0f;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            b[j * N + i] = (i + j) / 10.0f;
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

    printf("\n\nRunning Swizzle WGMMA M=64, N=8, K-32...\n\n");
    launch_swizzle_wgmma_m64n8k32<M, N, K>(d_a, d_b, d_c);
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
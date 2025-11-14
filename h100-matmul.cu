// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh", "wgmma-interface.cuh", "kernel.cu"]}
// TL+ {"compile_flags": ["-lcuda", "-lcublas"]}

#include "tma-interface.cuh"
#include "wgmma-interface.cuh"
#include <algorithm>
#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <vector>

typedef __nv_bfloat16 bf16;

////////////////////////////////////////////////////////////////////////////////
// Part 1: Matrix Multiplication for M = 8192, N = 8192, K = 8192
////////////////////////////////////////////////////////////////////////////////

#define UNROLLED_FOR(x) _Pragma("unroll") for(x)

template <typename T> constexpr __host__ __device__ T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

struct KernelTraits {

    static constexpr int WGMMA_WARP_GROUP_CNT = 3;
    static constexpr int TMA_LOAD_WARP_CNT = 4;
    static constexpr int TMA_LOAD_THREAD_CNT = TMA_LOAD_WARP_CNT * 32;

    static constexpr int TOTAL_WARP_CNT = 4 * WGMMA_WARP_GROUP_CNT + TMA_LOAD_WARP_CNT;
    static constexpr int TOTAL_THREAD_CNT = 32 * TOTAL_WARP_CNT;

    static constexpr int WGMMA_M = 64;
    static constexpr int WGMMA_N = 256;
    static constexpr int WGMMA_K = 16;

    static constexpr wgmmaSwizzle SWIZZLE_TYPE = SWIZZLE_64B;

    // depends on swizzle
    static constexpr int CORE_MATRIX_ROWS = 8;
    static constexpr int CORE_MATRIX_COLS_BYTES = 64;
    static constexpr int CORE_MATRIX_COLS = CORE_MATRIX_COLS_BYTES / sizeof(bf16);

    static constexpr int PHASE_M = WGMMA_M * WGMMA_WARP_GROUP_CNT;
    static constexpr int PHASE_N = WGMMA_N;
    static constexpr int PHASE_K = CORE_MATRIX_COLS;

    static constexpr int WGMMAS_PER_PHASE_PER_WARPGROUP = PHASE_K / WGMMA_K;

    static constexpr int PHASE_CNT = 2;

    static constexpr int SHMEM_NEEDED =
        PHASE_CNT * (PHASE_M * PHASE_K + PHASE_K * PHASE_N) * sizeof(bf16);
};

template <typename KernelTraits>
__device__ void tma_into_shmem(
    __grid_constant__ const CUtensorMap a_map,
    __grid_constant__ const CUtensorMap b_map,
    int lane_idx,
    bf16 *shmem_a,
    bf16 *shmem_b,
    int m_beg,
    int n_beg,
    int k_beg) {

    int m_end = m_beg + KernelTraits::PHASE_M;
    int n_end = n_beg + KernelTraits::PHASE_N;
    int k_end = k_beg + KernelTraits::PHASE_K;
}

template <typename KernelTraits>
__device__ void wgmma() {}

template <typename KernelTraits> __global__ void h100_matmul() {

    int lane = threadIdx.x % 32;
    int raw_warp_idx = threadIdx.x / 32;
    int warp_group = raw_warp_idx / 4;

    bool is_wgmma = warp_group <= KernelTraits::WGMMA_WARP_GROUP_CNT;
    bool is_tma = !is_wgmma;

    alignas(128) extern __shared__ bf16 shmem[];
}

void launch_h100_matmul(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {

    // <--- your code here --->
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

static constexpr size_t kNumOfWarmupIterations = 2;
static constexpr size_t kNumOfOuterIterations = 1;
static constexpr size_t kNumOfInnerIterations = 10;

#define BENCHPRESS(func, flops, ...) \
    do { \
        std::cout << "Running " << #func << " ...\n"; \
        for (size_t i = 0; i < kNumOfWarmupIterations; ++i) { \
            func(__VA_ARGS__); \
        } \
        cudaDeviceSynchronize(); \
        std::vector<float> times(kNumOfOuterIterations); \
        cudaEvent_t start, stop; \
        cudaEventCreate(&start); \
        cudaEventCreate(&stop); \
        for (size_t i = 0; i < kNumOfOuterIterations; ++i) { \
            cudaEventRecord(start); \
            for (size_t j = 0; j < kNumOfInnerIterations; ++j) { \
                func(__VA_ARGS__); \
            } \
            cudaEventRecord(stop); \
            cudaEventSynchronize(stop); \
            float elapsed_time; \
            cudaEventElapsedTime(&elapsed_time, start, stop); \
            times[i] = elapsed_time / kNumOfInnerIterations; \
        } \
        cudaEventDestroy(start); \
        cudaEventDestroy(stop); \
        std::sort(times.begin(), times.end()); \
        float best_time_ms = times[0]; \
        float tflops = (flops * 1e-9) / best_time_ms; \
        std::cout << "  Runtime: " << best_time_ms << " ms" << std::endl; \
        std::cout << "  TFLOP/s: " << tflops << std::endl; \
    } while (0)

void runCublasRef(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float alpha = 1, beta = 0;
    cublasStatus_t status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        M,
        N,
        K,
        &alpha,
        A,
        CUDA_R_16BF,
        K,
        B,
        CUDA_R_16BF,
        N,
        &beta,
        C,
        CUDA_R_16BF,
        M,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS error: " << status << std::endl;
        exit(1);
    }
}

void init_matrix(bf16 *mat, int N) {
    std::default_random_engine generator(0);
    std::normal_distribution<float> distribution(0, 1);
    for (int i = 0; i < N; i++) {
        mat[i] = distribution(generator);
    }
}

bool check_correctness(bf16 *ref, bf16 *test, int N, float tolerance = 0.1f) {
    int mismatches = 0;
    int total = N;
    for (int i = 0; i < N; i++) {
        float ref_val = __bfloat162float(ref[i]);
        float test_val = __bfloat162float(test[i]);
        float diff = std::abs(ref_val - test_val);
        if (diff > tolerance) {
            if (mismatches < 10) { // Print first 10 mismatches
                std::cout << "  Mismatch at index " << i << ": ref=" << ref_val
                          << ", test=" << test_val << ", diff=" << diff << std::endl;
            }
            mismatches++;
        }
    }
    std::cout << "Total mismatches: " << mismatches << " / " << total << " ("
              << (100.0 * mismatches / total) << "%)" << std::endl;
    return mismatches == 0;
}

int main() {

    const int M = 8192, N = 8192, K = 8192;

    bf16 *A = (bf16 *)malloc(sizeof(bf16) * M * K);
    bf16 *B = (bf16 *)malloc(sizeof(bf16) * K * N);
    bf16 *C = (bf16 *)malloc(sizeof(bf16) * M * N);

    init_matrix(A, M * K);
    init_matrix(B, K * N);
    memset(C, 0, sizeof(bf16) * M * N);

    bf16 *dA;
    bf16 *dB;
    bf16 *dC;
    bf16 *dCublas;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(bf16) * M * K));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(bf16) * K * N));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(bf16) * M * N));
    CUDA_CHECK(cudaMalloc(&dCublas, sizeof(bf16) * M * N));

    CUDA_CHECK(cudaMemcpy(dA, A, sizeof(bf16) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, sizeof(bf16) * K * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C, sizeof(bf16) * M * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCublas, C, sizeof(bf16) * M * N, cudaMemcpyHostToDevice));

    std::cout << "M = " << M << ", N = " << N << ", K = " << K << std::endl;

    bf16 *hCublas = (bf16 *)malloc(sizeof(bf16) * M * N);
    bf16 *hOurs = (bf16 *)malloc(sizeof(bf16) * M * N);

    runCublasRef(M, N, K, dA, dB, dCublas);
    launch_h100_matmul(M, N, K, dA, dB, dC);

    CUDA_CHECK(
        cudaMemcpy(hCublas, dCublas, sizeof(bf16) * M * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hOurs, dC, sizeof(bf16) * M * N, cudaMemcpyDeviceToHost));

    bool correct = check_correctness(hCublas, hOurs, M * N, 0.01f);
    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    long flops = 2LL * M * N * K;
    BENCHPRESS(runCublasRef, flops, M, N, K, dA, dB, dCublas);

    BENCHPRESS(launch_h100_matmul, flops, M, N, K, dA, dB, dC);

    free(hCublas);
    free(hOurs);

    return 0;
}
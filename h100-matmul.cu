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

#define UNROLLED_FOR(x) _Pragma("unroll") for (x)

template <typename T> constexpr __host__ __device__ T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

constexpr __host__ __device__ CUtensorMapSwizzle
convert_swizzle_enums(wgmmaSwizzle swizzle) {
    if (swizzle == NO_SWIZZLE) {
        return CU_TENSOR_MAP_SWIZZLE_NONE;
    } else if (swizzle == SWIZZLE_32B) {
        return CU_TENSOR_MAP_SWIZZLE_32B;
    } else if (swizzle == SWIZZLE_64B) {
        return CU_TENSOR_MAP_SWIZZLE_64B;
    } else if (swizzle == SWIZZLE_128B) {
        return CU_TENSOR_MAP_SWIZZLE_128B;
    }
}

struct KernelTraits {

    static constexpr int WGMMA_WARP_GROUP_CNT = 3;
    static constexpr int WGMMA_THREAD_CNT = WGMMA_WARP_GROUP_CNT * 4 * 32;

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
    static constexpr int CORE_MATRIX_N_ELEMENTS = CORE_MATRIX_ROWS * CORE_MATRIX_COLS;

    static constexpr int PHASE_M = WGMMA_M * WGMMA_WARP_GROUP_CNT;
    static constexpr int PHASE_N = WGMMA_N;
    static constexpr int PHASE_K = CORE_MATRIX_COLS;

    static constexpr int WGMMAS_PER_PHASE_PER_WARPGROUP = PHASE_K / WGMMA_K;

    static constexpr int PHASE_CNT = 2;

    static constexpr int BYTES_LOADED_PER_PHASE =
        (PHASE_M * PHASE_K + PHASE_K * PHASE_N) * sizeof(bf16);
    static constexpr int SMEM_SIZE = PHASE_CNT * BYTES_LOADED_PER_PHASE;
    
};

// We use define instead of function definition since __grid_constant__ variables can only be passed to __global__ functions.
#define TMA_LOAD(a_map, b_map, barrier, tidx, sa, sb, m_beg, n_beg, k_beg)                                              \
do {                                                                                                                    \
    int m_end = m_beg + KT::PHASE_M;                                                                                    \
    int n_end = n_beg + KT::PHASE_N;                                                                                    \
    int k_end = k_beg + KT::PHASE_K;                                                                                    \
    for (int m = m_beg + tidx * KT::CORE_MATRIX_ROWS; m < m_end; m += KT::TMA_LOAD_THREAD_CNT * KT::CORE_MATRIX_ROWS) { \
        cp_async_bulk_tensor_2d_global_to_shared(                                                                       \
            sa + (m - m_beg) * KT::CORE_MATRIX_COLS, &a_map, m, k_beg, barrier                                          \
        );                                                                                                              \
    }                                                                                                                   \
    for (int n = n_beg + tidx * KT::CORE_MATRIX_ROWS; n < n_end; n += KT::TMA_LOAD_THREAD_CNT * KT::CORE_MATRIX_ROWS) { \
        cp_async_bulk_tensor_2d_global_to_shared(                                                                       \
            sb + (n - n_beg) * KT::CORE_MATRIX_COLS, &b_map, n, k_beg, barrier                                          \
        );                                                                                                              \
    }                                                                                                                   \
} while (0);

template <typename KT>
__device__ void wgmma(int gidx, bf16 *sa, bf16 *sb, float d[16][8]) {

    sa += KT::PHASE_K * gidx * KT::WGMMA_M;

    async_proxy_fence();
    warpgroup_arrive();

    for (int k = 0; k < KT::PHASE_K; k += KT::WGMMA_K) {
        int stride_byte_offset = KT::PHASE_K * KT::CORE_MATRIX_ROWS * sizeof(bf16);
        uint64_t a_des = make_smem_desc<KT::SWIZZLE_TYPE>(sa + k, 1, stride_byte_offset);
        uint64_t b_des = make_smem_desc<KT::SWIZZLE_TYPE>(sb + k, 1, stride_byte_offset);
        wgmma_n256<1, 1, 1, 0, 0>(a_des, b_des, d);
    }

    wgmma_commit();
}

template <typename KT>
__global__ void h100_matmul(
    int M,
    int N,
    int K,
    __grid_constant__ const CUtensorMap a_map,
    __grid_constant__ const CUtensorMap b_map,
    __grid_constant__ const CUtensorMap c_map) {
    int block_m = blockIdx.x;
    int block_n = blockIdx.y;

    int m_beg = block_m * KT::PHASE_M;
    int n_beg = block_n * KT::PHASE_N;

    int tidx = threadIdx.x;
    int lidx = tidx % 32; // lane id within the warp
    int widx = (tidx / 32) % 4; // Warp id within the warp gorup
    int gidx = (tidx / 32) / 4; // Warp group id

    bool is_wgmma = gidx <= KT::WGMMA_WARP_GROUP_CNT;
    bool is_tma = !is_wgmma;

    alignas(128) extern __shared__ bf16 _smem[];

    bf16 *sa[2], *sb[2], *sc = _smem;

    sa[0] = _smem;
    sb[0] = sa[0] + KT::PHASE_M * KT::PHASE_K;
    sa[1] = sb[0] + KT::PHASE_N * KT::PHASE_K;
    sb[1] = sa[1] + KT::PHASE_M * KT::PHASE_K;

    int phase_bit = 0;

    // Init barrier
    __shared__ alignas(8) uint64_t barrier;
    if (tidx == 0) {
        init_barrier(&barrier, 1);
        expect_bytes_and_arrive(&barrier, KT::BYTES_LOADED_PER_PHASE);
    }
    async_proxy_fence();
    __syncthreads();

    float d[16][8] = {0.0f};
    
    for (int k_beg = 0; k_beg < K; k_beg += KT::PHASE_K, phase_bit ^= 1) {
        if (is_tma) {
            wgmma_wait<1>();
            
            int tma_tidx = tidx - KT::WGMMA_THREAD_CNT;
            TMA_LOAD(
                a_map, b_map, &barrier, tma_tidx, 
                sa[phase_bit], sb[phase_bit],
                m_beg, n_beg, k_beg
            );
        }

        if (is_wgmma) {
            wait(&barrier, phase_bit);
            if (tidx == 0) {
                expect_bytes_and_arrive(&barrier, KT::BYTES_LOADED_PER_PHASE);
            }
    
            wgmma<KT>(gidx, sa[phase_bit], sb[phase_bit], d);
        }
    
    }
    
    if (is_wgmma) {
        wgmma_wait<0>();
        UNROLLED_FOR(int i = 0; i < 16; i++) {
            UNROLLED_FOR(int j = 0; j < 8; j++) {
                int idx = i * 8 + j;
                // Legacy: IGNORE
                // int x = gidx * KT::WGMMA_M + widx * 16 + i + ((idx & 3) ? 8 : 0);
                // int y = (idx >> 2) * 8 + (idx & 1);
                // sc[x * KT::PHASE_N + y] = __float2bfloat16(d[i][j]);

                // This new reg->smem logic writes to sc in core matrix blocks
                int sidx = (gidx * KT::WGMMA_M + widx * 16) * KT::WGMMA_N;
                sidx += (idx & 3) ? KT::CORE_MATRIX_ROWS * KT::WGMMA_N : 0;
                sidx += (idx >> 2) * KT::CORE_MATRIX_N_ELEMENTS;
                sidx += lidx * 2 + (idx & 1);
                sc[sidx] = __float2bfloat16(d[i][j]);
            }
        }
    }
    __syncthreads();
    if (is_wgmma)
        return;

    int m_end = m_beg + KT::PHASE_M;
    int n_end = n_beg + KT::PHASE_N;
    for (int m = m_beg + tidx * KT::CORE_MATRIX_ROWS; m < m_end; m += KT::TMA_LOAD_THREAD_CNT * KT::CORE_MATRIX_ROWS) {
        for (int n = n_beg + tidx * KT::CORE_MATRIX_ROWS; n < n_end; n += KT::TMA_LOAD_THREAD_CNT * KT::CORE_MATRIX_COLS) {
            bf16 *core_matrix = sc + ((m - m_beg) * (KT::WGMMA_N / KT::CORE_MATRIX_COLS) + (n - n_beg)) * KT::CORE_MATRIX_N_ELEMENTS;
            cp_async_bulk_tensor_2d_shared_to_global(&c_map, m, n, core_matrix);
        }
    }
    tma_commit_group();
    tma_wait_until_pending<0>();
}

template<int block_size_x, int block_size_y>
CUtensorMap create_tensor_map(int rows, int cols, bf16* ptr, wgmmaSwizzle swizzle) {
    CUtensorMap map;
    const uint64_t globalDim     [] = {(uint64_t)cols, (uint64_t)rows};
    const uint64_t globalStrides [] = {(uint64_t)cols * sizeof(bf16)};
    const uint32_t boxDim        [] = {block_size_y, block_size_x};
    const uint32_t elementStrides[] = {1, 1};
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        ptr,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        convert_swizzle_enums(swizzle),
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    );

    return map;
}

void launch_h100_matmul(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
    using KT = KernelTraits;

    CUtensorMap a_map = create_tensor_map<KT::CORE_MATRIX_ROWS, KT::CORE_MATRIX_COLS>(M, K, A, KT::SWIZZLE_TYPE);
    CUtensorMap b_map = create_tensor_map<KT::CORE_MATRIX_ROWS, KT::CORE_MATRIX_COLS>(N, K, B, KT::SWIZZLE_TYPE);
    CUtensorMap c_map = create_tensor_map<KT::CORE_MATRIX_ROWS, KT::CORE_MATRIX_COLS>(M, N, C, NO_SWIZZLE);
    
    int block_cnt_m = ceil_div(M, KT::PHASE_M);
    int block_cnt_n = ceil_div(N, KT::PHASE_N);
    dim3 grid(block_cnt_m, block_cnt_n);

    constexpr int MAX_H100_SMEM_SIZE = 227000;
    static_assert(MAX_H100_SMEM_SIZE >= KT::SMEM_SIZE);
    cudaFuncSetAttribute(
        h100_matmul<KT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        KT::SMEM_SIZE
    );

    h100_matmul<KT><<<grid, KT::TOTAL_THREAD_CNT, KT::SMEM_SIZE>>>(M, N, K, a_map, b_map, c_map);
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
    cublasStatus_t status =
        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha,
                     A, CUDA_R_16BF, K, B, CUDA_R_16BF, K, &beta, C,
                     CUDA_R_16BF, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

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

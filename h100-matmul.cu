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

template <typename T> constexpr __host__ __device__ T ceil_div(T a, T b) { return (a + b - 1) / b; }

constexpr __host__ __device__ CUtensorMapSwizzle convert_swizzle_enums(wgmmaSwizzle swizzle) {
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

constexpr __host__ __device__ int get_swizzle_bytes(wgmmaSwizzle swizzle) {
  if (swizzle == NO_SWIZZLE) {
    return 0;
  } else if (swizzle == SWIZZLE_32B) {
    return 32;
  } else if (swizzle == SWIZZLE_64B) {
    return 64;
  } else if (swizzle == SWIZZLE_128B) {
    return 128;
  }
}

struct KernelTraits {

  static constexpr int WGMMA_WARP_GROUP_CNT = 2;
  static constexpr int TMA_WARP_GROUP_CNT = 1;

  static constexpr int WGMMA_THREAD_CNT = WGMMA_WARP_GROUP_CNT * 4 * 32;
  static constexpr int TMA_THREAD_CNT = TMA_WARP_GROUP_CNT * 4 * 32;

  static constexpr int TOTAL_THREAD_CNT = WGMMA_THREAD_CNT + TMA_THREAD_CNT;

  static constexpr int WGMMA_M = 64;
  static constexpr int WGMMA_N = 256;
  static constexpr int WGMMA_K = 16;

  static constexpr wgmmaSwizzle SWIZZLE_TYPE = SWIZZLE_128B;

  static constexpr int CORE_MATRIX_ROWS = 8;
  static constexpr int CORE_MATRIX_COLS = get_swizzle_bytes(SWIZZLE_TYPE) / sizeof(bf16);

  static constexpr int CALCS_PER_WARPGROUP = 1;

  static constexpr int PHASE_M = WGMMA_M * WGMMA_WARP_GROUP_CNT * CALCS_PER_WARPGROUP;
  static constexpr int PHASE_N = WGMMA_N;
  static constexpr int PHASE_K = CORE_MATRIX_COLS;

  static constexpr int PHASE_CNT = 2;

  static constexpr int BYTES_LOADED_PER_PHASE =
      (PHASE_M * PHASE_K + PHASE_K * PHASE_N) * sizeof(bf16);
  static constexpr int SHMEM_NEEDED = PHASE_CNT * BYTES_LOADED_PER_PHASE;

  static constexpr int BLOCK_CNT_M = 16;
  static constexpr int BLOCK_CNT_N = 8;
};

__always_inline
__device__ void tma_into_shmem(
    const CUtensorMap *a_map,
    const CUtensorMap *b_map,
    uint64_t *barrier,
    bf16 *shmem_a,
    bf16 *shmem_b,
    int m_beg,
    int n_beg,
    int k_beg) {

  cp_async_bulk_tensor_2d_global_to_shared(shmem_a, a_map, k_beg, m_beg, barrier);
  cp_async_bulk_tensor_2d_global_to_shared(shmem_b, b_map, k_beg, n_beg, barrier);
}

template <typename KT>
__always_inline
__device__ void wgmma(int warp_group, bf16 *shmem_a, bf16 *shmem_b, float d[KT::CALCS_PER_WARPGROUP][16][8]) {

  warpgroup_arrive();

  UNROLLED_FOR(int m_idx = 0; m_idx < KT::CALCS_PER_WARPGROUP; ++m_idx) {
    UNROLLED_FOR(int k_idx = 0; k_idx < KT::PHASE_K / KT::WGMMA_K; ++k_idx) {

      int stride_byte_offset = KT::PHASE_K * KT::CORE_MATRIX_ROWS * sizeof(bf16);

      bf16 *cur_shmem_a = shmem_a + (warp_group * KT::CALCS_PER_WARPGROUP + m_idx) * KT::WGMMA_M * KT::PHASE_K + k_idx * KT::WGMMA_K;
      bf16 *cur_shmem_b = shmem_b + k_idx * KT::WGMMA_K;

      uint64_t a_des = make_smem_desc<KT::SWIZZLE_TYPE>(cur_shmem_a, 1, stride_byte_offset);
      uint64_t b_des = make_smem_desc<KT::SWIZZLE_TYPE>(cur_shmem_b, 1, stride_byte_offset);

      wgmma_n256<1, 1, 1, 0, 0>(a_des, b_des, d[m_idx]);
    }
  }

  wgmma_commit();
}

template <typename KT>
__always_inline
__device__ void write_back_to_c(
  int M, int N, int m_beg, int n_beg, bf16 *C, int warp_group, int warp_idx, int lane_idx, float d[KT::CALCS_PER_WARPGROUP][16][8]
) {

  UNROLLED_FOR(int m_idx = 0; m_idx < KT::CALCS_PER_WARPGROUP; ++m_idx){
    UNROLLED_FOR(int i = 0; i < 32; ++i) {
      UNROLLED_FOR(int j = 0; j < 2; ++j) {
        UNROLLED_FOR(int k = 0; k < 2; ++k) {

          int idx = k + j * 2 + i * 2 * 2 + m_idx * 2 * 2 * 32;

          int m = (m_beg + (warp_group * KT::CALCS_PER_WARPGROUP + m_idx) * KT::WGMMA_M) + 16 * warp_idx + lane_idx / 4 + j * 8;
          int n = (n_beg) + (lane_idx % 4) * 2 + 8 * i + k;

          if (n < N && m < M) {
            C[n * N + m] = *(d[0][0] + idx);
          }
        }
      }
    }
  }

}

template <typename KT>
__launch_bounds__(KT::TOTAL_THREAD_CNT) __global__ void h100_matmul(
    int M,
    int N,
    int K,
    __grid_constant__ const CUtensorMap a_map,
    __grid_constant__ const CUtensorMap b_map,
    bf16 *C) {

  int BLOCK_M = ceil_div(M, KT::BLOCK_CNT_M);
  if (BLOCK_M % KT::PHASE_M) {
    BLOCK_M += KT::PHASE_M - BLOCK_M % KT::PHASE_M;
  }
  int BLOCK_N = ceil_div(N, KT::BLOCK_CNT_N);
  if (BLOCK_N % KT::PHASE_N) {
    BLOCK_N += KT::PHASE_N - BLOCK_N % KT::PHASE_N;
  }

  int block_m = blockIdx.x % KT::BLOCK_CNT_M;
  int block_n = blockIdx.x / KT::BLOCK_CNT_M;

  int lane_idx = threadIdx.x % 32;
  int raw_warp_idx = threadIdx.x / 32;
  int warp_idx = raw_warp_idx % 4;
  int warp_group = raw_warp_idx / 4;

  bool is_tma = warp_group < KT::TMA_WARP_GROUP_CNT;
  bool first_thread_in_role = is_tma ? threadIdx.x == 0 : threadIdx.x == KT::TMA_THREAD_CNT;

  alignas(128) extern __shared__ bf16 shmem[];

  bf16 *shmem_a[2], *shmem_b[2];

  shmem_a[0] = shmem;
  shmem_b[0] = shmem_a[0] + KT::PHASE_M * KT::PHASE_K;
  shmem_a[1] = shmem_b[0] + KT::PHASE_N * KT::PHASE_K;
  shmem_b[1] = shmem_a[1] + KT::PHASE_M * KT::PHASE_K;

  __shared__ alignas(8) uint64_t tma_tracker;
  __shared__ alignas(8) uint64_t wgmma_tracker;

  if (threadIdx.x == 0) {
    init_barrier(&tma_tracker, 1);
    init_barrier(&wgmma_tracker, KT::WGMMA_THREAD_CNT);
  }

  async_proxy_fence();
  __syncthreads();

  if (is_tma) {

    int phase_bit = 0;
    for (int m_beg = block_m * BLOCK_M; m_beg < (block_m + 1) * BLOCK_M; m_beg += KT::PHASE_M) {
      for (int n_beg = block_n * BLOCK_N; n_beg < (block_n + 1) * BLOCK_N; n_beg += KT::PHASE_N) {
        for (int k_beg = 0; k_beg < K; k_beg += KT::PHASE_K, phase_bit ^= 1) {

          if (first_thread_in_role) {
            expect_bytes_and_arrive(&tma_tracker, KT::BYTES_LOADED_PER_PHASE);
            tma_into_shmem(
                &a_map,
                &b_map,
                &tma_tracker,
                shmem_a[phase_bit],
                shmem_b[phase_bit],
                m_beg,
                n_beg,
                k_beg);
          }
          wait(&wgmma_tracker, phase_bit);
        }
      }
    }

  } else {
    --warp_group;

    int phase_bit = 0;
    for (int m_beg = block_m * BLOCK_M; m_beg < (block_m + 1) * BLOCK_M; m_beg += KT::PHASE_M) {
      for (int n_beg = block_n * BLOCK_N; n_beg < (block_n + 1) * BLOCK_N; n_beg += KT::PHASE_N) {
        float d[KT::CALCS_PER_WARPGROUP][16][8];
        memset(d, 0, sizeof(d));

        for (int k_beg = 0; k_beg < K; k_beg += KT::PHASE_K, phase_bit ^= 1) {
          wait(&tma_tracker, phase_bit);

          wgmma<KT>(warp_group, shmem_a[phase_bit], shmem_b[phase_bit], d);
          wgmma_wait<1>();
          arrive(&wgmma_tracker, 1);
          wgmma_wait<0>();
        }

        write_back_to_c<KT>(M, N, m_beg, n_beg, C, warp_group, warp_idx, lane_idx, d);
      }
    }
  }
}

CUtensorMap create_tensor_map(
    bf16 *array,
    uint64_t global_cols,
    uint64_t global_rows,
    uint32_t box_cols,
    uint32_t box_rows,
    wgmmaSwizzle swizzle_type) {
  CUtensorMap map;
  const uint64_t globalDim[] = {global_cols, global_rows};
  const uint64_t globalStrides[] = {global_cols * sizeof(bf16)};
  const uint32_t boxDim[] = {box_cols, box_rows};
  const uint32_t elementStrides[] = {1, 1};

  CUDA_CHECK(cuTensorMapEncodeTiled(
      &map,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      2,
      array,
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      convert_swizzle_enums(swizzle_type),
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  return map;
}

void launch_h100_matmul(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {

  using KT = KernelTraits;

  CUtensorMap a_map = create_tensor_map(A, K, M, KT::PHASE_K, KT::PHASE_M, KT::SWIZZLE_TYPE);
  CUtensorMap b_map = create_tensor_map(B, K, N, KT::PHASE_K, KT::PHASE_N, KT::SWIZZLE_TYPE);

  cudaFuncSetAttribute(
      h100_matmul<KT>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      KT::SHMEM_NEEDED);

  h100_matmul<KT><<<KT::BLOCK_CNT_M * KT::BLOCK_CNT_N, KT::TOTAL_THREAD_CNT, KT::SHMEM_NEEDED>>>(
      M,
      N,
      K,
      a_map,
      b_map,
      C);

  CUDA_CHECK(cudaGetLastError());
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE. ///
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
        std::cout << "  Mismatch at index " << i << ": ref=" << ref_val << ", test=" << test_val
                  << ", diff=" << diff << std::endl;
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

  CUDA_CHECK(cudaMemcpy(hCublas, dCublas, sizeof(bf16) * M * N, cudaMemcpyDeviceToHost));
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
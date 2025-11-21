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
  if (swizzle == SWIZZLE_32B) {
    return CU_TENSOR_MAP_SWIZZLE_32B;
  } else if (swizzle == SWIZZLE_64B) {
    return CU_TENSOR_MAP_SWIZZLE_64B;
  } else if (swizzle == SWIZZLE_128B) {
    return CU_TENSOR_MAP_SWIZZLE_128B;
  } else {
    return CU_TENSOR_MAP_SWIZZLE_NONE;
  }
}

constexpr __host__ __device__ int get_swizzle_bytes(wgmmaSwizzle swizzle) {
  if (swizzle == SWIZZLE_32B) {
    return 32;
  } else if (swizzle == SWIZZLE_64B) {
    return 64;
  } else if (swizzle == SWIZZLE_128B) {
    return 128;
  } else {
    return 0;
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

  static constexpr int PHASE_M = WGMMA_M * WGMMA_WARP_GROUP_CNT;
  static constexpr int PHASE_N = WGMMA_N;
  static constexpr int PHASE_K = CORE_MATRIX_COLS;

  static constexpr int PHASE_CNT = 3;

  static constexpr int BYTES_LOADED_PER_PHASE =
      (PHASE_M * PHASE_K + PHASE_K * PHASE_N) * sizeof(bf16);
  static constexpr int SHMEM_NEEDED = PHASE_CNT * BYTES_LOADED_PER_PHASE;

  static constexpr int BLOCK_CNT_M = 8;
  static constexpr int BLOCK_CNT_N = 16;

  static constexpr int TMA_REG_CNT = 24;
  static constexpr int WGMMA_REG_CNT = 240;
};

__always_inline __device__ void tma_into_shmem(
    const CUtensorMap *a_map,
    const CUtensorMap *b_map,
    uint64_t *barrier,
    bf16 *shmem_a,
    bf16 *shmem_b,
    int m,
    int n,
    int k) {

  cp_async_bulk_tensor_2d_global_to_shared(shmem_a, a_map, k, m, barrier);
  cp_async_bulk_tensor_2d_global_to_shared(shmem_b, b_map, k, n, barrier);
}

template <typename KT>
__always_inline __device__ void wgmma(
    int warp_group,
    bf16 *shmem_a,
    bf16 *shmem_b,
    float d[16][8],
    bool first_matmul) {

  warpgroup_arrive();
  UNROLLED_FOR(int k_idx = 0; k_idx < KT::PHASE_K / KT::WGMMA_K; ++k_idx) {

    int stride_byte_offset = KT::PHASE_K * KT::CORE_MATRIX_ROWS * sizeof(bf16);

    bf16 *cur_shmem_a = shmem_a + warp_group * KT::WGMMA_M * KT::PHASE_K +
        k_idx * KT::WGMMA_K;
    bf16 *cur_shmem_b = shmem_b + k_idx * KT::WGMMA_K;

    uint64_t a_des = make_smem_desc<KT::SWIZZLE_TYPE>(cur_shmem_a, 1, stride_byte_offset);
    uint64_t b_des = make_smem_desc<KT::SWIZZLE_TYPE>(cur_shmem_b, 1, stride_byte_offset);

    if (first_matmul && k_idx == 0) {
      wgmma_n256<0, 1, 1, 0, 0>(a_des, b_des, d);
    } else {
      wgmma_n256<1, 1, 1, 0, 0>(a_des, b_des, d);
    }
  }

  wgmma_commit();
}

template <typename KT>
__always_inline __device__ void write_back_to_c(
    int M,
    int N,
    int m_beg,
    int n_beg,
    bf16 *C,
    int warp_group,
    int warp_idx,
    int lane_idx,
    float d[16][8]) {

  UNROLLED_FOR(int i = 0; i < 32; ++i) {
    UNROLLED_FOR(int j = 0; j < 2; ++j) {
      UNROLLED_FOR(int k = 0; k < 2; ++k) {

        int idx = k + j * 2 + i * 2 * 2;

        int m = (m_beg + warp_group * KT::WGMMA_M) +
            16 * warp_idx + lane_idx / 4 + j * 8;
        int n = (n_beg) + (lane_idx % 4) * 2 + 8 * i + k;

        if (n < N && m < M) {
          C[n * N + m] = *(d[0] + idx);
        }
      }
    }
  }

}

template <typename KT> class BarrierWrapper {
private:
  uint64_t *tma_trackers;
  uint64_t *wgmma_trackers;

  int phase_idx = 0;
  int phase_parity = 0;

public:
  __always_inline __device__ BarrierWrapper(uint64_t *tma_trackers, uint64_t *wgmma_trackers)
      : tma_trackers(tma_trackers), wgmma_trackers(wgmma_trackers) {}

  __always_inline __device__ void init() {
    UNROLLED_FOR(int phase_idx = 0; phase_idx < KT::PHASE_CNT; ++phase_idx) {
      init_barrier(&tma_trackers[phase_idx], 1);

      init_barrier(&wgmma_trackers[phase_idx], KT::WGMMA_WARP_GROUP_CNT);
      arrive(&wgmma_trackers[phase_idx], KT::WGMMA_WARP_GROUP_CNT);
    }
  }

  __always_inline __device__ void advance_phase() {
    phase_idx++;
    if (phase_idx == KT::PHASE_CNT) {
      phase_idx = 0;
      phase_parity ^= 1;
    }
  }

  __always_inline __device__ void wait_wgmma() { wait(&wgmma_trackers[phase_idx], phase_parity); }
  __always_inline __device__ void arrive_wgmma() { arrive(&wgmma_trackers[phase_idx], 1); }

  __always_inline __device__ void wait_tma() { wait(&tma_trackers[phase_idx], phase_parity); }
  __always_inline __device__ void expect_bytes_and_arrive_tma() {
    expect_bytes_and_arrive(&tma_trackers[phase_idx], KT::BYTES_LOADED_PER_PHASE);
  }
  __always_inline __device__ uint64_t *get_tma() { return &tma_trackers[phase_idx]; }
  __always_inline __device__ int get_phase_idx() { return phase_idx; }
};

template <typename KT>
__always_inline __device__ void
init_shmem_arrays(bf16 *shmem, bf16 *shmem_a[KT::PHASE_CNT], bf16 *shmem_b[KT::PHASE_CNT]) {
  shmem_a[0] = shmem;
  shmem_b[0] = shmem_a[0] + KT::PHASE_M * KT::PHASE_K;
  UNROLLED_FOR(int phase_idx = 1; phase_idx < KT::PHASE_CNT; ++phase_idx) {
    shmem_a[phase_idx] = shmem_b[phase_idx - 1] + KT::PHASE_N * KT::PHASE_K;
    shmem_b[phase_idx] = shmem_a[phase_idx] + KT::PHASE_M * KT::PHASE_K;
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

  int block_idx_m = blockIdx.x % KT::BLOCK_CNT_M;
  int block_idx_n = blockIdx.x / KT::BLOCK_CNT_M;

#define BLOCK_LOOP \
  for (int m = block_idx_m * KT::PHASE_M; m < M; m += KT::BLOCK_CNT_M * KT::PHASE_M) \
    for (int n = block_idx_n * KT::PHASE_N; n < N; n += KT::BLOCK_CNT_N * KT::PHASE_N)

  int lane_idx = threadIdx.x % 32;
  int raw_warp_idx = threadIdx.x / 32;
  int warp_idx = raw_warp_idx % 4;
  int warp_group = raw_warp_idx / 4;

  bool is_tma = warp_group < KT::TMA_WARP_GROUP_CNT;
  bool is_first_thread_in_role = is_tma ? threadIdx.x == 0 : threadIdx.x == KT::TMA_THREAD_CNT;
  bool is_first_in_warpgroup = warp_idx == 0 && lane_idx == 0;

  alignas(128) extern __shared__ bf16 shmem[];
  bf16 *shmem_a[KT::PHASE_CNT], *shmem_b[KT::PHASE_CNT];
  init_shmem_arrays<KT>(shmem, shmem_a, shmem_b);

  __shared__ alignas(8) uint64_t tma_tracker_raw[KT::PHASE_CNT];
  __shared__ alignas(8) uint64_t wgmma_tracker_raw[KT::PHASE_CNT];

  BarrierWrapper<KT> barrier_wrapper{tma_tracker_raw, wgmma_tracker_raw};
  if (threadIdx.x == 0) {
    barrier_wrapper.init();
  }
  async_proxy_fence();
  __syncthreads();

  if (is_tma) {
    warpgroup_reg_dealloc<KT::TMA_REG_CNT>();

    if (is_first_thread_in_role) {
      BLOCK_LOOP {
        for (int k = 0; k < K; k += KT::PHASE_K, barrier_wrapper.advance_phase()) {
          barrier_wrapper.wait_wgmma();
          barrier_wrapper.expect_bytes_and_arrive_tma();
          tma_into_shmem(
              &a_map,
              &b_map,
              barrier_wrapper.get_tma(),
              shmem_a[barrier_wrapper.get_phase_idx()],
              shmem_b[barrier_wrapper.get_phase_idx()],
              m,
              n,
              k);
        }
      }
    }
  } else {
    warpgroup_reg_alloc<KT::WGMMA_REG_CNT>();

    float d[16][8];
    --warp_group;

    BLOCK_LOOP {
      for (int k = 0; k < K; k += KT::PHASE_K, barrier_wrapper.advance_phase()) {
        barrier_wrapper.wait_tma();
        wgmma<KT>(
            warp_group,
            shmem_a[barrier_wrapper.get_phase_idx()],
            shmem_b[barrier_wrapper.get_phase_idx()],
            d,
            k == 0);
        wgmma_wait<0>();
        if (is_first_in_warpgroup) {
          barrier_wrapper.arrive_wgmma();
        }
      }

      write_back_to_c<KT>(M, N, m, n, C, warp_group, warp_idx, lane_idx, d);
    }
  }

#undef BLOCK_LOOP
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
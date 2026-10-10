#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(CUresult err, const char *what) {
  if (err != CUDA_SUCCESS) {
    const char *name = NULL, *msg = NULL;
    cuGetErrorName(err, &name);
    cuGetErrorString(err, &msg);
    fprintf(stderr, "%s failed: %s: %s\n", what, name ? name : "?", msg ? msg : "?");
    exit(1);
  }
}

static CUtensorMap encode_tiled_2d(
    CUdeviceptr ptr,
    uint64_t dim0, uint64_t dim1,
    uint64_t stride1_bytes,
    uint32_t box0, uint32_t box1) {
  CUtensorMap map;
  uint64_t globalDim[2] = {dim0, dim1};
  uint64_t globalStrides[1] = {stride1_bytes};
  uint32_t boxDim[2] = {box0, box1};
  uint32_t elementStrides[2] = {1, 1};
  check(cuTensorMapEncodeTiled(
            &map,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            2,
            (void *)ptr,
            globalDim,
            globalStrides,
            boxDim,
            elementStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled");
  return map;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s MATMUL.cubin DYNAMIC_SMEM_BYTES\n", argv[0]);
    fprintf(stderr, "  emit_matmul_ptx --info FILE.gpu reports the byte count\n");
    return 2;
  }
  check(cuInit(0), "cuInit");
  CUdevice device;
  check(cuDeviceGet(&device, 0), "cuDeviceGet");
  CUcontext ctx;
  check(cuCtxCreate(&ctx, 0, device), "cuCtxCreate");
  CUmodule module;
  check(cuModuleLoad(&module, argv[1]), "cuModuleLoad");
  CUfunction fn;
  check(cuModuleGetFunction(&fn, module, "matmul_tiled"), "cuModuleGetFunction");

  /* Launch contract: hopper_f32 — 128x256x32, stages=3,
     consumers [0,256) + producers [256,288); elect tid=256 for TMA.
     WGMMA descriptor: K-major SW128, LBO=1, SBO=64 (16B units). */
  const int M = 256, K = 256, N = 256;
  const int BM = 128, BN = 256, BK = 32;
  /* Above 48 KiB the kernel's shared window is the dynamic one, so the launch
     sizes it and the opt-in below has to precede the first launch. */
  const int SMEM = atoi(argv[2]);
  const int THREADS = 288;
  if (SMEM > 0) {
    check(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                             SMEM), "cuFuncSetAttribute dynamic smem");
  }

  size_t a_bytes = (size_t)M * K * sizeof(float);
  size_t b_bytes = (size_t)K * N * sizeof(float);
  size_t c_bytes = (size_t)M * N * sizeof(float);
  float *A = (float *)malloc(a_bytes);
  float *B = (float *)malloc(b_bytes);
  /* B_T[n*K+k] = B[k*N+n] — K-contiguous for TMA 128B box0=BK. */
  float *B_T = (float *)malloc(b_bytes);
  float *C = (float *)malloc(c_bytes);
  float *expect = (float *)malloc(c_bytes);
  for (int i = 0; i < M * K; i++) A[i] = (float)((i % 7) + 1) * 0.01f;
  for (int i = 0; i < K * N; i++) B[i] = (float)((i % 5) + 1) * 0.01f;
  for (int k = 0; k < K; k++)
    for (int n = 0; n < N; n++)
      B_T[n * K + k] = B[k * N + n];
  /* TF32-aware reference: round inputs to TF32 then FP32 accumulate. */
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float acc = 0.f;
      for (int k = 0; k < K; k++) {
        float a = A[i * K + k];
        float b = B[k * N + j];
        union { float f; unsigned u; } ua = {.f = a}, ub = {.f = b};
        ua.u &= 0xffffe000u;
        ub.u &= 0xffffe000u;
        acc += ua.f * ub.f;
      }
      expect[i * N + j] = acc;
    }
  }

  CUdeviceptr dA, dBT, dC, dAMap, dBMap;
  check(cuMemAlloc(&dA, a_bytes), "cuMemAlloc A");
  check(cuMemAlloc(&dBT, b_bytes), "cuMemAlloc B_T");
  check(cuMemAlloc(&dC, c_bytes), "cuMemAlloc C");
  check(cuMemAlloc(&dAMap, sizeof(CUtensorMap)), "cuMemAlloc A map");
  check(cuMemAlloc(&dBMap, sizeof(CUtensorMap)), "cuMemAlloc B map");
  check(cuMemcpyHtoD(dA, A, a_bytes), "HtoD A");
  check(cuMemcpyHtoD(dBT, B_T, b_bytes), "HtoD B_T");
  check(cuMemsetD8(dC, 0, c_bytes), "memset C");

  CUtensorMap a_map = encode_tiled_2d(dA, (uint64_t)K, (uint64_t)M,
                                      (uint64_t)K * sizeof(float), BK, BM);
  CUtensorMap b_map = encode_tiled_2d(dBT, (uint64_t)K, (uint64_t)N,
                                      (uint64_t)K * sizeof(float), BK, BN);
  check(cuMemcpyHtoD(dAMap, &a_map, sizeof(a_map)), "HtoD A map");
  check(cuMemcpyHtoD(dBMap, &b_map, sizeof(b_map)), "HtoD B map");

  int m = M, n = N, k = K;
  void *args[] = {&dAMap, &dBMap, &dC, &m, &n, &k};
  check(cuLaunchKernel(fn, N / BN, M / BM, 1, THREADS, 1, 1, SMEM, 0, args, 0), "launch");
  check(cuCtxSynchronize(), "sync");
  check(cuMemcpyDtoH(C, dC, c_bytes), "DtoH C");

  for (int i = 0; i < M * N; i++) {
    if (fabsf(C[i] - expect[i]) > 5e-2f) {
      fprintf(stderr, "mismatch at %d: got %f expect %f\n", i, C[i], expect[i]);
      return 1;
    }
  }
  printf("H100 matmul_tiled TMA+WGMMA TF32 %dx%dx%d passed\n", M, N, K);
  return 0;
}

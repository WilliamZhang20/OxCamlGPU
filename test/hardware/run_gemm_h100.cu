/* Launch one OxCaml Hopper GEMM kernel and check it against a host reference.
 *
 * The element type is a parameter because the kernel's is: the only things
 * that change between the f32 and bf16 kernels are the tensor map's data
 * type, the global stride in bytes, BK, and which rounding the reference
 * applies to the inputs. C is f32 either way.
 */
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

/* Round-to-nearest-even into bf16, kept as raw bits so this file needs no
 * bf16 header and no device code of its own. */
static unsigned short to_bf16(float f) {
  union { float f; unsigned u; } v = {.f = f};
  unsigned lsb = (v.u >> 16) & 1u;
  return (unsigned short)((v.u + 0x7fffu + lsb) >> 16);
}

static float of_bf16(unsigned short h) {
  union { float f; unsigned u; } v;
  v.u = ((unsigned)h) << 16;
  return v.f;
}

/* Input rounding the matrix unit applies, so the reference sees what the
 * hardware saw. TF32 truncates the f32 mantissa to 10 bits. */
static float round_tf32(float f) {
  union { float f; unsigned u; } v = {.f = f};
  v.u &= 0xffffe000u;
  return v.f;
}

static CUtensorMap encode_tiled_2d(
    CUdeviceptr ptr,
    CUtensorMapDataType dtype,
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
            dtype,
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
  if (argc != 7) {
    fprintf(stderr,
            "usage: %s GEMM.cubin DYNAMIC_SMEM_BYTES KERNEL f32|bf16 BK THREADS\n",
            argv[0]);
    fprintf(stderr, "  emit_gemm_ptx --info FILE.gpu reports smem and THREADS\n");
    return 2;
  }
  const char *cubin = argv[1];
  const int SMEM = atoi(argv[2]);
  const char *kernel_name = argv[3];
  const int narrow = strcmp(argv[4], "bf16") == 0;
  const int BK = atoi(argv[5]);
  const int THREADS = atoi(argv[6]);
  if (!narrow && strcmp(argv[4], "f32") != 0) {
    fprintf(stderr, "element must be f32 or bf16\n");
    return 2;
  }
  const size_t ebytes = narrow ? 2 : 4;
  /* A 128-byte TMA/GMMA swizzle period is BK elements wide. */
  if ((size_t)BK * ebytes != 128) {
    fprintf(stderr, "BK %d does not make a 128-byte row at %zu bytes/element\n",
            BK, ebytes);
    return 2;
  }

  check(cuInit(0), "cuInit");
  CUdevice device;
  check(cuDeviceGet(&device, 0), "cuDeviceGet");
  CUcontext ctx;
  check(cuCtxCreate(&ctx, 0, device), "cuCtxCreate");
  CUmodule module;
  check(cuModuleLoad(&module, cubin), "cuModuleLoad");
  CUfunction fn;
  check(cuModuleGetFunction(&fn, module, kernel_name), "cuModuleGetFunction");

  /* Launch contract: BM x BN x BK, consumers [0,BM/64*128) then a producer
     warp; elect tid = consumers issues TMA. Above 48 KiB the kernel's shared
     window is the dynamic one, so the opt-in must precede the first launch. */
  const int M = 256, N = 256;
  const int BM = 128, BN = 256;
  /* Sixteen K-tiles, so the stage ring wraps several times and the barrier
     parity flips in both directions. A K of one ring's worth would pass
     without ever exercising that, and BK differs by width. */
  const int K = 16 * BK;
  if (SMEM > 0) {
    check(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                             SMEM), "cuFuncSetAttribute dynamic smem");
  }

  size_t c_bytes = (size_t)M * N * sizeof(float);
  float *A = (float *)malloc((size_t)M * K * sizeof(float));
  float *B = (float *)malloc((size_t)K * N * sizeof(float));
  float *C = (float *)malloc(c_bytes);
  float *expect = (float *)malloc(c_bytes);
  for (int i = 0; i < M * K; i++) A[i] = (float)((i % 7) + 1) * 0.01f;
  for (int i = 0; i < K * N; i++) B[i] = (float)((i % 5) + 1) * 0.01f;

  /* Device inputs: A row-major M x K and B transposed to N x K, both
     K-contiguous so a TMA box of BK elements is one 128-byte row. */
  size_t a_bytes = (size_t)M * K * ebytes;
  size_t b_bytes = (size_t)K * N * ebytes;
  void *A_dev = malloc(a_bytes);
  void *BT_dev = malloc(b_bytes);
  for (int i = 0; i < M * K; i++) {
    if (narrow) ((unsigned short *)A_dev)[i] = to_bf16(A[i]);
    else ((float *)A_dev)[i] = A[i];
  }
  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      float v = B[k * N + n];
      if (narrow) ((unsigned short *)BT_dev)[n * K + k] = to_bf16(v);
      else ((float *)BT_dev)[n * K + k] = v;
    }
  }

  /* Reference: the matrix unit's own input rounding, then f32 accumulate. */
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float acc = 0.f;
      for (int k = 0; k < K; k++) {
        float a = A[i * K + k], b = B[k * N + j];
        if (narrow) { a = of_bf16(to_bf16(a)); b = of_bf16(to_bf16(b)); }
        else { a = round_tf32(a); b = round_tf32(b); }
        acc += a * b;
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
  check(cuMemcpyHtoD(dA, A_dev, a_bytes), "HtoD A");
  check(cuMemcpyHtoD(dBT, BT_dev, b_bytes), "HtoD B_T");
  check(cuMemsetD8(dC, 0, c_bytes), "memset C");

  CUtensorMapDataType dtype =
      narrow ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  CUtensorMap a_map = encode_tiled_2d(dA, dtype, (uint64_t)K, (uint64_t)M,
                                      (uint64_t)K * ebytes, BK, BM);
  CUtensorMap b_map = encode_tiled_2d(dBT, dtype, (uint64_t)K, (uint64_t)N,
                                      (uint64_t)K * ebytes, BK, BN);
  check(cuMemcpyHtoD(dAMap, &a_map, sizeof(a_map)), "HtoD A map");
  check(cuMemcpyHtoD(dBMap, &b_map, sizeof(b_map)), "HtoD B map");

  int m = M, n = N, k = K;
  void *args[] = {&dAMap, &dBMap, &dC, &m, &n, &k};
  check(cuLaunchKernel(fn, N / BN, M / BM, 1, THREADS, 1, 1, SMEM, 0, args, 0), "launch");
  check(cuCtxSynchronize(), "sync");
  check(cuMemcpyDtoH(C, dC, c_bytes), "DtoH C");

  /* bf16 has 8 mantissa bits, so a K=256 dot product carries more rounding
     than the TF32 one; the tolerance follows the input precision. */
  const float tol = narrow ? 2e-1f : 5e-2f;
  for (int i = 0; i < M * N; i++) {
    if (fabsf(C[i] - expect[i]) > tol) {
      fprintf(stderr, "mismatch at %d: got %f expect %f\n", i, C[i], expect[i]);
      return 1;
    }
  }
  printf("H100 %s TMA+WGMMA %s %dx%dx%d passed\n", kernel_name, argv[4], M, N, K);
  return 0;
}

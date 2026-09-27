#include <cuda.h>
#include <stdio.h>

#define CUDA_CHECK(expr) do { \
  CUresult error = (expr); \
  if (error != CUDA_SUCCESS) { \
    const char *message; \
    cuGetErrorString(error, &message); \
    fprintf(stderr, "CUDA error at line %d: %s\n", __LINE__, message); \
    return 1; \
  } \
} while (0)

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s vector_add.cubin saxpy.cubin\n", argv[0]);
    return 2;
  }
  CUDA_CHECK(cuInit(0));
  CUdevice device;
  CUDA_CHECK(cuDeviceGet(&device, 0));
  CUcontext context;
  CUDA_CHECK(cuCtxCreate(&context, NULL, 0, device));

  float x[4] = {1, 2, 3, 4};
  float y[4] = {10, 20, 30, 40};
  float out[4] = {0};
  CUdeviceptr dx, dy, dout;
  CUDA_CHECK(cuMemAlloc(&dx, sizeof x));
  CUDA_CHECK(cuMemAlloc(&dy, sizeof y));
  CUDA_CHECK(cuMemAlloc(&dout, sizeof out));
  CUDA_CHECK(cuMemcpyHtoD(dx, x, sizeof x));
  CUDA_CHECK(cuMemcpyHtoD(dy, y, sizeof y));

  CUmodule module;
  CUfunction function;
  CUDA_CHECK(cuModuleLoad(&module, argv[1]));
  CUDA_CHECK(cuModuleGetFunction(&function, module, "vector_add"));
  void *vector_args[] = {&dx, &dy, &dout};
  CUDA_CHECK(cuLaunchKernel(function, 1, 1, 1, 4, 1, 1, 0, NULL, vector_args, NULL));
  CUDA_CHECK(cuCtxSynchronize());
  CUDA_CHECK(cuMemcpyDtoH(out, dout, sizeof out));
  for (int i = 0; i < 4; ++i) {
    if (out[i] != x[i] + y[i]) {
      fprintf(stderr, "vector_add mismatch at %d: %g\n", i, out[i]);
      return 3;
    }
  }
  CUDA_CHECK(cuModuleUnload(module));

  float alpha = 2.0f;
  CUDA_CHECK(cuModuleLoad(&module, argv[2]));
  CUDA_CHECK(cuModuleGetFunction(&function, module, "saxpy"));
  void *saxpy_args[] = {&dx, &dy, &alpha};
  CUDA_CHECK(cuLaunchKernel(function, 1, 1, 1, 4, 1, 1, 0, NULL, saxpy_args, NULL));
  CUDA_CHECK(cuCtxSynchronize());
  CUDA_CHECK(cuMemcpyDtoH(y, dy, sizeof y));
  for (int i = 0; i < 4; ++i) {
    float expected = 2.0f * x[i] + (10.0f * (i + 1));
    if (y[i] != expected) {
      fprintf(stderr, "saxpy mismatch at %d: %g, expected %g\n", i, y[i], expected);
      return 4;
    }
  }
  puts("H100 vector_add and saxpy passed");
  return 0;
}

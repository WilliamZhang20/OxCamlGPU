#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <vector>

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
  if (argc != 6) {
    fprintf(stderr, "usage: %s vector_add.cubin saxpy.cubin dot_product.cubin unique_reuse.cubin alias_reuse_aliased.cubin\n", argv[0]);
    return 2;
  }
  CUDA_CHECK(cuInit(0));
  CUdevice device;
  CUDA_CHECK(cuDeviceGet(&device, 0));
  CUcontext context;
  CUDA_CHECK(cuCtxCreate(&context, 0, device));

  float x[4] = {1, 2, 3, 4};
  float y[4] = {10, 20, 30, 40};
  CUdeviceptr dx, dy;
  CUDA_CHECK(cuMemAlloc(&dx, sizeof x));
  CUDA_CHECK(cuMemAlloc(&dy, sizeof y));
  CUDA_CHECK(cuMemcpyHtoD(dx, x, sizeof x));
  CUDA_CHECK(cuMemcpyHtoD(dy, y, sizeof y));

  CUmodule module;
  CUfunction function;
  CUDA_CHECK(cuModuleLoad(&module, argv[1]));
  CUDA_CHECK(cuModuleGetFunction(&function, module, "vector_add"));
  unsigned int vector_n = 1003;
  std::vector<float> vector_x(vector_n), vector_y(vector_n), vector_out(vector_n, 0.0f);
  for (unsigned int i = 0; i < vector_n; ++i) {
    vector_x[i] = (float)i * 0.25f;
    vector_y[i] = (float)(3 * (int)i - 7) * 0.125f;
  }
  CUdeviceptr dvx, dvy, dvz;
  CUDA_CHECK(cuMemAlloc(&dvx, vector_n * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&dvy, vector_n * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&dvz, vector_n * sizeof(float)));
  CUDA_CHECK(cuMemcpyHtoD(dvx, vector_x.data(), vector_n * sizeof(float)));
  CUDA_CHECK(cuMemcpyHtoD(dvy, vector_y.data(), vector_n * sizeof(float)));
  void *vector_args[] = {&dvx, &dvy, &dvz, &vector_n};
  unsigned int vector_threads = 128;
  unsigned int vector_blocks = (vector_n + vector_threads - 1) / vector_threads;
  CUDA_CHECK(cuLaunchKernel(function, vector_blocks, 1, 1, vector_threads, 1, 1, 0, NULL, vector_args, NULL));
  CUDA_CHECK(cuCtxSynchronize());
  CUDA_CHECK(cuMemcpyDtoH(vector_out.data(), dvz, vector_n * sizeof(float)));
  for (unsigned int i = 0; i < vector_n; ++i) {
    if (vector_out[i] != vector_x[i] + vector_y[i]) {
      fprintf(stderr, "vector_add mismatch at %u: %g\n", i, vector_out[i]);
      return 3;
    }
  }
  CUDA_CHECK(cuModuleUnload(module));

  float alpha = 2.0f;
  unsigned int saxpy_n = 4;
  CUDA_CHECK(cuModuleLoad(&module, argv[2]));
  CUDA_CHECK(cuModuleGetFunction(&function, module, "saxpy"));
  void *saxpy_args[] = {&dx, &dy, &alpha, &saxpy_n};
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

  float dot_x[32], dot_y[32], dot_output = 0.0f, expected_dot = 0.0f;
  for (int i = 0; i < 32; ++i) {
    dot_x[i] = (float)(i + 1) * 0.25f;
    dot_y[i] = (float)(2 * i - 7) * 0.5f;
    expected_dot += dot_x[i] * dot_y[i];
  }
  CUdeviceptr ddot_x, ddot_y, ddot_output;
  CUDA_CHECK(cuMemAlloc(&ddot_x, sizeof dot_x));
  CUDA_CHECK(cuMemAlloc(&ddot_y, sizeof dot_y));
  CUDA_CHECK(cuMemAlloc(&ddot_output, sizeof dot_output));
  CUDA_CHECK(cuMemcpyHtoD(ddot_x, dot_x, sizeof dot_x));
  CUDA_CHECK(cuMemcpyHtoD(ddot_y, dot_y, sizeof dot_y));
  CUDA_CHECK(cuModuleLoad(&module, argv[3]));
  CUDA_CHECK(cuModuleGetFunction(&function, module, "dot_product"));
  void *dot_args[] = {&ddot_x, &ddot_y, &ddot_output};
  CUDA_CHECK(cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, NULL, dot_args, NULL));
  CUDA_CHECK(cuCtxSynchronize());
  CUDA_CHECK(cuMemcpyDtoH(&dot_output, ddot_output, sizeof dot_output));
  if (fabsf(dot_output - expected_dot) > 1e-3f) {
    fprintf(stderr, "dot_product mismatch: %g, expected %g\n", dot_output, expected_dot);
    return 5;
  }
  CUDA_CHECK(cuModuleUnload(module));
  puts("H100 vector_add, saxpy, and dot_product passed");

  float aliased_data = 7.0f, aliased_result = 0.0f;
  CUdeviceptr daliased_data, daliased_result;
  CUDA_CHECK(cuMemAlloc(&daliased_data, sizeof aliased_data));
  CUDA_CHECK(cuMemAlloc(&daliased_result, sizeof aliased_result));
  CUDA_CHECK(cuMemcpyHtoD(daliased_data, &aliased_data, sizeof aliased_data));
  CUDA_CHECK(cuModuleLoad(&module, argv[5]));
  CUDA_CHECK(cuModuleGetFunction(&function, module, "alias_reuse_aliased"));
  void *aliased_args[] = {&daliased_data, &daliased_data, &daliased_result};
  CUDA_CHECK(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, NULL, aliased_args, NULL));
  CUDA_CHECK(cuCtxSynchronize());
  CUDA_CHECK(cuMemcpyDtoH(&aliased_result, daliased_result, sizeof aliased_result));
  if (aliased_result != 7.0f) {
    fprintf(stderr, "aliased baseline mismatch: %g, expected 7\n", aliased_result);
    return 6;
  }
  CUDA_CHECK(cuModuleUnload(module));
  puts("H100 aliased_reuse baseline passed");

  float reuse_source = 7.0f, reuse_target = 19.0f, reuse_result = 0.0f;
  CUdeviceptr dreuse_source, dreuse_target, dreuse_result;
  CUDA_CHECK(cuMemAlloc(&dreuse_source, sizeof reuse_source));
  CUDA_CHECK(cuMemAlloc(&dreuse_target, sizeof reuse_target));
  CUDA_CHECK(cuMemAlloc(&dreuse_result, sizeof reuse_result));
  CUDA_CHECK(cuMemcpyHtoD(dreuse_source, &reuse_source, sizeof reuse_source));
  CUDA_CHECK(cuMemcpyHtoD(dreuse_target, &reuse_target, sizeof reuse_target));
  CUDA_CHECK(cuModuleLoad(&module, argv[4]));
  CUDA_CHECK(cuModuleGetFunction(&function, module, "unique_reuse"));
  void *reuse_args[] = {&dreuse_source, &dreuse_target, &dreuse_result};
  CUDA_CHECK(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, NULL, reuse_args, NULL));
  CUDA_CHECK(cuCtxSynchronize());
  CUDA_CHECK(cuMemcpyDtoH(&reuse_result, dreuse_result, sizeof reuse_result));
  if (reuse_result != 14.0f) {
    fprintf(stderr, "unique_reuse mismatch: %g, expected 14\n", reuse_result);
    return 7;
  }
  CUDA_CHECK(cuModuleUnload(module));
  puts("H100 unique_reuse passed with noalias load reuse");
  return 0;
}

#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <stdint.h>

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
  if (argc != 7) {
    fprintf(stderr, "usage: %s vector_add.cubin saxpy.cubin dot_product.cubin unique_reuse.cubin alias_reuse_aliased.cubin control_flow_dir\n", argv[0]);
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

  // Direct-IR branch test validates lane-varying joins before source fixtures.
  std::string directory = argv[6];
  CUDA_CHECK(cuModuleLoad(&module, (directory + "/branch_ir.cubin").c_str()));
  CUDA_CHECK(cuModuleGetFunction(&function, module, "branch_ir"));
  void *branch_args[] = {&dvz};
  CUDA_CHECK(cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, NULL, branch_args, NULL));
  CUDA_CHECK(cuCtxSynchronize());
  CUDA_CHECK(cuMemcpyDtoH(vector_out.data(), dvz, 32 * sizeof(float)));
  for (int i=0;i<32;++i) if (vector_out[i] != (i<16 ? 1.0f : 2.0f)) return 8;
  CUDA_CHECK(cuModuleUnload(module));

  for (unsigned i=0;i<vector_n;++i) vector_x[i]=(float)((int)(i%7)-3);
  vector_x[0]=NAN; vector_x[1]=-0.0f; vector_x[2]=0.0f;
  CUDA_CHECK(cuMemcpyHtoD(dvx, vector_x.data(), vector_n*sizeof(float)));
  for (const char *name : {"guarded_saxpy", "short_circuit", "numeric", "float_compare", "uniform_branch"}) {
    CUDA_CHECK(cuModuleLoad(&module, (directory + "/" + name + ".cubin").c_str()));
    CUDA_CHECK(cuModuleGetFunction(&function, module, name));
    for (int32_t n : {0,1,31,32,33,1003}) {
      for (uint32_t gate : {0u,1u}) {
        std::fill(vector_out.begin(),vector_out.end(),123.0f);
        CUDA_CHECK(cuMemcpyHtoD(dvz,vector_out.data(),vector_n*sizeof(float)));
        // The zero-work guard and false short-circuit must protect a null input.
        CUdeviceptr input=(n==0 || (std::string(name)=="short_circuit" && !gate)) ? 0 : dvx;
        void *guarded[] = {&input,&dvz,&alpha,&n};
        void *short_args[] = {&input,&dvz,&gate,&n};
        void *numeric_args[] = {&dvz,&n};
        void *float_args[] = {&input,&dvz,&n};
        void *uniform_args[] = {&dvz,&gate,&n};
        std::string which=name;
        void **arguments=which=="guarded_saxpy" ? guarded : which=="short_circuit" ? short_args :
          which=="numeric" ? numeric_args : which=="float_compare" ? float_args : uniform_args;
        CUDA_CHECK(cuLaunchKernel(function, 8,1,1,128,1,1,0,NULL,arguments,NULL));
        CUDA_CHECK(cuCtxSynchronize());
        CUDA_CHECK(cuMemcpyDtoH(vector_out.data(),dvz,vector_n*sizeof(float)));
        for (unsigned i=0;i<vector_n;++i) {
          float expected=123.0f, v=vector_x[i];
          if (i<(unsigned)n) {
            if (which=="guarded_saxpy") expected=123.0f+(v>0 ? alpha*v : 0.0f);
            else if (which=="short_circuit") expected=!gate ? 7.0f : v>0 ? 99.0f : 88.0f;
            else if (which=="numeric") expected=1.0f;
            else if (which=="float_compare") expected=isnan(v) ? 3.0f : v==0 ? 2.0f : v<0 ? 1.0f : 0.0f;
            else expected=gate ? 9.0f : 4.0f;
          }
          if (vector_out[i]!=expected) {
            fprintf(stderr,"%s n=%d gate=%u i=%u got=%g expected=%g\n",name,n,gate,i,vector_out[i],expected);
            return 9;
          }
        }
      }
    }
    CUDA_CHECK(cuModuleUnload(module));
  }
  CUDA_CHECK(cuModuleLoad(&module, (directory + "/joined_reduction.cubin").c_str()));
  CUDA_CHECK(cuModuleGetFunction(&function, module, "joined_reduction"));
  void *joined_args[] = {&dvz};
  CUDA_CHECK(cuLaunchKernel(function,1,1,1,32,1,1,0,NULL,joined_args,NULL));
  CUDA_CHECK(cuCtxSynchronize());
  float joined_result=0;
  CUDA_CHECK(cuMemcpyDtoH(&joined_result,dvz,sizeof joined_result));
  if (joined_result!=48.0f) { fprintf(stderr,"reduction after join failed: %g\n",joined_result); return 10; }
  CUDA_CHECK(cuModuleUnload(module));
  CUDA_CHECK(cuModuleLoad(&module, (directory + "/rounding.cubin").c_str()));
  CUDA_CHECK(cuModuleGetFunction(&function, module, "rounding"));
  float almost_one=1.00000011920928955078125f;
  void *rounding_args[] = {&dvz,&almost_one};
  CUDA_CHECK(cuLaunchKernel(function,1,1,1,1,1,1,0,NULL,rounding_args,NULL));
  CUDA_CHECK(cuCtxSynchronize());
  float rounded_result=0;
  CUDA_CHECK(cuMemcpyDtoH(&rounded_result,dvz,sizeof rounded_result));
  if (rounded_result!=0.0f) { fprintf(stderr,"f32 contraction changed rounding: %g\n",rounded_result); return 11; }
  CUDA_CHECK(cuModuleUnload(module));
  puts("H100 branches, guarded SAXPY, short circuit, integer wrapping, NaN/signed-zero comparisons passed");
  return 0;
}

#include <cuda.h>
#include <cmath>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call) do { \
  CUresult cuda_status = (call); \
  if (cuda_status != CUDA_SUCCESS) { \
    const char *message = nullptr; \
    cuGetErrorString(cuda_status, &message); \
    std::fprintf(stderr, "CUDA error at line %d: %s\n", __LINE__, message); \
    return 1; \
  } \
} while (0)

static int run_multiply(CUcontext context, const char *path, const char *name,
                        const std::vector<float>& a, const std::vector<float>& b) {
  CUmodule module;
  CUfunction function;
  CUDA_CHECK(cuModuleLoad(&module, path));
  CUDA_CHECK(cuModuleGetFunction(&function, module, name));
  CUdeviceptr da, db, dc;
  size_t bytes=a.size()*sizeof(float);
  CUDA_CHECK(cuMemAlloc(&da, bytes));
  CUDA_CHECK(cuMemAlloc(&db, bytes));
  CUDA_CHECK(cuMemAlloc(&dc, bytes));
  CUDA_CHECK(cuMemcpyHtoD(da, a.data(), bytes));
  CUDA_CHECK(cuMemcpyHtoD(db, b.data(), bytes));
  void *args[]={&da,&db,&dc};
  CUDA_CHECK(cuLaunchKernel(function,1,1,1,32,1,1,0,nullptr,args,nullptr));
  CUDA_CHECK(cuCtxSynchronize());
  std::vector<float> output(a.size());
  CUDA_CHECK(cuMemcpyDtoH(output.data(),dc,bytes));
  for (size_t i=0;i<a.size();++i) {
    float expected=a[i]*b[i];
    if (output[i]!=expected) {
      std::fprintf(stderr,"%s mismatch at %zu: got %.9g expected %.9g\n",name,i,output[i],expected);
      return 2;
    }
  }
  CUDA_CHECK(cuMemFree(da));
  CUDA_CHECK(cuMemFree(db));
  CUDA_CHECK(cuMemFree(dc));
  CUDA_CHECK(cuModuleUnload(module));
  std::printf("H100 %s passed\n",name);
  (void)context;
  return 0;
}

int main(int argc, char **argv) {
  if (argc!=4) {
    std::fprintf(stderr,"usage: %s mul64_interleaved.cubin mul64_blocked.cubin dot64_interleaved.cubin\n",argv[0]);
    return 2;
  }
  CUDA_CHECK(cuInit(0));
  CUdevice device;
  CUDA_CHECK(cuDeviceGet(&device,0));
  CUcontext context;
  CUDA_CHECK(cuCtxCreate(&context,0,device));
  std::vector<float> a(64),b(64);
  for (int i=0;i<64;++i) { a[i]=float((i%9)-4); b[i]=float((i%5)-2); }
  if (run_multiply(context,argv[1],"mul64_interleaved",a,b) ||
      run_multiply(context,argv[2],"mul64_blocked",a,b)) return 3;

  CUmodule module;
  CUfunction function;
  CUDA_CHECK(cuModuleLoad(&module,argv[3]));
  CUDA_CHECK(cuModuleGetFunction(&function,module,"dot64_interleaved"));
  CUdeviceptr da,db,dout;
  CUDA_CHECK(cuMemAlloc(&da,a.size()*sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&db,b.size()*sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&dout,sizeof(float)));
  CUDA_CHECK(cuMemcpyHtoD(da,a.data(),a.size()*sizeof(float)));
  CUDA_CHECK(cuMemcpyHtoD(db,b.data(),b.size()*sizeof(float)));
  void *dot_args[]={&da,&db,&dout};
  CUDA_CHECK(cuLaunchKernel(function,1,1,1,32,1,1,0,nullptr,dot_args,nullptr));
  CUDA_CHECK(cuCtxSynchronize());
  float result=0,expected=0;
  CUDA_CHECK(cuMemcpyDtoH(&result,dout,sizeof(float)));
  for (int i=0;i<64;++i) expected+=a[i]*b[i];
  if (result!=expected) {
    std::fprintf(stderr,"dot64 mismatch: got %.9g expected %.9g\n",result,expected);
    return 4;
  }
  puts("H100 dot64_interleaved passed");
  CUDA_CHECK(cuMemFree(da));
  CUDA_CHECK(cuMemFree(db));
  CUDA_CHECK(cuMemFree(dout));
  CUDA_CHECK(cuModuleUnload(module));
  CUDA_CHECK(cuCtxDestroy(context));
  return 0;
}

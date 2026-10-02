#!/usr/bin/env python3
"""Launch-latency comparison for the OxGPU kernels and PyTorch references."""
import argparse
import ctypes
import ctypes.util
import statistics
import sys

import torch


def checked(result, operation):
    if result != 0:
        raise RuntimeError(f"{operation} failed with CUDA result {result}")


def driver_api(device_index):
    cuda = ctypes.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cuda.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
    cuda.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    checked(cuda.cuInit(0), "cuInit")
    device = ctypes.c_int()
    checked(cuda.cuDeviceGet(ctypes.byref(device), device_index), "cuDeviceGet")
    context = ctypes.c_void_p()
    checked(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device), "cuDevicePrimaryCtxRetain")
    checked(cuda.cuCtxSetCurrent(context), "cuCtxSetCurrent")
    current_context = ctypes.c_void_p()
    checked(cuda.cuCtxGetCurrent(ctypes.byref(current_context)), "cuCtxGetCurrent")
    if not current_context.value:
        raise RuntimeError("PyTorch has no current CUDA context")
    return cuda


def load_kernel(cuda, cubin, name):
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    checked(cuda.cuModuleLoad(ctypes.byref(module), cubin.encode()), "cuModuleLoad")
    checked(cuda.cuModuleGetFunction(ctypes.byref(function), module, name.encode()), "cuModuleGetFunction")
    return module, function


def launch(cuda, function, arguments, threads=128, blocks=1, stream=0):
    values = [ctypes.c_uint64(int(arg)) if isinstance(arg, int) else arg for arg in arguments]
    params = (ctypes.c_void_p * len(values))(*(ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in values))
    checked(cuda.cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0,
                                ctypes.c_void_p(stream), params, None), "cuLaunchKernel")


def timed_ms(operation, warmup, iterations):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(5):
        start.record()
        for _ in range(iterations):
            operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vector_add_cubin")
    parser.add_argument("saxpy_cubin")
    parser.add_argument("dot_product_cubin")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    torch.cuda.init()
    cuda = driver_api(torch.cuda.current_device())
    stream = torch.cuda.current_stream().cuda_stream
    n_elements = 1003
    vx, vy, vz = [torch.arange(n_elements, device="cuda", dtype=torch.float32) for _ in range(3)]
    sx, sy = [torch.ones(n_elements, device="cuda", dtype=torch.float32) for _ in range(2)]
    dx = torch.arange(1, 33, device="cuda", dtype=torch.float32) * 0.25
    dy = torch.arange(32, device="cuda", dtype=torch.float32) * 0.5 - 3.5
    dot_out = torch.zeros((), device="cuda", dtype=torch.float32)
    vector_module, vector = load_kernel(cuda, args.vector_add_cubin, "vector_add")
    saxpy_module, saxpy = load_kernel(cuda, args.saxpy_cubin, "saxpy")
    dot_module, dot = load_kernel(cuda, args.dot_product_cubin, "dot_product")
    alpha = ctypes.c_float(1e-6)
    if not 0 <= n_elements <= 2**31 - 1:
        raise ValueError("GPU native-int extent must fit signed i32")
    n = ctypes.c_int32(n_elements)
    saxpy_arguments = (ctypes.c_uint64(sx.data_ptr()), ctypes.c_uint64(sy.data_ptr()), alpha)
    saxpy_arguments += (n,)
    vector_arguments = (ctypes.c_uint64(vx.data_ptr()), ctypes.c_uint64(vy.data_ptr()), ctypes.c_uint64(vz.data_ptr()), n)

    vector_threads = 128
    vector_blocks = (n_elements + vector_threads - 1) // vector_threads
    if vector_blocks * vector_threads - 1 > 2**31 - 1:
        raise ValueError("launch indices exceed the supported signed i32 range")
    launch(cuda, vector, vector_arguments, threads=vector_threads, blocks=vector_blocks, stream=stream)
    torch.cuda.synchronize()
    if not torch.allclose(vz, vx + vy):
        raise RuntimeError("OxGPU vector_add result did not match PyTorch")

    launch(cuda, saxpy, saxpy_arguments, threads=vector_threads, blocks=vector_blocks, stream=stream)
    torch.cuda.synchronize()
    expected_saxpy = torch.ones_like(sy) + sx * 1e-6
    if not torch.allclose(sy, expected_saxpy):
        raise RuntimeError("OxGPU saxpy result did not match PyTorch")
    dot_arguments = (ctypes.c_uint64(dx.data_ptr()), ctypes.c_uint64(dy.data_ptr()), ctypes.c_uint64(dot_out.data_ptr()))
    launch(cuda, dot, dot_arguments, threads=32, stream=stream)
    torch.cuda.synchronize()
    expected_dot = torch.dot(dx, dy)
    if not torch.allclose(dot_out, expected_dot):
        raise RuntimeError("OxGPU dot_product result did not match torch.dot")

    vector_results = {
        "OxGPU vector_add": timed_ms(lambda: launch(cuda, vector, vector_arguments, threads=vector_threads, blocks=vector_blocks, stream=stream), args.warmup, args.iterations),
        "PyTorch add(out=)": timed_ms(lambda: torch.add(vx, vy, out=vz), args.warmup, args.iterations),
    }
    saxpy_results = {
        "OxGPU saxpy": timed_ms(lambda: launch(cuda, saxpy, saxpy_arguments, threads=vector_threads, blocks=vector_blocks, stream=stream), args.warmup, args.iterations),
        "PyTorch add_(alpha=)": timed_ms(lambda: torch.add(sy, sx, alpha=1e-6, out=sy), args.warmup, args.iterations),
    }
    dot_results = {
        "OxGPU dot_product": timed_ms(lambda: launch(cuda, dot, dot_arguments, threads=32, stream=stream), args.warmup, args.iterations),
        "PyTorch torch.dot (cuBLAS)": timed_ms(lambda: torch.dot(dx, dy, out=dot_out), args.warmup, args.iterations),
    }
    print("Median CUDA-event time per launch (microseconds; vector kernels N=1003, dot N=32):")
    for section, rows in (("vector_add", vector_results), ("saxpy", saxpy_results)):
        print(f"\n{section}")
        for name, micros in rows.items():
            print(f"  {name:24s} {micros:9.3f}")
    print("\ndot_product (N=32)")
    for name, micros in dot_results.items():
        print(f"  {name:24s} {micros:9.3f}")
    checked(cuda.cuModuleUnload(vector_module), "cuModuleUnload vector_add")
    checked(cuda.cuModuleUnload(saxpy_module), "cuModuleUnload saxpy")
    checked(cuda.cuModuleUnload(dot_module), "cuModuleUnload dot_product")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)

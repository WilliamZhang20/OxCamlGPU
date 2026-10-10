#!/usr/bin/env python3
"""OxGPU vs PyTorch/cuBLAS benchmarks (Tilus hopper_matmul timing protocol).

Protocol follows NVIDIA/tilus examples/hopper_matmul/benchmark.py:
  - CUDA-event timing (median of `repeat` iterations)
  - L2 flushed before each measured iteration (2 × device L2, ≥128 MiB)
  - Cooldown before each engine's measurement window (default 3s like Tilus)
  - Correctness checked before timing
  - Matmul reported as latency (ms) and TFLOPS vs torch.matmul (cuBLAS)
  - Default matmul sizes: 4096³ and 8192³ (not launch-noise microkernels)

TF32-on GEMM: OxGPU WGMMA (f32.tf32.tf32) vs torch.matmul with TF32 enabled
so %cuBLAS is an honest tensor-core race on Hopper.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import statistics
import sys
import time
from functools import lru_cache
from typing import Any, Callable, Sequence

import torch


WARMUP = 5
REPEAT = 30
COOLDOWN_S = 3.0

# Launch contract of the OxCaml GEMM kernel (examples/kernels/gemm.ml);
# emit_gemm_ptx --info reports threads and the dynamic shared window.
# Shared memory is static in the cubin; dynamic shared bytes at launch are 0.
# Default: bm=128,bn=256,bk=32,stages=3,producers=32 → 256 consumers + 32 TMA
# (consumers occupy tid [0,256); producers [256,288)).
MATMUL_BM = 128
MATMUL_BN = 256
MATMUL_BK = 32
MATMUL_THREADS = 288
# Dynamic shared memory is a compiled fact: emit_gemm_ptx --info reports it.
MATMUL_SMEM = 0
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8

# Driver enums for cuTensorMapEncodeTiled (cuda.h).
CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 7
CU_TENSOR_MAP_INTERLEAVE_NONE = 0
CU_TENSOR_MAP_SWIZZLE_NONE = 0
CU_TENSOR_MAP_SWIZZLE_32B = 1
CU_TENSOR_MAP_SWIZZLE_64B = 2
CU_TENSOR_MAP_SWIZZLE_128B = 3
CU_TENSOR_MAP_L2_PROMOTION_L2_128B = 2
CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0
CU_TENSOR_MAP_NUM_QWORDS = 16


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
    cuda.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
    cuda.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    cuda.cuMemcpyHtoD.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    cuda.cuMemcpyHtoD.restype = ctypes.c_int
    cuda.cuTensorMapEncodeTiled.argtypes = [
        ctypes.c_void_p,  # CUtensorMap*
        ctypes.c_int,  # data type
        ctypes.c_uint,  # rank
        ctypes.c_void_p,  # global address
        ctypes.c_void_p,  # globalDim
        ctypes.c_void_p,  # globalStrides
        ctypes.c_void_p,  # boxDim
        ctypes.c_void_p,  # elementStrides
        ctypes.c_int,  # interleave
        ctypes.c_int,  # swizzle
        ctypes.c_int,  # l2Promotion
        ctypes.c_int,  # oobFill
    ]
    cuda.cuTensorMapEncodeTiled.restype = ctypes.c_int
    cuda.cuLaunchKernel.argtypes = (
        [ctypes.c_void_p] + [ctypes.c_uint] * 7 +
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p])
    checked(cuda.cuInit(0), "cuInit")
    device = ctypes.c_int()
    checked(cuda.cuDeviceGet(ctypes.byref(device), device_index), "cuDeviceGet")
    context = ctypes.c_void_p()
    checked(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device),
            "cuDevicePrimaryCtxRetain")
    checked(cuda.cuCtxSetCurrent(context), "cuCtxSetCurrent")
    current_context = ctypes.c_void_p()
    checked(cuda.cuCtxGetCurrent(ctypes.byref(current_context)), "cuCtxGetCurrent")
    if not current_context.value:
        raise RuntimeError("PyTorch has no current CUDA context")
    return cuda


class CUtensorMap(ctypes.Structure):
    _fields_ = [("opaque", ctypes.c_uint64 * CU_TENSOR_MAP_NUM_QWORDS)]


def encode_tensor_map_2d(cuda, device_ptr: int, dim0: int, dim1: int,
                         stride1_bytes: int, box0: int, box1: int) -> CUtensorMap:
    tensor_map = CUtensorMap()
    global_dim = (ctypes.c_uint64 * 2)(dim0, dim1)
    global_strides = (ctypes.c_uint64 * 1)(stride1_bytes)
    box_dim = (ctypes.c_uint32 * 2)(box0, box1)
    element_strides = (ctypes.c_uint32 * 2)(1, 1)
    checked(cuda.cuTensorMapEncodeTiled(
        ctypes.byref(tensor_map),
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,
        ctypes.c_void_p(device_ptr),
        global_dim,
        global_strides,
        box_dim,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "cuTensorMapEncodeTiled")
    return tensor_map


def alloc_device_tensor_map(_cuda, host_map: CUtensorMap):
    """Copy a host CUtensorMap into a torch CUDA buffer (PyTorch memcpy/context)."""
    nbytes = ctypes.sizeof(CUtensorMap)
    cpu = torch.empty(nbytes, dtype=torch.uint8)
    ctypes.memmove(cpu.data_ptr(), ctypes.byref(host_map), nbytes)
    buf = cpu.cuda()
    return ctypes.c_uint64(buf.data_ptr()), buf


def load_kernel(cuda, cubin, name):
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    checked(cuda.cuModuleLoad(ctypes.byref(module), cubin.encode()), "cuModuleLoad")
    checked(cuda.cuModuleGetFunction(ctypes.byref(function), module, name.encode()),
            "cuModuleGetFunction")
    return module, function


def launch(cuda, function, arguments, threads=128, blocks=1, stream=0, shared_bytes=0):
    values = [ctypes.c_uint64(int(arg)) if isinstance(arg, int) else arg for arg in arguments]
    params = (ctypes.c_void_p * len(values))(
        *(ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in values))
    checked(cuda.cuLaunchKernel(
        function, blocks, 1, 1, threads, 1, 1, shared_bytes,
        ctypes.c_void_p(stream), params, None), "cuLaunchKernel")


@lru_cache(maxsize=None)
def _l2_clear_nbytes(device_index: int) -> int:
    floor = 128 * 1024 * 1024
    try:
        l2_size = int(torch.cuda.get_device_properties(device_index).L2_cache_size)
    except Exception:
        l2_size = 0
    return max(2 * l2_size, floor)


def benchmark_func(
    run_func: Callable[[], Any],
    warmup: int = WARMUP,
    repeat: int = REPEAT,
    clear_l2_cache: bool = True,
) -> float:
    """Median CUDA-event latency in milliseconds (Tilus benchmark_func protocol)."""
    device = torch.cuda.current_device()
    num_bytes = _l2_clear_nbytes(device)
    memory_slab = torch.empty(num_bytes, dtype=torch.int8, device="cuda")
    assert repeat >= 1
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2 * (repeat + warmup))]
    for event in events:
        event.record()
    memory_slab.zero_()
    torch.cuda.synchronize()
    for i in range(warmup + repeat):
        if clear_l2_cache:
            memory_slab.zero_()
        events[i * 2].record()
        run_func()
        events[i * 2 + 1].record()
    torch.cuda.synchronize()
    results = [
        events[i * 2].elapsed_time(events[i * 2 + 1])
        for i in range(warmup, warmup + repeat)]
    return float(statistics.median(results))


def tflops(ms: float, m: int, n: int, k: int) -> float:
    return 2.0 * m * n * k / ms * 1e-9


def configure_tf32_matmul():
    """Enable TF32 so cuBLAS uses Hopper tensor cores (matches WGMMA tf32 inputs)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def bench_matmul_size(
    cuda,
    matmul_fn,
    stream,
    stream_handle,
    m_size: int,
    n_size: int,
    k_size: int,
    warmup: int,
    repeat: int,
    cooldown: float,
    bm: int = MATMUL_BM,
    bn: int = MATMUL_BN,
    bk: int = MATMUL_BK,
    threads: int = MATMUL_THREADS,
    smem: int = MATMUL_SMEM,
):
    if m_size % bm or n_size % bn or k_size % bk:
        raise ValueError(
            f"matmul size {m_size}x{n_size}x{k_size} must be divisible by "
            f"CTA {bm}x{bn} and BK {bk}")
    scale = k_size ** 0.25
    a = torch.randn(m_size, k_size, device="cuda", dtype=torch.float32) / scale
    b = torch.randn(k_size, n_size, device="cuda", dtype=torch.float32) / scale
    c_ox = torch.empty(m_size, n_size, device="cuda", dtype=torch.float32)
    c_ref = torch.empty(m_size, n_size, device="cuda", dtype=torch.float32)

    # A[M,K] → TMA (K,M) box (BK,BM); B as K-contig via B^T[N,K] → TMA (K,N) box (BK,BN).
    a_map = encode_tensor_map_2d(
        cuda, a.data_ptr(), k_size, m_size, k_size * 4, bk, bm)
    b_k_major = b.T.contiguous()
    b_map = encode_tensor_map_2d(
        cuda, b_k_major.data_ptr(), k_size, n_size, k_size * 4, bk, bn)
    d_a_map, a_map_buf = alloc_device_tensor_map(cuda, a_map)
    d_b_map, b_map_buf = alloc_device_tensor_map(cuda, b_map)
    c_ptr = ctypes.c_uint64(c_ox.data_ptr())
    m_arg = ctypes.c_int32(m_size)
    n_arg = ctypes.c_int32(n_size)
    k_arg = ctypes.c_int32(k_size)
    grid_x = n_size // bn
    grid_y = m_size // bm

    def launch_mm():
        c_ptr.value = c_ox.data_ptr()
        params = (ctypes.c_void_p * 6)(
            ctypes.cast(ctypes.byref(d_a_map), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(d_b_map), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(m_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(n_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(k_arg), ctypes.c_void_p))
        checked(cuda.cuLaunchKernel(
            matmul_fn, grid_x, grid_y, 1, threads, 1, 1, smem,
            ctypes.c_void_p(stream_handle), params, None), "cuLaunchKernel matmul")

    launch_mm()
    stream.synchronize()
    torch.matmul(a, b, out=c_ref)
    stream.synchronize()
    # TF32 WGMMA vs TF32 cuBLAS: allow modest absolute error from rounding paths.
    if not torch.allclose(c_ox, c_ref, atol=5e-2, rtol=1e-2):
        max_err = (c_ox - c_ref).abs().max().item()
        raise RuntimeError(
            f"OxGPU matmul mismatch vs torch.matmul at {m_size}x{n_size}x{k_size} "
            f"(max abs err={max_err})")

    time.sleep(cooldown)
    ox_ms = benchmark_func(launch_mm, warmup=warmup, repeat=repeat)
    time.sleep(cooldown)
    cublas_ms = benchmark_func(
        lambda: torch.matmul(a, b, out=c_ref), warmup=warmup, repeat=repeat)
    # Keep map buffers and K-major B alive through timing.
    _ = (a_map_buf, b_map_buf, b_k_major)

    ox_tf = tflops(ox_ms, m_size, n_size, k_size)
    cublas_tf = tflops(cublas_ms, m_size, n_size, k_size)
    pct = 100.0 * ox_tf / cublas_tf if cublas_tf > 0 else float("nan")
    print(f"\nmatmul M={m_size} N={n_size} K={k_size} (TF32 on), "
          f"CTA {bm}x{bn}, grid ({grid_x},{grid_y}), "
          f"threads={threads}, smem={smem}")
    print(f"  {'version':24s} {'latency (ms)':>12s} {'tflops':>10s} {'% cublas':>10s}")
    print(f"  {'OxGPU IR→PTX tiled':24s} {ox_ms:12.4f} {ox_tf:10.2f} {pct:10.1f}")
    print(f"  {'torch.matmul (cuBLAS)':24s} {cublas_ms:12.4f} {cublas_tf:10.2f} {100.0:10.1f}")
    return pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vector_add_cubin", nargs="?", default=None)
    parser.add_argument("saxpy_cubin", nargs="?", default=None)
    parser.add_argument("dot_product_cubin", nargs="?", default=None)
    parser.add_argument("--matmul-cubin", default=None,
                        help="Optional OxGPU matmul cubin (entry gemm_fast)")
    parser.add_argument("--matmul-name", default="gemm_fast",
                        help="Matmul kernel symbol inside the cubin")
    parser.add_argument("--matmul-sizes", nargs="+", type=int, default=[4096, 8192],
                        help="Cube problem sizes to bench (default: 4096 8192)")
    parser.add_argument("--matmul-size", nargs=3, type=int, default=None,
                        metavar=("M", "N", "K"),
                        help="Single rectangular problem (overrides --matmul-sizes)")
    parser.add_argument("--matmul-bm", type=int, default=MATMUL_BM)
    parser.add_argument("--matmul-bn", type=int, default=MATMUL_BN)
    parser.add_argument("--matmul-bk", type=int, default=MATMUL_BK)
    parser.add_argument("--matmul-threads", type=int, default=MATMUL_THREADS)
    parser.add_argument("--matmul-smem", type=int, default=MATMUL_SMEM,
                        help="Dynamic shared memory bytes the kernel needs "
                             "(emit_gemm_ptx --info)")
    parser.add_argument("--skip-micro", action="store_true",
                        help="Skip vector_add/saxpy/dot microbenchmarks")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--repeat", type=int, default=REPEAT)
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_S)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    torch.cuda.init()
    configure_tf32_matmul()
    device = torch.cuda.current_device()
    cuda = driver_api(device)
    stream = torch.cuda.current_stream(device)
    stream_handle = stream.cuda_stream

    if not args.skip_micro:
        if not (args.vector_add_cubin and args.saxpy_cubin and args.dot_product_cubin):
            raise RuntimeError("microbenchmarks require vector_add/saxpy/dot cubins")
        n_elements = 1003
        vx = torch.arange(n_elements, device="cuda", dtype=torch.float32)
        vy = torch.arange(n_elements, device="cuda", dtype=torch.float32)
        vz = torch.empty(n_elements, device="cuda", dtype=torch.float32)
        sx = torch.ones(n_elements, device="cuda", dtype=torch.float32)
        sy_init = torch.ones(n_elements, device="cuda", dtype=torch.float32)
        sy = sy_init.clone()
        dx = torch.arange(1, 33, device="cuda", dtype=torch.float32) * 0.25
        dy = torch.arange(32, device="cuda", dtype=torch.float32) * 0.5 - 3.5
        dot_out = torch.empty((), device="cuda", dtype=torch.float32)

        vector_module, vector = load_kernel(cuda, args.vector_add_cubin, "vector_add")
        saxpy_module, saxpy = load_kernel(cuda, args.saxpy_cubin, "saxpy")
        dot_module, dot = load_kernel(cuda, args.dot_product_cubin, "dot_product")

        alpha = ctypes.c_float(1e-6)
        n = ctypes.c_int32(n_elements)
        saxpy_arguments = (
            ctypes.c_uint64(sx.data_ptr()), ctypes.c_uint64(sy.data_ptr()), alpha, n)
        vector_arguments = (
            ctypes.c_uint64(vx.data_ptr()), ctypes.c_uint64(vy.data_ptr()),
            ctypes.c_uint64(vz.data_ptr()), n)
        dot_arguments = (
            ctypes.c_uint64(dx.data_ptr()), ctypes.c_uint64(dy.data_ptr()),
            ctypes.c_uint64(dot_out.data_ptr()))
        vector_threads = 128
        vector_blocks = (n_elements + vector_threads - 1) // vector_threads

        launch(cuda, vector, vector_arguments, threads=vector_threads, blocks=vector_blocks,
               stream=stream_handle)
        stream.synchronize()
        if not torch.allclose(vz, vx + vy):
            raise RuntimeError("OxGPU vector_add result did not match PyTorch")

        sy.copy_(sy_init)
        launch(cuda, saxpy, saxpy_arguments, threads=vector_threads, blocks=vector_blocks,
               stream=stream_handle)
        stream.synchronize()
        if not torch.allclose(sy, sy_init + sx * 1e-6):
            raise RuntimeError("OxGPU saxpy result did not match PyTorch")

        launch(cuda, dot, dot_arguments, threads=32, stream=stream_handle)
        stream.synchronize()
        if not torch.allclose(dot_out, torch.dot(dx, dy)):
            raise RuntimeError("OxGPU dot_product result did not match torch.dot")

        print(
            f"Microkernels (CUDA-event median ms; warmup={args.warmup}, "
            f"repeat={args.repeat}, L2 flush on; launch-dominated at these sizes)")
        print(f"  vector_add/saxpy N={n_elements}; dot N=32")

        time.sleep(args.cooldown)
        vz.fill_(0)
        vector_ox = benchmark_func(
            lambda: launch(cuda, vector, vector_arguments, threads=vector_threads,
                           blocks=vector_blocks, stream=stream_handle),
            warmup=args.warmup, repeat=args.repeat)
        time.sleep(args.cooldown)
        vz.fill_(0)
        vector_pt = benchmark_func(
            lambda: torch.add(vx, vy, out=vz), warmup=args.warmup, repeat=args.repeat)

        time.sleep(args.cooldown)
        sy.copy_(sy_init)
        saxpy_ox = benchmark_func(
            lambda: launch(cuda, saxpy, saxpy_arguments, threads=vector_threads,
                           blocks=vector_blocks, stream=stream_handle),
            warmup=args.warmup, repeat=args.repeat)
        time.sleep(args.cooldown)
        sy.copy_(sy_init)
        saxpy_pt = benchmark_func(
            lambda: torch.add(sy, sx, alpha=1e-6, out=sy),
            warmup=args.warmup, repeat=args.repeat)

        time.sleep(args.cooldown)
        dot_ox = benchmark_func(
            lambda: launch(cuda, dot, dot_arguments, threads=32, stream=stream_handle),
            warmup=args.warmup, repeat=args.repeat)
        time.sleep(args.cooldown)
        dot_pt = benchmark_func(
            lambda: torch.dot(dx, dy, out=dot_out),
            warmup=args.warmup, repeat=args.repeat)

        print("\nvector_add (ms/launch)")
        print(f"  {'OxGPU':24s} {vector_ox:9.4f}")
        print(f"  {'PyTorch add(out=)':24s} {vector_pt:9.4f}")
        print("\nsaxpy (ms/launch)")
        print(f"  {'OxGPU':24s} {saxpy_ox:9.4f}")
        print(f"  {'PyTorch add(out=,alpha=)':24s} {saxpy_pt:9.4f}")
        print("\ndot_product N=32 (ms/launch; not a TFLOPS contest)")
        print(f"  {'OxGPU warp reduce':24s} {dot_ox:9.4f}")
        print(f"  {'torch.dot (cuBLAS)':24s} {dot_pt:9.4f}")

        checked(cuda.cuModuleUnload(vector_module), "cuModuleUnload vector_add")
        checked(cuda.cuModuleUnload(saxpy_module), "cuModuleUnload saxpy")
        checked(cuda.cuModuleUnload(dot_module), "cuModuleUnload dot_product")

    if args.matmul_cubin:
        matmul_module, matmul_fn = load_kernel(cuda, args.matmul_cubin, args.matmul_name)
        # Above 48 KiB a kernel's shared window is dynamic, and the opt-in
        # has to be explicit before the first launch.
        if args.matmul_smem > 0:
            checked(cuda.cuFuncSetAttribute(
                matmul_fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                args.matmul_smem),
                "cuFuncSetAttribute matmul smem")
        print(
            f"\nMatmul (Tilus protocol: warmup={args.warmup}, repeat={args.repeat}, "
            f"cooldown={args.cooldown}s, L2 flush on, TF32 on)")
        if args.matmul_size is not None:
            sizes: Sequence[tuple[int, int, int]] = [tuple(args.matmul_size)]  # type: ignore
        else:
            sizes = [(s, s, s) for s in args.matmul_sizes]
        for m_size, n_size, k_size in sizes:
            bench_matmul_size(
                cuda, matmul_fn, stream, stream_handle,
                m_size, n_size, k_size,
                args.warmup, args.repeat, args.cooldown,
                bm=args.matmul_bm, bn=args.matmul_bn, bk=args.matmul_bk,
                threads=args.matmul_threads, smem=args.matmul_smem)
        checked(cuda.cuModuleUnload(matmul_module), "cuModuleUnload matmul")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)

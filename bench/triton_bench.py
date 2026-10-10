"""Naive Triton TF32 GEMM, timed against cuBLAS under the same protocol as
bench.py so the percentages are comparable to the OxCaml kernel's.

Naive on purpose: one tile size, row-major program order, no autotuning, no
grouped CTA order, no warp specialization. It is the reference point for what
a straightforward tiled kernel gets you, not a tuned Triton GEMM.

Measured on an idle H100: about 20% of cuBLAS TF32 at 4096 cubed and 23% at
8192 cubed. The gap is what the omissions cost, mostly the row-major program
order, which makes every tile-row re-read all of B.

Measure on an idle GPU or not at all. An earlier run of this file overlapped a
neighbouring job and reported 8%, which said nothing about the kernel.
bench/run_frameworks.sh sources tools/gpu_idle.sh for that reason.

  bench/run.sh runs the OxCaml kernel; this runs standalone:
    .bench-venv/bin/python bench/triton_bench.py
"""

import sys

import torch
import triton
import triton.language as tl

from bench import benchmark_func, configure_tf32_matmul, tflops

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
SIZES = (4096, 8192)


@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """C = A @ B, one CTA per BLOCK_M x BLOCK_N output tile."""
    # Row-major program order. Grouping consecutive programs into column
    # strips is what buys L2 reuse; leaving it out is what makes this naive.
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0,
        )
        # TF32 tensor cores, matching the OxCaml kernel's WGMMA tf32 inputs.
        accumulator = tl.dot(a, b, accumulator, input_precision="tf32")
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_gemm(a, b, c):
    m, k = a.shape
    _, n = b.shape
    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
    gemm_kernel[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
    )
    return c


def bench_size(size: int, warmup: int, repeat: int) -> float:
    m = n = k = size
    scale = k ** 0.25
    a = torch.randn(m, k, device="cuda", dtype=torch.float32) / scale
    b = torch.randn(k, n, device="cuda", dtype=torch.float32) / scale
    c = torch.empty(m, n, device="cuda", dtype=torch.float32)
    reference = torch.empty_like(c)

    triton_gemm(a, b, c)
    torch.cuda.synchronize()
    torch.matmul(a, b, out=reference)
    torch.cuda.synchronize()
    if not torch.allclose(c, reference, atol=5e-2, rtol=1e-2):
        raise RuntimeError(
            f"triton gemm mismatch at {size}^3 "
            f"(max abs err={(c - reference).abs().max().item()})")

    triton_ms = benchmark_func(
        lambda: triton_gemm(a, b, c), warmup=warmup, repeat=repeat)
    cublas_ms = benchmark_func(
        lambda: torch.matmul(a, b, out=reference), warmup=warmup, repeat=repeat)
    triton_tf = tflops(triton_ms, m, n, k)
    cublas_tf = tflops(cublas_ms, m, n, k)
    percent = 100.0 * triton_tf / cublas_tf if cublas_tf > 0 else float("nan")

    print(f"\ngemm M={m} N={n} K={k} (TF32 on), "
          f"CTA {BLOCK_M}x{BLOCK_N}, BK={BLOCK_K}")
    print(f"  {'version':24s} {'latency (ms)':>12s} {'tflops':>10s} {'% cublas':>10s}")
    print(f"  {'Triton naive':24s} {triton_ms:12.4f} {triton_tf:10.2f} {percent:10.1f}")
    print(f"  {'torch.matmul (cuBLAS)':24s} {cublas_ms:12.4f} {cublas_tf:10.2f} {100.0:10.1f}")
    return percent


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1
    configure_tf32_matmul()
    print(f"Triton {triton.__version__} naive TF32 GEMM on "
          f"{torch.cuda.get_device_name(0)}")
    for size in SIZES:
        bench_size(size, warmup=5, repeat=30)
    return 0


if __name__ == "__main__":
    sys.exit(main())

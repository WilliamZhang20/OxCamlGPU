import triton
import triton.language as tl

@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M : tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr,
    BLOCK_SIZE_K : tl.constexpr
):
    # Tiled C = A x B
    pid = tl.program_id(axis=0)
    num_pid_in_group = 1  # Simplified mapping; can be increased to group PIDs for L2 reuse
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # compute pointer offsets
    offs_am = (pid_m * BLOCK_SIZE_M) + 
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # loop and compute


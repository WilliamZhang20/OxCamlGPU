"""A Hopper TMA + WGMMA GEMM in CuTeDSL, timed against cuBLAS at its own
precision. Unlike bench/triton_bench.py this is not naive: TMA loads, a
multi-stage mbarrier pipeline, warp specialization into a producer warpgroup
and consumer warpgroups, and a swizzled shared layout.

Two caveats, both real and neither incidental.

Precision. CuTeDSL 4.8.0 exposes warpgroup MMA for f16/bf16/f8/i8 only.
`make_trivial_tiled_mma` rejects Float32 outright ("unsupported a_dtype and
b_dtype") and the `warpgroup` module has no TF32 operation, so a TF32 WGMMA
kernel cannot be written here at all. This is bf16 in, f32 accumulate, which
is the best Hopper path CuTeDSL offers. bf16 WGMMA has roughly twice the
tensor-core throughput of TF32, so these TFLOPS are not comparable to the
OxCaml kernel's. Compare percentages of cuBLAS at each kernel's own precision.

Known limitation: K > 512 faults. The mainloop is verified bit-exact against
torch for K <= 512 at every M and N tried, including 1024x1024x512 and across
tile shapes and stage counts, and faults with CUDA_ERROR_LAUNCH_FAILED for
K = 1024 and above. The trigger is K alone, so it is the k-iteration count
rather than the tile or grid. Ruled out: the shared-memory budget and the
transaction count (both printed and correct, 200704 and 49152 bytes), the
tile size, the stage count, the cross-warpgroup barrier, and the epilogue
partition (its layout matches the WGMMA fragment exactly). Still open, and
the reason BENCH_K is 512 below rather than a square shape.

What it measures, honestly. At K = 512 it reaches about 36% of cuBLAS bf16.
That is not a verdict on the design, because a 512-deep K is only eight
mainloop iterations and the prologue, epilogue and launch dominate. Two things
would have to change before the number means much: the K limitation above, so
the mainloop can be measured where it dominates, and the per-iteration
`wait_group(0)`, which drains the WGMMA pipeline every tile instead of keeping
one group in flight the way examples/kernels/gemm.ml does. So: the structure
here is the right Hopper structure, but this file does not yet demonstrate a
good number, and should not be read as claiming one.

    .bench-venv/bin/python bench/cutedsl_bench.py
"""

import sys

import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as hh
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.cute.runtime import from_dlpack
from cutlass.tensor_utils.layout import LayoutEnum

from bench import benchmark_func, tflops

# Square shapes would need K = M, which the limitation above rules out, so
# the benchmark fixes K and sweeps M = N. Both sides get the same shape.
SIZES = (4096, 8192)
BENCH_K = 512
WARPGROUP = 128


class HopperGemm:
    """Warp-specialized TMA + WGMMA mainloop.

    One producer warpgroup issues TMA copies into a ring of shared-memory
    stages; two consumer warpgroups run WGMMA out of them. The pipeline's
    mbarriers carry the handshake both ways.
    """

    def __init__(self, tile_m=128, tile_n=256, tile_k=64, stages=4, debug=False):
        self.debug = debug
        self.tile_m, self.tile_n, self.tile_k = tile_m, tile_n, tile_k
        self.stages = stages
        self.a_dtype = cutlass.BFloat16
        self.b_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.c_dtype = cutlass.Float32
        # Consumers are one warpgroup per 64 rows; the producer gets its own.
        self.consumer_warpgroups = tile_m // 64
        self.threads = (self.consumer_warpgroups + 1) * WARPGROUP

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream):
        mma_tiler = (self.tile_m, self.tile_n, self.tile_k)
        tiled_mma = hh.make_trivial_tiled_mma(
            self.a_dtype, self.b_dtype,
            OperandMajorMode.K, OperandMajorMode.K, self.acc_dtype,
            (self.consumer_warpgroups, 1, 1),
            (64, self.tile_n),
            warpgroup.OperandSource.SMEM,
        )

        a_smem_staged = hh.make_smem_layout_a(
            LayoutEnum.ROW_MAJOR, mma_tiler, self.a_dtype, self.stages)
        # B is passed as (N, K) with K contiguous, which is K-major; that is
        # what ROW_MAJOR means here and what the MMA's OperandMajorMode.K wants.
        b_smem_staged = hh.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR, mma_tiler, self.b_dtype, self.stages)

        tma_a, a_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mA,
            cute.slice_(a_smem_staged, (None, None, 0)),
            (self.tile_m, self.tile_k))
        tma_b, b_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mB,
            cute.slice_(b_smem_staged, (None, None, 0)),
            (self.tile_n, self.tile_k))

        a_bytes = cute.size_in_bytes(self.a_dtype, cute.slice_(a_smem_staged, (None, None, 0)))
        b_bytes = cute.size_in_bytes(self.b_dtype, cute.slice_(b_smem_staged, (None, None, 0)))
        # Stages, plus the mbarrier array and the 1024-byte alignment each
        # tile is allocated on.
        smem_bytes = (a_bytes + b_bytes) * self.stages + 4096

        if cutlass.const_expr(self.debug):
            cute.printf("a_bytes {} b_bytes {} smem {} tx {}\n",
                        a_bytes, b_bytes, smem_bytes, a_bytes + b_bytes)
        grid = (
            cute.ceil_div(mC.shape[1], self.tile_n),
            cute.ceil_div(mC.shape[0], self.tile_m),
            1,
        )
        self.kernel(
            tiled_mma, tma_a, a_tma_tensor, tma_b, b_tma_tensor, mC,
            a_smem_staged, b_smem_staged, a_bytes + b_bytes,
        ).launch(
            grid=grid, block=(self.threads, 1, 1), smem=smem_bytes, stream=stream)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_b: cute.CopyAtom,
        mB: cute.Tensor,
        mC: cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        tx_count: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, _ = cute.arch.block_idx()

        # The WGMMA descriptor wants the swizzle on the pointer, not folded
        # into the layout, so hand the allocator the two halves separately.
        smem = utils.SmemAllocator()
        sA = smem.allocate_tensor(
            self.a_dtype, a_smem_layout.outer, byte_alignment=1024,
            swizzle=a_smem_layout.inner)
        sB = smem.allocate_tensor(
            self.b_dtype, b_smem_layout.outer, byte_alignment=1024,
            swizzle=b_smem_layout.inner)

        # PipelineTmaAsync elects exactly one signaling consumer thread, the
        # one whose index is 0 (is_signaling_thread = tidx < cluster size, and
        # there is no cluster here). Our consumers start at thread 128, so hand
        # the pipeline a consumer-relative index; otherwise nothing ever
        # signals the empty barrier and the producer blocks forever. The
        # arrival count therefore has to be 1, not the consumer thread count.
        mainloop = pipeline.PipelineTmaAsync.create(
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=tx_count,
            barrier_storage=smem.allocate_array(cutlass.Int64, 2 * self.stages),
            tidx=tidx - WARPGROUP,
        )

        # This CTA's tile of A and B, tiled over K.
        gA = cute.local_tile(mA, (self.tile_m, self.tile_k), (bidy, None))
        gB = cute.local_tile(mB, (self.tile_n, self.tile_k), (bidx, None))
        tAsA, tAgA = cpasync.tma_partition(
            tma_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
        tBsB, tBgB = cpasync.tma_partition(
            tma_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))

        k_tiles = cute.size(gA, mode=[2])
        if cutlass.const_expr(self.debug):
            cute.printf("gA {}  gB {}\n", gA.layout, gB.layout)
            cute.printf("tAsA {}  tAgA {}\n", tAsA.layout, tAgA.layout)
            cute.printf("k_tiles {}\n", k_tiles)

        if warp_idx < 4:
            # Producer warpgroup. The transaction-count arrive and the TMA
            # issue are both single-thread operations, and the producer group
            # above is declared as one thread, so exactly one thread runs this.
            # One thread drives the whole producer loop. The transaction-count
            # arrive and the TMA issue are both single-thread operations and
            # the producer group above is declared as one thread.
            #
            # Electing inside the loop instead, so the warp stays resident, is
            # what CUTLASS does and would be preferable, but the non-elected
            # lanes then run the loop without waiting on the pipeline and the
            # kernel faults at every size. Left as is pending the K limitation
            # noted at the top of this file.
            if tidx == 0:
                state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.stages)
                for k in cutlass.range(k_tiles, unroll=1):
                    mainloop.producer_acquire(state)
                    cute.copy(tma_a, tAgA[None, k], tAsA[None, state.index],
                              tma_bar_ptr=mainloop.producer_get_barrier(state))
                    cute.copy(tma_b, tBgB[None, k], tBsB[None, state.index],
                              tma_bar_ptr=mainloop.producer_get_barrier(state))
                    state.advance()
        else:
            # Consumer warpgroups: WGMMA out of the stages.
            thr_mma = tiled_mma.get_slice(tidx - WARPGROUP)
            acc = thr_mma.make_fragment_C(
                thr_mma.partition_shape_C((self.tile_m, self.tile_n)))
            acc.fill(0.0)
            tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
            tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB))

            # Without this each WGMMA overwrites the accumulator instead of
            # adding to it, so only the last K tile would survive.
            tiled_mma.set(warpgroup.Field.ACCUMULATE, True)

            state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            for k in cutlass.range(k_tiles, unroll=1):
                mainloop.consumer_wait(state)
                warpgroup.fence()
                cute.gemm(tiled_mma, acc,
                          tCrA[None, None, None, state.index],
                          tCrB[None, None, None, state.index], acc)
                warpgroup.commit_group()
                warpgroup.wait_group(0)
                # wait_group is warpgroup-collective, so this warpgroup is done
                # with the stage. One thread signals for all the consumers, so
                # every consumer warpgroup must arrive here before it does.
                cute.arch.barrier(
                    barrier_id=1,
                    number_of_threads=self.consumer_warpgroups * WARPGROUP)
                mainloop.consumer_release(state)
                state.advance()

            # Epilogue: each lane writes the accumulator elements it owns.
            gC = cute.local_tile(mC, (self.tile_m, self.tile_n), (bidy, bidx))
            tCgC = thr_mma.partition_C(gC)
            if cutlass.const_expr(self.debug):
                cute.printf("acc {}  tCgC {}  tCrA {}\n",
                            acc.layout, tCgC.layout, tCrA.layout)
            cute.autovec_copy(acc, tCgC)


def bench_size(size, warmup, repeat):
    m = n = size
    k = BENCH_K
    if k > 512:
        raise RuntimeError("this kernel faults for K > 512; see the module docstring")
    gemm = HopperGemm()
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    c = torch.zeros(m, n, device="cuda", dtype=torch.float32)

    stream = cutlass.torch.current_stream()
    compiled = cute.compile(
        gemm, from_dlpack(a), from_dlpack(b), from_dlpack(c), stream)

    def run():
        compiled(from_dlpack(a), from_dlpack(b), from_dlpack(c), stream)

    run()
    torch.cuda.synchronize()
    reference = (a.float() @ b.float().t())
    if not torch.allclose(c, reference, atol=1e-1, rtol=1e-2):
        raise RuntimeError(
            f"cutedsl gemm mismatch at {size}^3 "
            f"(max abs err={(c - reference).abs().max().item()})")

    ms = benchmark_func(run, warmup=warmup, repeat=repeat)
    bt = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
    bt.copy_(b)
    cref = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    cublas_ms = benchmark_func(
        lambda: torch.matmul(a, bt.t(), out=cref), warmup=warmup, repeat=repeat)
    ours, theirs = tflops(ms, m, n, k), tflops(cublas_ms, m, n, k)
    print(f"\ngemm M={m} N={n} K={k} (bf16 in, f32 acc), "
          f"CTA {gemm.tile_m}x{gemm.tile_n}, BK={gemm.tile_k}, "
          f"stages={gemm.stages}, threads={gemm.threads}")
    print(f"  {'version':24s} {'latency (ms)':>12s} {'tflops':>10s} {'% cublas':>10s}")
    print(f"  {'CuTeDSL TMA+WGMMA':24s} {ms:12.4f} {ours:10.2f} "
          f"{100.0 * ours / theirs:10.1f}")
    print(f"  {'torch.matmul (cuBLAS)':24s} {cublas_ms:12.4f} {theirs:10.2f} {100.0:10.1f}")


def main():
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1
    print(f"CuTeDSL {cutlass.__version__} Hopper bf16 GEMM on "
          f"{torch.cuda.get_device_name(0)}")
    for size in SIZES:
        bench_size(size, warmup=5, repeat=30)
    return 0


if __name__ == "__main__":
    sys.exit(main())

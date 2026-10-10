# GEMM frontend design

Status: design revision — **authors keep hardware control**. There is no
“call `matmul` and kaboom” primary path. The compiler verifies modes/layouts and
lowers what you wrote; it does not invent a CTA tile schedule behind your back.

## Where the performant kernel lives

**[`examples/kernels/gemm.ml`](../examples/kernels/gemm.ml)** —
Hopper TMA producers + WGMMA consumers, software-pipelined K loop, written as
ordinary OxCaml + `Gpu.*`. `hopper_gemm ~bm ~bn ~bk ~stages_n ~group_m` is the
schedule.
Each catalog tile is a binding that applies it to constants (`gemm_fast` is
128×256×32, stages=3, 288 threads). Shared extents, WGMMA `n`, and accumulator
indexes are those constants. Stages are 2 or 3; the mbarrier ladder stays three
deep because the stage index is dynamic.

Emit an OxCaml specialization with `emit_gemm_ptx --gpu …/NAME.gpu`.
`emit_gemm_ptx --print-choose M N K` prints `name bm bn bk threads`
(see [gemm-kernel-status.md](gemm-kernel-status.md)).

## Frontend surface (what authors control)

| Hierarchy | Author knobs (`Gpu.*`) |
| --- | --- |
| **Compute** | `thread_idx_*` / `block_idx_*`, `barrier_cta`, elect predicates, WGMMA fence/wait/commit |
| **Memory** | global load/store (+f32x4), `shared`, shared load/store, `cp_async_*`, TMA, indexed mbarrier sets, GMMA descriptors |

What the compiler supplies rather than asks for: the GMMA descriptor's
leading/stride/layout fields, which follow from the shared tile; the row and
column of each accumulator register inside a warpgroup's tile, which is the
WGMMA layout; and vector widths for stores. Those are hardware encodings, not
schedule choices. Tile sizes, stage depth, warp roles, barrier order, CTA
order and which rows a warpgroup owns stay with the author.

There is no `Gpu.matmul`, and no IR-builder path either. Schedules are
ordinary OxCaml using those ops; the Hopper kernel is
[`examples/kernels/gemm.ml`](../examples/kernels/gemm.ml),
beside the other kernels.
Autotune walks that catalog through the OxCaml specializations.

## Goal

A fast TF32 CTA GEMM is ordinary OxCaml **plus a small set of GPU-visible
ops**. Authors choose BM/BN/BK, pipeline stages, shared layout, and barriers.
OxCaml modes are facts about those choices (unique `C`, read-only `A`/`B`,
shared layout legality) — not a substitute for writing the algorithm.

| Author controls | Compiler does |
| --- | --- |
| CTA tile sizes, thread mapping | Verify launch / shared sizing against layouts |
| Shared alloc + layout (A^T, pad, swizzle) | Lower addressing; reject illegal layouts |
| K-loop, prefetch, barriers | `Ir.For` / `Barrier`; uniformity checks |
| Register tile + FMA/WGMMA | Emit the ops you wrote |
| Global f32x4 / TMA | Lower the ops you called |
| Modes on buffers | Provenance, permissions, noalias at verify/launch |

## Where the pieces live

| Path | Owns |
| --- | --- |
| `examples/kernels/gemm.ml` | **The schedule**: TMA producer elect + WGMMA consumers, and the catalog's bindings as applications of it to constants |
| `test/gemm_catalog.ml` | Tile shapes, `stages`/`producers`, catalog, `choose_config`, `kernel_binding`, parse |
| `test/emit_gemm_ptx.ml` | The emitter: `--gpu`, `--info`, `--print-choose`, `--print-config`, `--list-catalog` |

Every kernel is in `examples/kernels/`, including this one. The catalog is a
module of the emitter, its only consumer, rather than a library of its own.

Warp specialization: threads `[0, consumers)` run WGMMA (HW-aligned 128-thread
warpgroups); `[consumers, threads)` issue TMA (`cp.async.bulk.tensor`).

## Launch contract

The kernel's shared window is 148112 bytes at the default tile, far over
sm_90's 48 KiB static cap, so it is emitted as the dynamic window and **the
launch must request it**. `emit_gemm_ptx --info FILE.gpu` prints
`threads dynamic_smem total_smem`; the PTX also carries
`// oxgpu.shared.dynamic N`. A launcher must call
`cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, N)`
before the first launch and pass `N` as the dynamic shared argument.
`bench/run.sh` and `test/hardware/run_gemm_h100.sh` both do this. See
[shared memory](architecture.md#shared-memory).

## Three idioms worth knowing

**Keep one WGMMA group in flight.** The K loop commits a group per stage and
then waits with `wgmma_wait_group 1`, releasing the *previous* stage rather
than the current one. The tile's MMA therefore overlaps the next tile's
barrier wait instead of draining into it. `wgmma_wait_group 0` and a drain
after the loop retire the last group before the epilogue reads accumulators.
This costs one stage of producer runway, which is why stages=3 matters.

**State the stride's divisibility.** C's row stride is written
`(N / BN) * BN`, not `N`. The launch already requires N to be a multiple of
BN, since the grid is N/BN wide; writing it this way states that precondition
in a form the compiler can use. It proves the stride is a multiple of BN,
which makes `row * stride + col` provably even, which is what lets the store
vectorizer fuse each accumulator column pair into one 8-byte store. Without
it the epilogue stays scalar and costs about 8% of the 4096³ runtime.

Authors write plain `Gpu.store` in the epilogue, and ask
`wgmma_acc_row` / `wgmma_acc_col` where each accumulator register sits.
Vectorization and the register layout are the backend's job, not the
kernel's.

**Group the CTA order.** In the default row-major CTA order every tile-row
streams the whole of B, so B's DRAM traffic scales with the number of
tile-rows and gets worse as the problem grows. The kernel instead remaps the
linear CTA id so that `group_m` tile-rows are walked before advancing along N,
which keeps one A row-block resident across a group and cuts B's re-reads by
the same factor. `group_m = 16` measured best on an H100; at 8192³ it is worth
14 points of cuBLAS (80% to 94%), and 2 points at 4096³. The remap is ordinary
integer `div`/`rem` plus one scalar conditional for the short last group, so
it needs nothing from the compiler.

## Status

1. Ascending OxCaml `for` is in the Typedtree adapter. `ref` is still out
   ([loops.md](loops.md)).
2. `examples/kernels/gemm.ml` is the TMA + WGMMA schedule. All eight
   catalog specializations assemble for `sm_90a` with no register spills, and
   256³ is checked against a CPU reference on hardware by
   `test/hardware/run_gemm_h100.sh`. The 4096³ and 8192³ bench runs check against
   `torch.matmul` before timing.
3. `bench/run.sh` and `emit_gemm_ptx --gpu` use those `.gpu` files.
   `--print-choose` picks the specialization, always at stages=3.
4. A repeatable mid-90s percentage of cuBLAS at 4096³, and parity at 8192³
   ([numbers](architecture.md#measured-gemm-performance)). Under Nsight
   Compute the durations match cuBLAS within 1%, so what is left is launch and
   clock behaviour rather than arithmetic.

## Still open

- A persistent CTA loop, 132 CTAs over the 132 SMs, which is how cuBLAS
  launches. Worth the last few percent; needs no new compiler support.
- The grouped CTA order is still 15 lines of `div`/`rem` in the kernel, and
  the warp roles are still derived from raw thread arithmetic. Both are
  candidates for the same treatment the accumulator layout got.

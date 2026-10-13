# What the GPU mapping is missing

Status: scope. The inventory below was taken from `lib/base/gpu_type.ml` and
`lib/backend/ptx.ml`, not from memory. Items 1 and 2 of the original list are
done and are recorded under "Done since this list was written" rather than
deleted, because what they settled about *how* to add an operation is the part
worth keeping.

## Where it stands

The compiler covers a GEMM well at either input width — TF32 at mid-90s
percent of cuBLAS, bf16 on the same schedule — with TMA, WGMMA, an mbarrier
pipeline, warp specialization, swizzled shared memory, proven store
vectorization and compile-time staging of the whole schedule, plus the float
intrinsics that softmax and normalization need. The machinery underneath is
general. The *surface* is still narrow: outside those, it is close to the set
of operations two kernel families needed.

What the IR actually knows:

| | Covered | Missing |
| --- | --- | --- |
| Element types | `Int32`, `Float32`, `Bfloat16`, `Float16` | f8, f64, i8 |
| Float math | `add.rn`, `mul.rn`, `fma.rn`, `sub.rn`, `div.rn`, `max`, `min`, `neg`, `abs`, `sqrt.rn`, `rsqrt`, `ex2`, `lg2`, `tanh`, `rcp`, `cvt.rn.f32.s32` | f32→int, narrow-float conversions |
| Global memory | load/store, masked, v2/v4, `nc` | atomics of any kind; narrow-float load/store |
| Shared memory | load/store, v4, swizzled, padded | `ldmatrix`, `stmatrix` |
| Async copy | `cp.async`, TMA 2D G2S | TMA store (S2G), TMA 3D-5D, multicast |
| Tensor cores | `wgmma.mma_async` m64nN, k8 TF32 and k16 bf16/f16 | f8, `mma.sync`, sparse |
| Collectives | `bar.sync`, warp butterfly sum | cluster sync, DSMEM, generalized reductions |

## Done since this list was written

**Float math intrinsics.** `Unary_f32` and `Binary_f32` carry a small opcode
enum rather than one constructor per operation: nothing in the compiler
reasons about their algebra, so they need one shape, not thirteen. `Add_f32`,
`Mul_f32` and `Mad_f32` stayed separate because passes *do* reason about those
— commutative value numbering, and the tensor path aliasing onto `mul` — which
is the line LLVM draws between `fadd` and an `llvm.*` intrinsic. Value
numbering covers the intrinsics, so a repeated `rsqrt` collapses.

The split between Stdlib names and `_approx` names is the design decision
worth keeping: `-.`, `/.`, `sqrt`, `Float.max` and friends lower to the
correctly-rounded instruction and keep their OCaml meaning, down to emitting
`max.NaN.f32` so NaN behaves the way `Float.max` says it does. The
special-function unit is reachable only by a name that says so. Base-2 only:
`exp x` is `Gpu.exp2_approx (x *. 1.4426950408889634)`, which keeps the change
of base at the call site instead of hiding it in a primitive.
[`examples/kernels/row_norm.ml`](../examples/kernels/row_norm.ml) is softmax
and RMS norm; `test/fixtures/float_math.ml` pins all thirteen instructions.

**bf16, f16 and WGMMA k16.** Width is a property of the type and appears
nowhere as a literal: see the element-width section of
[architecture.md](architecture.md). The element type rides from a
`bf16 tensor_map` formal through the tile and the descriptor into the
instruction, so there is one `wgmma_mma`, one `shared_wgmma` and one
`tma_load_2d` rather than a set per width.
[`examples/kernels/gemm_bf16.ml`](../examples/kernels/gemm_bf16.ml) is the f32
schedule at bf16, verified on an H100 at sixteen K-tiles so the stage ring and
barrier parity are exercised, and with a byte-identical shared window. f16 is
accepted and selects `m64nNk16.f32.f16.f16`, pinned by
`test/fixtures/f16_wgmma.ml` through ptxas; there is no hardware test for it,
only for bf16.

Two pieces of this are still open, and they are frontend work rather than
GPU mapping:

- **Narrow-float global load and store**, so a bf16 kernel can write bf16
  output instead of f32. Needs the element dtype on `Gep`, `Load` and `Store`
  plus `cvt.rn.bf16.f32` at the boundary; mechanical, and the only reason it
  is not here is that f32 output is what a bf16 GEMM benchmark compares
  against anyway.
- **One schedule over both widths.** `gemm_bf16.ml` duplicates `gemm.ml`
  because the adapter reads types off a single Typedtree and does not
  substitute type variables while inlining, so a `'a tensor_map` formal stays
  unresolved and the allocation inside the body has no ground element type.
  Fixing it means threading a type-variable substitution through the adapter's
  partial evaluator — the same machinery that would let any polymorphic helper
  be shared, so it is worth doing for its own sake. Until then the duplication
  is real and should be read as a frontend gap, not a GPU one.

## Ordered by value per unit of work

**1. Atomics.** `atom.global.add` on f32 and u32, and compare-and-swap.
Unlocks split-K GEMM, histograms, scatter-add and grid-level reductions. The
interesting part is not the instruction, it is the mode interaction: an atomic
is the one operation where `aliased` is the point rather than a limitation, so
the verifier gets a rule that says something new.

**2. TMA store, S2G.** Needed for a staged epilogue that writes C through
shared memory instead of scattering from registers, and a prerequisite for a
persistent kernel that writes tiles as it goes. Mirrors the existing G2S path.

**3. Generalized reductions.** `warp_reduce ~op` and `block_reduce ~op` as
library code rather than primitives. Mostly surface work, because the
recursion and higher-order machinery that
[`block_reduce.ml`](../examples/kernels/block_reduce.ml) already uses is the
hard part and it exists. Needs the warp shuffle exposed, and ideally the Tier 1
frontend item that lets a direct kernel see structure-level helpers, so this
can be a shared module instead of copied into each kernel.

**4. Clusters, DSMEM and TMA multicast.** This is the measured remaining GEMM
gap. Nsight puts the kernel at 93.6% tensor-pipe utilization against cuBLAS's
93.2%, so the arithmetic is done; cuBLAS wins on scheduling by launching one
persistent wave of 132 CTAs with multicast loads where this launches a grid of
512. Highest complexity on the list: cluster launch dimensions, distributed
shared memory, multicast masks, and a tile scheduler. Worth doing last, and
worth pairing with a persistent tile-scheduler helper.

## Compiler work that is not a GPU operation

- The Tier 1 items in [functions.md](functions.md), in particular letting a
  direct kernel see structure-level functions. One line, and it is what turns
  kernels from single files into a library.
- Cross-iteration CSE. Value numbering is cleared at loop boundaries, so
  loop-invariant address arithmetic is recomputed each iteration. ptxas hoists
  some of it; doing it in the IR would make the emitted PTX honest about what
  the loop costs.
- Type-variable substitution in the adapter's partial evaluator, which is
  what one schedule over several element widths needs; see above.
- A persistent tile loop, which needs no new operations and pairs with item 4.

## Not planned

Device-side allocation and collection, exceptions, effects, and runtime
first-class functions beyond the uniformity-gated dispatch sketched in
[functions.md](functions.md). These are not gaps in the mapping; they are
things a GPU does not have.

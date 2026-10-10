# OxCamlGPU architecture

OxCamlGPU turns a supported subset of OxCaml programs into GPU kernels. The compiler checks source types and modes; OxCamlGPU imports those facts, verifies GPU-specific rules, and emits PTX. It does not run the OCaml runtime on the GPU or compile arbitrary OCaml programs.

## Source layout

`lib/` is one library over a directory per stage, in dependency order. The
graph across the folders is acyclic, and `lib/dune` lists the modules grouped
the same way.

| Folder | Holds |
| --- | --- |
| `base/` | Target-independent types, modes, layouts, source spans |
| `ir/` | Semantic GPU SSA and its alias analysis |
| `frontend/` | OxCaml `.gpu` metadata, transported and lowered into IR |
| `verify/` | Semantic verification, uniformity, launch contracts |
| `transform/` | IR-to-IR passes |
| `backend/` | PTX target IR, lowering, register mapping, printer |
| `compiler.ml` | The driver, at the root as the library's front door |
| `dsl/` | The author-facing `Gpu` API |

Every module carries an `.mli`, which is the house convention, except the four
that exist only to define IR datatypes: `ir`, `ptx_ir`, `kernel_ast`, and
`gpu_type`. Every constructor in those is matched on somewhere downstream, so a
signature could abstract nothing and would only duplicate the definitions and
force a second edit for each new opcode. Where a module has internals worth
hiding the signature stays and earns its place: `store_vector` exports 1 of its
24 definitions, `optimizer` 1 of 10.

`dsl/gpu_dsl.ml` is OxCaml source carrying mode syntax that ordinary OCaml
cannot parse. It is deliberately absent from the library's `(modules)` list
and is compiled only by `tools/compile_oxcaml_kernels.sh`, which also compiles
the schema modules it shares with the adapter.

Under `test/`, `hardware/` holds the on-device harnesses, which are shell and
CUDA rather than dune targets. The dune unit tests, the PTX emitters, and the
OxCaml `fixtures/` stay at the top of `test/`; the fixtures also carry mode
syntax, so that directory is not swept into the host build.

## Compilation stages

```text
kernel.ml
  → OxCaml typechecker and .cmt Typedtree
  → version-matched Typedtree adapter
  → versioned .gpu metadata
  → Oxcaml_frontend.import
  → Kernel_frontend.lower
  → semantic GPU IR and verification
  → optimizer
  → store vectorization
  → target strategy selection
  → physical/register mapping
  → PTX IR and printer
```

- `tools/typedtree_support.ml` recognizes supported OxCaml types, modes, and primitive declarations. It depends on the matching compiler-libs.
- `tools/export_typedtree_modes.ml` walks Typedtree expressions and builds the shared `Kernel_ast`. `Gpu_metadata` encodes the complete artifact; output is buffered so a failed import cannot leave partial metadata.
- `Oxcaml_frontend.import` decodes metadata. `Kernel_frontend.lower` creates semantic IR values and calls the verifier.
- `Verifier` checks types, SSA scope, modes, memory access, layouts, and shared-memory initialization. `Uniformity` rejects collectives reached by insufficiently uniform participation.
- `Optimizer` reuses a prior load only when intervening writes cannot alias it, and value-numbers pure integer arithmetic so a repeated computation is emitted once. The second matters because compiler-generated expansions repeat themselves: an accumulator-position query re-derives the lane from the thread id at every call site, which without value numbering turned the GEMM epilogue into 4483 lines of PTX instead of 1219.
- `Store_vector` fuses consecutive scalar f32 stores into `st.global.v2` / `v4`. It proves two things before fusing: that the element indices are consecutive, using an affine form over structurally keyed atoms, and that the first index is a multiple of the vector width, using a congruence `v = r (mod m)`. Atoms are keyed by structure, so an index subexpression recomputed per store still unifies without a prior CSE pass. Anything it cannot prove stays scalar.
- `Ptx_lowering` selects and validates PTX-specific strategies. `Physical_ir` maps semantic SSA values to per-participant register tuples and rejects tensors the current PTX backend cannot expand. `Ptx_ir` contains target operations; `Ptx` prints PTX and its launch contract.

The representation boundaries have distinct responsibilities: `Kernel_ast` is the versioned frontend transport format; `Ir` is semantic GPU SSA; `Physical_ir` expands supported tensors into per-participant scalar registers; `Ptx_ir` contains legal target operations. Tensor arithmetic keeps its semantic operation kind in target IR through legality checks.

The compiler adapter uses compiler APIs and Typedtree structures that are specific to an OxCaml build. Compile it with the matching compiler-libs. The shared schema modules are compiled separately for the host and adapter; host `.cmx` files are not linked into the adapter.

## Source subset and numeric contract

The adapter accepts simple nonrecursive `let` bindings, fully applied local functions (inlined at the call), sequencing, unit and boolean literals, scalar comparisons, scalar/unit `if`, short-circuit boolean operators, selected standard arithmetic, and the declared `Gpu` operations. A kernel binding may itself be a fully applied call of an earlier structure-level function; the arguments become constants in the inlined body, which is how one GEMM schedule specializes to several tiles. A `for` whose bounds are compile-time constants and whose trip count is at most 64 is unrolled, so an index expression such as `4 * i` can select a register. Constant `Int32` arithmetic folds the same way, with i32 wrap. Constants that folding leaves unused are dropped. Larger or dynamic loops stay `Ir.For`. A branch whose condition is grid-, CTA-, or subgroup-uniform lowers to `bra.uni`, including a warp-aligned `thread_idx` compare (`tid < 256`). A one-lane compare (`tid = 256`) stays divergent. It identifies API declarations by compiler identity, so a similarly named user function does not gain GPU meaning. Unsupported syntax reports a source location.

Current source buffers are dynamic one-dimensional f32 arrays. Supported operations include indices, unmasked or explicitly masked loads/stores, `warp_sum_f32`, `block_idx_x` / `block_idx_y`, `Int32.div` / `Int32.rem`, ascending `for` loops, and a grid-leader store. Kernels return `unit`. `[@@gpu.threads N]` on the kernel binding sets `threads_per_cta`. `downto`, recursive helpers, shaped source tensors, buffer-valued conditionals, and escaping mutable references are unsupported.

GPU `float` uses f32 semantics and separate round-to-nearest-even operations. It does not promise host OCaml binary64 behavior. Native `int` uses a signed i32 ABI; ordinary arithmetic is accepted only when interval analysis proves intermediates fit. Explicit `Int32.add/sub/mul` wrap as i32. Callers of raw PTX must provide in-range arguments and launch dimensions. Boolean parameters use a u32 slot, with zero representing false.

The metadata adapter transports uniqueness, locality, OxCaml domain portability, and visibility from supported signature modes. It does not transport every local Typedtree mode. OxCaml portability means crossing an OxCaml domain boundary; it does not mean host/device portability. GPU-boundary portability is a separate IR fact: the adapter leaves it unspecified on the wire, and `Kernel_frontend.lower` derives `Boundary_portable` for buffer and pointer formals that cross the host/device launch boundary. Scalar formals remain unspecified.

## Types, modes, and memory

`Gpu_type` separates logical `Tensor(shape, dtype)` values from addressable `MemRef(shape, dtype, address_space)` storage. Shapes can be static, symbolic, or dynamic. The current source ABI erases buffer shapes to device pointers; extents and strides are not passed. A mask checks nonnegative index and `index < bound`, but the verifier cannot prove that the supplied bound fits the allocation.

`Layout.Register` preserves the original factorized distribution. `Layout.Mapped_register` additionally specifies blocked or interleaved ownership and independent dimension orders for lane, register, and subgroup coordinates. Both expose invertible logical-to-hardware coordinate conversions. The physical IR stores per-participant register slots from this distribution. `Layout.Shared` describes vector width, order, and optional swizzle. `Layout.Padded_shared` adds explicit row padding to allocation sizing; movement through padded or swizzled shared layouts remains unsupported by the current PTX backend. `Layout.Global` carries explicit element strides for global MemRefs. These layouts describe data distribution and organization; they do not describe value uniformity or physical locality modes. Bit-level lane mappings and MMA layouts remain future extensions.

Mode facts are also separate from address space:

- `unique` gives a formal buffer root a noalias promise relative to other distinct unique roots. Derived pointers retain root provenance but are themselves treated as aliased.
- `aliased` allows another reference to designate the same storage.
- `local` and `global` describe lifetime. They do not select CUDA local or global memory.
- Visibility is a permission: read, write, read-write, or immutable. Immutable means readable and never writable, with no mutable aliases.
- OxCaml `portable` describes domain portability. GPU-boundary portability is independent and derived for kernel buffer formals at lower.
- Body SSA scalars default to aliased, local, and read-only. Branch joins meet modal facts from both yields and never strengthen either arm. GEP-derived pointers stay aliased and local while inheriting root permission, provenance, domain portability, and GPU-boundary.

`Verifier.check_call` and `Launch_contract.check` validate abstract argument facts before a host launch. There is still no checked OxCaml launcher wired to the raw CUDA ABI; callers must honor unique/noalias promises for complete accessed ranges and pass `Boundary_portable` buffer actuals for lowered buffer formals.

## Control flow and collectives

Source conditionals become scoped SSA regions with explicit yields. PTX lowering emits branches and edge moves. Memory operations stay in the selected arm, so loads and stores are not speculated. Uniformity tracks value agreement and active participation separately: CTA barriers require CTA-wide participation, and full-subgroup reductions require all needed lanes to reach the collective. A normal branch join restores the enclosing participation requirement; it is not a memory fence.

## Shared memory

A kernel's shared allocations all live in one window, laid out by ascending
MemRef id and each started on a 128-byte boundary, which TMA and GMMA
descriptors require. One declaration then decides placement for the whole
window.

sm_90 caps a *statically* declared `.shared` array at 48 KiB. The 228 KiB an
SM actually has is reachable only through the dynamic window, so a window
over the cap is emitted as `.extern .shared` and the launch must request its
size. `Compiler.compile` reports that size as `dynamic_shared_bytes`, which is
0 for a window that fits statically; `emit_matmul_ptx --info FILE.gpu` prints
it, and the emitted PTX carries it as `// oxgpu.shared.dynamic N` so shell
harnesses can read it without linking the compiler. A launch that requests too
little fails at `cuLaunchKernel`, so the number is not optional: the Hopper
GEMM needs 148112 bytes and cannot run as a static allocation at all.

`Ptx_ir.shared_plan` is the single source of this layout. Both the launch
contract (`launch.shared_bytes`) and the emitter read it, so the declared size
and the contract cannot disagree. A window larger than an SM's capacity is
rejected at lowering.

## Current backend limits

The PTX backend covers the operations used by current examples and selected direct-IR cases. Elementwise indexing and masked tails can span multiple CTAs. Tensor movement, elementwise scale/multiply, and reductions support 1D f32 tiles distributed across one 32-thread CTA, including multiple values per lane under blocked or interleaved mappings. Rank-two f32 tiles are supported for direct-IR movement when lanes own the tile in row-major order: global MemRefs may carry an explicit `Layout.Global` stride layout, and `Load_tensor_masked` / `Store_tensor_masked` predicate row/col bounds. Physicalization expands each lane’s registers and emits mapping-derived global/shared offsets (linear for 1D, div/mod plus row stride for 2D). Swizzled/padded shared movement and shaped tensor expressions imported from OxCaml are not yet supported. Reductions sum each lane’s local registers, then use butterfly shuffles across the warp. `Gpu.warp_sum_f32` remains the source-level scalar escape hatch.

The core `oxgpu` compiler has **no matmul-specific expansion**. It only
lowers general IR ops authors write (`Shared_*`, `Barrier`, `Tma_load_2d`,
`Gmma_descriptor`, `Wgmma_mma_tf32`, `Mad_f32`, …).

The **tile catalog** names the shapes, the size-to-shape choice, and the
kernel binding each shape maps to. It builds no IR, and its only consumer is
`test/emit_matmul_ptx.ml`, so it is a module of that executable rather than a
library. The schedule is the OxCaml kernel
[`examples/kernels/matmul_tiled.ml`](../examples/kernels/matmul_tiled.ml).
`bench/run.sh` emits a specialization (`--print-choose` per size, or
`MATMUL_BM` / `BN` / `BK` / `STAGES` to pin one). See
[matmul frontend design](matmul-frontend.md).
Default tile: BM=128, BN=256, BK=32, stages=3, 288 threads; PTX target `sm_90a`.

## Validation and benchmarks

`dune runtest` runs host-side verifier, optimizer, frontend, and control-flow checks. OxCaml bridge checks run when a compatible toolchain is configured; otherwise those checks are skipped.

`test/hardware/run_h100.sh` exercises generated PTX on a local H100 when CUDA tools are available. `test/hardware/run_matmul_h100.sh` checks the compiler-emitted tiled TF32 WGMMA GEMM cubin for correctness. `bench/run.sh` compares vector-add and SAXPY against PyTorch operations, the 32-element dot product against `torch.dot`, and matmul at 4096³ / 8192³ against `torch.matmul` (TF32 on) using the Tilus CUDA-event + L2-flush protocol. The harness checks results before timing and reports median latency and TFLOPS. Elementwise benchmarks use a non-multiple extent to exercise masked tails.

Every hardware script sources [`tools/gpu_idle.sh`](../tools/gpu_idle.sh) and
refuses to run on a contended GPU, because a neighbouring job skews the
percentage badly.

### Measured GEMM performance

H100 80GB HBM3, TF32 on both sides, CUDA 12.9, `torch.matmul` as the cuBLAS
reference, on an otherwise idle GPU. `matmul_tiled` at BM=128, BN=256, BK=32,
stages=4, group_m=16, 288 threads.

One `bench/run.sh` run:

| Problem | oxgpu | cuBLAS | % cuBLAS |
| --- | --- | --- | --- |
| 4096³ | 384.6 TFLOPS | 398.1 TFLOPS | 96.6 |
| 8192³ | 395.1 TFLOPS | 400.6 TFLOPS | 98.6 |

Read those ratios with the spread in mind. 4096³ is stable: this kernel lands
at 384-391 TFLOPS across runs against cuBLAS's 398-401, so the mid-90s is
repeatable. 8192³ moves much more on both sides, 368-404 against 366-412, so a
single run there can read anywhere from 89% to 109%. The profiler numbers
below, taken back to back in one process, are the better evidence.

Nsight Compute on the same launches, against the cuBLAS kernel each was timed
against. cuBLAS independently selects the same 128×256×32 tile with two
consumer warpgroups.

| Metric | 4096³ oxgpu | 4096³ cuBLAS | 8192³ oxgpu | 8192³ cuBLAS |
| --- | --- | --- | --- | --- |
| Duration | 412 us | 410 us | 3.03 ms | 3.04 ms |
| Compute (SM) throughput | 88.9 % | 88.6 % | 93.7 % | 93.2 % |
| L2 throughput | 57.9 % | 55.1 % | 74.6 % | 62.9 % |
| DRAM throughput | 25.5 % | 25.4 % | 26.8 % | 26.1 % |

Both kernels are compute-bound at the same SM throughput, and the durations
match within 1% at 4096³ while 8192³ is a tie. The higher L2 throughput is
this kernel doing more L2 traffic for the same work, which is the cost of
launching a 16×32 or 32×64 grid and leaning on the grouped CTA order for
locality; cuBLAS launches 132 CTAs as one persistent wave over the 132 SMs. A
persistent tile loop is the remaining structural difference and needs no new
compiler support.

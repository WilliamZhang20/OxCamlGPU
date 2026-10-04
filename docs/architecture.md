# OxCamlGPU architecture

OxCamlGPU turns a supported subset of OxCaml programs into GPU kernels. The compiler checks source types and modes; OxCamlGPU imports those facts, verifies GPU-specific rules, and emits PTX. It does not run the OCaml runtime on the GPU or compile arbitrary OCaml programs.

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
  → target strategy selection
  → physical/register mapping
  → PTX IR and printer
```

- `tools/typedtree_support.ml` recognizes supported OxCaml types, modes, and primitive declarations. It depends on the matching compiler-libs.
- `tools/export_typedtree_modes.ml` walks Typedtree expressions and builds the shared `Kernel_ast`. `Gpu_metadata` encodes the complete artifact; output is buffered so a failed import cannot leave partial metadata.
- `Oxcaml_frontend.import` decodes metadata. `Kernel_frontend.lower` creates semantic IR values and calls the verifier.
- `Verifier` checks types, SSA scope, modes, memory access, layouts, and shared-memory initialization. `Uniformity` rejects collectives reached by insufficiently uniform participation.
- `Optimizer` currently reuses a prior load only when intervening writes cannot alias it.
- `Ptx_lowering` selects and validates PTX-specific strategies. `Physical_ir` maps semantic SSA values to per-participant register tuples and rejects tensors the current PTX backend cannot expand. `Ptx_ir` contains target operations; `Ptx` prints PTX and its launch contract.

The representation boundaries have distinct responsibilities: `Kernel_ast` is the versioned frontend transport format; `Ir` is semantic GPU SSA; `Physical_ir` expands supported tensors into per-participant scalar registers; `Ptx_ir` contains legal target operations. Tensor arithmetic keeps its semantic operation kind in target IR through legality checks.

The compiler adapter uses compiler APIs and Typedtree structures that are specific to an OxCaml build. Compile it with the matching compiler-libs. The shared schema modules are compiled separately for the host and adapter; host `.cmx` files are not linked into the adapter.

## Source subset and numeric contract

The adapter accepts simple nonrecursive `let` bindings, sequencing, unit and boolean literals, scalar comparisons, scalar/unit `if`, short-circuit boolean operators, selected standard arithmetic, and the declared `Gpu` operations. It identifies API declarations by compiler identity, so a similarly named user function does not gain GPU meaning. Unsupported syntax reports a source location.

Current source buffers are dynamic one-dimensional f32 arrays. Supported operations include indices, unmasked or explicitly masked loads/stores, `warp_sum_f32`, and a grid-leader store. Kernels return `unit`. Loops, general helper calls, shaped source tensors, buffer-valued conditionals, and escaping mutable references are unsupported.

GPU `float` uses f32 semantics and separate round-to-nearest-even operations. It does not promise host OCaml binary64 behavior. Native `int` uses a signed i32 ABI; ordinary arithmetic is accepted only when interval analysis proves intermediates fit. Explicit `Int32.add/sub/mul` wrap as i32. Callers of raw PTX must provide in-range arguments and launch dimensions. Boolean parameters use a u32 slot, with zero representing false.

The metadata adapter transports uniqueness, locality, OxCaml domain portability, and visibility from supported signature modes. It does not transport every local Typedtree mode. OxCaml portability means crossing an OxCaml domain boundary; it does not mean host/device portability. GPU-boundary portability is a separate IR fact: the adapter leaves it unspecified on the wire, and `Kernel_frontend.lower` derives `Boundary_portable` for buffer and pointer formals that cross the host/device launch boundary. Scalar formals remain unspecified.

## Types, modes, and memory

`Gpu_type` separates logical `Tensor(shape, dtype)` values from addressable `MemRef(shape, dtype, address_space)` storage. Shapes can be static, symbolic, or dynamic. The current source ABI erases buffer shapes to device pointers; extents and strides are not passed. A mask checks nonnegative index and `index < bound`, but the verifier cannot prove that the supplied bound fits the allocation.

`Layout.Register` preserves the original factorized distribution. `Layout.Mapped_register` additionally specifies blocked or interleaved ownership and independent dimension orders for lane, register, and subgroup coordinates. Both expose invertible logical-to-hardware coordinate conversions. The physical IR stores per-participant register slots from this distribution. Multi-register tensor expansion is explicitly rejected by the current PTX backend. `Layout.Shared` describes vector width, order, and optional swizzle. `Layout.Padded_shared` adds explicit row padding to allocation sizing; movement through padded or swizzled shared layouts remains unsupported by the current PTX backend. These layouts describe data distribution and organization; they do not describe value uniformity or physical locality modes. Global MemRefs do not yet carry explicit layouts. Bit-level lane mappings and MMA layouts remain future extensions.

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

## Current backend limits

The PTX backend covers the operations used by current examples and selected direct-IR cases. Elementwise indexing and masked tails can span multiple CTAs. Tensor movement, elementwise scale/multiply, and reductions support 1D f32 tiles distributed across one 32-thread CTA, including multiple values per lane under blocked or interleaved mappings. Physicalization expands each lane’s registers and emits mapping-derived global/shared offsets. Rank-two physicalization, swizzled/padded shared movement, and shaped tensor expressions imported from OxCaml are not yet supported. Reductions sum each lane’s local registers, then use butterfly shuffles across the warp. `Gpu.warp_sum_f32` remains the source-level scalar escape hatch.

There is no matmul operation, tiled matmul strategy, multi-CTA reduction strategy, or tensor-core lowering. These require both a richer source/IR model and target lowering; adding source syntax alone is insufficient. Shared-memory launch sizing is derived from static shape and dtype; layout-specific padding and swizzle storage requirements are not yet modeled.

## Validation and benchmarks

`dune runtest` runs host-side verifier, optimizer, frontend, and control-flow checks. OxCaml bridge checks run when a compatible toolchain is configured; otherwise those checks are skipped.

`test/run_h100.sh` exercises generated PTX on a local H100 when CUDA tools are available. `bench/run.sh` compares vector-add and SAXPY against PyTorch operations and compares the 32-element dot product against `torch.dot`. The harness checks results before timing and reports CUDA-event medians. Elementwise benchmarks use a non-multiple extent to exercise masked tails.

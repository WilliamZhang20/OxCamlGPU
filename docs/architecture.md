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
  → PTX lowering and printer
```

- `tools/typedtree_support.ml` recognizes supported OxCaml types, modes, and primitive declarations. It depends on the matching compiler-libs.
- `tools/export_typedtree_modes.ml` walks Typedtree expressions and builds the shared `Kernel_ast`. `Gpu_metadata` encodes the complete artifact; output is buffered so a failed import cannot leave partial metadata.
- `Oxcaml_frontend.import` decodes metadata. `Kernel_frontend.lower` creates semantic IR values and calls the verifier.
- `Verifier` checks types, SSA scope, modes, memory access, layouts, and shared-memory initialization. `Uniformity` rejects collectives reached by insufficiently uniform participation.
- `Optimizer` currently reuses a prior load only when intervening writes cannot alias it.
- `Ptx_lowering` selects and validates PTX-specific strategies. `Ptx` prints PTX and its launch contract.

The compiler adapter uses compiler APIs and Typedtree structures that are specific to an OxCaml build. Compile it with the matching compiler-libs. The shared schema modules are compiled separately for the host and adapter; host `.cmx` files are not linked into the adapter.

## Source subset and numeric contract

The adapter accepts simple nonrecursive `let` bindings, sequencing, unit and boolean literals, scalar comparisons, scalar/unit `if`, short-circuit boolean operators, selected standard arithmetic, and the declared `Gpu` operations. It identifies API declarations by compiler identity, so a similarly named user function does not gain GPU meaning. Unsupported syntax reports a source location.

Current source buffers are dynamic one-dimensional f32 arrays. Supported operations include indices, unmasked or explicitly masked loads/stores, `warp_sum_f32`, and a grid-leader store. Kernels return `unit`. Loops, general helper calls, shaped source tensors, buffer-valued conditionals, and escaping mutable references are unsupported.

GPU `float` uses f32 semantics and separate round-to-nearest-even operations. It does not promise host OCaml binary64 behavior. Native `int` uses a signed i32 ABI; ordinary arithmetic is accepted only when interval analysis proves intermediates fit. Explicit `Int32.add/sub/mul` wrap as i32. Callers of raw PTX must provide in-range arguments and launch dimensions. Boolean parameters use a u32 slot, with zero representing false.

The metadata adapter transports uniqueness, locality, OxCaml domain portability, and visibility from supported signature modes. It does not transport every local Typedtree mode. OxCaml portability means crossing an OxCaml domain boundary; it does not mean host/device portability. GPU-boundary portability is a separate IR fact and is currently unspecified by the adapter.

## Types, modes, and memory

`Gpu_type` separates logical `Tensor(shape, dtype)` values from addressable `MemRef(shape, dtype, address_space)` storage. Shapes can be static, symbolic, or dynamic. The current source ABI erases buffer shapes to device pointers; extents and strides are not passed. A mask checks nonnegative index and `index < bound`, but the verifier cannot prove that the supplied bound fits the allocation.

`Layout.Register` describes elements per lane, lanes per subgroup, subgroups per CTA, and dimension order. `Layout.Shared` describes vector width, order, and optional swizzle. These layouts describe data distribution and organization; they do not describe value uniformity or physical locality modes. Global MemRefs do not yet carry explicit layouts.

Mode facts are also separate from address space:

- `unique` gives a formal buffer root a noalias promise relative to other distinct unique roots. Derived pointers retain root provenance but are themselves treated as aliased.
- `aliased` allows another reference to designate the same storage.
- `local` and `global` describe lifetime. They do not select CUDA local or global memory.
- Visibility is a permission: read, write, read-write, or immutable. Immutable means readable and never writable, with no mutable aliases.
- OxCaml `portable` describes domain portability. GPU-boundary portability is independent.

`Verifier.check_call` checks abstract argument facts, but there is no checked OxCaml launcher wired to the raw CUDA ABI. Callers must honor unique/noalias promises for complete accessed ranges.

## Control flow and collectives

Source conditionals become scoped SSA regions with explicit yields. PTX lowering emits branches and edge moves. Memory operations stay in the selected arm, so loads and stores are not speculated. Uniformity tracks value agreement and active participation separately: CTA barriers require CTA-wide participation, and full-subgroup reductions require all needed lanes to reach the collective. A normal branch join restores the enclosing participation requirement; it is not a memory fence.

## Current backend limits

The PTX backend covers the operations used by current examples and selected direct-IR cases. Elementwise indexing and masked tails can span multiple CTAs. Tensor movement and reductions currently support only a 32-element f32 tile distributed one element per lane across one 32-thread CTA. The reduction lowers to lane extraction and butterfly shuffles. `Gpu.warp_sum_f32` is the source-level scalar escape hatch; shaped tensor expressions are not yet imported from OxCaml.

There is no matmul operation, tiled matmul strategy, multi-CTA reduction strategy, or tensor-core lowering. These require both a richer source/IR model and target lowering; adding source syntax alone is insufficient.

## Validation and benchmarks

`dune runtest` runs host-side verifier, optimizer, frontend, and control-flow checks. OxCaml bridge checks run when a compatible toolchain is configured; otherwise those checks are skipped.

`test/run_h100.sh` exercises generated PTX on a local H100 when CUDA tools are available. `bench/run.sh` compares vector-add and SAXPY against PyTorch operations and compares the 32-element dot product against `torch.dot`. The harness checks results before timing and reports CUDA-event medians. Elementwise benchmarks use a non-multiple extent to exercise masked tails.

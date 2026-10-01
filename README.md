# OxCamlGPU semantic core

This prototype checks whether facts assigned by the OxCaml compiler can be carried from typed source into GPU IR and used by verification. Kernel authors write ordinary OxCaml functions. GPU-specific operations include index queries, warp reduction, and explicitly masked memory access.

Each OxCaml-authored example kernel lives in one source file: [SAXPY](examples/kernels/saxpy.ml), [vector_add](examples/kernels/vector_add.ml), and [dot_product](examples/kernels/dot_product.ml). Its modal function type annotation sits directly on the `let` binding, beside the body, so there is no separate per-kernel `.mli`. The shared source API is [lib/gpu_dsl.mli](lib/gpu_dsl.mli) and its primitive declarations are in [lib/gpu_dsl.ml](lib/gpu_dsl.ml). The examples use ordinary `float` and `int` types and operators; only thread indexing, warp reduction, and buffer access are `Gpu.*` calls. OxCaml typechecks the prelude and each kernel and writes typedtree artifacts. A small compiler-libs adapter reads each kernel's `.cmt` and exports its inferred argument/result modes and supported body as stable metadata. Our importer builds the source AST and mode-bearing signature from that representation, then lowers to GPU IR.

```text
OxCaml kernel .ml
        ↓ OxCaml typechecker
.cmt Typedtree
        ↓ version-matched compiler-libs adapter
stable GPU metadata
        ↓ Kernel_ast + Oxcaml_frontend
Kernel_frontend → Ir → Verifier → Optimizer → PTX backend
```

## OxCaml integration

The adapter must use an OxCaml compiler and compiler-libs from the same build, since its Typedtree format and APIs are build-specific. `tools/compile_oxcaml_kernels.sh` accepts either a source build root with the conventional OxCaml layout, or explicit paths to an OxCaml installation:

```text
OXCAML_ROOT=/path/to/oxcaml-src
# or, for a nonstandard/installed layout:
OXCC=/path/to/ocamlc.opt
OXOPT=/path/to/ocamlopt.opt
OXCAML_MAIN_BUILD=/path/to/matching/compiler-libs
OXCAML_STDLIB_DIR=/path/to/matching/stdlib
```

When `ocamlc.opt` and `ocamlopt.opt` are on `PATH`, the script discovers the compiler executables. If their location identifies the conventional OxCaml build tree, it also derives the compiler-libs and standard-library paths. Otherwise set `OXCAML_ROOT`, or provide all four explicit paths above. The tested compiler source revision is recorded in [tools/oxcaml-revision](tools/oxcaml-revision); source-tree builds are checked against it by default. Set `OXCAML_ALLOW_UNPINNED=1` only after validating the adapter against another revision. The repository does not assume a particular user's home directory or keep compiler build products under version control.

The integration currently works as a compiler invocation plus a small adapter, rather than embedding OxCaml inside OxCamlGPU:

1. The script compiles `lib/gpu_dsl.mli/.ml` once as a shared kernel API, then runs `ocamlc.opt -bin-annot` on each kernel `.ml`. Each function's mode-annotated type and body are captured in one `.cmt` Typedtree artifact. Dune's host OCaml build excludes `Gpu_dsl` because its mode annotations require OxCaml.
2. It builds `tools/export_typedtree_modes.ml` against that same build's `ocamlcommon.cmxa`, `ocamlfrontend.cmxa`, and `oxcaml_common.cmxa`.
3. The adapter reads each `.cmt` with `Cmt_format`, exports argument/result types and modes plus the supported function body as tab-separated `.gpu` metadata.
4. `Oxcaml_frontend` and `Kernel_ast` import that metadata; the existing lowering, verifier, and PTX stages then run.

The compiler and compiler-libs must come from the **same OxCaml build**: their artifact formats and Typedtree APIs are build-specific. Example:

```sh
tools/compile_oxcaml_kernels.sh /tmp/oxgpu-metadata
dune exec examples/saxpy.exe -- /tmp/oxgpu-metadata/saxpy.gpu
```

This is an initial adaptation point, not a replacement OxCaml parser or a general typedtree-to-GPU lowering. The adapter currently accepts the examples' types and a small expression subset: local `let` bindings, sequenced stores, `thread_idx_x`, `global_idx_x`, `warp_sum_f32`, grid-leader stores, masked and unmasked buffer loads/stores, `+.`/`*.`, and integer or float literals. It extracts uniqueness, locality, portability, and visibility directly from structured Typedtree mode constants. `shareable` and `corruptible` portability are rejected because the GPU mode model has no mapping for them; other OxCaml mode axes are not currently transported. Kernels must return `unit`. Unsupported typedtree nodes and primitive calls fail rather than being silently reinterpreted.

## Mode contract

| Mode | GPU semantic fact |
| --- | --- |
| `unique` | A formal buffer root is noalias with distinct formal roots. Derived pointers retain root provenance but are conservatively `aliased`; mutation permission is separate. |
| `aliased` | Other references may designate the same storage. |
| `local` | The value cannot escape its current region or kernel scope. |
| `global` | The value may outlive its current region. |
| OxCaml `portable` | The value may cross an OxCaml domain boundary (`Domain_portable`). This mode does not establish host/device portability. |
| GPU-boundary portable | The value may cross a host/device or asynchronous execution boundary. This is a separate IR fact and is currently unspecified by the OxCaml adapter. |

These facts do not select physical placement. `local` does not mean CUDA local memory; `global` does not mean CUDA global memory. Address space (`Global`, `Shared`, `Local`) remains an independent IR property. `unique` does not guarantee race freedom across GPU threads, and `aliased` does not imply read-only access. `immutable` is readable but never writable; a read-only formal may accept an immutable actual, while an immutable formal requires an immutable actual.

Permission compatibility is capability based: read-only accepts read-only, read-write, or immutable actuals; write-only accepts write-only or read-write; read-write accepts only read-write; immutable accepts only immutable. This keeps the stronger “no mutable aliases exist” promise of `immutable` distinct from ordinary read-only access.

Raw PTX entry points have no checked OxCaml launcher yet. Callers must uphold each `unique` argument's noalias precondition for the full accessed buffer range; violating it is outside the kernel ABI contract, just as violating a `restrict` contract is undefined. `Verifier.check_call` checks abstract buffer identities, but it is not currently wired into the CUDA driver ABI. The H100 harness uses distinct allocations for `unique` arguments and deliberately passes the same allocation only to the aliased baseline.

## Core GPU objects

### Memory Hierarchy

`Gpu_type` distinguishes `Tensor(shape, dtype)` logical values from `MemRef(shape, dtype, address_space)` addressable storage. Dimensions can be static, symbolic, or dynamic. OxCaml `float gpu_array` arguments currently import as dynamic one-dimensional global f32 MemRefs; element-address calculations produce temporary pointer types during lowering.

The current PTX ABI erases that MemRef shape to a device pointer; runtime extents and strides are not passed yet. The elementwise kernels use scalar-per-lane values and predicated accesses, while dot product reduces a full warp. The predicates check nonnegative indices and `index < bound`, but the verifier does not prove that the supplied bound is within the allocation extent. Ownership, locality, permission, and portability remain facts on IR values, separate from both the MemRef address space and Tensor shape.

`Layout.Register` records size per thread, threads per warp, warps per CTA, and dimension order. `Layout.Shared` records vector width, dimension order, and an optional 32/64/128-byte swizzle.

The verifier checks rank, positive factors, permutation validity, and exact coverage for static register-tile dimensions. `Shared_alloc`, `Load_tensor`, `Store_tensor`, and `Barrier Cta` are explicit IR operations; the straight-line verifier rejects shared reads before initialization or before a CTA barrier. Global MemRefs intentionally have no explicit layout descriptor yet.

The dot product source example is still a backend smoke test: its current OxCaml adapter recognizes a scalar lane value and the explicitly named `Gpu.warp_sum_f32`. The typed GPU IR also has a shape-level `Reduce_sum_f32` operation over a laid-out Tensor. The PTX backend selects a one-warp strategy only for `Tensor<[32], f32>` with one element per lane, then lowers it to lane extraction plus butterfly shuffles. The PTX declares `.reqntid 32,1,1`; `Gpu.store_grid_leader` makes one grid-wide writer. Other reduction shapes/layouts are rejected by this strategy selection. CTA and multi-CTA strategy selection, and transporting shaped Tensor expressions from OxCaml source, remain future work. `Gpu.warp_sum_f32` remains the current source-level escape hatch and does not yet express a shaped reduction.

### Execution Hierarchy

`Execution` defines the separate scope hierarchy: `Grid → Cta → Warpgroup → Warp → Lane`. These levels describe execution coordinates/scopes and do not choose storage. `global_idx_x` combines CTA id, CTA width, and lane index. Warpgroup remains a target-defined grouping for future lowering.

The importer preserves modal facts that OxCaml exposes in its typedtree. Kernel-call verification now checks locality, OxCaml domain portability, and GPU-boundary portability independently: a `global` formal requires a global-lifetime actual, while a `local` formal accepts either lifetime; a `portable` formal requires an actual explicitly marked portable in that same portability domain. OxCaml domain portability does not imply host/device portability. The current metadata adapter leaves GPU-boundary portability unspecified, so it cannot satisfy a GPU-boundary-portable formal without an explicit IR-side fact. The body importer currently supports only the source constructs used in these examples.

## Build and run

Run the semantic verifier and OxCaml bridge checks with `dune runtest`. The bridge check runs when `OXCAML_ROOT` is set, the compiler path identifies the conventional OxCaml build tree, or all four explicit toolchain paths are set; otherwise it reports that the external-toolchain checks were skipped. The focused `unique_reuse` fixture lives under `test/`, not `examples/`: it compares OxCaml kernels whose target buffer is aliased versus unique and checks that only the unique version removes a global load. With CUDA modules and a local H100, `test/run_h100.sh` also executes both cases to check their distinct aliasing semantics.

`bench/run.sh` builds the OxCaml kernels and compares vector_add/SAXPY with PyTorch operations. It also compares the 32-element OxGPU dot product with `torch.dot`, which uses PyTorch's CUDA dot implementation (cuBLAS for contiguous f32 vectors). The harness validates results before timing and reports CUDA-event medians. Elementwise kernels now take `n`, use `global_idx_x`, and mask tail loads/stores; the benchmark uses `N=1003` to exercise multiple blocks and a partial final block. Dot product still uses one full warp (`N=32`).

The PTX backend remains small and supports only the operations used by these examples. Tensor and reduction lowering requires exactly one 32-thread, one-dimensional CTA; the backend emits `.reqntid 32, 1, 1`, which the CUDA driver enforces at launch. Launch indexing and predicated tails now work across multiple CTAs. The first `unique`-driven optimization reuses a prior load across a store only when alias analysis proves the store cannot alias the loaded buffer; the test checks that this removes one `ld.global` from PTX while the aliased case retains it.

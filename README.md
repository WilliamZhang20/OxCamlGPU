# OxCamlGPU semantic core

This prototype checks whether facts assigned by the OxCaml compiler can be carried from typed source into GPU IR and used by verification. Kernel authors write ordinary OxCaml functions. Only GPU operations such as `Gpu.thread_idx_x`, `Gpu.load`, and `Gpu.store` are special.

Each OxCaml-authored kernel lives in one source file: [SAXPY](examples/kernels/saxpy.ml), [vector_add](examples/kernels/vector_add.ml), and [dot_product](examples/kernels/dot_product.ml). Its modal function type annotation sits directly on the `let` binding, beside the body, so there is no separate per-kernel `.mli`. The shared source API is [lib/gpu_dsl.mli](lib/gpu_dsl.mli) and its primitive declarations are in [lib/gpu_dsl.ml](lib/gpu_dsl.ml). The examples use ordinary `float` and `int` types and operators; only thread indexing, warp reduction, and buffer access are `Gpu.*` calls. OxCaml typechecks the prelude and each kernel and writes typedtree artifacts. A small compiler-libs adapter reads each kernel's `.cmt` and exports its inferred argument/result modes and supported body as stable metadata. Our importer builds the source AST and mode-bearing signature from that representation, then lowers to GPU IR.

```text
OxCaml kernel .ml
        ↓ OxCaml typechecker
.cmt Typedtree
        ↓ version-matched compiler-libs adapter
stable GPU metadata
        ↓ Kernel_ast + Oxcaml_frontend
Kernel_frontend → Ir → Verifier → PTX
```

## OxCaml integration

OxCaml is built locally from source at `$HOME/src/oxcaml-src` (currently `/home/wzhang20/src/oxcaml-src`). It is **not installed globally or into the project's opam switch**. The compiler entry points used here are:

```text
$HOME/src/oxcaml-src/_build/_bootinstall/bin/ocamlc.opt
$HOME/src/oxcaml-src/_build/_bootinstall/bin/ocamlopt.opt
```

Those are build-tree symlinks to OxCaml's native compiler executables. The matching compiler-libs used to build the Typedtree adapter are in `$HOME/src/oxcaml-src/_build/main`; the matching standard library artifacts are in `$HOME/src/oxcaml-src/_build/runtime_stdlib_install/lib/ocaml_runtime_stdlib`. `tools/compile_oxcaml_kernels.sh` defaults to this path via `OXCAML_ROOT=$HOME/src/oxcaml-src`.

The integration currently works as a compiler invocation plus a small adapter, rather than embedding OxCaml inside OxCamlGPU:

1. The script compiles `lib/gpu_dsl.mli/.ml` once as a shared kernel API, then runs `ocamlc.opt -bin-annot` on each kernel `.ml`. Each function's mode-annotated type and body are captured in one `.cmt` Typedtree artifact. Dune's host OCaml build excludes `Gpu_dsl` because its mode annotations require OxCaml.
2. It builds `tools/export_typedtree_modes.ml` against that same build's `ocamlcommon.cmxa`, `ocamlfrontend.cmxa`, and `oxcaml_common.cmxa`.
3. The adapter reads each `.cmt` with `Cmt_format`, exports argument/result types and modes plus the supported function body as tab-separated `.gpu` metadata.
4. `Oxcaml_frontend` and `Kernel_ast` import that metadata; the existing lowering, verifier, and PTX stages then run.

The compiler and compiler-libs must come from the **same OxCaml build**: their artifact formats and Typedtree APIs are build-specific. For another local build, set `OXCAML_ROOT` or override all paths with `OXCAML_MAIN_BUILD`, `OXCAML_STDLIB_DIR`, `OXCC`, and `OXOPT`. Example:

```sh
tools/compile_oxcaml_kernels.sh /tmp/oxgpu-metadata
dune exec examples/saxpy.exe -- /tmp/oxgpu-metadata/saxpy.gpu
```

This is an initial adaptation point, not a replacement OxCaml parser or a general typedtree-to-GPU lowering. The adapter currently accepts the examples' types and a small expression subset: local `let` bindings, `thread_idx_x`, `warp_sum_f32`, lane-zero stores, buffer loads/stores, `+.`/`*.`, and integer or float literals. Unsupported typedtree nodes, primitive calls, or mode axes fail with an error rather than being silently reinterpreted.

## Mode contract

| Mode | GPU semantic fact |
| --- | --- |
| `unique` | A formal buffer root is noalias with distinct formal roots. Derived pointers retain root provenance but are conservatively `aliased`; mutation permission is separate. |
| `aliased` | Other references may designate the same storage. |
| `local` | The value cannot escape its current region or kernel scope. |
| `global` | The value may outlive its current region. |
| `portable` | The value may cross a host/device or asynchronous execution boundary. |

These facts do not select physical placement. `local` does not mean CUDA local memory; `global` does not mean CUDA global memory. Address space (`Global`, `Shared`, `Local`) remains an independent IR property. `unique` does not guarantee race freedom across GPU threads, and `aliased` does not imply read-only access. `immutable` is readable but never writable; a read-only formal may accept an immutable actual, while an immutable formal requires an immutable actual.

Permission compatibility is capability based: read-only accepts read-only, read-write, or immutable actuals; write-only accepts write-only or read-write; read-write accepts only read-write; immutable accepts only immutable. This keeps the stronger “no mutable aliases exist” promise of `immutable` distinct from ordinary read-only access.

## Core GPU objects

`Gpu_type` distinguishes `Tensor(shape, dtype)` logical values from `MemRef(shape, dtype, address_space)` addressable storage. Dimensions can be static, symbolic, or dynamic. OxCaml `float gpu_array` arguments currently import as dynamic one-dimensional global f32 MemRefs; element-address calculations produce temporary pointer types during lowering. The current PTX ABI erases that MemRef shape to a device pointer; runtime extents and strides are not passed yet. These elementwise kernels operate on scalar-per-lane values, so they do not construct Tensor values yet. Ownership, locality, permission, and portability remain facts on IR values, separate from both the MemRef address space and Tensor shape.

`Layout.Register` records size per thread, threads per warp, warps per CTA, and dimension order. `Layout.Shared` records vector width, dimension order, and an optional 32/64/128-byte swizzle. The verifier checks rank, positive factors, permutation validity, and exact coverage for static register-tile dimensions. `Shared_alloc`, `Load_tensor`, `Store_tensor`, and `Barrier Cta` are explicit IR operations; the straight-line verifier rejects shared reads before initialization or before a CTA barrier. Global MemRefs intentionally have no explicit layout descriptor yet.

The next source-level GPU operation is [dot_product](examples/kernels/dot_product.ml): one 32-lane warp multiplies corresponding f32 values, reduces their sum with butterfly shuffles, and lane zero writes the result. `Gpu.warp_sum_f32` and `Gpu.store_lane0` are explicit GPU primitives; the arithmetic, loads, and function structure remain ordinary OxCaml. The current implementation requires exactly one full warp and exactly 32 input elements.

Execution levels are a separate hierarchy: `Grid → Cta → Warpgroup → Warp → Lane`. These levels describe execution coordinates/scopes and do not choose storage. Warpgroup is a target-defined grouping; the current kernels only use CTA-local thread indexing, and the other levels are semantic vocabulary for future lowering.

The importer preserves modal facts that OxCaml exposes in its typedtree. OxCaml domain portability is recorded separately from the intended GPU host/device portability contract; that GPU boundary fact remains unspecified unless a future frontend extension supplies it. The adapter and body importer currently support only the source constructs used in these two examples.

## Build and run

Run the semantic verifier tests with `dune runtest`. `test/check_oxcaml_signature.sh` checks the compiler-to-IR mode path, including the reduction. With CUDA modules and a local H100, `test/run_h100.sh` assembles `vector_add`, `saxpy`, and `dot_product` PTX for `sm_90` and executes them.

`bench/run.sh` builds the OxCaml kernels and compares vector_add/SAXPY with PyTorch operations. It also compares the 32-element OxGPU dot product with `torch.dot`, which uses PyTorch's CUDA dot implementation (cuBLAS for contiguous f32 vectors). The harness validates results before timing and reports CUDA-event medians. The elementwise kernels still launch exactly four lanes with no grid-stride loop or bounds check, so their `N=4` measurements are launch-latency wiring checks, not throughput comparisons; dot product uses one full warp (`N=32`).

The PTX backend remains small and assumes launch geometry compatible with each kernel. The next work, in order, is: generalize launch indexing and bounds checks; extend the dot reduction from one warp to CTA/multi-warp shapes; then use retained `unique` facts for a load/store optimization and show the resulting PTX difference. Matvec follows naturally once shape-aware tiling and scalable launches exist.

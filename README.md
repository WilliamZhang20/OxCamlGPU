# OxCamlGPU

OxCamlGPU is a GPU programming DSL built from OxCaml’s high-performance programming model. Kernel authors write ordinary OxCaml functions, with modal types describing buffer access and a small `Gpu` API for indexing, memory operations, and collectives. The project carries compiler-derived facts into GPU IR, verifies them, and emits PTX.

This is an active prototype. The source language and PTX backend support a
narrow kernel subset. The OxCaml frontend exposes **compute hierarchy**
(indices, barriers, elect) and **memory hierarchy** (global/shared, vector
and async copies, TMA/mbarrier, WGMMA descriptors) — not a matmul expand.
The performant Hopper GEMM schedule is the OxCaml kernel
[`examples/kernels/gemm.ml`](examples/kernels/gemm.ml):
one function of the tile constants, specialized to the catalog bindings.
[`gemm_bf16.ml`](examples/kernels/gemm_bf16.ml) is the same schedule at bf16
inputs — the element type rides on the kernel's `bf16 tensor_map` formals and
picks `wgmma.mma_async.…k16.f32.bf16.bf16`, with a byte-identical shared
window.

On an H100 that kernel holds a repeatable **mid-90s percentage of cuBLAS** at
4096³ TF32 and runs at parity at 8192³. The bf16 schedule reaches
**764 TFLOPS at 8192³, 98.6% of cuBLAS bf16** — 2.03× the TF32 kernel in the
same run, which is the bf16-to-TF32 WGMMA ratio, so the width is all that
changed. Under Nsight Compute the durations
match within 1%, at 88.9% and 93.7% SM throughput against cuBLAS's 88.6% and
93.2%, and cuBLAS independently picks the same 128×256×32 tile. See
[measured GEMM performance](docs/architecture.md#measured-gemm-performance).

## How it works

```text
OxCaml source → OxCaml Typedtree → metadata → Kernel_ast → GPU IR
                                      → verify → optimize → PTX IR → PTX
```

The version-matched adapter in `tools/` reads OxCaml `.cmt` files. The host library imports the metadata, lowers the supported source subset, verifies GPU semantics, and emits PTX. See [the architecture guide](docs/architecture.md) for stage responsibilities and current limits.

## Current programming model

Examples live in [`examples/kernels/`](examples/kernels/). They use ordinary `let`, scalar arithmetic, and conditionals. GPU-specific operations come from [`Gpu_dsl`](lib/dsl/gpu_dsl.mli): thread and global indices, buffer loads and stores, masks, and a warp sum.

Authors write scalar loads and stores; the backend fuses consecutive stores
into vector stores where it can prove the indices are consecutive and the
first is aligned, so hand-vectorizing an epilogue is not part of the
authoring model. See [compilation stages](docs/architecture.md#compilation-stages).

The accepted subset includes scalar f32/i32/bool values, one-dimensional f32
buffers, simple `let` bindings, local and higher-order functions, bounded
recursion, sequencing, scalar conditionals, ascending `for` loops, and the
listed GPU operations. Shared tiles, TMA maps and WGMMA operands may also be
`bf16` or `f16`: narrow floats are a storage format with no register form, so
the accumulator stays f32 and the element type is a phantom tag rather than a
scalar. Float math covers the Stdlib operations that round the way OCaml does
(`-.`, `/.`, `sqrt`, `Float.max`, `float_of_int`, …) plus the hardware's
approximate unit under names that say so (`Gpu.exp2_approx`,
`rsqrt_approx`, `tanh_approx`, …). Kernels return `unit`. A kernel binding may
carry `[@@gpu.threads N]` to set the CTA width. Constant `for` loops of at
most 64 iterations are unrolled. `downto`, `ref` cells, shaped source tensors,
and buffer-valued branches are not supported yet. The matmul authoring model (explicit shared, barriers,
and WGMMA — not a black-box matmul) is in
[GEMM frontend design](docs/gemm-frontend.md).

OxCaml `unique`, `aliased`, locality, portability, and visibility modes are represented as GPU semantic facts. They remain distinct from physical memory placement and GPU address space. See [mode and memory semantics](docs/architecture.md#types-modes-and-memory).

## Build and run

The OxCaml compiler and its compiler-libs must come from the same build. Configure `OXCAML_ROOT`, or set `OXCC`, `OXOPT`, `OXCAML_MAIN_BUILD`, and `OXCAML_STDLIB_DIR` for a nonstandard installation. The tested source revision is recorded in [`tools/oxcaml-revision`](tools/oxcaml-revision).

The bridge scripts also read an optional, git-ignored `.oxcaml-config` in the project root. This keeps a machine-local compiler path out of global shell startup files:

```sh
printf 'OXCAML_ROOT=%q\n' /path/to/oxcaml-checkout > .oxcaml-config
```

The checkout must be at the revision in `tools/oxcaml-revision` and built with `autoconf && ./configure --prefix=... && make install`. `make install` also populates the configured prefix that the built compiler needs at runtime. On machines using opam, activate the switch that provides OCaml 5.4.x, Dune 3.23+, and Menhir 20231231 before building. The test and metadata scripts prepend that checkout's `_build/_bootinstall/bin` to `PATH` automatically.

```sh
tools/compile_oxcaml_kernels.sh /tmp/oxcamlgpu-metadata
dune exec test/emit_ptx.exe -- /tmp/oxcamlgpu-metadata/saxpy.gpu
```

Run the host-side checks with `dune runtest`. CUDA execution and benchmarking instructions are in [the architecture guide](docs/architecture.md#validation-and-benchmarks).

## Project notes

- [Architecture, semantics, and implementation limits](docs/architecture.md)
- [Matmul frontend design](docs/gemm-frontend.md)
- [Frontend expansion plan](docs/frontend-expansion.md)
- [Loop design](docs/loops.md)

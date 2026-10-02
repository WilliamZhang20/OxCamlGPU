# OxCamlGPU

OxCamlGPU is a GPU programming DSL built from OxCaml’s high-performance programming model. Kernel authors write ordinary OxCaml functions, with modal types describing buffer access and a small `Gpu` API for indexing, memory operations, and collectives. The project carries compiler-derived facts into GPU IR, verifies them, and emits PTX.

This is an active prototype. The source language and PTX backend support a narrow kernel subset; shaped tensor operations, tiled matmul, and tensor-core lowering are not available yet.

## How it works

```text
OxCaml source → OxCaml Typedtree → metadata → Kernel_ast → GPU IR
                                      → verify → optimize → PTX IR → PTX
```

The version-matched adapter in `tools/` reads OxCaml `.cmt` files. The host library imports the metadata, lowers the supported source subset, verifies GPU semantics, and emits PTX. See [the architecture guide](docs/architecture.md) for stage responsibilities and current limits.

## Current programming model

Examples live in [`examples/kernels/`](examples/kernels/). They use ordinary `let`, scalar arithmetic, and conditionals. GPU-specific operations come from [`Gpu_dsl`](lib/gpu_dsl.mli): thread and global indices, buffer loads and stores, masks, and a warp sum.

The accepted subset includes scalar f32/i32/bool values, one-dimensional f32 buffers, simple `let` bindings, sequencing, scalar conditionals, and the listed GPU operations. Kernels return `unit`. Loops, general helper calls, shaped source tensors, and buffer-valued branches are not supported.

OxCaml `unique`, `aliased`, locality, portability, and visibility modes are represented as GPU semantic facts. They remain distinct from physical memory placement and GPU address space. See [mode and memory semantics](docs/architecture.md#types-modes-and-memory).

## Build and run

The OxCaml compiler and its compiler-libs must come from the same build. Configure `OXCAML_ROOT`, or set `OXCC`, `OXOPT`, `OXCAML_MAIN_BUILD`, and `OXCAML_STDLIB_DIR` for a nonstandard installation. The tested source revision is recorded in [`tools/oxcaml-revision`](tools/oxcaml-revision).

```sh
tools/compile_oxcaml_kernels.sh /tmp/oxcamlgpu-metadata
dune exec examples/saxpy.exe -- /tmp/oxcamlgpu-metadata/saxpy.gpu
```

Run the host-side checks with `dune runtest`. CUDA execution and benchmarking instructions are in [the architecture guide](docs/architecture.md#validation-and-benchmarks).

## Project notes

- [Architecture, semantics, and implementation limits](docs/architecture.md)
- [Frontend expansion plan](docs/frontend-expansion.md)
- [Loop design](docs/loops.md)

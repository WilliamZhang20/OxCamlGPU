# OxCamlGPU semantic core

This prototype checks whether facts assigned by the OxCaml compiler can be carried from typed source into GPU IR and used by verification. Kernel authors write ordinary OxCaml functions. Only GPU operations such as `Gpu.thread_idx_x`, `Gpu.load`, and `Gpu.store` are special.

The sample kernels are [SAXPY](examples/kernels/saxpy.ml) and [vector_add](examples/kernels/vector_add.ml); their interfaces live beside them. OxCaml typechecks each `.mli` and `.ml` and writes `.cmti` and `.cmt` typedtree artifacts. A small compiler-libs adapter reads those artifacts and exports a stable tab-separated representation. Our importer builds the source AST and mode-bearing signature from that representation, then lowers to GPU IR.

```text
OxCaml .mli/.ml
        ↓ OxCaml typechecker
.cmti/.cmt Typedtree
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

1. The script runs `ocamlc.opt -bin-annot` on each kernel interface and implementation. This makes `.cmti` and `.cmt` artifacts containing compiler Typedtrees.
2. It builds `tools/export_typedtree_modes.ml` against that same build's `ocamlcommon.cmxa`, `ocamlfrontend.cmxa`, and `oxcaml_common.cmxa`.
3. The adapter reads the artifacts with `Cmt_format`, exports argument/result types and modes plus the supported function body as tab-separated `.gpu` metadata.
4. `Oxcaml_frontend` and `Kernel_ast` import that metadata; the existing lowering, verifier, and PTX stages then run.

The compiler and compiler-libs must come from the **same OxCaml build**: their artifact formats and Typedtree APIs are build-specific. For another local build, set `OXCAML_ROOT` or override all paths with `OXCAML_MAIN_BUILD`, `OXCAML_STDLIB_DIR`, `OXCC`, and `OXOPT`. Example:

```sh
tools/compile_oxcaml_kernels.sh /tmp/oxgpu-metadata
dune exec examples/saxpy.exe -- /tmp/oxgpu-metadata/saxpy.gpu
```

This is an initial adaptation point, not a replacement OxCaml parser or a general typedtree-to-GPU lowering. The adapter currently accepts only the two examples' types and expression forms. The next integration increment should broaden the typedtree mapping while retaining compiler-provided mode information.

## Mode contract

| Mode | GPU semantic fact |
| --- | --- |
| `unique` | Exclusive writable access / noalias. Mutation permission remains a separate fact. |
| `aliased` | Other references may designate the same storage. |
| `local` | The value cannot escape its current region or kernel scope. |
| `global` | The value may outlive its current region. |
| `portable` | The value may cross a host/device or asynchronous execution boundary. |

These facts do not select physical placement. `local` does not mean CUDA local memory; `global` does not mean CUDA global memory. Address space (`Global`, `Shared`, `Local`) remains an independent IR property. `unique` does not guarantee race freedom across GPU threads, and `aliased` does not imply read-only access.

The importer preserves modal facts that OxCaml exposes in its typedtree. OxCaml domain portability is recorded separately from the intended GPU host/device portability contract; that GPU boundary fact remains unspecified unless a future frontend extension supplies it. The adapter and body importer currently support only the source constructs used in these two examples.

## Build and run

Run the semantic verifier tests with `dune runtest`. `test/check_oxcaml_signature.sh` checks the compiler-to-IR mode path. With CUDA modules and a local H100, `test/run_h100.sh` compiles both kernels, assembles their PTX for `sm_90`, and executes them.

The PTX backend is intentionally small, assumes a launch grid sized for the input, and supports only the operations used by these examples. The next research step is a load/store optimization whose legality depends on a retained `unique` fact, making the modal path observable in generated code.

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

This invokes OxCaml as the frontend; we do not embed or fork the full compiler. The adapter uses OxCaml `Typedtree`/`Cmt_format` libraries and therefore must be built against the same OxCaml build that typechecks the source. Set `OXCAML_ROOT`, `OXCAML_MAIN_BUILD`, `OXCAML_STDLIB_DIR`, `OXCC`, and `OXOPT` when your installation is not at the local development defaults in `tools/compile_oxcaml_kernels.sh`.

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

Run the semantic verifier tests with `dune runtest`. To typecheck, import, verify, and emit PTX for the OxCaml examples, run `tools/compile_oxcaml_kernels.sh /tmp/oxgpu-metadata`, then `dune exec examples/saxpy.exe -- /tmp/oxgpu-metadata/saxpy.gpu` (or `vector_add.exe`). `test/check_oxcaml_signature.sh` checks the compiler-to-IR mode path. With CUDA modules and a local H100, `test/run_h100.sh` compiles both kernels, assembles their PTX for `sm_90`, and executes them.

The PTX backend is intentionally small, assumes a launch grid sized for the input, and supports only the operations used by these examples. The next research step is a load/store optimization whose legality depends on a retained `unique` fact, making the modal path observable in generated code.

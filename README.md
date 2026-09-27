# OxCamlGPU semantic core

This prototype makes modal facts explicit in a small, manually constructed GPU IR. It tests ownership and lifetime rules before building a full OxCaml compiler plugin or a substantial backend.

## Proposed GPU mode contract

The initial intended mapping is:

| Mode | GPU value fact |
| --- | --- |
| `unique` | Exclusive reference/noalias fact. Write permission is a separate fact, represented by `permission`; a writable unique buffer therefore needs both facts. |
| `aliased` | Other references may designate the same storage. Aliasing alone does not grant or deny mutation; `permission` is represented separately. |
| `local` | The value cannot escape the current region or kernel scope. |
| `global` | The value may outlive the current region. This describes lifetime only. |
| `portable` | The intended GPU meaning is that a value may cross a host/device or asynchronous execution boundary. |

These modes do not choose a physical address space. In particular, `local` does not mean CUDA local memory, `global` does not mean CUDA global memory, and `portable` does not mean unified memory or imply a transfer. Address space (`Global`, `Shared`, `Local`) is an independent IR type property. `unique` does not itself guarantee race freedom between GPU threads, and `aliased` is not synonymous with read-only. IR stores OxCaml domain portability separately from the GPU boundary fact so an OxCaml annotation cannot silently claim device portability.

## Current prototype

The IR has explicit values, pointer address spaces, ownership, locality, portability, mutation permission, and instructions for scalar f32 arithmetic and global-memory element access. `Verifier` checks IR well-formedness, local returns, writes through read-only values, and call-site aliasing of unique formals. `Kernel.vector_add` and `Kernel.saxpy` are manually authored. `Ptx.emit` prints a deliberately small PTX subset.

`Oxcaml_frontend` imports a restricted subset of OxCaml `.mli` function signatures and maps explicit `@ unique`, `@ aliased`, `@ local`, `@ global`, `@ portable`, and `@ nonportable` facts into corresponding source-mode fields in the IR. It also maps OxCaml visibility modes `read`, `write`, `read_write`, and `immutable` to access permission. On the axes it represents, omitted modes use OxCaml's legacy defaults (`global aliased nonportable read_write`); other axes such as contention, linearity, and statefulness are outside this IR. It currently supports one arrow type per declaration and a few primitive types. It is a source adapter, not an OxCaml compiler plugin: OxCaml must type-check the interface separately. Try `dune exec examples/import_signature.exe -- frontend/saxpy.mli` to inspect the imported facts.

OxCaml's `portable` means that a value can move between domains. The importer preserves this as `domain_portability`; it does not infer the separate `gpu_boundary` fact, which remains `Boundary_unspecified`. Thus the proposed GPU `portable` contract is not yet carried by ordinary OxCaml modes. OxCaml `unique` is likewise only imported as the uniqueness/noalias fact; it does not grant write permission. The SAXPY interface states `read` and `read_write` explicitly so the source modes support the intended read-only input and writable output. OxCaml locality is a future mode and may mode-cross for some types; this importer records the signature constraint but does not reproduce the compiler's type-dependent mode-crossing analysis.

Build and run the semantic checks with `dune runtest`. Print PTX with `dune exec examples/vector_add.exe` or `dune exec examples/saxpy.exe`.

With a CUDA toolkit module loaded, run `test/run_h100.sh` to assemble both PTX files for `sm_90` and launch them through the CUDA driver API. The checked-in runner expects `ptxas`, `nvcc`, and access to an NVIDIA GPU. With an OxCaml compiler on `PATH` (or `OXCC` set to its `ocamlc`), run `test/check_oxcaml_signature.sh` to check the sample interfaces using OxCaml's parser and type checker. The local build from this environment is at `/tmp/oxcaml-src/_build/_bootinstall/bin/ocamlc.opt`; use `OXCC=/tmp/oxcaml-src/_build/_bootinstall/bin/ocamlc.opt test/check_oxcaml_signature.sh` to run the check here.

The PTX is intentionally an early prototype: it maps one thread to one element, does not include a length/bounds check, and requires a launch grid sized for the input. `ptxas` validation and H100 execution depend on CUDA toolkit/runtime availability. OxCaml was built locally from its upstream source with `make hacking` against OCaml 5.4.1, Dune 3.23.0, and Menhir 20231231. The source checkout and compiler live under `/tmp/oxcaml-src`; the system compiler was left untouched. `make hacking` provides the compiler frontend used for `.mli` checks here, without building/installing the full standard library.

## Stretch Goals

Implement the following kernels in the OxCaml DSL and beat corresp. NVIDIA libraries (all on H100 80GB HBM3):
- Matmul
- Einsum
- QR Decomp
- Orthogonal Decomp and SVD
- Cross-entropy loss
- Flash Attention Fwd/Bwd

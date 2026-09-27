# OxCamlGPU semantic core

This prototype tests whether OxCaml modal facts can survive a typed kernel source layer, reach GPU IR, and affect verification. Kernel authors write ordinary OCaml bindings and floating point expressions. Only GPU concepts are explicit: thread indexing and buffer access.

```ocaml
let saxpy x y a =
  let open Kernel_source.F32 in
  let i = Kernel_source.Gpu.thread_idx_x () in
  Kernel_source.Gpu.store y i
    (a *. Kernel_source.Gpu.load x i +. Kernel_source.Gpu.load y i)
```

This is staged OCaml: values are symbolic while the function runs under `Kernel_source.compile3_bbf`. Kernel authors do not construct low-level instructions or SSA values. The current staging API covers only three argument shapes and f32 elementwise kernels; it is an experiment, not a general OCaml quotation system.

## Mode contract

The initial meaning assigned to GPU values is deliberately small:

| Mode | Semantic fact |
| --- | --- |
| `unique` | Exclusive writable access / noalias, subject to a separate permission fact for whether this particular reference may be written. |
| `aliased` | Other references may designate the same storage. |
| `local` | The value cannot escape its current region or kernel scope. |
| `global` | The value may outlive its current region. |
| `portable` | Intended GPU contract: the value may cross a host/device or asynchronous execution boundary. |

These modes do not determine physical placement. `local` does not mean CUDA local memory, and `global` does not mean CUDA global memory. Address space (`Global`, `Shared`, `Local`) is an independent IR type property. `unique` does not guarantee race freedom between GPU threads; `aliased` does not mean read-only. Lifetime and ownership facts stay separate from memory placement and mutation permission.

OxCaml currently assigns `portable` to movement between domains. The signature importer preserves that as `domain_portability`; it does not infer GPU boundary portability, which remains unspecified. This means the intended GPU `portable` contract is documented but not yet encoded by ordinary OxCaml modes. `unique` imports only the uniqueness fact and does not grant write permission. The sample SAXPY interface uses `read` and `read_write` explicitly. The importer handles a small `.mli` signature subset and does not invoke OxCaml typedtree internals or reproduce type-dependent mode crossing.

## Compiler layers

```text
kernel author OCaml function
        ↓
Kernel_source staged expressions
        ↓
Kernel_ast typed source AST
        ↓
Kernel_frontend attaches imported modal facts
        ↓
Ir typed instruction representation
        ↓
Verifier
        ↓
PTX printer
```

`Oxcaml_frontend` imports explicit facts from one restricted `val` declaration. `Kernel_frontend` checks argument types and attaches those signature facts to IR values. `Verifier` checks well-formed uses, local escapes, mutation permissions, call-site unique requirements, and aliased actual buffers passed to unique formals. The compiler examples live under `examples/`; there are no sample kernel definitions in the compiler implementation modules.

## Build and run

Run semantic and lowering checks with `dune runtest`. Print PTX with `dune exec examples/vector_add.exe` or `dune exec examples/saxpy.exe`. With the CUDA toolkit module and H100 available, `test/run_h100.sh` assembles both outputs for `sm_90` and executes them through the CUDA driver API.

The sample interfaces can be checked with OxCaml using `OXCC=/path/to/ocamlc.opt test/check_oxcaml_signature.sh`. In the original development environment, the locally built compiler was at `/tmp/oxcaml-src/_build/_bootinstall/bin/ocamlc.opt`.

The PTX backend remains intentionally small. It has no bounds checks, assumes a launch grid sized for the input, and supports only the instructions needed by these examples. The first research follow-up is an optimization whose legality depends on a retained `unique` fact, so the modal path has an observable code generation consequence.

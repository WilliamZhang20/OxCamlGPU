# Frontend expansion: adapter hardening and structured conditionals

Status: implemented for the scalar/conditional subset described below. The original
plan is retained as design context. See README for the supported numeric/ABI
restrictions and `docs/loops.md` for the follow-up plan. Target branches currently
use a linear label/edge instruction stream with explicit scalar edge moves;
multiple block parameters are deferred until loops need parallel copies.
Signature modes are transported; arbitrary local Typedtree modal axes are not.

The completion milestone is an OxCaml kernel using nested ordinary `let` and
`if` expressions, including a guarded memory access, compiled through the real
Typedtree adapter and executed on the H100. Loops, helper functions, and new
collective strategies follow this milestone.

## Branches and divergence

A source `if` becomes semantic control flow with a condition and two regions.
Uniformity analysis determines whether participating threads agree on that
condition. Target lowering emits a predicate and conditional PTX branch; a
short pure expression may eventually be predicated, but that is an optimization.
The initial implementation uses branches and never speculates loads or stores
from an unselected arm.

PTX `bra` permits divergent execution. The `.uni` modifier asserts uniformity;
it is not a mechanism for making a divergent branch uniform. Initially emit
ordinary `bra` for all conditionals, including proven uniform ones. Retain the
analysis for collective legality before using it for code-generation shortcuts.
`ptxas` selects the machine control-flow implementation. Explicit vote, shuffle,
and synchronization instructions are for collective semantics, not a mandatory
translation of every source branch.

Track agreement separately at grid, CTA, and subgroup scope. Constants and
scalar kernel parameters are grid-uniform. Thread/global indices and memory
loads are conservatively varying. Pure operations inherit the weakest agreement
of their operands. A branch result also depends on its selector: selecting
different uniform constants with a varying condition produces a varying result.
Uniformity is derived from canonical definitions, never trusted from metadata.

Execution participation is a separate fact from value uniformity. Even a constant
condition nested inside divergent control flow does not make the active set a
full subgroup. Full-mask shuffle reduction is legal only where every required
lane reaches the same collective. CTA barriers require CTA-wide participation.
Initially reject collectives inside control flow that is not proven uniform at
their required scope. Do not substitute an active mask and silently change the
meaning of a full-group reduction.

With both arms completing normally, a structured join restores the enclosing
participation obligation. It does not itself synchronize memory or promise
lockstep execution; subsequent collectives still use their synchronization
semantics. Source early exits are excluded from this increment, so partial
termination cannot invalidate that rule.

References: NVIDIA's [PTX branch specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#control-flow-instructions-bra)
and [shuffle synchronization specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync).

## Step 1: one typed adapter contract

Current problems: `Gpu_metadata` owns the operation definitions, `Kernel_ast`
re-exports them, and the exporter manually writes matching strings. Example
drivers call the metadata parser separately for the signature and body. Source
locations are absent. Primitive and type recognition rely on textual names.

| File | Planned responsibility/change |
| --- | --- |
| `lib/base/source_span.ml` (new) | Compiler-independent file/start/end positions, plus an explicit synthetic location for manually constructed fixtures. No dependency on OxCaml `Location`. |
| `lib/mode.ml` → `lib/base/gpu_mode.ml` | Rename the project mode module to avoid collision with OxCaml compiler-libs' `Mode` when compiling shared schema sources into the adapter. Preserve its semantics. Update imports mechanically. |
| `lib/frontend/kernel_ast.ml` | Own the canonical frontend types: complete signature, typed value definitions/references, supported modal facts, source spans, and body. Remove aliases to codec-owned operation types and redundant argument-kind descriptions. No dependency on backend `Ir`. |
| `lib/frontend/gpu_metadata.ml` | Encode/decode `Kernel_ast.t`; own wire syntax and version validation only. Add a version 3 format with explicit region boundaries, escaped source paths, locations, declared result types, and typed mode fields. The format remains a transport for operations, not a recursive expression language. |
| `tools/export_typedtree_modes.ml` | Construct the shared AST from resolved Typedtree nodes, then invoke the shared encoder once. Buffer the entire result so failure cannot leave a plausible partial artifact. Convert compiler locations and supported structured mode constants explicitly. |
| `tools/compile_oxcaml_kernels.sh` | Compile the small shared schema/codec source closure with the pinned OxCaml toolchain in its build directory. Do not link host-built `.cmx` files into the compiler adapter. |
| `lib/frontend/oxcaml_frontend.ml` | Provide one import entry point returning the complete typed frontend kernel from one decode. Stop manufacturing a separate signature from a second parse. |
| `lib/frontend/kernel_frontend.ml` | Consume that complete typed kernel. Validate IDs, argument references, source/kernel identity, types, and supported modes before constructing semantic IR. Resolve operands before publishing their result definition. |
| `lib/dune` | Register the renamed/new modules with the dependency direction `source types → codec → import/lowering`. |
| `examples/*.ml`, `test/emit_ptx.ml`, `test/check_compiler_metadata.ml` | Use the single import entry point; remove duplicate parsing. |

Resolve GPU API types and primitives through the compiler environment and
declaration identity, following supported module aliases. Local names or another
module's similarly named `Gpu.load`/`gpu_array` must not acquire GPU semantics.
Treat standard arithmetic and comparisons the same way; reject unsupported
overloads or calls with the location of the actual expression.

Preserve supported source facts as facts about their particular binding/use.
Formal-root uniqueness remains distinct from derived-reference aliasing. If the
pinned compiler does not expose a required local fact through a stable API,
report it as unsupported or unknown; do not infer an ownership guarantee from a
printed annotation. Other modal axes remain explicitly outside this subset.

Keep the schema implementation portable between the host OCaml build and the
pinned OxCaml build. `Gpu_type`, `Gpu_mode`, `Source_span`, `Kernel_ast`, and
`Gpu_metadata` form the small shared source closure. This avoids both a second
handwritten encoder and dependency on backend/compiler-libs internals in the
host compiler.

Step 1 acceptance:

- All current examples retain their modes, verification outcomes, and results.
- Unsupported source reports its real file and span; primitive/type lookalikes
  are rejected, while supported aliases resolve correctly.
- Malformed versions, missing mode fields, duplicate/negative IDs, undefined
  references, and argument/type mismatches are rejected deterministically.
- Adapter serialization and host decoding agree using the real pinned compiler.
- No generated compiler artifacts enter the repository.

## Step 2: ordinary scalar expressions and structured branches

Accepted source additions: nested nonrecursive single-name `let`, boolean
literals, supported scalar comparisons, short-circuit `&&`/`||`, `unit`, and
`if` with scalar or unit results. `if` without `else` has an implicit unit arm.
Buffer-valued branch results, source early exits, loops, exceptions, and general
calls remain rejected until their corresponding semantics are implemented.

Use structured SSA regions in semantic IR. An `If` has a condition, two regions,
and zero or one scalar result. Each region ends in an explicit `Yield`; both
arms must produce the same type. Ordinary kernel completion is distinct from a
region yield. Branch-local definitions cannot escape except through the declared
result. This leaves an appropriate structure for later loop-carried values.

Numeric policy must be explicit before expanding operators. Existing GPU
`float` expressions have f32 semantics, not host OCaml binary64 semantics;
preserve that documented subset and specify comparison behavior for NaNs and
signed zero. Do not add reassociation or contraction in this increment.
Existing `int` arguments/indices have a bounded i32 ABI. Do not silently give
arbitrary OCaml native-int arithmetic wrapping-i32 semantics. Initially expose
explicit `Int32` arithmetic with its defined wrapping behavior; accept ordinary
`int` arithmetic only when range analysis proves that narrowing and all
intermediate results are safe, otherwise emit a source diagnostic. Native-int
ABI range checks are required for supported entry arguments. General native-int
semantics or a wider ABI need a separate design.

| File | Planned responsibility/change |
| --- | --- |
| `lib/base/gpu_type.ml` | Add internal `Bool`; distinguish supported source numeric kinds in the frontend mapping. Define which types may be branch results. |
| `lib/frontend/kernel_ast.ml` | Add typed comparisons, scalar integer operations, structured conditional regions and yields, carrying source spans. Keep lexical bindings in the adapter environment and explicit values in the AST. |
| `lib/frontend/gpu_metadata.ml` | Encode/decode the structured AST through the shared versioned codec; validate nested region structure rather than flattening both arms into unconditional operations. |
| `tools/export_typedtree_modes.ml` | Visit nested `let`, `if`, unit, boolean and supported operator nodes in expression/effect positions. Recognize short-circuit operators before ordinary call handling. Preserve sequencing and supported evaluation semantics. |
| `lib/frontend/kernel_frontend.ml` | Lower both arms in independent lexical environments; create fresh result definitions and explicit yields. Apply numeric range restrictions and preserve modal/provenance facts. |
| `lib/ir/ir.ml` | Introduce regions, typed `If`, comparisons, and explicit terminators. Provide region-aware traversal; distinguish immediate operands from nested uses for dominance checks. Keep canonical SSA identities. |
| `lib/verify/verifier.ml` | Verify nested scopes, operand/result types, yields, terminators and dominance. At joins, definite shared initialization is the intersection of both paths; pending writes are conservatively combined. Neither a store nor a barrier in one arm establishes a fact on the other arm. |
| `lib/verify/uniformity.ml` (new) | Compute value agreement and region participation from verified SSA. Return analysis keyed by value/region IDs; avoid duplicating untrusted annotations on operands. Model selector dependence at joins. |
| `lib/transform/optimizer.ml` | Recurse through regions. Initially clear cached-load facts at branch boundaries and joins, and keep substitutions scoped to their arm. Retain current straight-line unique reuse. No speculative loads or cross-arm reuse. |
| `lib/backend/ptx_ir.ml` | Add predicate comparisons and explicit target blocks, edges and join parameters. Semantic `If` stays out of target IR. Target labels/predicate identifiers must be allocated centrally. |
| `lib/backend/ptx_lowering.ml` (new) | Move existing target legalization out of the printer. Lower structured branches to target blocks, translating scalar yields into join-edge moves. Check collective participation, recursively validate layouts, and derive launch/shared-memory requirements over every region. |
| `lib/backend/ptx.ml` | Print target labels, predicates, branches and edge moves. Preserve explicit returns. Keep memory operations inside their selected arms; emit ordinary `bra` initially. |
| `lib/compiler.ml` | Orchestrate import/lowering, semantic verification and uniformity, optimization, re-verification/re-analysis, target legalization, and emission. Derived analyses cannot survive transformations without recomputation. |
| `lib/dsl/gpu_dsl.ml`, `lib/dsl/gpu_dsl.mli` | Keep ordinary conditionals/operators out of the GPU API. Add only required explicit numeric conversions/range contracts. No `Gpu.if` or divergence syntax for kernel authors. |

Branch result permission/ownership joins are deliberately avoided initially by
restricting results to scalars/unit. Argument buffer accesses inside either arm
still use their original permission and provenance. Ordinary aliasing of named
buffer locals must never create a new unique root.

The first kernel milestone is a guarded SAXPY variant:

```ocaml
let i = Gpu.global_idx_x () in
if i < n then begin
  let xi = Gpu.load x i in
  let contribution = if xi > 0. then a *. xi else 0. in
  Gpu.store y i (contribution +. Gpu.load y i)
end
```

The surrounding binding retains its existing OxCaml mode annotation. The outer
branch must prevent out-of-range memory operations on the tail. The inner
branch exercises a varying condition and a scalar join. Keep ordinary SAXPY as
the existing example; this focused conditional kernel can be a test fixture.

Step 2 acceptance:

- Uniform parameter conditions, lane-varying conditions, nested conditionals,
  zero-work cases, and nonmultiple launch sizes produce correct H100 results.
- Untaken arms cannot perform stores or invalid loads. Short-circuit expressions
  do not evaluate their right operand when the left operand determines the result.
- Branch-local values, mismatched yields, uninitialized join values, forged SSA
  facts, and incompatible scalar operations fail before code generation.
- Branches selecting distinct uniform constants remain varying when their
  selector varies. A uniform inner condition cannot repair divergent outer
  participation. Unsafe barriers/full-group reductions are rejected.
- A collective after a normally completing join is tested separately from a
  collective inside a divergent arm; a join is never treated as a memory fence.
- Unique load reuse still passes its existing tests, and a store in either arm
  cannot leave a stale cached load after the join.
- Test f32 comparison edge cases and integer range/wrapping boundaries according
  to the declared source subset, rather than comparing to incompatible host
  numeric semantics.

## Implementation and validation order

1. Shared schema, collision-free mode module, source spans, single import API.
   Add focused cases in `test/test_frontend.ml` and adapter fixtures under
   `test/fixtures/`; wire them into `test/dune` and the existing bridge script.
2. Resolved primitive/type identities and source diagnostics; validate the real
   OxCaml bridge before adding source constructs.
3. Bool/comparisons and manually built semantic `If` regions, verifier rules,
   uniformity analysis, conservative optimizer traversal. Add adversarial
   control-flow cases in `test/test_control_flow.ml`.
4. PTX target blocks, edge moves, region-aware launch legalization and branch
   printing. Assemble generated branches and execute direct-IR control-flow
   fixtures before enabling source lowering.
5. Typedtree lowering for nested expressions, `if`, and short-circuit operators;
   run the guarded SAXPY fixture through the full pipeline.
6. Numeric operations and their range rules; update `README.md` with the precise
   accepted source subset. Extend `test/hardware/run_h100.sh` and `test/hardware/run_h100.cu` for
   correctness checks using the existing harness, with PyTorch where applicable.

No loop or new reduction primitive is needed to pass these gates. The key result
is a trustworthy control-flow path from OxCaml through verification to PTX, on
which the next loop/matvec increment can depend.

# Functions, closures and higher-order code on the GPU

Status: assessment and plan. The measurements below were taken by running the
adapter over small fixtures; `test/fixtures/functional.ml` pins the accepted
cases in the test suite.

## The model the adapter already has

`tools/export_typedtree_modes.ml` is a partial evaluator over the Typedtree,
not a code generator for functions. A `Texp_function` evaluates to a
compile-time value

```ocaml
| Func of { params; body; env }
```

carrying the environment it was defined in, and `local_apply` traces the body
with the arguments bound. Every call is inlined. That is defunctionalization
by total inlining, and it is why a good deal of functional code already works
with no runtime representation at all: no closure record, no function
pointer, no indirect branch.

## What reaches PTX today

Verified by running the adapter:

| Construct | Status |
| --- | --- |
| Local function, fully applied | works |
| Closure over a kernel formal or local | works |
| Higher-order: function passed as an argument, then applied | works |
| Lambda passed as an argument | works |
| Polymorphic local function used at several types | works |
| `\|>` and composition chains | works |
| Structure-level function as the kernel's specialization | works |

The last row is how `examples/kernels/gemm.ml` gets nine tile bindings out of
one `hopper_gemm`: the binding is a partial application at structure level,
and `prior_functions` resolves it.

`test/fixtures/functional.ml` applies `twice` to a closure and to a lambda and
uses a polymorphic identity. It lowers to two multiplies, two adds and a
store. The identity leaves nothing behind.

## What is rejected today, and why

| Construct | Message | Real cause |
| --- | --- | --- |
| `(make 5) i` | call target must be a resolved declaration | `local_apply` requires the callee to be a `Texp_ident`, so a callee that is an expression is never evaluated to a `Func` |
| `let inc = add 1` | functions must be fully applied | no representation for a partially applied `Func` |
| Function in a tuple or record | unsupported GPU expression | `binding` has no compound case |
| `if c then f else g` | functions cannot be branch results | a deliberate guard |
| `let rec` | unsupported call declaration | recursion is rejected outright |
| Structure-level helper called inside a direct kernel body | unsupported call declaration | deliberate: a direct kernel is traced with only its formals in scope |

Only the last one is a stated policy decision. Its comment says a user
function sharing a primitive's name must stay rejected, but primitives are
identified by compiler identity (`path_uid`), not by name, so shadowing is
already impossible. This looks safe to relax and is the single cheapest
usability win here.

## Tier 1: more of the same staging

All of this stays inside the existing model. Nothing reaches run time.

1. **Let direct kernels see structure-level functions.** Pass
   `prior_functions` as the initial environment in the `Texp_function` arm of
   the kernel entry, as the specialization arm already does. Unlocks ordinary
   helper libraries.
2. **Evaluate the callee.** In `local_apply`, evaluate `callee` through
   `expression` and accept a `Func` result, instead of pattern-matching on
   `Texp_ident`. Gets `(make 5) i` and any computed-but-statically-known
   callee.
3. **Partial application.** Represent a `Func` with a prefix of its
   parameters already bound. `add 1` becomes a `Func` of one remaining
   parameter. Falls out of (2) once `Func` carries bound arguments.
4. **Compound compile-time values.** Add `Tuple of binding list` (and a
   record case) so functions can travel in tuples and be destructured. This
   is what lets a kernel return several values, which is useful well beyond
   functions.
5. **Bounded recursion.** Inline `let rec` to a compile-time depth limit and
   fail past it, the way constant `for` loops are already unrolled to 64
   iterations. Covers structural recursion over a statically known bound.

## Tier 2: compile-time data structures

Staged metaprogramming, where the data exists only in the compiler. A list of
stage descriptors, or `List.fold_left` over a statically known list, unrolled
at trace time. This is the natural way to express a schedule: the pipeline
depth, the tile ladder, an epilogue plan. It needs the compound values from
Tier 1 plus recognition of a few `List` operations on literal lists. Modules
and functors applied at known arguments are the same mechanism again, just at
module level.

## Tier 3: genuine runtime first-class functions

Possible, and mostly a bad idea. What it would take:

- **Defunctionalization.** Collect the lambdas that can flow to a call site,
  tag them, and replace the indirect call with a switch on the tag. Needs a
  control-flow analysis the compiler does not have.
- **Closure environments.** Registers when the closure does not escape;
  otherwise shared or local memory. There is no heap and no collector, so an
  escaping closure has nowhere to live unless the kernel allocates a slab for
  it by hand.
- **Recursion.** PTX has `.local` stack frames and an ABI for real calls, so
  it is expressible. Divergent recursion on a GPU is ruinous in practice.

The one case worth building is narrower and already half-supported. A
defunctionalized dispatch is as cheap as a branch when its tag is uniform,
and `lib/verify/uniformity.ml` already proves grid, CTA and subgroup
uniformity and lowers uniform conditions to `bra.uni`. So the rule would be:
allow an indirect call when the analysis proves the tag uniform, and reject it
otherwise with that reason. That turns "first-class functions on a GPU" from
a performance trap into a checked property, and reuses analysis that exists.

## What stays out

Allocation and collection on device. Exceptions and effects. Polymorphism
over representations that are not known at the call site.

That last one is where OxCaml specifically helps, and it is worth being
precise about why this language rather than stock OCaml:

- **Unboxed types and layouts** make monomorphization decidable. Stock
  OCaml's uniform boxed representation is what forces a compiler to either
  box everything or specialize everything; OxCaml lets the type say which
  representation is meant.
- **Modes already carry to GPU facts.** `unique` on an output buffer is what
  licenses the no-alias reasoning in the backend today. For closures the
  relevant mode is locality: a `local` closure provably does not escape, so
  its environment can live in registers with no storage question to answer.
  That is a type-level answer to the hardest part of Tier 3, and it is not
  available in stock OCaml.

## Suggested order

Tier 1 items 1 and 2 are small, independent, and unlock ordinary functional
style inside kernels. Item 4 is worth doing for its own sake. Tier 2 is where
schedules start to read like programs rather than constants. Tier 3 should
wait for a kernel that genuinely needs it, and should be gated on uniformity
when it comes.

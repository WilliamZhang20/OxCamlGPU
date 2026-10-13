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
| Recursive function whose base case folds | works |

The last row is how `examples/kernels/gemm.ml` gets nine tile bindings out of
one `hopper_gemm`: the binding is a partial application at structure level,
and `prior_functions` resolves it.

`test/fixtures/functional.ml` applies `twice` to a closure and to a lambda and
uses a polymorphic identity. It lowers to two multiplies, two adds and a
store. The identity leaves nothing behind.

## Recursion

A recursive function is inlined like any other, so it terminates exactly when
its controlling test folds at compile time. Two pieces of the adapter already
made that work: a comparison of two compile-time constants folds to a
`Const_bool`, and `Texp_ifthenelse` on a `Const_bool` traces only the taken
arm. So a recursion stepped by a constant unrolls and the base case
disappears, while one stepped by a thread index traces both arms forever.

That is the whole rule, and it is checked. A call nested deeper than 64, the
same limit constant `for` loops unroll to, fails with the source location and
says why:

```
reject_unbounded_recursion.ml:11:45: call nested deeper than 64; a recursion
whose base case is not a compile-time constant cannot be inlined
```

### What bounds the inlining, and what deliberately does not

The compiler owes termination and a diagnostic. It does not owe cost control.
A kernel that is expensive on purpose is the author's business, the same way a
C++ compiler will instantiate a template that produces a megabyte of code
without arguing. So a limit here has to answer one of exactly two questions,
and anything that only answers "is this big?" was removed.

| Limit | Default | Why it exists |
| --- | --- | --- |
| `OXGPU_MAX_INLINE_DEPTH` | 64 | A recursion whose base case never folds does not terminate. Termination is the compiler's problem. |
| `OXGPU_MAX_NESTING` | 8192 | Deep enough expression nesting exhausts the adapter's own stack, giving an uncatchable error with no source location. |
| `OXGPU_MAX_OPERATIONS` | 33554432 | Runaway backstop only. Matches gcc's `-fconstexpr-ops-limit`. |

They are environment overrides because that is what they are: compiler flags
on compile-time evaluation, like `-fconstexpr-depth`, `-fbracket-depth`,
`-ftemplate-depth` and `recursion_limit`. Note the shape of those precedents,
which this follows: tight on depth, generous on operations.

What that buys, measured. Three nested loops each inside their own
64-iteration unroll limit multiply to 1,048,576 operations and 164 MB of
metadata; `fib 24` with a constant argument stays at depth 24 while inlining
exponentially many calls and takes about four seconds. Both are **accepted**.
They terminate, they are what the author asked for, and an earlier budget that
refused them was legislating style. A 20000-term expression is **refused**,
because that one is a host crash rather than an expense.

Catching `Stack_overflow` was tried first and does not work: the handler never
ran, since native-code OCaml cannot catch it dependably. The nesting counter
replaced it, and the dead handler was removed rather than left as false
reassurance.

None of this can reach the GPU. A kernel emits one `.visible .entry` with no
`.func`, no `call` and no `.local`, and ptxas reports a 0-byte stack frame for
every kernel here, so there is no device-side stack to overflow. When the
adapter does die unguarded it exits 2, not by a signal, with no core dump and
zero bytes on stdout, so the bridge fails and no partial artifact is left. The
only thing lost is the source location, which is the whole reason the counters
exist.

This is a different property from the one the transform passes hold, and the
two are worth not confusing. The passes never *guess*: the store vectorizer
fuses only when consecutiveness and alignment are proven and stays scalar
otherwise, and each refusal has a negative fixture. That is about correctness.
The limits above are about termination. Neither is about taste.

[`examples/kernels/block_reduce.ml`](../examples/kernels/block_reduce.ml) is
the practical form: a block reduction as a recursive tree fold parameterized
by its combining function. The same fold is reused at three different
operators in [`row_norm.ml`](../examples/kernels/row_norm.ml), where softmax
needs a maximum and then a sum over the same shared tile, and RMS norm needs
a sum of squares — one definition, no call left anywhere.

```ocaml
let rec fold stride =
  if stride = 0 then ()
  else begin
    if tid < stride then
      Gpu.shared_store sm tid
        (combine (Gpu.shared_load sm tid)
           (Gpu.shared_load sm (tid +: stride)));
    Gpu.barrier_cta ();
    fold (stride / 2)
  end
```

For size 256 that unrolls into eight steps with the operator inlined once per
step, and nothing recursive, no call and no closure survive into PTX.
`block_sum` and `block_max` are the same fold under `(fun a b -> a +. b)` and
a maximum, and differ in the emitted code only by those eight operations.
`test/hardware/run_h100.sh` checks both against a CPU reference on an H100,
and the signature test pins the step count so a regression in inlining fails
rather than quietly costing performance.

This is the shape worth noticing: log-step algorithms, tree reductions,
scans, butterflies, are all naturally recursive and all have compile-time
bounds, so they are exactly the class this handles.

## What is rejected today, and why

| Construct | Message | Real cause |
| --- | --- | --- |
| `(make 5) i` | call target must be a resolved declaration | `local_apply` requires the callee to be a `Texp_ident`, so a callee that is an expression is never evaluated to a `Func` |
| `let inc = add 1` | functions must be fully applied | no representation for a partially applied `Func` |
| Function in a tuple or record | unsupported GPU expression | `binding` has no compound case |
| `if c then f else g` | functions cannot be branch results | a deliberate guard |
| Structure-level helper called inside a direct kernel body | unsupported call declaration | deliberate: a direct kernel is traced with only its formals in scope |

Only the last one is a stated policy decision. Its comment says a user
function sharing a primitive's name must stay rejected, but primitives are
identified by compiler identity (`path_uid`), not by name, so shadowing is
already impossible. This looks safe to relax and is the single cheapest
usability win here.

## Writing CPU code in a kernel

The GPU surface is a whitelist, and `typedtree_support.ml` matches it by
compiler identity (`Types.Uid.equal` on the value description) rather than by
name. So an ordinary Stdlib function has no GPU meaning however well it
typechecks, and a user function that shadows a primitive's name is a different
uid and stays rejected. Tried against the adapter:

| In the kernel | Result |
| --- | --- |
| `print_endline`, `failwith`, `sqrt` | rejected, with a source location |
| `ref` / `!` / `:=` | rejected |
| `List.map`, `String.length` | rejected |
| `raise`, `try ... with` | rejected |
| An OCaml `array` literal, a string literal | rejected |
| A mutable record and a field assignment | rejected |
| A CPU helper defined but never applied | accepted, never traced |
| A module-level `let () = ...` initialiser | accepted, never traced |

So the answer to "what if I write CPU primitives in a kernel" is that it is a
compile error naming the line, not a miscompile and not a silent fallback to
the host. `test/fixtures/reject_cpu_primitive.ml` pins that, and
`reject_operator.ml` pins the shadowing case.

There is exactly one gap, and it follows from the design rather than being an
oversight. The adapter checks what survives specialization, so a CPU call in a
branch whose condition folds away is never traced and never seen:

```ocaml
if 1 = 2 then print_endline "unreachable";   (* accepted *)
```

That is semantically harmless, since the call cannot execute and nothing is
emitted for it. It does mean "the kernel compiled" is a claim about the
specialized program, not about every line of source. The same is true of a
helper that is defined but never applied, which is the usual and correct
behaviour for a partial evaluator.

## Tier 1: more of the same staging

All of this stays inside the existing model. Nothing reaches run time.

Bounded recursion is done, as described above. What remains:

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
5. **Substitute type variables while inlining.** The adapter reads types off
   a single Typedtree, so a polymorphic helper's body still holds `Tvar`
   nodes after the call site has fixed them. Any binding whose GPU meaning
   comes from a type argument then has nothing to read: `Gpu.shared_wgmma`
   cannot tell whether its tile is f32 or bf16, so one schedule cannot serve
   both widths and
   [`examples/kernels/gemm_bf16.ml`](../examples/kernels/gemm_bf16.ml)
   duplicates [`gemm.ml`](../examples/kernels/gemm.ml). The fix is a
   substitution carried alongside `Func`'s environment: unify each parameter's
   declared type with the argument's actual type at the call, and apply the
   result in `kind`. This is the one Tier 1 item that is about types rather
   than values, and it is the only thing standing between the two GEMMs.
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
style inside kernels. Item 5 is the one with a concrete debt behind it — a
duplicated GEMM — so it is the one to do first. Item 4 is worth doing for its
own sake. Tier 2 is where
schedules start to read like programs rather than constants. Tier 3 should
wait for a kernel that genuinely needs it, and should be gated on uniformity
when it comes.

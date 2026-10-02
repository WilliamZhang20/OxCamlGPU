# Loop frontend expansion

Status: follow-up plan only. No loop syntax is enabled by the conditional work.
The real adapter currently rejects `for`; a negative fixture keeps that boundary
explicit. The design below uses the implemented `Ir.If`, scoped regions,
`Uniformity.check`, `Ptx_lowering`, and per-region optimizer caches.

## Source subset and first useful kernel

Support ordinary bounded `for i = lo to hi do ... done` first. Evaluate both
bounds once, preserve inclusive endpoints and zero-trip behavior. Start with
unit bodies and scalar local `ref` accumulators; use f32 or explicit Int32 for
accumulation unless native-int range proofs succeed. Support `downto` through
the same representation once the ascending cases pass.

The first useful kernel computes one matrix row per thread:

```ocaml
let row = Gpu.global_idx_x () in
if row < rows then begin
  let acc = ref 0. in
  for col = 0 to cols - 1 do
    acc := !acc +. Gpu.load matrix (row * cols + col) *. Gpu.load x col
  done;
  Gpu.store y row !acc
end
```

This is a correctness/expressiveness milestone. It needs dimension/extent and
integer-range contracts before `row * cols + col` can be lowered safely. The
current scalar i32 ABI alone does not prove the product fits or the allocation
is large enough. Add a narrow checked launch path carrying those constraints,
or reject invocations without the required facts. Do not relax the existing
native-int range check to make this example compile.

## Representation changes before syntax

`Ir.region.yield` is currently a single optional scalar and `Ir.If` has at most
one result. A loop with an accumulator inside a conditional needs both its
ordinary expression result and updated cells. Generalize region yields and
operation results to lists before adding loop bodies. Ordinary source `if`
still returns one language value; additional results are compiler-managed
promoted cells.

Add a structured `For` operation with:

- Fresh induction definition, evaluated lower/upper bound operands and direction.
- Initial carried operands and fresh region parameter definitions.
- Region body ending in exactly one yield per carried value.
- Fresh loop result definitions corresponding to the carried state after exit.

Keep induction and region parameters scoped to the loop. At zero iterations,
results equal the initial carried values. For an inclusive ascending loop, check
`i == hi` after the body and exit before incrementing; do not compute `hi + 1`,
which can overflow at the endpoint. Use the analogous rule for `downto`.

## File-level work

| File | Change |
| --- | --- |
| `lib/kernel_ast.ml` | Generalize yields/results; add a typed structured loop and explicit carried parameters. Keep source spans. |
| `lib/gpu_metadata.ml` | Version the changed schema; encode/decode bounds, direction, parameters, initial values and yields with strict arity checks. The adapter and host still share this implementation. |
| `tools/export_typedtree_modes.ml` | Handle Typedtree `for`. Recognize `ref`, `!`, and `:=` by resolved declaration identity. Represent eligible local scalar cells in the lexical environment and thread updated values through branches/loops. Reject escaping, aliased, heap-stored or buffer-valued cells initially. No general heap allocation is introduced. |
| `lib/kernel_frontend.ml` | Allocate globally fresh induction/carried/result IDs, lower a separate loop environment, and translate mutable source cells into explicit SSA state. Import extent/range preconditions from checked invocation metadata. |
| `lib/ir.ml` | Add loop records and generalized region parameters/yields. Distinguish values defined at the loop boundary from nested definitions in traversal APIs. |
| `lib/verifier.ml` | Check bound/induction types, region parameter and result arity, canonical identities, dominance, no loop-local escape, and modal compatibility of initial/yielded state. Zero-trip loops cannot establish initialization. Use conservative loop memory facts first. |
| `lib/uniformity.ml` | Compute carried-value agreement to a fixed point. Trip-count agreement and enclosing participation jointly govern collectives. A condition uniform only on the first iteration is insufficient. |
| `lib/optimizer.ml` | Start with cache invalidation on loop entry/backedge/exit and local optimization of each body. No loop-invariant load motion or cross-iteration reuse until a separate effects/alias proof is implemented. |
| `lib/ptx_ir.ml` | Generalize edge moves to parallel copies for multiple carried values and join results. Define label/edge invariants explicitly. |
| `lib/ptx_lowering.ml` | Emit preheader, header, body, latch and exit labels. Initialize carried registers before the header, preserve zero trips, and lower edge copies with cycle-breaking temporaries. Recurse through loops for layout/launch requirements. |
| `lib/ptx.ml` | Print legalized branches/copies and reserve the extra scratch registers through the existing allocation scheme. No semantic loop cases in the printer. |
| `lib/gpu_dsl.ml/.mli` | Keep loops and scalar refs as ordinary OxCaml syntax; do not add a kernel-author instruction-list API. Add only memory descriptors/checks genuinely needed by the launch boundary. |
| `test/test_control_flow.ml` | Extend with multiple yields, simultaneous swaps, loop scope/dominance, zero trips, invalid carried modes and iteration-dependent uniformity. |
| `test/fixtures/`, bridge script | Compile actual OxCaml loops/accumulators; retain negative tests for escaping refs, while loops, recursion and unsupported calls. |
| H100 harness | Compare empty/single/multiple-iteration loops and rectangular matvec against a reference; validate untouched outputs and tail guards. Add PyTorch matvec comparison for correctness before performance measurements. |

## Uniformity and memory rules

Analyze loop-carried values to a fixed point before allowing any collective.
For example, a carried predicate initialized to `true` can become lane-varying
after an iteration; its initial uniformity is not a valid loop-wide proof.
A CTA barrier requires every CTA thread to execute the same sequence of barrier
instances. A full-subgroup reduction similarly requires the same participating
lanes across corresponding iterations. Initially reject loops for which this
cannot be proven. Continue rejecting early return/break/continue in this subset.

A store in a potentially zero-trip loop does not definitely initialize shared
storage after the loop. A barrier in the body does not clear pending writes on
paths that skip the body. Until a precise loop transfer analysis exists, reject
new shared-memory producer/consumer patterns that depend on cross-iteration
initialization or synchronization. Scalar global-memory loops can land first.

Scalar loop results cannot gain global lifetime, portability, permission or
uniqueness from a weaker initial/backedge value. Memory-valued carried state is
excluded initially, avoiding an unimplemented provenance join.

## Ordered completion gates

1. Generalized region results and parallel edge copies pass existing branch tests
   plus two-value swaps/cycles. No source loops yet.
2. Manually constructed loops verify and execute: zero trips, one trip, nested
   conditionals, multiple accumulators, and inclusive endpoint safety. Check
   near-boundary endpoints with one/two iterations, not billion-iteration tests.
3. Actual OxCaml `for` and local scalar `ref` syntax reaches those same IR forms.
   Reject aliasing/escaping cells with a source location.
4. Add checked dimensions/extents and range propagation sufficient for indexing.
   Implement scalar matvec for rectangular and zero-sized shapes, with guards.
5. Only after those pass, plan cooperative/tiled matvec and collectives in loops.
   They need layout mappings and stronger synchronization analysis; a working
   scalar loop is not evidence that those analyses are already sound.

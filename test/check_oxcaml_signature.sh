#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/tools/oxcaml-env.sh"
cd "$repo_root"
metadata_checker="${1:-}"
ptx_emitter="${2:-}"
matmul_emitter="${3:-}"

run_metadata_checker() {
  if [[ -n "$metadata_checker" ]]; then
    "$metadata_checker" "$@"
  else
    dune exec test/check_compiler_metadata.exe -- "$@"
  fi
}

run_ptx_emitter() {
  if [[ -n "$ptx_emitter" ]]; then
    "$ptx_emitter" "$@"
  else
    dune exec test/emit_ptx.exe -- "$@"
  fi
}

run_matmul_emitter() {
  if [[ -n "$matmul_emitter" ]]; then
    "$matmul_emitter" "$@"
  else
    dune exec test/emit_gemm_ptx.exe -- "$@"
  fi
}

root="${OXCAML_ROOT:-}"
compiler="${OXCC:-}"
if [[ -z "$compiler" ]]; then
  if [[ -n "$root" ]]; then
    compiler="$root/_build/_bootinstall/bin/ocamlc.opt"
  else
    compiler="$(command -v ocamlc.opt || true)"
  fi
fi
if [[ -z "$compiler" ]]; then
  echo "OxCaml bytecode compiler not found. Set OXCC or OXCAML_ROOT." >&2
  exit 1
fi
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

tools/compile_oxcaml_kernels.sh "$tmpdir"

# The debug dump is used only as an additional assertion. The actual compiler
# bridge reads the kernel .cmt typedtrees via compiler-libs.
"$compiler" -nostdlib -nopervasives -bin-annot -dtypedtree -c \
  -o "$tmpdir/mode_probe.cmi" test/fixtures/mode_probe.mli 2>"$tmpdir/mode_probe.typedtree"
rg -q 'local,.*unique' "$tmpdir/mode_probe.typedtree"
rg -q 'portable' "$tmpdir/mode_probe.typedtree"

rg -q $'^arg\t0\tbuffer_f32\taliased\tglobal\tnonportable\tread\t' "$tmpdir/saxpy.gpu"
rg -q $'^arg\t1\tbuffer_f32\tunique\tglobal\tnonportable\tread_write\t' "$tmpdir/saxpy.gpu"
rg -q $'^format\t3$' "$tmpdir/saxpy.gpu"
rg -q $'^arg\t1\tbuffer_f32\tunique\tglobal\tnonportable\tread_write\t' "$tmpdir/unique_reuse.gpu"
rg -q $'^arg\t1\tbuffer_f32\taliased\tglobal\tnonportable\tread_write\t' "$tmpdir/alias_reuse_aliased.gpu"
run_metadata_checker "$tmpdir/saxpy.gpu" "$tmpdir/vector_add.gpu" "$tmpdir/literal_probe.gpu" "$tmpdir/dot_product.gpu"
rg -q $'^body\tshared_alloc\t' "$tmpdir/hierarchy_smoke.gpu"
rg -q $'^body\tshared_store_f32\t' "$tmpdir/hierarchy_smoke.gpu"
rg -q $'^body\tbarrier_cta$' "$tmpdir/hierarchy_smoke.gpu"
rg -q $'^body\tshared_load_f32\t' "$tmpdir/hierarchy_smoke.gpu"
rg -q $'^body\tmad_f32\t' "$tmpdir/hierarchy_smoke.gpu"
run_ptx_emitter "$tmpdir/hierarchy_smoke.gpu" > "$tmpdir/hierarchy_smoke.ptx"
rg -q 'bar\.sync' "$tmpdir/hierarchy_smoke.ptx"
rg -q 'fma\.rn\.f32' "$tmpdir/hierarchy_smoke.ptx"
rg -q '\.shared' "$tmpdir/hierarchy_smoke.ptx"
unique_reuse_ptx="$tmpdir/unique_reuse.ptx"
run_ptx_emitter "$tmpdir/unique_reuse.gpu" > "$unique_reuse_ptx"
if [[ $(rg -c 'ld\.global\.f32' "$unique_reuse_ptx") != 1 ]]; then
  echo "unique_reuse should emit one global load after noalias load reuse" >&2
  exit 1
fi
aliased_reuse_ptx="$tmpdir/alias_reuse_aliased.ptx"
run_ptx_emitter "$tmpdir/alias_reuse_aliased.gpu" > "$aliased_reuse_ptx"
if [[ $(rg -c 'ld\.global\.f32' "$aliased_reuse_ptx") != 2 ]]; then
  echo "aliased baseline must retain the second global load" >&2
  exit 1
fi
echo "OxCaml Typedtree modes, bodies, literals, and reduction reached verified GPU IR"
for kernel in guarded_saxpy short_circuit numeric float_compare uniform_branch; do
  run_ptx_emitter "$tmpdir/$kernel.gpu" > "$tmpdir/$kernel.ptx"
  rg -q 'bra L' "$tmpdir/$kernel.ptx"
done

run_ptx_emitter "$tmpdir/joined_reduction.gpu" > "$tmpdir/joined_reduction.ptx"
if run_ptx_emitter "$tmpdir/divergent_reduction.gpu" > "$tmpdir/divergent.ptx" 2> "$tmpdir/divergent.err"; then
  echo "Divergent full-warp reduction unexpectedly accepted" >&2; exit 1
fi
rg -q 'E_DIVERGENT_COLLECTIVE' "$tmpdir/divergent.err"
rg -q 'control_flow.ml:[0-9]+:[0-9]+:' "$tmpdir/divergent.err"

run_ptx_emitter "$tmpdir/rounding.gpu" > "$tmpdir/rounding.ptx"
rg -q 'mul.rn.f32' "$tmpdir/rounding.ptx"
rg -q 'add.rn.f32' "$tmpdir/rounding.ptx"

# The epilogue writes BN/8 groups of four accumulator registers, two per
# output row. Each row's pair is column-adjacent, so the store vectorizer must
# fuse it into one 8-byte store and leave no scalar store behind.
expect_tile () {
  local name="$1" threads="$2" wgmma_n="$3" pair_stores="$4"
  local gpu="$tmpdir/$name.gpu"
  local ptx="$tmpdir/$name.ptx"
  if [[ ! -f "$gpu" ]]; then
    echo "missing $name.gpu" >&2
    exit 1
  fi
  rg -q $'^threads\t'"$threads"'$' "$gpu"
  run_ptx_emitter "$gpu" > "$ptx"
  rg -q "\\.reqntid ${threads}, 1, 1" "$ptx"
  rg -q "wgmma.mma_async.sync.aligned.m64n${wgmma_n}k8" "$ptx"
  rg -q 'cp.async.bulk.tensor.2d' "$ptx"
  rg -q 'bra\.uni' "$ptx"
  if rg -q 'bra L' "$ptx"; then
    echo "$name has a divergent branch" >&2
    exit 1
  fi
  if [[ $(rg -c 'wgmma\.mma_async' "$ptx") != 4 ]]; then
    echo "$name: expected 4 wgmma instructions" >&2
    exit 1
  fi
  if [[ $(rg -c 'st\.global\.v2\.f32' "$ptx") != "$pair_stores" ]]; then
    echo "$name: expected $pair_stores vectorized epilogue stores" >&2
    exit 1
  fi
  if rg -q 'st\.global\.f32' "$ptx"; then
    echo "$name: epilogue left an unvectorized scalar store" >&2
    exit 1
  fi
}
# Every float intrinsic reaches exactly the instruction its name promises:
# the Stdlib names round, the [_approx] names are the special-function unit.
run_ptx_emitter "$tmpdir/float_math.gpu" > "$tmpdir/float_math.ptx"
for instruction in 'neg\.f32' 'abs\.f32' 'sqrt\.rn\.f32' 'sub\.rn\.f32' \
  'div\.rn\.f32' 'max\.NaN\.f32' 'min\.NaN\.f32' 'cvt\.rn\.f32\.s32' \
  'ex2\.approx\.f32' 'lg2\.approx\.f32' 'tanh\.approx\.f32' \
  'rsqrt\.approx\.f32' 'rcp\.approx\.f32'; do
  if ! rg -q "$instruction" "$tmpdir/float_math.ptx"; then
    echo "float_math: no $instruction in the emitted PTX" >&2
    exit 1
  fi
done
# The intrinsics are pure, so value numbering must collapse the repeats:
# three [abs_float b] and two [abs_float a] are two instructions, not five.
if [[ $(rg -c 'abs\.f32' "$tmpdir/float_math.ptx") != 2 ]]; then
  echo "float_math: redundant float intrinsics were not eliminated" >&2
  exit 1
fi

# The [.NaN] modifier is the whole of the claim that Float.max keeps its OCaml
# meaning, so it is checked on its own; run_h100.cu verifies the behaviour.
run_ptx_emitter "$tmpdir/float_extremes.gpu" > "$tmpdir/float_extremes.ptx"
rg -q 'max\.NaN\.f32' "$tmpdir/float_extremes.ptx"
rg -q 'min\.NaN\.f32' "$tmpdir/float_extremes.ptx"

# A row softmax is the exponential, the maximum and the reciprocal together;
# RMS norm is the reciprocal square root and the one int-to-float conversion.
run_ptx_emitter "$tmpdir/softmax_256.gpu" > "$tmpdir/softmax_256.ptx"
rg -q 'ex2\.approx\.f32' "$tmpdir/softmax_256.ptx"
rg -q 'rcp\.approx\.f32' "$tmpdir/softmax_256.ptx"
rg -q 'sub\.rn\.f32' "$tmpdir/softmax_256.ptx"
if [[ $(rg -c 'max\.NaN\.f32' "$tmpdir/softmax_256.ptx") != 8 ]]; then
  echo "softmax_256: the maximum should inline once per fold step" >&2
  exit 1
fi
run_ptx_emitter "$tmpdir/rms_norm_256.gpu" > "$tmpdir/rms_norm_256.ptx"
rg -q 'rsqrt\.approx\.f32' "$tmpdir/rms_norm_256.ptx"
rg -q 'cvt\.rn\.f32\.s32' "$tmpdir/rms_norm_256.ptx"

rg -q $'^arg\t0\ttensor_map\taliased\tglobal\tnonportable\tread\t' "$tmpdir/gemm_fast.gpu"
rg -q $'^arg\t2\tbuffer_f32\tunique\tglobal\tnonportable\tread_write\t' "$tmpdir/gemm_fast.gpu"
expect_tile gemm_fast 288 256 64
expect_tile gemm_bm128_bn256_s3 288 256 64
expect_tile gemm_bm128_bn256_s2 288 256 64
expect_tile gemm_bm128_bn128_s3 288 128 32
expect_tile gemm_bm128_bn128_s2 288 128 32
expect_tile gemm_bm64_bn256_s3 160 256 64
expect_tile gemm_bm64_bn256_s2 160 256 64
expect_tile gemm_bm256_bn128_s3 544 128 32
expect_tile gemm_bm256_bn128_s2 544 128 32
# bf16 is the same schedule at the other input width. The element type is
# what picks the instruction, and because a swizzle period is 128 bytes at
# either width the shared window comes out byte-identical — which is the
# claim worth pinning, since it is what makes the two tiles interchangeable.
rg -q $'^arg\t0\ttensor_map\taliased\tglobal\tnonportable\tread\t' \
  "$tmpdir/gemm_bf16_fast.gpu"
rg -q $'^body\tshared_alloc\t[0-9]+\t512\t64\tbf16\twgmma$' "$tmpdir/gemm_bf16_fast.gpu"
rg -q $'^threads\t288$' "$tmpdir/gemm_bf16_fast.gpu"
run_ptx_emitter "$tmpdir/gemm_bf16_fast.gpu" > "$tmpdir/gemm_bf16_fast.ptx"
rg -q 'wgmma\.mma_async\.sync\.aligned\.m64n256k16\.f32\.bf16\.bf16' \
  "$tmpdir/gemm_bf16_fast.ptx"
if [[ $(rg -c 'wgmma\.mma_async' "$tmpdir/gemm_bf16_fast.ptx") != 4 ]]; then
  echo "gemm_bf16_fast: expected 4 wgmma instructions" >&2
  exit 1
fi
if [[ $(rg -c 'st\.global\.v2\.f32' "$tmpdir/gemm_bf16_fast.ptx") != 64 ]]; then
  echo "gemm_bf16_fast: expected 64 vectorized epilogue stores" >&2
  exit 1
fi
if rg -q 'st\.global\.f32' "$tmpdir/gemm_bf16_fast.ptx"; then
  echo "gemm_bf16_fast: epilogue left an unvectorized scalar store" >&2
  exit 1
fi
f32_window="$(run_matmul_emitter --info "$tmpdir/gemm_fast.gpu" | cut -d" " -f3)"
bf16_window="$(run_matmul_emitter --info "$tmpdir/gemm_bf16_fast.gpu" | cut -d" " -f3)"
if [[ "$f32_window" != "$bf16_window" ]]; then
  echo "shared window differs across widths: f32 $f32_window, bf16 $bf16_window" >&2
  exit 1
fi

# Only TMA and GMMA may touch a swizzled tile. A software store into one is
# well-typed and would silently read the wrong bytes, so the verifier has to
# be the thing that stops it, with a source location.
if run_ptx_emitter "$tmpdir/swizzled_shared.gpu" > "$tmpdir/swizzled.ptx" \
     2> "$tmpdir/swizzled.err"; then
  echo "Software store into a swizzled shared tile was accepted" >&2; exit 1
fi
rg -q 'E_SHARED_SWIZZLE' "$tmpdir/swizzled.err"
rg -q 'swizzled_shared.ml:[0-9]+:[0-9]+:' "$tmpdir/swizzled.err"

# f16 picks the same k16 shape as bf16 and differs only in the suffix.
rg -q $'^body\tshared_alloc\t[0-9]+\t64\t64\tf16\twgmma$' "$tmpdir/f16_wgmma.gpu"
run_ptx_emitter "$tmpdir/f16_wgmma.gpu" > "$tmpdir/f16_wgmma.ptx"
rg -q 'wgmma\.mma_async\.sync\.aligned\.m64n8k16\.f32\.f16\.f16' \
  "$tmpdir/f16_wgmma.ptx"

choose_4096="$(run_matmul_emitter --print-choose 4096 4096 4096)"
choose_8192="$(run_matmul_emitter --print-choose 8192 8192 8192)"
choose_64="$(run_matmul_emitter --print-config 'bm=64,bn=256,bk=32,stages=2,producers=32,threads=160')"
if [[ "$choose_4096" != "gemm_fast 128 256 32 288" ]]; then
  echo "print-choose 4096: $choose_4096" >&2
  exit 1
fi
# The square shapes take the deepest measured pipeline, which is gemm_fast
# at four stages; a shallower ring leaves the TMA producers no slack once the
# K loop keeps a WGMMA group in flight.
if [[ "$choose_8192" != "gemm_fast 128 256 32 288" ]]; then
  echo "print-choose 8192: $choose_8192" >&2
  exit 1
fi
if [[ "$choose_64" != "gemm_bm64_bn256_s2 64 256 32 160" ]]; then
  echo "print-config 64x256 s2: $choose_64" >&2
  exit 1
fi
# A recursive tree fold parameterized by its operator must leave no call, no
# closure and no recursion: eight fold steps for size 256, the operator
# inlined once per step, and the two reductions differing only in that
# operator. A regression in inlining shows up here rather than as slower code.
for kernel in block_sum block_max; do
  rg -q $'^threads\t256$' "$tmpdir/$kernel.gpu"
  if [[ $(rg -c '^body\tbarrier_cta$' "$tmpdir/$kernel.gpu") != 9 ]]; then
    echo "$kernel: expected 8 fold barriers plus the load barrier" >&2
    exit 1
  fi
  run_ptx_emitter "$tmpdir/$kernel.gpu" > "$tmpdir/$kernel.ptx"
  rg -q '\.shared' "$tmpdir/$kernel.ptx"
done
if [[ $(rg -c '^body\tadd_f32\t' "$tmpdir/block_sum.gpu") != 8 ]]; then
  echo "block_sum: the summing operator should inline once per fold step" >&2
  exit 1
fi
if rg -q '^body\tadd_f32\t' "$tmpdir/block_max.gpu"; then
  echo "block_max: should carry no addition, only comparisons" >&2
  exit 1
fi

# Higher-order use, closures and polymorphism are erased by inlining: twice
# applied to two different functions leaves two multiplies and two adds, the
# closure over the formal reads it directly, and the polymorphic identity
# leaves nothing at all. No indirect call, no closure record.
run_ptx_emitter "$tmpdir/functional.gpu" > "$tmpdir/functional.ptx"
if [[ $(rg -c '^body\tmul_i32\t' "$tmpdir/functional.gpu") != 2 ]]; then
  echo "functional: twice over a closure should inline to two multiplies" >&2
  exit 1
fi
if [[ $(rg -c '^body\tadd_i32\t' "$tmpdir/functional.gpu") != 2 ]]; then
  echo "functional: twice over a lambda should inline to two adds" >&2
  exit 1
fi
rg -q 'st\.global\.f32' "$tmpdir/functional.ptx"

# The constant for-loop unrolls into stores at y[0..3]: consecutive, and the
# first index is a multiple of four, so they fuse into one 16-byte store.
run_ptx_emitter "$tmpdir/indexed_stores.gpu" > "$tmpdir/indexed_stores.ptx"
if [[ $(rg -c 'st\.global\.v4\.f32' "$tmpdir/indexed_stores.ptx") != 1 ]]; then
  echo "unrolled constant for-loop should fuse into one v4 store" >&2
  exit 1
fi
if rg -q 'st\.global\.f32' "$tmpdir/indexed_stores.ptx"; then
  echo "unrolled constant for-loop left an unvectorized scalar store" >&2
  exit 1
fi

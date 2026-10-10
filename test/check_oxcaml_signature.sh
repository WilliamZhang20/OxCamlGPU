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

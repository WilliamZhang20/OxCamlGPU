#!/usr/bin/env bash
set -euo pipefail

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
dune exec test/check_compiler_metadata.exe -- "$tmpdir/saxpy.gpu" "$tmpdir/vector_add.gpu" "$tmpdir/literal_probe.gpu" "$tmpdir/dot_product.gpu"
unique_reuse_ptx="$tmpdir/unique_reuse.ptx"
dune exec test/emit_ptx.exe -- "$tmpdir/unique_reuse.gpu" > "$unique_reuse_ptx"
if [[ $(rg -c 'ld\.global\.f32' "$unique_reuse_ptx") != 1 ]]; then
  echo "unique_reuse should emit one global load after noalias load reuse" >&2
  exit 1
fi
aliased_reuse_ptx="$tmpdir/alias_reuse_aliased.ptx"
dune exec test/emit_ptx.exe -- "$tmpdir/alias_reuse_aliased.gpu" > "$aliased_reuse_ptx"
if [[ $(rg -c 'ld\.global\.f32' "$aliased_reuse_ptx") != 2 ]]; then
  echo "aliased baseline must retain the second global load" >&2
  exit 1
fi
echo "OxCaml Typedtree modes, bodies, literals, and reduction reached verified GPU IR"
for kernel in guarded_saxpy short_circuit numeric float_compare uniform_branch; do
  dune exec test/emit_ptx.exe -- "$tmpdir/$kernel.gpu" > "$tmpdir/$kernel.ptx"
  rg -q 'bra L' "$tmpdir/$kernel.ptx"
done

dune exec test/emit_ptx.exe -- "$tmpdir/joined_reduction.gpu" > "$tmpdir/joined_reduction.ptx"
if dune exec test/emit_ptx.exe -- "$tmpdir/divergent_reduction.gpu" > "$tmpdir/divergent.ptx" 2> "$tmpdir/divergent.err"; then
  echo "Divergent full-warp reduction unexpectedly accepted" >&2; exit 1
fi
rg -q 'E_DIVERGENT_COLLECTIVE' "$tmpdir/divergent.err"
rg -q 'control_flow.ml:[0-9]+:[0-9]+:' "$tmpdir/divergent.err"

dune exec test/emit_ptx.exe -- "$tmpdir/rounding.gpu" > "$tmpdir/rounding.ptx"
rg -q 'mul.rn.f32' "$tmpdir/rounding.ptx"
rg -q 'add.rn.f32' "$tmpdir/rounding.ptx"

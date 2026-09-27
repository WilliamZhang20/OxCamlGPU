#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
  echo "usage: compile_oxcaml_kernels.sh OUTPUT_DIR" >&2
  exit 2
fi
out="$1"
mkdir -p "$out"

root="${OXCAML_ROOT:-$HOME/src/oxcaml-src}"
main_build="${OXCAML_MAIN_BUILD:-$root/_build/main}"
stdlib_dir="${OXCAML_STDLIB_DIR:-$root/_build/runtime_stdlib_install/lib/ocaml_runtime_stdlib}"
compiler="${OXCC:-$root/_build/_bootinstall/bin/ocamlc.opt}"
native_compiler="${OXOPT:-$root/_build/_bootinstall/bin/ocamlopt.opt}"

for path in "$compiler" "$native_compiler" "$main_build/ocamlcommon.cmxa" \
  "$main_build/ocamlfrontend.cmxa" "$main_build/oxcaml_common.cmxa" \
  "$stdlib_dir/stdlib.cmi"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing matching OxCaml compiler artifact: $path" >&2
    echo "Set OXCAML_ROOT, OXCAML_MAIN_BUILD, OXCAML_STDLIB_DIR, OXCC, and OXOPT." >&2
    exit 1
  fi
done

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp tools/export_typedtree_modes.ml "$tmp/export_typedtree_modes.ml"

"$native_compiler" -I "$stdlib_dir" \
  -I "$main_build/.ocamlcommon.objs/byte" -I "$main_build/.ocamlcommon.objs/native" \
  -I "$main_build/.ocamlfrontend.objs/byte" -I "$main_build/.ocamlfrontend.objs/native" \
  -I "$main_build/.oxcaml_common.objs/byte" -I "$main_build/.oxcaml_common.objs/native" \
  "$main_build/ocamlcommon.cmxa" "$main_build/ocamlfrontend.cmxa" \
  "$main_build/oxcaml_common.cmxa" "$tmp/export_typedtree_modes.ml" \
  -ccopt "-L$stdlib_dir" -o "$tmp/export_typedtree_modes"

# The OxCaml-only DSL API lives in lib/ alongside the compiler library. Dune's
# host OCaml build excludes it because ordinary OCaml cannot parse mode syntax.
"$compiler" -nostdlib -I "$stdlib_dir" -bin-annot -c \
  -o "$tmp/gpu_dsl.cmi" lib/gpu_dsl.mli
"$compiler" -nostdlib -I "$stdlib_dir" -I "$tmp" -bin-annot -c \
  -o "$tmp/gpu_dsl.cmo" lib/gpu_dsl.ml

for kernel in saxpy vector_add dot_product; do
  "$compiler" -nostdlib -I "$stdlib_dir" -I "$tmp" -bin-annot -c \
    -o "$tmp/$kernel.cmo" "examples/kernels/$kernel.ml"
  "$tmp/export_typedtree_modes" "$tmp/$kernel.cmt" "$kernel" > "$out/$kernel.gpu"
done

# A small fixture keeps non-kernel examples from carrying adapter-only syntax
# just to exercise literal lowering.
"$compiler" -nostdlib -I "$stdlib_dir" -I "$tmp" -bin-annot -c \
  -o "$tmp/literal_probe.cmo" test/fixtures/literal_probe.ml
"$tmp/export_typedtree_modes" "$tmp/literal_probe.cmt" literal_probe > "$out/literal_probe.gpu"

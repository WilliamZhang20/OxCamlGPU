#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
  echo "usage: compile_oxcaml_kernels.sh OUTPUT_DIR" >&2
  exit 2
fi
out="$1"
mkdir -p "$out"

root="${OXCAML_ROOT:-/tmp/oxcaml-src}"
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

"$native_compiler" -I "$stdlib_dir" \
  -I "$main_build/.ocamlcommon.objs/byte" -I "$main_build/.ocamlcommon.objs/native" \
  -I "$main_build/.ocamlfrontend.objs/byte" -I "$main_build/.ocamlfrontend.objs/native" \
  -I "$main_build/.oxcaml_common.objs/byte" -I "$main_build/.oxcaml_common.objs/native" \
  "$main_build/ocamlcommon.cmxa" "$main_build/ocamlfrontend.cmxa" \
  "$main_build/oxcaml_common.cmxa" tools/export_typedtree_modes.ml \
  -ccopt "-L$stdlib_dir" -o "$tmp/export_typedtree_modes"

for kernel in saxpy vector_add; do
  "$compiler" -nostdlib -nopervasives -bin-annot -c \
    -o "$tmp/$kernel.cmi" "examples/kernels/$kernel.mli"
  "$compiler" -nostdlib -nopervasives -I "$tmp" -bin-annot -c \
    -o "$tmp/$kernel.cmo" "examples/kernels/$kernel.ml"
  "$tmp/export_typedtree_modes" "$tmp/$kernel.cmti" "$kernel" > "$out/$kernel.gpu"
  "$tmp/export_typedtree_modes" "$tmp/$kernel.cmt" "$kernel" >> "$out/$kernel.gpu"
done

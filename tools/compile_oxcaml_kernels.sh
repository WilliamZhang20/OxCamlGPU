#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
  echo "usage: compile_oxcaml_kernels.sh OUTPUT_DIR" >&2
  exit 2
fi
out="$1"
mkdir -p "$out"

root="${OXCAML_ROOT:-}"
compiler="${OXCC:-}"
native_compiler="${OXOPT:-}"

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

# If OXCC points into OxCaml's conventional build tree, infer the source root.
# Installed compilers can instead be paired with explicit compiler-libs paths.
if [[ -z "$root" ]]; then
  build_suffix="/_build/_bootinstall/bin/ocamlc.opt"
  if [[ "$compiler" == *"$build_suffix" ]]; then
    root="${compiler%$build_suffix}"
  fi
fi

if [[ -n "$root" && -e "$root/.git" && "${OXCAML_ALLOW_UNPINNED:-0}" != 1 ]]; then
  expected_revision="$(<tools/oxcaml-revision)"
  actual_revision="$(git -C "$root" rev-parse HEAD)"
  if [[ "$actual_revision" != "$expected_revision" ]]; then
    echo "OxCaml revision $actual_revision is not the tested revision $expected_revision." >&2
    echo "Set OXCAML_ALLOW_UNPINNED=1 only after validating the adapter against another revision." >&2
    exit 1
  fi
fi

if [[ -z "$native_compiler" ]]; then
  if [[ -n "$root" ]]; then
    native_compiler="$root/_build/_bootinstall/bin/ocamlopt.opt"
  else
    native_compiler="$(command -v ocamlopt.opt || true)"
  fi
fi
main_build="${OXCAML_MAIN_BUILD:-${root:+$root/_build/main}}"
stdlib_dir="${OXCAML_STDLIB_DIR:-${root:+$root/_build/runtime_stdlib_install/lib/ocaml_runtime_stdlib}}"

if [[ -z "$main_build" || -z "$stdlib_dir" ]]; then
  echo "OxCaml compiler-libs paths are unknown. Set OXCAML_ROOT or both OXCAML_MAIN_BUILD and OXCAML_STDLIB_DIR." >&2
  exit 1
fi

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

"$compiler" -nostdlib -I "$stdlib_dir" -I "$tmp" -bin-annot -c \
  -o "$tmp/unique_reuse.cmo" test/fixtures/unique_reuse.ml
"$tmp/export_typedtree_modes" "$tmp/unique_reuse.cmt" unique_reuse > "$out/unique_reuse.gpu"
"$tmp/export_typedtree_modes" "$tmp/unique_reuse.cmt" alias_reuse_aliased > "$out/alias_reuse_aliased.gpu"

# A small fixture keeps non-kernel examples from carrying adapter-only syntax
# just to exercise literal lowering.
"$compiler" -nostdlib -I "$stdlib_dir" -I "$tmp" -bin-annot -c \
  -o "$tmp/literal_probe.cmo" test/fixtures/literal_probe.ml
"$tmp/export_typedtree_modes" "$tmp/literal_probe.cmt" literal_probe > "$out/literal_probe.gpu"

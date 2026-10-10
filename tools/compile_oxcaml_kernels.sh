#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/oxcaml-env.sh"

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
cp tools/typedtree_support.ml "$tmp/typedtree_support.ml"
# Schema modules shared between the host library and this adapter. Paths are
# stage-relative under lib/; they are flattened into $tmp, which is also how
# the library sees them (lib/dune uses include_subdirs unqualified).
shared=(base/gpu_type base/gpu_mode base/source_span
  frontend/kernel_ast frontend/gpu_metadata)
shared_objects=()
for path in "${shared[@]}"; do
  name="${path##*/}"
  cp "lib/$path.ml" "$tmp/$name.ml"
  "$native_compiler" -I "$stdlib_dir" -I "$tmp" -c "$tmp/$name.ml"
  shared_objects+=("$tmp/$name.cmx")
done

"$native_compiler" -I "$stdlib_dir" -I "$tmp" \
  -I "$main_build/.ocamlcommon.objs/byte" -I "$main_build/.ocamlcommon.objs/native" \
  -I "$main_build/.ocamlfrontend.objs/byte" -I "$main_build/.ocamlfrontend.objs/native" \
  -I "$main_build/.oxcaml_common.objs/byte" -I "$main_build/.oxcaml_common.objs/native" \
  "$main_build/ocamlcommon.cmxa" "$main_build/ocamlfrontend.cmxa" \
  "$main_build/oxcaml_common.cmxa" "${shared_objects[@]}" \
  "$tmp/typedtree_support.ml" "$tmp/export_typedtree_modes.ml" \
  -ccopt "-L$stdlib_dir" -o "$tmp/export_typedtree_modes"

# The OxCaml-only DSL API lives in lib/dsl/ alongside the compiler library.
# Dune's host OCaml build excludes it because ordinary OCaml cannot parse mode
# syntax, so this is the only thing that compiles it.
"$compiler" -nostdlib -I "$stdlib_dir" -bin-annot -c \
  -o "$tmp/gpu_dsl.cmi" lib/dsl/gpu_dsl.mli
"$compiler" -nostdlib -I "$stdlib_dir" -I "$tmp" -bin-annot -c \
  -o "$tmp/gpu_dsl.cmo" lib/dsl/gpu_dsl.ml

for kernel in saxpy vector_add dot_product; do
  "$compiler" -nostdlib -I "$stdlib_dir" -I "$tmp" -bin-annot -c \
    -o "$tmp/$kernel.cmo" "examples/kernels/$kernel.ml"
  "$tmp/export_typedtree_modes" "$tmp/$kernel.cmt" "$kernel" > "$out/$kernel.gpu"
done

# A smoke kernel for the shared/barrier/mad path. It is a test fixture rather
# than an example: nothing demonstrates it, the signature checks assert on it.
"$compiler" -nostdlib -I "$stdlib_dir" -I "$tmp" -bin-annot -c \
  -o "$tmp/hierarchy_smoke.cmo" test/fixtures/hierarchy_smoke.ml
"$tmp/export_typedtree_modes" "$tmp/hierarchy_smoke.cmt" hierarchy_smoke \
  > "$out/hierarchy_smoke.gpu"

# Keep these names in sync with Gemm_catalog.kernel_binding.
"$compiler" -nostdlib -I "$stdlib_dir" -I "$tmp" -bin-annot -c \
  -o "$tmp/gemm_fast.cmo" examples/kernels/gemm.ml
for kernel in \
  gemm_fast \
  gemm_bm128_bn256_s2 \
  gemm_bm128_bn256_s3 \
  gemm_bm128_bn128_s3 \
  gemm_bm128_bn128_s2 \
  gemm_bm64_bn256_s3 \
  gemm_bm64_bn256_s2 \
  gemm_bm256_bn128_s3 \
  gemm_bm256_bn128_s2
do
  "$tmp/export_typedtree_modes" "$tmp/gemm_fast.cmt" "$kernel" > "$out/$kernel.gpu"
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

"$compiler" -nostdlib -I "$stdlib_dir" -I "$tmp" -bin-annot -c \
  -o "$tmp/control_flow.cmo" test/fixtures/control_flow.ml
for kernel in guarded_saxpy short_circuit numeric float_compare uniform_branch joined_reduction divergent_reduction rounding indexed_stores; do
  "$tmp/export_typedtree_modes" "$tmp/control_flow.cmt" "$kernel" > "$out/$kernel.gpu"
done

# Real Typedtree negative fixtures: reject declaration lookalikes, unsupported
# native integer ranges, and source constructs not yet implemented.
for fixture in reject_shadow reject_overflow reject_fake_type reject_loop reject_operator; do
  "$compiler" -nostdlib -I "$stdlib_dir" -I "$tmp" -bin-annot -c \
    -o "$tmp/$fixture.cmo" "test/fixtures/$fixture.ml"
  if "$tmp/export_typedtree_modes" "$tmp/$fixture.cmt" "$fixture" > "$tmp/rejected.gpu" 2> "$tmp/rejected.err"; then
    echo "Unexpectedly accepted $fixture" >&2; exit 1
  fi
  if [[ -s "$tmp/rejected.gpu" ]]; then
    echo "Failed adapter left a partial artifact for $fixture" >&2; exit 1
  fi
  rg -q "$fixture.ml:[0-9]+:[0-9]+:" "$tmp/rejected.err"
done

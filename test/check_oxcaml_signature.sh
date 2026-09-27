#!/usr/bin/env bash
set -euo pipefail

compiler="${OXCC:-ocamlc}"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

# -dtypedtree is an OxCaml compiler diagnostic format, not a stable interchange
# format. Here it proves that the actual typechecker sees the annotated modes;
# the GPU compiler still has its own restricted importer for now.
"$compiler" -nostdlib -nopervasives -dtypedtree -c \
  -o "$tmpdir/saxpy.cmi" frontend/saxpy.mli 2>"$tmpdir/saxpy.typedtree"
"$compiler" -nostdlib -nopervasives -dtypedtree -c \
  -o "$tmpdir/mode_probe.cmi" frontend/mode_probe.mli 2>"$tmpdir/mode_probe.typedtree"

rg -q 'uniqueness: aliased' "$tmpdir/saxpy.typedtree"
rg -q 'visibility: read' "$tmpdir/saxpy.typedtree"
rg -q 'uniqueness: unique' "$tmpdir/saxpy.typedtree"
rg -q 'visibility: read_write' "$tmpdir/saxpy.typedtree"
rg -q 'global,.*aliased' "$tmpdir/mode_probe.typedtree"
rg -q 'uniqueness: unique' "$tmpdir/mode_probe.typedtree"
rg -q 'portable' "$tmpdir/mode_probe.typedtree"
echo "OxCaml typechecked the sample signatures and retained their modes in Typedtree"

#!/usr/bin/env bash
set -euo pipefail

root="${OXCAML_ROOT:-$HOME/src/oxcaml-src}"
compiler="${OXCC:-$root/_build/_bootinstall/bin/ocamlc.opt}"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

tools/compile_oxcaml_kernels.sh "$tmpdir"

# The debug dump is used only as an additional assertion. The actual compiler
# bridge reads .cmti/.cmt typedtrees via compiler-libs.
"$compiler" -nostdlib -nopervasives -bin-annot -dtypedtree -c \
  -o "$tmpdir/mode_probe.cmi" test/fixtures/mode_probe.mli 2>"$tmpdir/mode_probe.typedtree"
rg -q 'local,.*unique' "$tmpdir/mode_probe.typedtree"
rg -q 'portable' "$tmpdir/mode_probe.typedtree"

rg -q $'^arg\t0\tbuffer_f32\taliased\tglobal\tnonportable\tread$' "$tmpdir/saxpy.gpu"
rg -q $'^arg\t1\tbuffer_f32\tunique\tglobal\tnonportable\tread_write$' "$tmpdir/saxpy.gpu"
rg -q $'^body\tstore\t1\tthread_idx_x\tadd ' "$tmpdir/saxpy.gpu"
rg -q $'^body\tstore\t2\tthread_idx_x\tadd ' "$tmpdir/vector_add.gpu"
dune exec test/check_compiler_metadata.exe -- "$tmpdir/saxpy.gpu" "$tmpdir/vector_add.gpu"
echo "OxCaml typechecked both kernels; Typedtree bodies and modes reached verified GPU IR"

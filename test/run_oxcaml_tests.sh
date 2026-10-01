#!/usr/bin/env bash
set -euo pipefail

compiler="${OXCC:-$(command -v ocamlc.opt || true)}"
has_conventional_root=false
if [[ "$compiler" == *"/_build/_bootinstall/bin/ocamlc.opt" ]]; then
  has_conventional_root=true
fi
has_explicit_paths=false
if [[ -n "${OXCC:-}" && -n "${OXOPT:-}" && -n "${OXCAML_MAIN_BUILD:-}" && -n "${OXCAML_STDLIB_DIR:-}" ]]; then
  has_explicit_paths=true
fi
if [[ -z "${OXCAML_ROOT:-}" && "$has_conventional_root" != true && "$has_explicit_paths" != true ]]; then
  echo "OxCaml bridge tests skipped; set OXCAML_ROOT or OXCC to enable them."
  exit 0
fi

repo_root="${DUNE_SOURCEROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
exec test/check_oxcaml_signature.sh

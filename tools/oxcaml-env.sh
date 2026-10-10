#!/usr/bin/env bash
# Load a working-tree OxCaml checkout configuration without changing global
# shell startup files. Create .oxcaml-config with OXCAML_ROOT=/path/to/oxcaml.

_oxcaml_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$_oxcaml_repo_root/.oxcaml-config" ]]; then
  # shellcheck disable=SC1091
  source "$_oxcaml_repo_root/.oxcaml-config"
fi

if [[ -n "${OXCAML_ROOT:-}" ]]; then
  export PATH="$OXCAML_ROOT/_build/_bootinstall/bin:$PATH"
fi
unset _oxcaml_repo_root

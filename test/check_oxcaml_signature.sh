#!/usr/bin/env bash
set -euo pipefail

compiler="${OXCC:-ocamlc}"
"$compiler" -nostdlib -nopervasives -c -o /tmp/Oxgpu_saxpy.cmi frontend/saxpy.mli
"$compiler" -nostdlib -nopervasives -c -o /tmp/Oxgpu_mode_probe.cmi frontend/mode_probe.mli
echo "OxCaml accepted the sample GPU signatures"

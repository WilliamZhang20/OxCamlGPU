#!/usr/bin/env bash
set -euo pipefail

if ! command -v ptxas >/dev/null; then
  echo "Load a CUDA toolkit module providing ptxas first." >&2
  exit 1
fi

temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT
metadata_dir="$temp_dir/metadata"
mkdir -p "$metadata_dir"
tools/compile_oxcaml_kernels.sh "$metadata_dir"
dune exec examples/vector_add.exe -- "$metadata_dir/vector_add.gpu" > "$temp_dir/vector_add.ptx"
dune exec examples/saxpy.exe -- "$metadata_dir/saxpy.gpu" > "$temp_dir/saxpy.ptx"
dune exec examples/dot_product.exe -- "$metadata_dir/dot_product.gpu" > "$temp_dir/dot_product.ptx"
ptxas -arch=sm_90 "$temp_dir/vector_add.ptx" -o "$temp_dir/vector_add.cubin"
ptxas -arch=sm_90 "$temp_dir/saxpy.ptx" -o "$temp_dir/saxpy.cubin"
ptxas -arch=sm_90 "$temp_dir/dot_product.ptx" -o "$temp_dir/dot_product.cubin"
python="${PYTHON:-python3}"
"$python" bench/bench.py "$temp_dir/vector_add.cubin" "$temp_dir/saxpy.cubin" "$temp_dir/dot_product.cubin" "$@"

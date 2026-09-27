#!/usr/bin/env bash
set -euo pipefail

if ! command -v ptxas >/dev/null || ! command -v nvcc >/dev/null; then
  echo "Load a CUDA toolkit module (providing ptxas and nvcc) first." >&2
  exit 1
fi

metadata_dir="$(mktemp -d)"
trap 'rm -rf "$metadata_dir"' EXIT
tools/compile_oxcaml_kernels.sh "$metadata_dir"
dune exec examples/vector_add.exe -- "$metadata_dir/vector_add.gpu" > /tmp/oxgpu-vector-add.ptx
dune exec examples/saxpy.exe -- "$metadata_dir/saxpy.gpu" > /tmp/oxgpu-saxpy.ptx
ptxas -arch=sm_90 /tmp/oxgpu-vector-add.ptx -o /tmp/oxgpu-vector-add.cubin
ptxas -arch=sm_90 /tmp/oxgpu-saxpy.ptx -o /tmp/oxgpu-saxpy.cubin
nvcc -O2 test/run_h100.cu -lcuda -o /tmp/oxgpu-run-h100
/tmp/oxgpu-run-h100 /tmp/oxgpu-vector-add.cubin /tmp/oxgpu-saxpy.cubin

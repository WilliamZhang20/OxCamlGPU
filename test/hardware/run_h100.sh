#!/usr/bin/env bash
set -euo pipefail

if ! command -v ptxas >/dev/null || ! command -v nvcc >/dev/null; then
  echo "Load a CUDA toolkit module (providing ptxas and nvcc) first." >&2
  exit 1
fi

root="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=../../tools/oxcaml-env.sh
source "$root/tools/oxcaml-env.sh"
# Refuse to run on a contended GPU.
# shellcheck source=../../tools/gpu_idle.sh
source "$root/tools/gpu_idle.sh"
cd "$root"

metadata_dir="$(mktemp -d)"
trap 'rm -rf "$metadata_dir"' EXIT
tools/compile_oxcaml_kernels.sh "$metadata_dir"
dune exec examples/vector_add.exe -- "$metadata_dir/vector_add.gpu" > /tmp/oxgpu-vector-add.ptx
dune exec examples/saxpy.exe -- "$metadata_dir/saxpy.gpu" > /tmp/oxgpu-saxpy.ptx
dune exec examples/dot_product.exe -- "$metadata_dir/dot_product.gpu" > /tmp/oxgpu-dot-product.ptx
dune exec test/emit_ptx.exe -- "$metadata_dir/unique_reuse.gpu" > /tmp/oxgpu-unique-reuse.ptx
dune exec test/emit_ptx.exe -- "$metadata_dir/alias_reuse_aliased.gpu" > /tmp/oxgpu-alias-reuse-aliased.ptx
ptxas -arch=sm_90 /tmp/oxgpu-vector-add.ptx -o /tmp/oxgpu-vector-add.cubin
ptxas -arch=sm_90 /tmp/oxgpu-saxpy.ptx -o /tmp/oxgpu-saxpy.cubin
ptxas -arch=sm_90 /tmp/oxgpu-dot-product.ptx -o /tmp/oxgpu-dot-product.cubin
ptxas -arch=sm_90 /tmp/oxgpu-unique-reuse.ptx -o /tmp/oxgpu-unique-reuse.cubin
ptxas -arch=sm_90 /tmp/oxgpu-alias-reuse-aliased.ptx -o /tmp/oxgpu-alias-reuse-aliased.cubin
for kernel in guarded_saxpy short_circuit numeric float_compare uniform_branch joined_reduction rounding; do
  dune exec test/emit_ptx.exe -- "$metadata_dir/$kernel.gpu" > "$metadata_dir/$kernel.ptx"
  ptxas -arch=sm_90 "$metadata_dir/$kernel.ptx" -o "$metadata_dir/$kernel.cubin"
done
dune exec test/test_control_flow.exe -- --ptx > "$metadata_dir/branch_ir.ptx"
ptxas -arch=sm_90 "$metadata_dir/branch_ir.ptx" -o "$metadata_dir/branch_ir.cubin"
nvcc -O2 test/hardware/run_h100.cu -lcuda -o /tmp/oxgpu-run-h100
/tmp/oxgpu-run-h100 /tmp/oxgpu-vector-add.cubin /tmp/oxgpu-saxpy.cubin /tmp/oxgpu-dot-product.cubin /tmp/oxgpu-unique-reuse.cubin /tmp/oxgpu-alias-reuse-aliased.cubin "$metadata_dir"

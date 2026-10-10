#!/usr/bin/env bash
set -euo pipefail

module load StdEnv/2023 gcc/12.3 cuda/12.6 2>/dev/null || module load StdEnv/2023 gcc/12.3 cuda/12.9

if ! command -v nvcc >/dev/null || ! command -v ptxas >/dev/null; then
  echo "CUDA module did not provide nvcc and ptxas." >&2
  exit 1
fi

root="$(cd "$(dirname "$0")/../.." && pwd)"
# Prefer the OxCaml toolchain after module load (modules may prepend system ocamlc).
# shellcheck source=../../tools/oxcaml-env.sh
source "$root/tools/oxcaml-env.sh"
# shellcheck source=../../tools/gpu_idle.sh
source "$root/tools/gpu_idle.sh"
artifact_dir="$(mktemp -d)"
trap 'rm -rf "$artifact_dir"' EXIT

metadata_dir="$artifact_dir/metadata"
(cd "$root" && tools/compile_oxcaml_kernels.sh "$metadata_dir")
(cd "$root" && dune exec test/emit_matmul_ptx.exe -- \
  --gpu "$metadata_dir/matmul_tiled.gpu") \
  > "$artifact_dir/matmul_tiled.ptx"
ptxas -arch=sm_90a "$artifact_dir/matmul_tiled.ptx" -o "$artifact_dir/matmul_tiled.cubin"
# The launch has to request the kernel's dynamic shared window.
read -r _threads smem _total < <(cd "$root" && dune exec test/emit_matmul_ptx.exe -- \
  --info "$metadata_dir/matmul_tiled.gpu")

nvcc -O2 "$root/test/hardware/run_matmul_h100.cu" -lcuda -o "$artifact_dir/run_matmul_h100"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$artifact_dir/run_matmul_h100" \
  "$artifact_dir/matmul_tiled.cubin" "$smem"

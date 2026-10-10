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
(cd "$root" && dune exec test/emit_gemm_ptx.exe -- \
  --gpu "$metadata_dir/gemm_fast.gpu") \
  > "$artifact_dir/gemm_fast.ptx"
ptxas -arch=sm_90a "$artifact_dir/gemm_fast.ptx" -o "$artifact_dir/gemm_fast.cubin"
# The launch has to request the kernel's dynamic shared window.
read -r _threads smem _total < <(cd "$root" && dune exec test/emit_gemm_ptx.exe -- \
  --info "$metadata_dir/gemm_fast.gpu")

nvcc -O2 "$root/test/hardware/run_gemm_h100.cu" -lcuda -o "$artifact_dir/run_gemm_h100"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$artifact_dir/run_gemm_h100" \
  "$artifact_dir/gemm_fast.cubin" "$smem"

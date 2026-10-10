#!/usr/bin/env bash
set -euo pipefail

module load StdEnv/2023 gcc/12.3 cuda/12.9

if ! command -v ptxas >/dev/null || ! command -v nvcc >/dev/null; then
  echo "CUDA module did not provide ptxas and nvcc." >&2
  exit 1
fi

root="$(cd "$(dirname "$0")/../.." && pwd)"
# Prefer the OxCaml toolchain after module load, as the other harnesses do.
# shellcheck source=../../tools/oxcaml-env.sh
source "$root/tools/oxcaml-env.sh"
# Refuse to run on a contended GPU, like the other hardware harnesses.
# shellcheck source=../../tools/gpu_idle.sh
source "$root/tools/gpu_idle.sh"
cd "$root"

artifact_dir="$(mktemp -d)"
trap 'rm -rf "$artifact_dir"' EXIT

for kernel in mul64_interleaved mul64_blocked dot64_interleaved; do
  dune exec test/emit_multireg_ptx.exe -- "$kernel" > "$artifact_dir/$kernel.ptx"
  ptxas -arch=sm_90 "$artifact_dir/$kernel.ptx" -o "$artifact_dir/$kernel.cubin"
done

nvcc -O2 "$root/test/hardware/run_multireg_h100.cu" -lcuda -o "$artifact_dir/run_multireg_h100"
"$artifact_dir/run_multireg_h100" \
  "$artifact_dir/mul64_interleaved.cubin" \
  "$artifact_dir/mul64_blocked.cubin" \
  "$artifact_dir/dot64_interleaved.cubin"

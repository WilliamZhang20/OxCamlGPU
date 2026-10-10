#!/usr/bin/env bash
# Framework comparison: a naive Triton GEMM and a tuned CuTeDSL GEMM, each
# against cuBLAS at its own precision. bench/run.sh covers the OxCaml kernel.
#
# Precisions differ and that is not incidental: CuTeDSL 4.8.0 has no TF32
# warpgroup MMA, so its Hopper path is bf16. Read the percentages of cuBLAS,
# not the raw TFLOPS, when comparing across the three.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../tools/gpu_idle.sh
source "$root/tools/gpu_idle.sh"
cd "$root"

if [[ -n "${PYTHON:-}" ]]; then
  python="$PYTHON"
elif [[ -x "$root/.bench-venv/bin/python" ]]; then
  python="$root/.bench-venv/bin/python"
else
  python=python3
fi

run_one () {
  local name="$1" script="$2"
  if ! "$python" -c "import $name" >/dev/null 2>&1; then
    echo "skipping $script: $name is not installed" >&2
    return 0
  fi
  echo
  echo "######## $script ########"
  (cd "$root/bench" && "$python" "$(basename "$script")")
}

run_one triton bench/triton_bench.py
run_one cutlass bench/cutedsl_bench.py

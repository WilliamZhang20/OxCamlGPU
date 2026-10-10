#!/usr/bin/env bash
# Framework comparison against cuBLAS, under the same timing protocol as
# bench/run.sh, which covers the OxCaml kernel.
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

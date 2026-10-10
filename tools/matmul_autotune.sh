#!/usr/bin/env bash
# Autotune the Hopper tile catalog vs cuBLAS (idle GPU only).
# Default: OxCaml specializations from examples/kernels/matmul_tiled.ml.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=gpu_idle.sh
source "$root/tools/gpu_idle.sh"

module load StdEnv/2023 gcc/12.3 cuda/12.6 2>/dev/null \
  || module load StdEnv/2023 gcc/12.3 cuda/12.9 || true
# shellcheck source=oxcaml-env.sh
source "$root/tools/oxcaml-env.sh"

M="${1:-4096}"
N="${2:-$M}"
K="${3:-$M}"

if [[ -n "${PYTHON:-}" ]]; then
  python="$PYTHON"
elif [[ -x "$root/.bench-venv/bin/python" ]]; then
  python="$root/.bench-venv/bin/python"
else
  python=python3
fi

temp="$(mktemp -d)"
trap 'rm -rf "$temp"' EXIT

echo "matmul_autotune: problem ${M}x${N}x${K} on GPU ${CUDA_VISIBLE_DEVICES}" >&2

best_tf=0
best_cfg=""
best_pct=0

(cd "$root" && tools/compile_oxcaml_kernels.sh "$temp/metadata")

while IFS= read -r cfg; do
  [[ -z "$cfg" ]] && continue
  echo "--- trying $cfg" >&2
  read -r name bm bn _bk threads < <(cd "$root" && dune exec test/emit_matmul_ptx.exe -- --print-config "$cfg") \
    || { echo "no OxCaml kernel for $cfg" >&2; continue; }
  (cd "$root" && dune exec test/emit_matmul_ptx.exe -- --gpu "$temp/metadata/$name.gpu") \
    >"$temp/matmul.ptx" 2>"$temp/emit.err" || {
      echo "emit failed for $cfg ($name)" >&2
      cat "$temp/emit.err" >&2
      continue
    }
  if (( M % bm != 0 || N % bn != 0 )); then
    echo "skip: size not divisible by tile $bm x $bn" >&2
    continue
  fi
  ptxas -arch=sm_90a "$temp/matmul.ptx" -o "$temp/matmul.cubin" 2>"$temp/ptxas.err" || {
    echo "ptxas failed for $cfg" >&2
    cat "$temp/ptxas.err" >&2
    continue
  }
  out="$("$python" "$root/bench/bench.py" \
    --skip-micro \
    --matmul-cubin "$temp/matmul.cubin" \
    --matmul-name "$name" \
    --matmul-size "$M" "$N" "$K" \
    --matmul-bm "$bm" --matmul-bn "$bn" --matmul-threads "$threads" \
    --warmup 3 --repeat 10 --cooldown 1 2>&1)" || {
      echo "bench failed for $cfg" >&2
      echo "$out" >&2
      continue
    }
  echo "$out"
  # Parse "OxGPU ... TFLOPS ... %cuBLAS" line
  pct=$(echo "$out" | sed -n 's/.*OxGPU[^0-9]*\([0-9.]*\)[^0-9]*\([0-9.]*\).*/\2/p' | tail -1)
  tf=$(echo "$out" | awk '/OxGPU/{for(i=1;i<=NF;i++) if($i+0==$i){t=$i}} END{print t}')
  # Prefer the %cuBLAS column from the table (last float on OxGPU line often %)
  line=$(echo "$out" | grep -E 'OxGPU|oxgpu' | tail -1 || true)
  if [[ -n "$line" ]]; then
    # columns: name ms tflops pct
    tf=$(echo "$line" | awk '{print $(NF-1)}')
    pct=$(echo "$line" | awk '{print $NF}')
  fi
  echo "result: cfg=$cfg tflops=$tf pct=$pct" >&2
  # numeric compare
  if awk "BEGIN{exit !($tf > $best_tf)}"; then
    best_tf=$tf
    best_cfg=$cfg
    best_pct=$pct
  fi
done < <(cd "$root" && dune exec test/emit_matmul_ptx.exe -- --list-catalog)

echo "BEST $best_cfg  TFLOPS=$best_tf  %cuBLAS=$best_pct" >&2
echo "$best_cfg"

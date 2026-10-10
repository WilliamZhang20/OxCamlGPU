#!/usr/bin/env bash
set -euo pipefail

if ! command -v ptxas >/dev/null; then
  echo "Load a CUDA toolkit module providing ptxas first." >&2
  exit 1
fi
if ! command -v nvcc >/dev/null; then
  echo "Load a CUDA toolkit module providing nvcc first." >&2
  exit 1
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
# Prefer OxCaml after any prior module load (system ocamlc lacks Printexc.Safe).
# shellcheck source=../tools/oxcaml-env.sh
source "$root/tools/oxcaml-env.sh"
# Refuse to bench on a contended GPU (skews %cuBLAS badly).
# shellcheck source=../tools/gpu_idle.sh
source "$root/tools/gpu_idle.sh"

if [[ -n "${PYTHON:-}" ]]; then
  python="$PYTHON"
elif [[ -x "$root/.bench-venv/bin/python" ]]; then
  python="$root/.bench-venv/bin/python"
else
  python=python3
fi

temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT
metadata_dir="$temp_dir/metadata"
mkdir -p "$metadata_dir"
tools/compile_oxcaml_kernels.sh "$metadata_dir"
dune exec test/emit_ptx.exe -- "$metadata_dir/vector_add.gpu" > "$temp_dir/vector_add.ptx"
dune exec test/emit_ptx.exe -- "$metadata_dir/saxpy.gpu" > "$temp_dir/saxpy.ptx"
dune exec test/emit_ptx.exe -- "$metadata_dir/dot_product.gpu" > "$temp_dir/dot_product.ptx"
ptxas -arch=sm_90 "$temp_dir/vector_add.ptx" -o "$temp_dir/vector_add.cubin"
ptxas -arch=sm_90 "$temp_dir/saxpy.ptx" -o "$temp_dir/saxpy.cubin"
ptxas -arch=sm_90 "$temp_dir/dot_product.ptx" -o "$temp_dir/dot_product.cubin"

# Microbenchmarks (vector/saxpy/dot) once.
"$python" bench/bench.py "$temp_dir/vector_add.cubin" "$temp_dir/saxpy.cubin" \
  "$temp_dir/dot_product.cubin" "$@"

# Tiled GEMM vs cuBLAS TF32 at 4096³ and 8192³.
# choose_config picks gemm_fast (128x256, stages=4) at both sizes. Set
# MATMUL_BM/BN/BK/STAGES/THREADS to pin one tile for both sizes.
bench_args=("$@")
# The emitter records the dynamic shared window in the PTX, so any harness can
# read the number it must pass to cuLaunchKernel.
dynamic_smem () {
  awk '/^\/\/ oxgpu.shared.dynamic /{print $3; exit}' "$1"
}
run_matmul_bench () {
  local cubin="$1" size="$2" name="$3" bm="$4" bn="$5" bk="$6" threads="$7" smem="$8"
  "$python" bench/bench.py --skip-micro \
    --matmul-cubin "$cubin" \
    --matmul-name "$name" \
    --matmul-sizes "$size" \
    --matmul-bm "$bm" \
    --matmul-bn "$bn" \
    --matmul-bk "$bk" \
    --matmul-threads "$threads" \
    --matmul-smem "$smem" \
    "${bench_args[@]}"
}
emit_oxcaml_tile () {
  local name="$1"
  (cd "$root" && dune exec test/emit_gemm_ptx.exe -- --gpu "$metadata_dir/$name.gpu") \
    > "$temp_dir/$name.ptx"
  ptxas -arch=sm_90a "$temp_dir/$name.ptx" -o "$temp_dir/$name.cubin"
}
if [[ -n "${MATMUL_BM:-}${MATMUL_BN:-}${MATMUL_BK:-}${MATMUL_STAGES:-}${MATMUL_THREADS:-}" ]]; then
  spec="bm=${MATMUL_BM:-128},bn=${MATMUL_BN:-256},bk=${MATMUL_BK:-32},stages=${MATMUL_STAGES:-4},producers=32"
  read -r name bm bn bk threads < <(cd "$root" && dune exec test/emit_gemm_ptx.exe -- --print-config "$spec")
  if [[ -n "${MATMUL_THREADS:-}" && "$MATMUL_THREADS" != "$threads" ]]; then
    echo "MATMUL_THREADS=$MATMUL_THREADS does not match $name ($threads threads)." >&2
    exit 1
  fi
  emit_oxcaml_tile "$name"
  smem="$(dynamic_smem "$temp_dir/$name.ptx")"
  for size in 4096 8192; do
    run_matmul_bench "$temp_dir/$name.cubin" "$size" "$name" "$bm" "$bn" "$bk" "$threads" "$smem"
  done
else
  for size in 4096 8192; do
    read -r name bm bn bk threads < <(cd "$root" && dune exec test/emit_gemm_ptx.exe -- --print-choose "$size" "$size" "$size")
    if [[ ! -f "$temp_dir/$name.cubin" ]]; then
      emit_oxcaml_tile "$name"
    fi
    run_matmul_bench "$temp_dir/$name.cubin" "$size" "$name" "$bm" "$bn" "$bk" "$threads" \
      "$(dynamic_smem "$temp_dir/$name.ptx")"
  done
fi

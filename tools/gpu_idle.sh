#!/usr/bin/env bash
# Pick a CUDA GPU that is both idle and has memory headroom, and export
# CUDA_VISIBLE_DEVICES. Utilization alone is not enough: a GPU reads 0% busy
# while a neighbour is still allocating, and the benchmark then OOMs.
# Usage: source tools/gpu_idle.sh
# If CUDA_VISIBLE_DEVICES is already set but that GPU is busy, falls through
# to another idle GPU (avoids sticky contended picks across scripts).
set -euo pipefail

max_util="${GPU_IDLE_MAX_UTIL:-15}"
# A GPU can read 0% busy while a neighbour has already reserved its memory,
# and a benchmark that then OOMs wastes the run. Require headroom too.
min_free_mib="${GPU_IDLE_MIN_FREE_MIB:-8192}"
prefer="${CUDA_VISIBLE_DEVICES-}"

if ! command -v nvidia-smi >/dev/null; then
  echo "gpu_idle: nvidia-smi not found" >&2
  return 1 2>/dev/null || exit 1
fi

pick=""
pick_util=999
pick_free=0
prefer_util=""
prefer_free=0
while IFS= read -r line; do
  idx=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
  util=$(echo "$line" | cut -d',' -f2 | tr -d ' ')
  free=$(echo "$line" | cut -d',' -f3 | tr -d ' ')
  [[ "$util" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] || continue
  if [[ -n "$prefer" && "$idx" == "$prefer" ]]; then
    prefer_util=$util
    prefer_free=$free
  fi
  # Only consider GPUs with room; among those take the least busy.
  if (( free >= min_free_mib )) && (( util < pick_util )); then
    pick="$idx"
    pick_util="$util"
    pick_free="$free"
  fi
done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits)

# Honor an already-idle preferred device; otherwise take the global idle pick.
if [[ -n "$prefer" && -n "$prefer_util" ]] && (( prefer_util <= max_util )) \
   && (( prefer_free >= min_free_mib )); then
  pick=$prefer
  pick_util=$prefer_util
  pick_free=$prefer_free
fi

if [[ -z "$pick" ]] || (( pick_util > max_util )); then
  echo "gpu_idle: no GPU is both under ${max_util}% busy and has ${min_free_mib} MiB free" >&2
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv >&2
  return 1 2>/dev/null || exit 1
fi

echo "gpu_idle: using GPU $pick (util ${pick_util}%, ${pick_free} MiB free)" >&2
export CUDA_VISIBLE_DEVICES="$pick"

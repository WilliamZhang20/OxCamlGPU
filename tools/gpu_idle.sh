#!/usr/bin/env bash
# Pick a CUDA GPU with low utilization. Exports CUDA_VISIBLE_DEVICES.
# Usage: source tools/gpu_idle.sh
# If CUDA_VISIBLE_DEVICES is already set but that GPU is busy, falls through
# to another idle GPU (avoids sticky contended picks across scripts).
set -euo pipefail

max_util="${GPU_IDLE_MAX_UTIL:-15}"
prefer="${CUDA_VISIBLE_DEVICES-}"

if ! command -v nvidia-smi >/dev/null; then
  echo "gpu_idle: nvidia-smi not found" >&2
  return 1 2>/dev/null || exit 1
fi

pick=""
pick_util=999
prefer_util=""
while IFS= read -r line; do
  idx=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
  util=$(echo "$line" | cut -d',' -f2 | tr -d ' ')
  [[ "$util" =~ ^[0-9]+$ ]] || continue
  if [[ -n "$prefer" && "$idx" == "$prefer" ]]; then
    prefer_util=$util
  fi
  if (( util < pick_util )); then
    pick="$idx"
    pick_util="$util"
  fi
done < <(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)

if [[ -z "$pick" ]]; then
  echo "gpu_idle: no GPU found" >&2
  return 1 2>/dev/null || exit 1
fi

# Honor an already-idle preferred device; otherwise take the global idle pick.
if [[ -n "$prefer" && -n "$prefer_util" ]] && (( prefer_util <= max_util )); then
  pick=$prefer
  pick_util=$prefer_util
fi

if (( pick_util > max_util )); then
  echo "gpu_idle: all GPUs busy (best gpu $pick at ${pick_util}% > max ${max_util}%)" >&2
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv >&2
  return 1 2>/dev/null || exit 1
fi

echo "gpu_idle: using GPU $pick (util ${pick_util}%)" >&2
export CUDA_VISIBLE_DEVICES="$pick"

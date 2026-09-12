#!/usr/bin/env bash
# Extract frames from a headerless YUV420p sequence as PNG.
# Raw .yuv carries no resolution/framerate, so ffmpeg must be told -- getting
# this wrong silently produces skewed garbage rather than an error.
#   usage: yuv2png.sh <in.yuv> <WxH> <num_frames> <out_dir>
set -euo pipefail
IN="$1"; SIZE="$2"; N="$3"; OUT="$4"
mkdir -p "${OUT}"
ffmpeg -hide_banner -loglevel error -y \
  -f rawvideo -pix_fmt yuv420p -s "${SIZE}" -i "${IN}" \
  -frames:v "${N}" "${OUT}/frame_%04d.png"
echo "extracted ${N} frame(s) from $(basename "${IN}") -> ${OUT}/"
ls "${OUT}" | head -5

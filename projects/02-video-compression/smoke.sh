#!/usr/bin/env bash
# Smoke test for DCVC-UF video coding: one intra frame + one full chunk.
set -euo pipefail
cd /opt/DCVC

if [[ -f /work/weights/dcvc/cvpr2026_video_ld.pth.tar ]]; then
  echo "=== LD model (chunk size 1) ==="
  python /work/projects/02-video-compression/run_video.py --structure ld --frames 5
  echo
  echo "=== HTS model (chunk size 8 -- the chunk-based design) ==="
  python /work/projects/02-video-compression/run_video.py --structure hts --frames 9
else
  echo "no checkpoints in weights/dcvc/ -- running environment preflight only"
  python /work/projects/01-image-compression/preflight.py
fi

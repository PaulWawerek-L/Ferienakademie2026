#!/usr/bin/env bash
# Smoke test for DCVC-UF-Intra.
# Falls back to the checkpoint-free preflight so this still passes for a student
# who has cloned the repo but not yet done the manual OneDrive download.
set -euo pipefail
cd /opt/DCVC

if [[ -f /work/weights/dcvc/cvpr2026_image.pth.tar ]]; then
  echo "checkpoint found -- running the real RD sweep"
  python /work/projects/01-image-compression/run_image.py --rate-num 4
else
  echo "no checkpoint in weights/dcvc/ -- running environment preflight only"
  echo "(run 'make weights' for the download instructions)"
  python /work/projects/01-image-compression/preflight.py
fi

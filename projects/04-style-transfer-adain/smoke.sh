#!/usr/bin/env bash
# Smoke test: AdaIN style transfer, kodim19 (content) x RaceHorses frame (style).
set -euo pipefail
OUT=/work/outputs/04-style-transfer
mkdir -p "${OUT}"

# The style image comes from the video sequence, so the test needs nothing
# beyond what is already in Dataset/.
bash /work/scripts/yuv2png.sh \
  /work/Dataset/RaceHorses_416x240_30.yuv 416x240 1 "${OUT}/style_src"

cd /opt/AdaIN
python test.py \
  --content /work/Dataset/kodim19.png \
  --style   "${OUT}/style_src/frame_0001.png" \
  --vgg     /work/weights/adain/vgg_normalised.pth \
  --decoder /work/weights/adain/decoder.pth \
  --output  "${OUT}" \
  --alpha   1.0

echo "--- produced ---"
ls -la "${OUT}"/*.jpg "${OUT}"/*.png 2>/dev/null | grep -v style_src || true

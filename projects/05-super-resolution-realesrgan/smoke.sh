#!/usr/bin/env bash
# Smoke test: Real-ESRGAN x4 on a small crop.
# A crop, not the full 768x512 kodim19: x4 on CPU would mean a 3072x2048 output
# and several minutes. The crop proves the pipeline in seconds; students can
# scale up once they know it works.
set -euo pipefail
OUT=/work/outputs/05-super-resolution
mkdir -p "${OUT}/in"

python - <<'PY'
from PIL import Image
im = Image.open('/work/Dataset/kodim19.png').convert('RGB')
im.crop((200, 150, 328, 278)).save('/work/outputs/05-super-resolution/in/kodim19_crop128.png')
print('input crop: 128x128 from kodim19.png')
PY

cd /opt/Real-ESRGAN
python inference_realesrgan.py \
  -n RealESRGAN_x4plus \
  --model_path /work/weights/realesrgan/RealESRGAN_x4plus.pth \
  -i "${OUT}/in" \
  -o "${OUT}" \
  --outscale 4 \
  --fp32

echo "--- produced ---"
python - <<'PY'
import glob
from PIL import Image
for f in sorted(glob.glob('/work/outputs/05-super-resolution/*.png')):
    print(f, Image.open(f).size)
PY

#!/usr/bin/env bash
# Task 5 -- AdaIN across several styles, and across the alpha range.
#
# Style images come from the ones the AdaIN repo ships in /opt/AdaIN/input/style,
# so nothing has to be downloaded. Content is kodim19 plus one RaceHorses frame,
# which keeps the whole experiment inside this repo's own Dataset.
#   bash exp5_adain_styles.sh [--styles "mondrian sketch ..."] [--alphas "0.25 0.5 1.0"]
set -euo pipefail

STYLES="brushstrokes mondrian picasso_self_portrait sketch woman_with_hat_matisse la_muse scene_de_rue flower_of_life"
ALPHAS="1.0"
ALPHA_SWEEP_STYLE="brushstrokes"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --styles) STYLES="$2"; shift 2 ;;
    --alphas) ALPHAS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

OUT=/work/outputs/experiments/05-adain-styles
mkdir -p "${OUT}/grid" "${OUT}/alpha" "${OUT}/styles_used"
STYLE_DIR=/opt/AdaIN/input/style

# Content 2 comes from the video sequence, so both contents are ours.
mkdir -p "${OUT}/content"
CONTENT_A=/work/Dataset/kodim19.png
CONTENT_B="${OUT}/content/racehorses_f0.png"
ffmpeg -hide_banner -loglevel error -y -f rawvideo -pix_fmt yuv420p -s 416x240 \
  -i /work/Dataset/RaceHorses_416x240_30.yuv -frames:v 1 "${CONTENT_B}"

cd /opt/AdaIN

echo "=== style sweep (alpha=1.0) ==="
for STYLE in ${STYLES}; do
  SP=""
  for ext in jpg png jpeg; do
    [[ -f "${STYLE_DIR}/${STYLE}.${ext}" ]] && SP="${STYLE_DIR}/${STYLE}.${ext}" && break
  done
  if [[ -z "${SP}" ]]; then
    echo "  [skip] ${STYLE} -- not in ${STYLE_DIR}"
    continue
  fi
  cp "${SP}" "${OUT}/styles_used/" 2>/dev/null || true
  echo "  ${STYLE}"
  for CONTENT in "${CONTENT_A}" "${CONTENT_B}"; do
    python test.py \
      --content "${CONTENT}" --style "${SP}" \
      --vgg /work/weights/adain/vgg_normalised.pth \
      --decoder /work/weights/adain/decoder.pth \
      --output "${OUT}/grid" --alpha 1.0 > /dev/null
  done
done

# alpha controls the content/style mix and is the cheapest way to see what AdaIN
# actually does -- at 0 it is the identity, at 1 full statistics transfer.
echo
echo "=== alpha sweep on ${ALPHA_SWEEP_STYLE} ==="
SP="${STYLE_DIR}/${ALPHA_SWEEP_STYLE}.jpg"
for A in 0.0 0.25 0.5 0.75 1.0; do
  echo "  alpha=${A}"
  python test.py \
    --content "${CONTENT_A}" --style "${SP}" \
    --vgg /work/weights/adain/vgg_normalised.pth \
    --decoder /work/weights/adain/decoder.pth \
    --output "${OUT}/alpha" --alpha "${A}" > /dev/null
  # test.py names the file only by content+style, so each alpha would overwrite
  # the previous one.
  LAST=$(ls -t "${OUT}/alpha"/*.jpg 2>/dev/null | head -1)
  [[ -n "${LAST}" ]] && mv "${LAST}" "${OUT}/alpha/alpha_${A}.jpg"
done

echo
echo "produced:"
ls "${OUT}/grid" | wc -l | xargs echo "  style-transfer results:"
ls "${OUT}/alpha"

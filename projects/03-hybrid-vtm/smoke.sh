#!/usr/bin/env bash
# Smoke test: VTM encode -> decode round-trip on 4 frames of RaceHorses.
# 4 frames, not 300: VTM is a reference implementation, not a fast encoder --
# it is ~100x slower than a production codec, and the point here is to prove
# the binaries work, not to produce an RD curve.
set -euo pipefail
OUT=/work/outputs/03-hybrid-vtm
mkdir -p "${OUT}"

SEQ=/work/Dataset/RaceHorses_416x240_30.yuv
W=416; H=240; FPS=30; FRAMES=4; QP=37

echo "=== EncoderApp (QP=${QP}, ${FRAMES} frames) ==="
EncoderApp \
  -c "${VTM_CFG}/encoder_randomaccess_vtm.cfg" \
  -i "${SEQ}" \
  --SourceWidth=${W} --SourceHeight=${H} \
  --InputBitDepth=8 --FrameRate=${FPS} \
  --FramesToBeEncoded=${FRAMES} \
  --QP=${QP} \
  -b "${OUT}/rh_qp${QP}.bin" \
  -o "${OUT}/rh_qp${QP}_rec.yuv" 2>&1 | tail -20

echo
echo "=== DecoderApp (round-trip) ==="
DecoderApp -b "${OUT}/rh_qp${QP}.bin" -o "${OUT}/rh_qp${QP}_dec.yuv" 2>&1 | tail -8

echo
echo "=== verifying decoder output matches encoder reconstruction ==="
if cmp -s "${OUT}/rh_qp${QP}_rec.yuv" "${OUT}/rh_qp${QP}_dec.yuv"; then
  echo "PASS: decoded YUV is bit-exact with the encoder reconstruction"
else
  echo "FAIL: decoder output differs from encoder reconstruction"; exit 1
fi

BITS=$(wc -c < "${OUT}/rh_qp${QP}.bin")
python3 -c "print(f'bitstream: ${BITS} bytes -> {${BITS}*8/(${W}*${H}*${FRAMES}):.4f} bpp over ${FRAMES} frames')"

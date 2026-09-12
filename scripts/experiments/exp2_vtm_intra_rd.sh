#!/usr/bin/env bash
# Task 2 -- VTM all-intra RD curve on kodim19.
#
# VTM only eats raw YUV, so the PNG is converted to YUV420 first. That same YUV
# is what task 1 feeds to DCVC-UF, so the two image curves are measured on
# identical input in an identical colour space.
#   bash exp2_vtm_intra_rd.sh [--qps "22 27 32 37 42"]
set -euo pipefail

QPS="22 27 32 37 42"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --qps) QPS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

OUT=/work/outputs/experiments/02-vtm-intra
mkdir -p "${OUT}"
SRC=/work/Dataset/kodim19.png
REF=/work/outputs/experiments/kodim19_420.yuv

W=$(python3 -c "
import struct,sys
d=open('${SRC}','rb').read()
print(struct.unpack('>I', d[16:20])[0])")
H=$(python3 -c "
import struct,sys
d=open('${SRC}','rb').read()
print(struct.unpack('>I', d[20:24])[0])")
echo "kodim19: ${W}x${H}"

mkdir -p "$(dirname "${REF}")"
if [[ ! -f "${REF}" ]]; then
  ffmpeg -hide_banner -loglevel error -y -i "${SRC}" \
    -pix_fmt yuv420p -f rawvideo "${REF}"
  echo "converted to YUV420: ${REF} ($(wc -c < "${REF}") bytes)"
fi

echo "VTM all-intra | QPs: ${QPS}"
POINTS=""
for QP in ${QPS}; do
  BIN="${OUT}/kodim19_qp${QP}.bin"
  REC="${OUT}/kodim19_qp${QP}_rec.yuv"
  echo "--- QP ${QP} ---"
  start=$(date +%s)
  EncoderApp \
    -c "${VTM_CFG}/encoder_intra_vtm.cfg" \
    -i "${REF}" \
    --SourceWidth=${W} --SourceHeight=${H} \
    --InputBitDepth=8 --FrameRate=1 \
    --FramesToBeEncoded=1 \
    --QP=${QP} \
    --OutputBitDepth=8 \
    -b "${BIN}" -o "${REC}" > "${OUT}/enc_qp${QP}.log" 2>&1
  dur=$(( $(date +%s) - start ))
  BYTES=$(wc -c < "${BIN}")
  python3 /work/scripts/experiments/yuv_psnr.py "${REF}" "${REC}" ${W} ${H} \
      --frames 1 --json "${OUT}/psnr_qp${QP}.json"
  echo "QP ${QP}: ${BYTES} bytes, ${dur}s"
  POINTS="${POINTS}${QP} "
done

python3 - "${OUT}" "${W}" "${H}" "${POINTS}" <<'PY'
import json, os, sys
outdir, w, h, points = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split()
pts = []
for qp in sorted(int(q) for q in points):
    p = json.load(open(f"{outdir}/psnr_qp{qp}.json"))
    nbytes = os.path.getsize(f"{outdir}/kodim19_qp{qp}.bin")
    pts.append({"qp": qp, "bpp": nbytes * 8 / (w * h), "psnr_yuv": p["psnr_yuv"],
                "psnr_y": p["psnr_y"], "psnr_u": p["psnr_u"], "psnr_v": p["psnr_v"],
                "real_bitstream": True})
json.dump({"label": "VTM intra", "sequence": "kodim19", "frames": 1,
           "rate": "real bitstream", "points": pts},
          open(f"{outdir}/rd.json", "w"), indent=2)
print(f"wrote {outdir}/rd.json with {len(pts)} points")
PY

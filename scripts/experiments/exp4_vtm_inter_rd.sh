#!/usr/bin/env bash
# Task 4 -- VTM inter (random access) RD curve on the first N frames of RaceHorses.
#
# Runs inside the hybrid-vtm container.
#   bash exp4_vtm_inter_rd.sh [--qps "22 27 32 37 42"] [--frames 64] [--tag std]
#
# COST: VTM is a reference encoder -- correctness first, speed never. Measured
# here on 416x240 random access: 8 frames take 83 s at QP 37 and 509 s at QP 0;
# 64 frames took 494 s at QP 42 alone, and the full sweep was heading past five
# hours. The default is therefore 8 frames, which produces a real curve in
# minutes. Pass --frames 64 to match the original brief if you have the time.
# QPs are encoded cheapest-first and rd.json is rewritten after each point, so
# an interrupted run still leaves a usable curve.
set -euo pipefail

QPS="22 27 32 37 42"
FRAMES=8
TAG="std"
SEQ=/work/Dataset/RaceHorses_416x240_30.yuv
W=416; H=240; FPS=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qps) QPS="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

OUT=/work/outputs/experiments/04-vtm-inter
mkdir -p "${OUT}/${TAG}"
REF="${OUT}/ref_${FRAMES}f.yuv"

# Trim the reference to exactly FRAMES so PSNR compares like with like.
head -c $(( W * H * 3 / 2 * FRAMES )) "${SEQ}" > "${REF}"

# cheapest (highest QP) first
SORTED=$(echo "${QPS}" | tr ' ' '\n' | sort -rn | tr '\n' ' ')
echo "VTM inter (RA) | ${FRAMES} frames | QPs (cheapest first): ${SORTED}"

POINTS=""
for QP in ${SORTED}; do
  BIN="${OUT}/${TAG}/rh_qp${QP}.bin"
  REC="${OUT}/${TAG}/rh_qp${QP}_rec.yuv"
  echo
  echo "--- QP ${QP} ---"
  start=$(date +%s)
  EncoderApp \
    -c "${VTM_CFG}/encoder_randomaccess_vtm.cfg" \
    -i "${SEQ}" \
    --SourceWidth=${W} --SourceHeight=${H} \
    --InputBitDepth=8 --FrameRate=${FPS} \
    --FramesToBeEncoded=${FRAMES} \
    --QP=${QP} \
    --OutputBitDepth=8 \
    -b "${BIN}" -o "${REC}" > "${OUT}/${TAG}/enc_qp${QP}.log" 2>&1
  dur=$(( $(date +%s) - start ))

  BYTES=$(wc -c < "${BIN}")
  python3 /work/scripts/experiments/yuv_psnr.py "${REF}" "${REC}" ${W} ${H} \
      --frames ${FRAMES} --json "${OUT}/${TAG}/psnr_qp${QP}.json"
  BPP=$(python3 -c "print(${BYTES}*8/(${W}*${H}*${FRAMES}))")
  echo "QP ${QP}: ${BYTES} bytes -> ${BPP} bpp, ${dur}s"

  POINTS="${POINTS}${QP} "
  # Write the curve after every point, so an interrupted run is still usable.
  python3 - "${OUT}/${TAG}" "${W}" "${H}" "${FRAMES}" "${POINTS}" <<'PY'
import json, os, sys
outdir, w, h, frames, points = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5].split()
pts = []
for qp in sorted(int(q) for q in points):
    p = json.load(open(f"{outdir}/psnr_qp{qp}.json"))
    nbytes = os.path.getsize(f"{outdir}/rh_qp{qp}.bin")
    pts.append({"qp": qp, "bpp": nbytes * 8 / (w * h * frames),
                "kbps": nbytes * 8 * 30 / frames / 1000,
                "psnr_yuv": p["psnr_yuv"], "psnr_y": p["psnr_y"],
                "psnr_u": p["psnr_u"], "psnr_v": p["psnr_v"], "real_bitstream": True})
json.dump({"label": "VTM inter (RA)", "sequence": "RaceHorses_416x240",
           "frames": frames, "rate": "real bitstream", "points": pts},
          open(f"{outdir}/rd.json", "w"), indent=2)
print(f"  -> {outdir}/rd.json now has {len(pts)} point(s)")
PY
done

echo
echo "done: ${OUT}/${TAG}/rd.json"

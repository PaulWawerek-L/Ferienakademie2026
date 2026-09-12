"""Rebuild an rd.json from bitstreams and reconstructions already on disk.

Encoding is the expensive part (a VTM inter point costs minutes to an hour), so
when only the measurement changes there is no reason to re-encode. Point this at
an experiment directory and it re-measures every .bin/_rec.yuv pair it finds.

  python rebuild_rd.py --dir out/02-vtm-intra --ref ref.yuv \
      --width 512 --height 768 --frames 1 --label "VTM intra" --sequence kodim19
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bins = sorted(glob.glob(os.path.join(args.dir, "*.bin")))
    if not bins:
        sys.exit(f"no .bin files in {args.dir}")

    points = []
    for b in bins:
        m = re.search(r"qp(\d+)\.bin$", os.path.basename(b))
        if not m:
            print(f"  [skip] cannot read a qp from {os.path.basename(b)}")
            continue
        qp = int(m.group(1))
        rec = b[:-4] + "_rec.yuv"
        if not os.path.exists(rec):
            print(f"  [skip] qp {qp}: no reconstruction next to the bitstream")
            continue

        pj = os.path.join(args.dir, f"psnr_qp{qp}.json")
        subprocess.run([sys.executable, os.path.join(HERE, "yuv_psnr.py"),
                        args.ref, rec, str(args.width), str(args.height),
                        "--frames", str(args.frames), "--quiet", "--json", pj],
                       check=True)
        p = json.load(open(pj))
        nbytes = os.path.getsize(b)
        bpp = nbytes * 8 / (args.width * args.height * args.frames)
        points.append({"qp": qp, "bpp": bpp,
                       "kbps": bpp * args.width * args.height * args.fps / 1000,
                       "bytes": nbytes,
                       "psnr_yuv": p["psnr_yuv"], "psnr_y": p["psnr_y"],
                       "psnr_u": p["psnr_u"], "psnr_v": p["psnr_v"],
                       "real_bitstream": True})
        print(f"  qp {qp:>2}: {bpp:.4f} bpp, {p['psnr_yuv']:.2f} dB")

    points.sort(key=lambda x: x["qp"])
    out = args.out or os.path.join(args.dir, "rd.json")
    json.dump({"label": args.label, "sequence": args.sequence,
               "frames": args.frames, "rate": "real bitstream", "points": points},
              open(out, "w"), indent=2)
    print(f"wrote {out} with {len(points)} points")


if __name__ == "__main__":
    main()

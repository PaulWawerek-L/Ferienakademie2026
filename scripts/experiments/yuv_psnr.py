"""PSNR between two headerless YUV420 8-bit files, one formula for every codec.

Why this exists: VTM prints its own YUV-PSNR and DCVC's test_video.py uses
(6*Y + U + V)/8. They are different definitions and disagree by roughly half a
dB, so reading each codec's own log would silently bias any comparison between
them. Every experiment here instead dumps a reconstructed .yuv and calls this.

  python yuv_psnr.py ref.yuv rec.yuv 416 240 [--frames N] [--json out.json]
"""
import argparse
import json
import os
import sys

import numpy as np


def detect_sample_bytes(path, w, h, frames):
    """1 byte per sample, or 2?

    VTM's cfg files set InternalBitDepth 10 and, unless told otherwise, write the
    reconstruction at that depth -- 10 bits inside 16-bit words. Reading such a
    file as 8-bit silently produces noise and a PSNR around 7 dB that looks like
    a broken codec rather than a broken reader. Size makes it unambiguous when
    the frame count is known.
    """
    frame8 = w * h * 3 // 2
    size = os.path.getsize(path)
    if size == frames * frame8:
        return 1
    if size == frames * frame8 * 2:
        return 2
    raise SystemExit(
        f"{os.path.basename(path)} is {size} bytes; expected {frames * frame8} "
        f"(8-bit) or {frames * frame8 * 2} (16-bit) for {w}x{h} x{frames}. "
        f"Check the resolution and frame count.")


def read_frames(path, w, h, n, sample_bytes):
    """Yield (y, u, v) planes as float64 on an 8-bit scale, frame by frame."""
    y_size = w * h
    uv_size = y_size // 4
    dtype = np.uint8 if sample_bytes == 1 else np.uint16
    frame_size = (y_size + 2 * uv_size) * sample_bytes
    with open(path, "rb") as f:
        i = 0
        while n is None or i < n:
            buf = f.read(frame_size)
            if len(buf) < frame_size:
                return
            a = np.frombuffer(buf, dtype=dtype).astype(np.float64)
            if sample_bytes == 2:
                # 10-bit -> 8-bit with rounding, which is what an 8-bit output
                # pipeline does. Comparing against the 8-bit source demands it.
                a = np.floor(a / 4.0 + 0.5)
            y = a[:y_size].reshape(h, w)
            u = a[y_size:y_size + uv_size].reshape(h // 2, w // 2)
            v = a[y_size + uv_size:].reshape(h // 2, w // 2)
            yield y, u, v
            i += 1


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-10:
        return 999.9
    return 10.0 * np.log10(255.0 * 255.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("rec")
    ap.add_argument("width", type=int)
    ap.add_argument("height", type=int)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    n = args.frames
    if n is None:
        n = os.path.getsize(args.ref) // (args.width * args.height * 3 // 2)
    ref_sb = detect_sample_bytes(args.ref, args.width, args.height, n)
    rec_sb = detect_sample_bytes(args.rec, args.width, args.height, n)
    if not args.quiet and (ref_sb == 2 or rec_sb == 2):
        print(f"  (ref {ref_sb * 8}-bit container, rec {rec_sb * 8}-bit container "
              f"-- 16-bit samples are scaled down to 8-bit for the comparison)")

    ys, us, vs, yuvs = [], [], [], []
    for (ry, ru, rv), (cy, cu, cv) in zip(
            read_frames(args.ref, args.width, args.height, n, ref_sb),
            read_frames(args.rec, args.width, args.height, n, rec_sb)):
        py, pu, pv = psnr(ry, cy), psnr(ru, cu), psnr(rv, cv)
        ys.append(py); us.append(pu); vs.append(pv)
        # DCVC's convention. Stated explicitly so the number is reproducible;
        # VTM's own YUV-PSNR is a different weighting and is NOT used here.
        yuvs.append((6 * py + pu + pv) / 8)

    if not ys:
        sys.exit(f"no frames compared -- check size {args.width}x{args.height} and file lengths")

    out = {
        "frames": len(ys),
        "psnr_y": float(np.mean(ys)),
        "psnr_u": float(np.mean(us)),
        "psnr_v": float(np.mean(vs)),
        "psnr_yuv": float(np.mean(yuvs)),
        "per_frame_yuv": [float(x) for x in yuvs],
    }
    if not args.quiet:
        print(f"frames {out['frames']}  Y {out['psnr_y']:.4f}  U {out['psnr_u']:.4f}  "
              f"V {out['psnr_v']:.4f}  YUV {out['psnr_yuv']:.4f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

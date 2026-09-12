"""Task 1 -- DCVC-UF-Intra RD curve on kodim19.

Measured on the SAME YUV420 file that task 2 hands to VTM, so the two image
curves are directly comparable. Rate is the entropy model's estimate (see
dcvc_yuv.estimated_bits); VTM's is a real bitstream.

  python exp1_dcvc_image_rd.py [--qps 0 15 30 45 63]
"""
import argparse
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dcvc_yuv import (crop, estimated_bits, eval_rate, pad, padding_for, read_yuv420,
                      to_model_input, write_yuv420)

from src.models.image_model import DMCI
from src.utils.common import get_state_dict

OUT = "/work/outputs/experiments/01-dcvc-image"
REF = "/work/outputs/experiments/kodim19_420.yuv"
SRC = "/work/Dataset/kodim19.png"


def ensure_ref():
    """Make the YUV420 reference if task 2 has not already produced it."""
    if os.path.exists(REF):
        return
    os.makedirs(os.path.dirname(REF), exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", SRC, "-pix_fmt", "yuv420p", "-f", "rawvideo", REF], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 15, 30, 45, 63])
    ap.add_argument("--model", default="/work/weights/dcvc/cvpr2026_image.pth.tar")
    args = ap.parse_args()

    torch.manual_seed(0)   # add_noise() in the rate estimate is stochastic
    os.makedirs(OUT, exist_ok=True)
    ensure_ref()

    from PIL import Image
    W, H = Image.open(SRC).size
    print(f"kodim19: {W}x{H}  |  qps: {args.qps}")

    net = DMCI().eval()
    net.load_state_dict(get_state_dict(args.model))

    frames = read_yuv420(REF, W, H, 1)
    if not frames:
        sys.exit("could not read the YUV reference")
    yuv444, _, _, _ = frames[0]
    x = to_model_input(yuv444)
    pad_r, pad_b = padding_for(H, W)
    if pad_r or pad_b:
        print(f"padding {W}x{H} -> {W + pad_r}x{H + pad_b}")

    points = []
    print(f"\n{'qp':>4} {'est. bpp':>10} {'YUV-PSNR':>10} {'kB':>8}")
    print("-" * 36)
    for qp in args.qps:
        with torch.no_grad(), eval_rate():
            out = net.forward_one_frame(pad(x, pad_r, pad_b), torch.tensor([qp]))
        rec_path = f"{OUT}/kodim19_qp{qp:02d}_rec.yuv"
        with open(rec_path, "wb") as fh:
            write_yuv420(fh, crop(out["x_hat"], H, W))

        bpp = estimated_bits(out) / (H * W)
        res = subprocess.run(
            [sys.executable, "/work/scripts/experiments/yuv_psnr.py", REF, rec_path,
             str(W), str(H), "--frames", "1", "--quiet",
             "--json", f"{OUT}/psnr_qp{qp:02d}.json"], check=True)
        p = json.load(open(f"{OUT}/psnr_qp{qp:02d}.json"))
        print(f"{qp:>4} {bpp:>10.4f} {p['psnr_yuv']:>9.2f}dB {bpp*H*W/8/1024:>7.1f}")
        points.append({"qp": qp, "bpp": bpp, "psnr_yuv": p["psnr_yuv"],
                       "psnr_y": p["psnr_y"], "psnr_u": p["psnr_u"],
                       "psnr_v": p["psnr_v"], "real_bitstream": False})

    json.dump({"label": "DCVC-UF-Intra", "sequence": "kodim19", "frames": 1,
               "rate": "entropy-model estimate (no bitstream on CPU)",
               "points": points},
              open(f"{OUT}/rd.json", "w"), indent=2)
    print(f"\nwrote {OUT}/rd.json")


if __name__ == "__main__":
    main()

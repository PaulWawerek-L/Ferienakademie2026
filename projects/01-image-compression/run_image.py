"""DCVC-UF-Intra on a real image: rate-distortion sweep over the qp range.

Runs the forward path only -- see README.md for why there is no real bitstream
without CUDA. What you get is the entropy model's *estimated* bpp plus the true
PSNR of the reconstruction, which is enough to draw an RD curve.

  python run_image.py                          # defaults: kodim19, 6 rate points
  python run_image.py --rate-num 12 --image ...
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from src.models.image_model import DMCI
from src.utils.common import get_state_dict
from src.utils.metrics import calc_psnr
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb

import sys
# Rate estimation: use the quantised symbols, not the training-time noise proxy.
# forward_one_frame() adds uniform noise before estimating bits, which inflates the
# rate badly at low qp (8x at qp 0 on kodim19, checked against real bitstreams --
# see scripts/bitstream/uf_bitstream_demo.py). eval_rate() swaps the noise for
# rounding; the estimate then lands within 0.7% of the arithmetic-coded payload.
sys.path.insert(0, "/work/scripts/experiments")
from dcvc_yuv import eval_rate  # noqa: E402


def load_rgb(path):
    """PNG -> (1, 3, H, W) float tensor in [0, 1]."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="/work/Dataset/kodim19.png")
    ap.add_argument("--model", default="/work/weights/dcvc/cvpr2026_image.pth.tar")
    ap.add_argument("--out-dir", default="/work/outputs/01-image-compression")
    ap.add_argument("--rate-num", type=int, default=6,
                    help="how many qp points to sample across the model's range")
    ap.add_argument("--save-recon", action="store_true", default=True)
    args = ap.parse_args()

    # forward_one_frame estimates the rate with add_noise(), the training-time
    # quantisation proxy, so the bpp is stochastic. Seed it or two runs of the
    # same command disagree in the third decimal.
    torch.manual_seed(0)

    os.makedirs(args.out_dir, exist_ok=True)

    net = DMCI().eval()
    net.load_state_dict(get_state_dict(args.model))
    print(f"loaded {os.path.basename(args.model)}  "
          f"({sum(p.numel() for p in net.parameters())/1e6:.1f}M params, "
          f"{DMCI.qp_num()} qp levels)")

    rgb = load_rgb(args.image)
    _, _, H, W = rgb.shape
    pad_r, pad_b = DMCI.get_padding_size(H, W, p=64)
    if pad_r or pad_b:
        raise SystemExit(
            f"{os.path.basename(args.image)} is {W}x{H}; this script expects both "
            f"sides to be multiples of 64 (kodim19 is 768x512). Padding support "
            f"would need the pad/crop dance that test_video.py does."
        )

    x = rgb2ycbcr(rgb) - 0.5          # model works on centred YCbCr 4:4:4
    rgb_ref = (rgb[0].numpy() * 255.0)

    qps = [int(round(q)) for q in np.linspace(0, DMCI.qp_num() - 1, args.rate_num)]
    print(f"\n{os.path.basename(args.image)}  {W}x{H}  |  qp points: {qps}\n")
    print(f"{'qp':>4} {'est. bpp':>10} {'PSNR-RGB':>10} {'kB @ this bpp':>15}")
    print("-" * 44)

    results = []
    for qp in qps:
        with torch.no_grad(), eval_rate():
            out = net.forward_one_frame(x, torch.tensor([qp]))

        rgb_rec = ycbcr2rgb(out["x_hat"] + 0.5)
        rgb_rec = torch.clamp(rgb_rec * 255.0, 0, 255)[0].numpy()

        bpp = out["bpp"].item()
        psnr = calc_psnr(rgb_ref, rgb_rec)
        kb = bpp * H * W / 8 / 1024
        print(f"{qp:>4} {bpp:>10.4f} {psnr:>9.2f}dB {kb:>14.1f}")
        results.append({"qp": qp, "bpp": bpp, "psnr_rgb": psnr, "kbytes": kb})

        if args.save_recon:
            img = Image.fromarray(rgb_rec.transpose(1, 2, 0).round().astype(np.uint8))
            img.save(os.path.join(args.out_dir, f"recon_qp{qp:02d}.png"))

    with open(os.path.join(args.out_dir, "rd.json"), "w") as f:
        json.dump({"image": args.image, "width": W, "height": H,
                   "points": results}, f, indent=2)

    print(f"\nwrote {len(results)} reconstructions + rd.json to {args.out_dir}")
    print("reminder: bpp is the entropy model's ESTIMATE -- no bitstream is written "
          "on CPU (see README.md).")


if __name__ == "__main__":
    main()

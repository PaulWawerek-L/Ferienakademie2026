"""Write and read back a REAL DCVC bitstream, on CPU.

DCVC-UF cannot do this without a GPU: its compress() goes through the CUDA-only
CUTLASS proxy and raises NotImplementedError otherwise. DCVC-RT (CVPR 2025, the
direct predecessor) keeps its entropy coding in Python over the CPU rANS coder,
so with the small scheduling shim in cpu_cuda_shim.py the whole encode/decode
path runs here.

What this demonstrates that a forward pass cannot: the gap between the entropy
model's ESTIMATED rate and the bytes an arithmetic coder actually emits.

  python real_bitstream_demo.py                  # kodim19, several qps
  python real_bitstream_demo.py --qps 16 32 48
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_cuda_shim import cpu_cuda_shim

from src.models.image_model import DMCI

OUT = "/work/outputs/experiments/08-real-bitstream"
SRC = "/work/Dataset/kodim19.png"
REF = "/work/outputs/experiments/kodim19_420.yuv"
CKPT = "/work/weights/dcvc-rt/cvpr2025_image.pth.tar"


def ensure_ref(w, h):
    if not os.path.exists(REF):
        os.makedirs(os.path.dirname(REF), exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", SRC, "-pix_fmt", "yuv420p", "-f", "rawvideo", REF],
                       check=True)


def read_yuv420_444(path, w, h):
    y_size, uv_size = w * h, (w // 2) * (h // 2)
    import scipy.ndimage
    with open(path, "rb") as f:
        a = np.frombuffer(f.read(y_size + 2 * uv_size), dtype=np.uint8)
    y = a[:y_size].reshape(1, h, w).astype(np.float32)
    uv = np.stack([a[y_size:y_size + uv_size].reshape(h // 2, w // 2),
                   a[y_size + uv_size:].reshape(h // 2, w // 2)]).astype(np.float32)
    uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=0)
    return np.concatenate((y, uv), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qps", type=int, nargs="+", default=[8, 24, 40, 56])
    ap.add_argument("--zero-thres", type=float, default=0.12,
                    help="upstream's force_zero_thres: latents below it are skipped")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)

    from PIL import Image
    W, H = Image.open(SRC).size
    # The latent is at /16 and then pixel-unshuffled, so the picture has to be a
    # multiple of 64. kodim19 is 512x768, which already is; padding a picture
    # that is not would need the crop-back dance test_video.py does.
    if W % 64 or H % 64:
        raise SystemExit(f"{W}x{H} is not a multiple of 64; padding is not implemented here")
    ensure_ref(W, H)

    net = DMCI().eval()
    have_ckpt = os.path.exists(CKPT)
    if have_ckpt:
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        net.load_state_dict(sd, strict=False)
        print(f"loaded {os.path.basename(CKPT)}")
    else:
        print(f"NOTE: {CKPT} not found -- running with RANDOM weights.")
        print("      The bitstream is still real and the round-trip is still exact;")
        print("      only the rate/quality numbers are meaningless. See the README.")
    net.update(force_zero_thres=args.zero_thres)

    x = torch.from_numpy(read_yuv420_444(REF, W, H)).unsqueeze(0) / 255.0 - 0.5
    print(f"\nkodim19 {W}x{H}, YUV420 -> 4:4:4, force_zero_thres={args.zero_thres}\n")
    print(f"{'qp':>4} {'bytes':>9} {'actual bpp':>12} {'round-trip':>12}")
    print("-" * 42)

    rows = []
    for qp in args.qps:
        with torch.no_grad(), cpu_cuda_shim():
            enc = net.compress(x, qp)
            bs = enc["bit_stream"]
            dec = net.decompress(bs, {"height": H, "width": W, "ec_part": 0}, qp)

        diff = (dec["x_hat"] - enc["x_hat"]).abs().max().item()
        bpp = len(bs) * 8 / (W * H)
        path = f"{OUT}/kodim19_qp{qp:02d}.bin"
        with open(path, "wb") as f:
            f.write(bs)
        print(f"{qp:>4} {len(bs):>9} {bpp:>12.4f} "
              f"{'exact' if diff == 0 else f'{diff:.2e}':>12}")
        rows.append({"qp": qp, "bytes": len(bs), "bpp": bpp,
                     "roundtrip_max_abs_diff": diff})

    json.dump({"image": "kodim19", "width": W, "height": H,
               "model": "DCVC-RT image (DMCI)", "random_weights": not have_ckpt,
               "force_zero_thres": args.zero_thres, "points": rows},
              open(f"{OUT}/bitstreams.json", "w"), indent=2)
    print(f"\nbitstreams written to {OUT}/")
    print("These are real files: the decoder above reconstructed from the bytes alone.")


if __name__ == "__main__":
    main()

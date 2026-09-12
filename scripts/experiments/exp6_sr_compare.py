"""Task 6 -- Real-ESRGAN vs classical resampling, x4.

Protocol: downsample kodim19 by 4 (bicubic), then bring it back to the original
size with each method and score against the original.

Read the PSNR column with care. Real-ESRGAN is a GAN trained for perceptual
quality on *real-world* degradations; this benchmark feeds it a clean bicubic
downsample, which is not its training distribution. A GAN that invents plausible
texture is penalised by PSNR even when the result looks better, so it is normal
and expected for bicubic to win on PSNR/SSIM while losing visibly on detail.
That gap between "scores well" and "looks good" is the point of the experiment --
which is why crops are written out too.

  python exp6_sr_compare.py [--scale 4]
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

OUT = "/work/outputs/experiments/06-sr-compare"
SRC = "/work/Dataset/kodim19.png"
MODEL = "/work/weights/realesrgan/RealESRGAN_x4plus.pth"

CLASSICAL = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def scores(ref, test):
    ref_a = np.asarray(ref, dtype=np.float64)
    test_a = np.asarray(test, dtype=np.float64)
    mse = np.mean((ref_a - test_a) ** 2)
    psnr = 999.9 if mse <= 1e-10 else 10 * np.log10(255.0 ** 2 / mse)
    ssim = structural_similarity(
        np.asarray(ref.convert("L")), np.asarray(test.convert("L")), data_range=255)
    return psnr, ssim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--crop", default="200,150,328,278",
                    help="x0,y0,x1,y1 region saved for visual comparison")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    hr = Image.open(SRC).convert("RGB")
    W, H = hr.size
    # Crop to a multiple of the scale so the round trip is exact.
    W, H = W - W % args.scale, H - H % args.scale
    hr = hr.crop((0, 0, W, H))
    lr = hr.resize((W // args.scale, H // args.scale), Image.Resampling.BICUBIC)
    hr.save(f"{OUT}/00_original.png")
    lr.save(f"{OUT}/01_lowres_input.png")
    print(f"kodim19 {W}x{H}  ->  LR {lr.size[0]}x{lr.size[1]}  (bicubic /{args.scale})\n")

    rows = []
    print(f"{'method':>14} {'PSNR':>9} {'SSIM':>8} {'time':>9}")
    print("-" * 44)

    for name, filt in CLASSICAL.items():
        t0 = time.perf_counter()
        sr = lr.resize((W, H), filt)
        dt = time.perf_counter() - t0
        psnr, ssim = scores(hr, sr)
        sr.save(f"{OUT}/sr_{name}.png")
        print(f"{name:>14} {psnr:>8.2f}dB {ssim:>8.4f} {dt:>8.3f}s")
        rows.append({"method": name, "kind": "classical", "psnr": psnr,
                     "ssim": ssim, "seconds": dt})

    # --- Real-ESRGAN ---
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                    num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path=MODEL, model=model,
                             tile=0, tile_pad=10, pre_pad=0, half=False)
    t0 = time.perf_counter()
    # RealESRGANer works in BGR uint8, the OpenCV convention.
    out_bgr, _ = upsampler.enhance(np.asarray(lr)[:, :, ::-1], outscale=args.scale)
    dt = time.perf_counter() - t0
    sr = Image.fromarray(out_bgr[:, :, ::-1])
    if sr.size != (W, H):
        sr = sr.resize((W, H), Image.Resampling.BICUBIC)
    psnr, ssim = scores(hr, sr)
    sr.save(f"{OUT}/sr_realesrgan.png")
    print(f"{'Real-ESRGAN':>14} {psnr:>8.2f}dB {ssim:>8.4f} {dt:>8.3f}s")
    rows.append({"method": "Real-ESRGAN", "kind": "learned", "psnr": psnr,
                 "ssim": ssim, "seconds": dt})

    # Side-by-side crops: the fidelity metrics and the eye disagree here, so the
    # comparison is not complete without pixels to look at.
    x0, y0, x1, y1 = (int(v) for v in args.crop.split(","))
    for f in ["00_original"] + [f"sr_{m}" for m in list(CLASSICAL) + ["realesrgan"]]:
        Image.open(f"{OUT}/{f}.png").crop((x0, y0, x1, y1)).save(f"{OUT}/crop_{f}.png")

    best_psnr = max(rows, key=lambda r: r["psnr"])
    re_row = rows[-1]
    print("-" * 44)
    print(f"best PSNR: {best_psnr['method']} ({best_psnr['psnr']:.2f} dB)")
    if best_psnr["method"] != "Real-ESRGAN":
        print(f"Real-ESRGAN is {best_psnr['psnr'] - re_row['psnr']:.2f} dB BELOW it "
              f"and {re_row['seconds'] / max(best_psnr['seconds'], 1e-9):.0f}x slower "
              f"-- see the note at the top of this file, and look at crop_*.png "
              f"before drawing a conclusion.")

    json.dump({"image": "kodim19", "width": W, "height": H, "scale": args.scale,
               "protocol": "bicubic downsample, then upsample back",
               "results": rows}, open(f"{OUT}/results.json", "w"), indent=2)
    print(f"\nwrote {OUT}/results.json")


if __name__ == "__main__":
    main()

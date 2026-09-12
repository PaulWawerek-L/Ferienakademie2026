"""Why Real-ESRGAN scores worse on PSNR/SSIM and still looks better.

Four measurements on the task-6 outputs, each isolating one reason:

  A  LPIPS      - a perceptual metric. Does anything rank the GAN first?
  B  phase test - shift the GROUND TRUTH by one pixel and score it. A perfect
                  image, misplaced by 1 px, against a blurry reconstruction.
  C  blur sweep - blur the Real-ESRGAN output and watch PSNR. If the metric goes
                  UP when detail is destroyed, it is not measuring detail.
  D  sharpness  - gradient energy per method, against its PSNR. If PSNR rank is
                  the inverse of sharpness rank, PSNR is ranking smoothness.

  python exp6b_perception_distortion.py
"""
import json
import os

import numpy as np
import torch
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity

SRC = "/work/outputs/experiments/06-sr-compare"
OUT = "/work/outputs/experiments/06b-perception-distortion"
METHODS = ["nearest", "bilinear", "bicubic", "lanczos", "realesrgan"]


def psnr(a, b):
    mse = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return 999.9 if mse <= 1e-10 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    return structural_similarity(np.asarray(a.convert("L")),
                                 np.asarray(b.convert("L")), data_range=255)


def sharpness(img):
    """Gradient energy -- a plain, reference-free measure of how much detail is
    present. It says nothing about whether the detail is CORRECT."""
    g = np.asarray(img.convert("L"), dtype=np.float64)
    gx = np.diff(g, axis=1)
    gy = np.diff(g, axis=0)
    return float(np.mean(gx ** 2) + np.mean(gy ** 2))


def main():
    os.makedirs(OUT, exist_ok=True)
    hr = Image.open(f"{SRC}/00_original.png").convert("RGB")
    outs = {m: Image.open(f"{SRC}/sr_{m}.png").convert("RGB") for m in METHODS}
    report = {}

    # ---- A. a perceptual metric -------------------------------------------
    import lpips
    net = lpips.LPIPS(net="alex", verbose=False)

    def to_t(img):
        a = np.asarray(img, np.float32) / 127.5 - 1.0
        return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)

    hr_t = to_t(hr)
    print("A. LPIPS -- perceptual distance, LOWER is better\n")
    print(f"   {'method':>12} {'PSNR':>9} {'SSIM':>8} {'LPIPS':>9}")
    print("   " + "-" * 40)
    rows = []
    for m in METHODS:
        with torch.no_grad():
            d = net(hr_t, to_t(outs[m])).item()
        p, s = psnr(hr, outs[m]), ssim(hr, outs[m])
        rows.append({"method": m, "psnr": p, "ssim": s, "lpips": d,
                     "sharpness": sharpness(outs[m])})
        print(f"   {m:>12} {p:>8.2f}dB {s:>8.4f} {d:>9.4f}")
    best_fid = max(rows, key=lambda r: r["psnr"])["method"]
    best_per = min(rows, key=lambda r: r["lpips"])["method"]
    print(f"\n   best PSNR: {best_fid}     best LPIPS: {best_per}")
    print("   The two metrics disagree because they measure different things.\n")
    report["a_metrics"] = rows

    # ---- B. PSNR is phase-sensitive ---------------------------------------
    print("B. Phase test -- the ORIGINAL image, shifted, scored against itself\n")
    print(f"   {'shift':>12} {'PSNR':>9} {'SSIM':>8}")
    print("   " + "-" * 32)
    shifts = []
    for px in (1, 2, 4):
        moved = Image.fromarray(np.roll(np.asarray(hr), px, axis=1))
        p, s = psnr(hr, moved), ssim(hr, moved)
        shifts.append({"shift_px": px, "psnr": p, "ssim": s})
        print(f"   {px:>10} px {p:>8.2f}dB {s:>8.4f}")
    re_psnr = next(r["psnr"] for r in rows if r["method"] == "realesrgan")
    print(f"\n   A pixel-perfect photograph moved 1 px scores {shifts[0]['psnr']:.2f} dB.")
    print(f"   Real-ESRGAN scores {re_psnr:.2f} dB. PSNR is not measuring 'does this")
    print("   look right' -- it is measuring 'is each pixel in the same place'.\n")
    report["b_shift"] = shifts

    # ---- C. blurring the GAN output raises its score -----------------------
    print("C. Blur sweep -- Gaussian blur applied to the Real-ESRGAN output\n")
    print(f"   {'sigma':>12} {'PSNR':>9} {'SSIM':>8} {'sharpness':>11}")
    print("   " + "-" * 44)
    sweep = []
    for sigma in (0.0, 0.3, 0.5, 0.8, 1.2, 1.6):
        img = outs["realesrgan"] if sigma == 0 else \
            outs["realesrgan"].filter(ImageFilter.GaussianBlur(sigma))
        p, s = psnr(hr, img), ssim(hr, img)
        sweep.append({"sigma": sigma, "psnr": p, "ssim": s, "sharpness": sharpness(img)})
        print(f"   {sigma:>12.1f} {p:>8.2f}dB {s:>8.4f} {sharpness(img):>11.1f}")
        if sigma > 0:
            img.save(f"{OUT}/realesrgan_blur{sigma}.png")
    best = max(sweep, key=lambda r: r["psnr"])
    if best["sigma"] > 0:
        print(f"\n   PSNR PEAKS at sigma={best['sigma']} ({best['psnr']:.2f} dB), "
              f"not at sigma=0 ({sweep[0]['psnr']:.2f} dB).")
        print("   Destroying detail IMPROVES the score. That is the whole argument.\n")
    else:
        print("\n   PSNR is highest unblurred here; the gradient is still shallow --")
        print("   compare how little it costs to blur versus how much detail is lost.\n")
    report["c_blur"] = sweep

    # ---- D. PSNR rank vs sharpness rank ------------------------------------
    print("D. Is PSNR just ranking smoothness?\n")
    by_psnr = [r["method"] for r in sorted(rows, key=lambda r: -r["psnr"])]
    by_sharp = [r["method"] for r in sorted(rows, key=lambda r: r["sharpness"])]
    print(f"   by PSNR      (best first): {', '.join(by_psnr)}")
    print(f"   by SMOOTHNESS(most first): {', '.join(by_sharp)}")
    corr = np.corrcoef([r["psnr"] for r in rows], [r["sharpness"] for r in rows])[0, 1]
    print(f"\n   correlation(PSNR, sharpness) = {corr:+.3f}")
    print("   Negative means: across these five methods, the smoother the output,")
    print("   the better its PSNR.\n")
    report["d_sharpness"] = {"by_psnr": by_psnr, "by_smoothness": by_sharp,
                             "corr_psnr_sharpness": float(corr)}

    json.dump(report, open(f"{OUT}/report.json", "w"), indent=2)
    print(f"wrote {OUT}/report.json")


if __name__ == "__main__":
    main()

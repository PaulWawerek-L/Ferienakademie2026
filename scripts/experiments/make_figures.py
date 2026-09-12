"""Contact sheets for the two image-domain experiments (tasks 5 and 6).

  python make_figures.py styles   # AdaIN: style grid + alpha sweep
  python make_figures.py sr       # Real-ESRGAN vs classical resampling
  python make_figures.py all
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

EXP = "/work/outputs/experiments"
FIG = "/work/results/figures"
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def panel(ax, path, title=None, subtitle=None):
    ax.imshow(Image.open(path))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#e3e2de"); sp.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=9, color=INK, pad=4)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=8, color=INK_2, labelpad=4)


def fig_styles():
    base = f"{EXP}/05-adain-styles"
    styles = sorted(os.path.splitext(f)[0] for f in os.listdir(f"{base}/styles_used"))
    rows = len(styles)
    fig, axes = plt.subplots(rows, 3, figsize=(8.4, 2.5 * rows), dpi=130)
    fig.patch.set_facecolor(SURFACE)

    for r, st in enumerate(styles):
        sp = f"{base}/styles_used/{st}.jpg"
        if not os.path.exists(sp):
            sp = f"{base}/styles_used/{st}.png"
        panel(axes[r][0], sp, title="style" if r == 0 else None,
              subtitle=st.replace("_", " "))
        panel(axes[r][1], f"{base}/grid/kodim19_stylized_{st}.jpg",
              title="kodim19" if r == 0 else None)
        panel(axes[r][2], f"{base}/grid/racehorses_f0_stylized_{st}.jpg",
              title="RaceHorses frame 0" if r == 0 else None)

    fig.suptitle("AdaIN: one decoder, eight styles, no retraining",
                 fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(f"{FIG}/adain_style_grid.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {FIG}/adain_style_grid.png")

    # alpha sweep
    alphas = ["0.0", "0.25", "0.5", "0.75", "1.0"]
    have = [a for a in alphas if os.path.exists(f"{base}/alpha/alpha_{a}.jpg")]
    if have:
        fig, axes = plt.subplots(1, len(have), figsize=(2.1 * len(have), 3.7), dpi=140)
        fig.patch.set_facecolor(SURFACE)
        for ax, a in zip(axes, have):
            panel(ax, f"{base}/alpha/alpha_{a}.jpg", subtitle=f"alpha = {a}")
        fig.suptitle("AdaIN alpha: content statistics -> style statistics",
                     fontsize=12, fontweight="bold", color=INK, x=0.02, ha="left")
        fig.text(0.02, 0.02, "alpha interpolates the normalised feature between the "
                 "content's own statistics and the style's. It is not a blend of two images.",
                 fontsize=8, color=INK_2)
        fig.tight_layout(rect=(0, 0.10, 1, 0.93))
        fig.savefig(f"{FIG}/adain_alpha_sweep.png", facecolor=SURFACE)
        plt.close(fig)
        print(f"wrote {FIG}/adain_alpha_sweep.png")


def fig_sr():
    base = f"{EXP}/06-sr-compare"
    res = {r["method"]: r for r in json.load(open(f"{base}/results.json"))["results"]}
    order = ["00_original", "sr_nearest", "sr_bicubic", "sr_lanczos", "sr_realesrgan"]
    names = ["original", "nearest", "bicubic", "lanczos", "Real-ESRGAN"]

    fig, axes = plt.subplots(1, len(order), figsize=(2.3 * len(order), 3.9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, key, name in zip(axes, order, names):
        r = res.get(name)
        sub = "reference" if r is None else f"{r['psnr']:.2f} dB / SSIM {r['ssim']:.3f}"
        panel(ax, f"{base}/crop_{key}.png", title=name, subtitle=sub)
        if name == "Real-ESRGAN":
            ax.title.set_color(BLUE)
            ax.title.set_fontweight("bold")

    fig.suptitle("x4 super-resolution: the metric and the eye disagree",
                 fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.02,
             "Lanczos wins PSNR by 0.65 dB. Look at the railing. Real-ESRGAN is a GAN "
             "trained for perceptual quality on real-world\ndegradations, and this "
             "benchmark feeds it a clean bicubic downsample -- PSNR penalises invented "
             "texture even when it is right.",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.115, 1, 0.93))
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(f"{FIG}/sr_comparison.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {FIG}/sr_comparison.png")

    # Metrics as a chart: one measure, one axis, emphasis on the learned method.
    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    rows = json.load(open(f"{base}/results.json"))["results"]
    labels = [r["method"] for r in rows]
    vals = [r["psnr"] for r in rows]
    colors = [BLUE if r["kind"] == "learned" else "#c9c8c3" for r in rows]
    bars = ax.barh(labels, vals, color=colors, height=0.55)
    ax.set_xlim(22.5, 24.0)
    ax.grid(True, axis="x", color="#e3e2de", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#e3e2de"); ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_xlabel("PSNR (dB)  -- higher is better", color=INK_2, fontsize=10, labelpad=8)
    ax.set_title("Fidelity scores, x4 on kodim19", fontsize=12, fontweight="bold",
                 color=INK, loc="left", pad=14)
    for b, v in zip(bars, vals):
        ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=9, color=INK)
    fig.text(0.01, 0.015, "Emphasis, not eight hues: the learned method is the subject, "
             "the classical filters are context.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{FIG}/sr_psnr_bars.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {FIG}/sr_psnr_bars.png")


def fig_perception_distortion():
    """Three panels, three separate reasons PSNR and the eye disagree."""
    base = f"{EXP}/06b-perception-distortion"
    if not os.path.exists(f"{base}/report.json"):
        print("  [skip] run exp6b_perception_distortion.py first")
        return
    r = json.load(open(f"{base}/report.json"))
    GREY = "#c9c8c3"

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color="#e3e2de", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#e3e2de"); ax.spines[sp].set_linewidth(0.8)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)

    # 1. the two metrics disagree. Emphasis: the learned method is the subject.
    ax = axes[0]
    # bicubic and lanczos sit almost on top of each other, so the labels are
    # placed per method rather than with one shared offset.
    LABEL_OFFSET = {"nearest": (9, -3), "bilinear": (-12, 9), "bicubic": (-14, -14),
                    "lanczos": (9, 2), "realesrgan": (10, -3)}
    for m in r["a_metrics"]:
        learned = m["method"] == "realesrgan"
        ax.scatter(m["psnr"], m["lpips"], s=110 if learned else 70,
                   color=BLUE if learned else GREY, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(m["method"], (m["psnr"], m["lpips"]),
                    textcoords="offset points",
                    xytext=LABEL_OFFSET.get(m["method"], (8, -3)), fontsize=8,
                    color=INK if learned else INK_2,
                    fontweight="bold" if learned else "normal")
    ax.set_xlabel("PSNR (dB)  -  higher is better", color=INK_2, fontsize=9.5, labelpad=7)
    ax.set_ylabel("LPIPS  -  LOWER is better", color=INK_2, fontsize=9.5, labelpad=7)
    ax.set_title("Fidelity and perception rank it oppositely",
                 fontsize=10.5, fontweight="bold", color=INK, loc="left", pad=10)
    ax.set_xlim(22.88, 24.2)
    ax.set_ylim(0.20, 0.60)

    # 2. the metric rewards destroying detail
    ax = axes[1]
    sw = r["c_blur"]
    xs = [p["sigma"] for p in sw]; ys = [p["psnr"] for p in sw]
    ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
    peak = max(sw, key=lambda p: p["psnr"])
    ax.scatter([peak["sigma"]], [peak["psnr"]], s=150, facecolor="none",
               edgecolor=ORANGE, linewidth=2, zorder=4)
    ax.annotate(f"peak: sigma {peak['sigma']}\n{peak['psnr']:.2f} dB",
                (peak["sigma"], peak["psnr"]), textcoords="offset points",
                xytext=(10, -26), fontsize=8.5, color=ORANGE, fontweight="bold")
    ax.annotate(f"unblurred\n{sw[0]['psnr']:.2f} dB", (xs[0], ys[0]),
                textcoords="offset points", xytext=(6, 4), fontsize=8.5, color=INK_2)
    ax.set_xlabel("Gaussian blur applied to the output (sigma, px)",
                  color=INK_2, fontsize=9.5, labelpad=7)
    ax.set_ylabel("PSNR (dB)", color=INK_2, fontsize=9.5, labelpad=7)
    ax.set_title("Blurring the result IMPROVES its PSNR",
                 fontsize=10.5, fontweight="bold", color=INK, loc="left", pad=10)

    # 3. the metric is phase-sensitive
    ax = axes[2]
    re = next(m for m in r["a_metrics"] if m["method"] == "realesrgan")
    labels = ["Real-ESRGAN"] + [f"original\nshifted {s['shift_px']} px" for s in r["b_shift"]]
    vals = [re["psnr"]] + [s["psnr"] for s in r["b_shift"]]
    colors = [BLUE] + [GREY] * len(r["b_shift"])
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}",
                ha="center", fontsize=8.5, color=INK)
    ax.set_ylim(17, 25)
    ax.set_ylabel("PSNR (dB)", color=INK_2, fontsize=9.5, labelpad=7)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("A perfect photo, moved 1 px, scores worse",
                 fontsize=10.5, fontweight="bold", color=INK, loc="left", pad=10)

    fig.suptitle("Why Real-ESRGAN loses on PSNR and wins on sight",
                 fontsize=13.5, fontweight="bold", color=INK, x=0.006, ha="left")
    fig.text(0.006, 0.015,
             "Left: LPIPS, a learned perceptual metric, ranks Real-ESRGAN FIRST "
             f"({re['lpips']:.3f}) while PSNR ranks it last.   "
             "Middle: blurring its output raises PSNR by "
             f"{peak['psnr'] - sw[0]['psnr']:.2f} dB while removing "
             f"{100 * (1 - peak['sharpness'] / sw[0]['sharpness']):.0f}% of the "
             "detail.\nRight: shifting the untouched original by one pixel scores "
             f"{r['b_shift'][0]['psnr']:.2f} dB - below Real-ESRGAN. PSNR asks "
             "'is each pixel where it was', not 'does this look like the scene'. "
             f"Across the five methods, corr(PSNR, sharpness) = "
             f"{r['d_sharpness']['corr_psnr_sharpness']:+.2f}.",
             fontsize=8.2, color=INK_2)
    fig.tight_layout(rect=(0, 0.105, 1, 0.92))
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(f"{FIG}/sr_perception_distortion.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {FIG}/sr_perception_distortion.png")


def fig_timing():
    """Inference time per project, plus throughput so the bars are comparable.

    Two panels because the projects do not share an input size: seconds alone
    answers "how long do I wait", megapixels/second answers "which is actually
    faster". A single bar chart of seconds would quietly compare a 512x768 image
    against a 128x128 crop.

    Log scale on both: VTM is a reference encoder doing full RDO and lands two
    orders of magnitude from the neural codecs. On a linear axis every other bar
    collapses to a sliver.
    """
    base = f"{EXP}/07-timing"
    # Short names: the full model name plus the input size does not fit under a
    # bar without colliding with its neighbour.
    SHORT = {
        "image-compression": ("01 Image\nDCVC-UF-Intra", "512x768"),
        "video-compression": ("02 Video\nDCVC-UF HTS", "416x240 / frame"),
        "hybrid-vtm": ("03 Hybrid\nVTM intra", "512x768, QP 32"),
        "style-transfer": ("04 Style\nAdaIN", "512x768"),
        "super-resolution": ("05 Super-res\nReal-ESRGAN", "128x128 -> 512x512"),
    }
    order = list(SHORT)
    rows = []
    for name in order:
        f = f"{base}/{name}.json"
        if not os.path.exists(f):
            print(f"  [skip] {name} not measured yet")
            continue
        d = json.load(open(f))
        t = d["timing"]["median_s"]
        short, detail = SHORT[name]
        rows.append({"name": name, "short": short, "detail": detail, "t": t,
                     # Throughput on OUTPUT pixels: for super-resolution the work
                     # is proportional to what comes out, not what goes in.
                     "mpx_s": d["output_px"] / 1e6 / t,
                     "learned": name != "hybrid-vtm"})
    if not rows:
        print("  [skip] no timing results -- run scripts/experiments/run_timing.sh")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    labels = [r["short"] for r in rows]
    # Emphasis, not a rainbow: the neural models are the subject, the traditional
    # reference encoder is the context they are measured against.
    colors = [BLUE if r["learned"] else ORANGE for r in rows]

    def style(ax, ylabel, title):
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color="#e3e2de", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#e3e2de"); ax.spines[sp].set_linewidth(0.8)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel, color=INK_2, fontsize=10, labelpad=8)
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK,
                     loc="left", pad=12)
        ax.set_xticks(range(len(labels)))
        # Input size is the third line of the tick label, not floating text in
        # the plot: inside the axes it collides with whichever bar is shortest.
        ax.set_xticklabels([f"{r['short']}\n{r['detail']}" for r in rows],
                           fontsize=8.3, color=INK_2, linespacing=1.6)

    ax = axes[0]
    vals = [r["t"] for r in rows]
    bars = ax.bar(range(len(rows)), vals, color=colors, width=0.6)
    style(ax, "inference time (s, log scale)", "Time per run")
    for i, (b, r) in enumerate(zip(bars, rows)):
        ax.text(i, r["t"] * 1.25,
                f"{r['t']:.2f} s" if r["t"] >= 1 else f"{r['t'] * 1000:.0f} ms",
                ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_ylim(min(vals) * 0.28, max(vals) * 4.0)

    ax = axes[1]
    vals = [r["mpx_s"] for r in rows]
    bars = ax.bar(range(len(rows)), vals, color=colors, width=0.6)
    style(ax, "throughput (output megapixels / s, log scale)",
          "Throughput - what the seconds hide")
    for i, r in enumerate(rows):
        ax.text(i, r["mpx_s"] * 1.25, f"{r['mpx_s']:.3f}", ha="center",
                fontsize=10, color=INK, fontweight="bold")
    ax.set_ylim(min(vals) * 0.28, max(vals) * 4.0)

    vtm = next((r for r in rows if not r["learned"]), None)
    vid = next((r for r in rows if r["name"] == "video-compression"), None)
    note = ("CPU only, no GPU. Warm-up runs discarded, median of the timed runs "
            "reported; model construction, checkpoint loading and file I/O sit "
            "outside the timed region.\nOrange is the traditional reference "
            "encoder, blue the neural models.")
    if vtm and vid and vtm["mpx_s"] > 0:
        note += (f" Per pixel VTM is {vid['mpx_s'] / vtm['mpx_s']:.0f}x slower "
                 f"than DCVC-UF video - it is built for compression research, "
                 f"not speed.")
    fig.suptitle("Inference time per project", fontsize=13.5, fontweight="bold",
                 color=INK, x=0.007, ha="left", y=0.985)
    fig.text(0.007, 0.014, note, fontsize=8.3, color=INK_2)
    fig.tight_layout(rect=(0, 0.115, 1, 0.945))
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(f"{FIG}/inference_time.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {FIG}/inference_time.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["styles", "sr", "pd", "timing", "all"],
                    nargs="?", default="all")
    a = ap.parse_args()
    if a.which in ("styles", "all"):
        fig_styles()
    if a.which in ("sr", "all"):
        fig_sr()
    if a.which in ("pd", "all"):
        fig_perception_distortion()
    if a.which in ("timing", "all"):
        fig_timing()

"""Draw the rate-distortion figures from the rd.json files the experiments write.

  python plot_rd.py image     # task 1 vs task 2
  python plot_rd.py video     # task 3 vs task 4
  python plot_rd.py all

Colour follows the validated categorical palette (slots 1-3). Each series also
gets its own marker shape, so identity never rests on hue alone -- which matters
here because one of the three slots sits below 3:1 against the surface.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "/work/outputs/experiments"
# Figures go to results/, which is tracked by git. outputs/ holds the bulk
# intermediates (reconstructed .yuv, bitstreams) and is ignored -- a figure left
# there is invisible to anyone who clones the repo.
FIG = "/work/results/figures"
DATA = "/work/results/data"

# Validated categorical slots 1-3 (light mode), all-pairs clean.
SERIES = [
    {"color": "#2a78d6", "marker": "o"},   # blue
    {"color": "#eb6834", "marker": "s"},   # orange
    {"color": "#1baf7a", "marker": "^"},   # aqua
]
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e3e2de"


def style_axes(ax, title, subtitle, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.patch.set_facecolor(SURFACE)
    # Recessive chrome: hairline grid, two spines, no dashes.
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10, labelpad=8)
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=18)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=INK_2,
            fontsize=9, va="bottom", ha="left")


def plot_curves(curves, title, subtitle, footnote, out_path, annotate_qp=True,
                logx=False):
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=160)
    # A log rate axis when the sampled rates span orders of magnitude. Here VTM
    # QP 0 sits at 4.36 bpp while the neural curves top out near 0.3; on a linear
    # axis they would be crushed into the leftmost few percent of the plot.
    xlabel = "rate  (bits per pixel, log scale)" if logx else "rate  (bits per pixel)"
    style_axes(ax, title, subtitle, xlabel, "YUV-PSNR  (dB)")
    if logx:
        ax.set_xscale("log")

    for i, c in enumerate(curves):
        s = SERIES[i % len(SERIES)]
        pts = sorted(c["points"], key=lambda p: p["bpp"])
        xs = [p["bpp"] for p in pts]
        ys = [p["psnr_yuv"] for p in pts]
        ax.plot(xs, ys, color=s["color"], linewidth=2, marker=s["marker"],
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=c["label"], zorder=3 + i)
        if annotate_qp:
            # Label the ends only -- a number on every point is noise.
            for p in (pts[0], pts[-1]):
                ax.annotate(f"qp {p['qp']}", (p["bpp"], p["psnr_yuv"]),
                            textcoords="offset points", xytext=(6, -11),
                            fontsize=8, color=INK_2)

    # A legend for one series is noise -- the title already names it. Two or more
    # need it, since colour alone must never be the only identity cue.
    if len(curves) > 1:
        leg = ax.legend(frameon=False, fontsize=10, loc="lower right",
                        labelcolor=INK, handlelength=2.2)
        for t in leg.get_texts():
            t.set_color(INK)

    fig.text(0.008, 0.012, footnote, fontsize=8, color=INK_2, va="bottom")
    n_lines = footnote.count("\n") + 1
    fig.tight_layout(rect=(0, 0.035 + 0.026 * n_lines, 1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


def load(path):
    if not os.path.exists(path):
        print(f"  [skip] missing {path}")
        return None
    return json.load(open(path))


def merge_vtm_inter():
    """VTM inter is ONE codec; the two QP sets just sample its curve.

    Plotting them as separate series would imply two codecs. Merging gives one
    curve over the union of the sampled rates -- which is also the only way the
    requested QP set (0/15/30/45/63) shares an axis with anything else.
    """
    pts, seen, frames = [], set(), None
    for tag in ("std", "requested"):
        d = load(f"{EXP}/04-vtm-inter/{tag}/rd.json")
        if not d:
            continue
        frames = d.get("frames", frames)
        for p in d["points"]:
            if p["qp"] not in seen:
                seen.add(p["qp"])
                pts.append(p)
    if not pts:
        return None
    return {"label": "VTM inter (RA)", "points": pts, "frames": frames}


def publish(src, name):
    """Copy an rd.json next to the figures so the numbers ship with the plots."""
    d = load(src)
    if d:
        os.makedirs(DATA, exist_ok=True)
        json.dump(d, open(f"{DATA}/{name}", "w"), indent=2)
    return d


def do_image():
    # One figure per task, as asked -- each codec's own curve stands alone --
    # plus the overlay, which is the only one that answers "which is better".
    dcvc = publish(f"{EXP}/01-dcvc-image/rd.json", "task1_dcvc_image.json")
    vtm = publish(f"{EXP}/02-vtm-intra/rd.json", "task2_vtm_intra.json")

    if dcvc:
        plot_curves([dcvc], "Task 1 - DCVC-UF-Intra on kodim19",
                    "512x768 YUV420, qp 0/15/30/45/63",
                    "Rate is the entropy model's estimate of the coded bits "
                    "(quantised symbols, not the training-time noise proxy).\n"
                    "No real bitstream is written on CPU - the encoder needs the "
                    "CUDA-only CUTLASS extension.",
                    f"{FIG}/task1_rd_dcvc_image.png")
    if vtm:
        plot_curves([vtm], "Task 2 - VTM all-intra on kodim19",
                    "512x768 YUV420, QP 22/27/32/37/42",
                    "Rate is real bytes on disk. PSNR is (6Y+U+V)/8 measured from "
                    "the reconstruction by the shared tool,\nnot VTM's own "
                    "YUV-PSNR, which uses a different weighting.",
                    f"{FIG}/task2_rd_vtm_intra.png")

    curves = [c for c in (dcvc, vtm) if c]
    if not curves:
        return
    plot_curves(
        curves,
        "Image coding on kodim19",
        "512x768, YUV420 - both codecs on the identical source file",
        "DCVC-UF rate is the entropy model's ESTIMATE (no bitstream without CUDA); "
        "VTM rate is real bytes on disk.\nPSNR is (6Y+U+V)/8 computed by one shared "
        "tool, not each codec's own log.",
        f"{FIG}/compare_image_kodim19.png")


def do_video():
    # Task 3's own figure uses the full 64 frames that were asked for.
    hts64 = publish(f"{EXP}/03-dcvc-video/rd_hts.json", "task3_dcvc_video_hts_64f.json")
    ld64 = publish(f"{EXP}/03-dcvc-video/rd_ld.json", "task3_dcvc_video_ld_64f.json")
    if hts64 or ld64:
        plot_curves([c for c in (hts64, ld64) if c],
                    "Task 3 - DCVC-UF video on RaceHorses, first 64 frames",
                    "416x240 YUV420 30 fps, qp 0/15/30/45/63 - one intra frame, "
                    "the rest inter",
                    "HTS codes 8 frames into one latent; LD codes one at a time. "
                    "Rates are entropy-model estimates\nof the coded bits; no "
                    "bitstream is written on CPU.",
                    f"{FIG}/task3_rd_dcvc_video.png")

    vtm = merge_vtm_inter()
    if vtm:
        os.makedirs(DATA, exist_ok=True)
        json.dump(vtm, open(f"{DATA}/task4_vtm_inter.json", "w"), indent=2)
        plot_curves([vtm], "Task 4 - VTM inter (random access) on RaceHorses",
                    f"416x240 YUV420 30 fps, first {vtm['frames']} frames - "
                    "both QP sets merged",
                    "One codec at ten operating points: the requested QP "
                    "0/15/30/45/63 plus the JVET common-test\n22/27/32/37/42. "
                    "Rates are real bytes on disk. Log axis - QP 0 reaches 4.36 bpp.",
                    f"{FIG}/task4_rd_vtm_inter.png", logx=True)

    # The overlay must compare matched content: frames 0-7 and 0-63 of RaceHorses
    # are different sequences, so pick the DCVC run whose frame count matches VTM.
    n = vtm["frames"] if vtm else 64
    suffix = "" if n == 64 else f"_{n}f"
    hts = load(f"{EXP}/03-dcvc-video/rd_hts{suffix}.json")
    ld = load(f"{EXP}/03-dcvc-video/rd_ld{suffix}.json")
    curves = [c for c in (hts, ld, vtm) if c]
    if not curves:
        return
    plot_curves(
        curves,
        f"Video coding on RaceHorses, first {n} frames",
        "416x240 YUV420, 30 fps - intra frame plus inter frames",
        "Read with care: 8 frames means 1 intra + 7 inter, so the intra frame dominates the "
        "rate and DCVC-UF's\ntemporal advantage barely shows. Its own numbers say so - the same "
        "model at 64 frames costs 0.0041 bpp at\nqp 0 against 0.0113 here, purely from "
        "amortising the intra frame. The published comparison uses long sequences.\n"
        "DCVC-UF rates are entropy-model ESTIMATES; VTM rates are real bitstreams. The VTM curve "
        "merges both QP sets\n(0/15/30/45/63 and 22/27/32/37/42) - one codec, ten operating "
        "points. Log rate axis: VTM QP 0 reaches 4.36 bpp.",
        f"{FIG}/compare_video_racehorses.png", logx=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["image", "video", "all"], default="all", nargs="?")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    if a.which in ("image", "all"):
        do_image()
    if a.which in ("video", "all"):
        do_video()

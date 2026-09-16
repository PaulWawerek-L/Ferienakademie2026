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
    if not vtm or not (hts or ld):
        # An overlay with one side missing is a comparison that says nothing, and
        # it looks finished. Refuse rather than write it.
        print(f"  [skip] compare_video_racehorses.png NOT written: needs VTM inter and "
              f"DCVC video on the same {n} frames (run_all.sh task 3 produces the "
              f"matched run, task 4 the VTM one)")
        return
    curves = [c for c in (hts, ld, vtm) if c]
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


def do_uf_bitstream():
    """Real DCVC-UF bytes against its own estimate and against VTM's real bytes."""
    d = publish("/work/outputs/experiments/09-uf-real-bitstream/rd.json",
                "task8_uf_real_bitstream.json")
    vtm = load(f"{EXP}/02-vtm-intra/rd.json")
    if not d:
        return
    px = d["width"] * d["height"]
    hdr = d["header_bytes"]
    pts = d["points"]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), dpi=160,
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    style_axes(ax, "DCVC-UF-Intra, real bitstream on CPU",
               "kodim19 512x768 YUV420 - bytes on disk, decoded back bit-exactly",
               "rate  (bits per pixel)", "YUV-PSNR  (dB)")
    series = [
        ("DCVC-UF, real bytes", [(r["full"]["bpp"], r["full"]["psnr_yuv"], r["qp"]) for r in pts]),
        ("VTM intra, real bytes", [(p["bpp"], p["psnr_yuv"], p["qp"]) for p in (vtm or {}).get("points", [])]),
        ("DCVC-UF, entropy estimate", [(r["estimated_bpp"], r["full"]["psnr_yuv"], r["qp"]) for r in pts]),
    ]
    for i, (label, xy) in enumerate(series):
        if not xy:
            continue
        xy = sorted(xy)
        s_ = SERIES[i]
        est = "estimate" in label
        # The estimate sits almost exactly on the real-bytes curve -- that overlap
        # is the result. Thin line and small markers on top keep both visible.
        ax.plot([a for a, _, _ in xy], [b for _, b, _ in xy], color=s_["color"],
                linewidth=1.2 if est else 2.2, marker=s_["marker"],
                markersize=4.5 if est else 8, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=label, zorder=5 if est else 3 + i)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK)

    # One measure, one axis: how far the estimate is from the coded payload.
    ax = axes[1]
    style_axes(ax, "Estimate vs coded payload",
               "rANS payload only - the 13-byte container header excluded",
               "qp", "payload / estimate - 1  (%)")
    qps = [r["qp"] for r in pts]
    ovh = [((r["full"]["bytes"] - hdr) * 8 / px / r["estimated_bpp"] - 1) * 100 for r in pts]
    bars = ax.bar(range(len(qps)), ovh, color=SERIES[0]["color"], width=0.55)
    ax.set_xticks(range(len(qps)))
    ax.set_xticklabels([str(q) for q in qps])
    ax.set_ylim(0, max(ovh) * 1.35)
    for b, v in zip(bars, ovh):
        ax.text(b.get_x() + b.get_width() / 2, v + max(ovh) * 0.03, f"+{v:.2f}%",
                ha="center", fontsize=9, color=INK)

    fig.text(0.008, 0.012,
             "The estimate and the real bytes overlap: arithmetic coding adds 0.2-0.7% to what the "
             "entropy model predicts. At equal 33.00 dB DCVC-UF\nneeds "
             f"{1 - pts[1]['full']['bpp'] / next(p['bpp'] for p in vtm['points'] if p['qp'] == 42):.0%} "
             "fewer bits than VTM - now measured in bytes on both sides. Upstream writes this "
             "stream only with a CUDA kernel; this is a CPU port of that path.",
             fontsize=8, color=INK_2, va="bottom")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(f"{FIG}/uf_real_bitstream.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {FIG}/uf_real_bitstream.png")


def do_uf_video_bitstream():
    """Real DCVC-UF video bytes against the estimate, and where the extra bytes go."""
    import sys
    sys.path.insert(0, "/work/scripts/bitstream")
    import numpy as np
    import MLCodec_extensions_cpp
    import uf_codec
    import uf_video_codec as V

    base = "/work/outputs/experiments/10-uf-video-bitstream"
    data = {st: publish(f"{base}/rd_{st}.json", f"task8_uf_video_bitstream_{st}.json")
            for st in ("hts", "ld")}
    data = {k: v for k, v in data.items() if v}
    if not data:
        return

    # the rANS coder's fixed cost per packet, measured rather than assumed
    e = MLCodec_extensions_cpp.RansEncoder()
    e.reset(); e.flush()
    flush = len(np.asarray(e.get_encoded_stream()))

    # Colour follows the structure in BOTH panels; real vs estimate is line weight.
    colour = {"hts": SERIES[0]["color"], "ld": SERIES[1]["color"]}
    marker = {"hts": "o", "ld": "s"}
    name = {"hts": "HTS (8-frame chunks)", "ld": "LD (one frame at a time)"}

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), dpi=160,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    any_d = next(iter(data.values()))
    style_axes(ax, "DCVC-UF video, real bitstream on CPU",
               f"RaceHorses 416x240, {any_d['frames']} frames - bytes on disk, every frame decoded back bit-exactly",
               "rate  (bits per pixel, log scale)", "YUV-PSNR  (dB)")
    ax.set_xscale("log")
    for st, d in data.items():
        pts = sorted(d["points"], key=lambda r: r["qp"])
        ax.plot([r["full"]["bpp"] for r in pts], [r["full"]["psnr_yuv"] for r in pts],
                color=colour[st], linewidth=2.2, marker=marker[st], markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=f"{name[st]}, real bytes", zorder=3)
        est = [(r["estimated_bpp"], r["full"]["psnr_yuv"]) for r in pts if r["estimated_bpp"]]
        if est:
            ax.plot([a for a, _ in est], [b for _, b in est], color=colour[st], linewidth=1.1,
                    marker="^", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8,
                    label=f"{name[st]}, estimate", zorder=4)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK)

    # Right: bytes above the estimate at the lowest rate, by cause. Grouped, not
    # stacked, so colour keeps meaning "structure" as it does on the left.
    ax = axes[1]
    style_axes(ax, "Bytes above the estimate at qp 0",
               "fixed per-packet costs dominate at very low rate",
               "bytes", "")
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    n = any_d["frames"]
    causes = ["container header\n(this codec)", "rANS flush\n(4 B per packet)",
              "coding + padded frame\n(remainder)"]
    ypos = np.arange(len(causes))
    height = 0.36
    for i, st in enumerate(("ld", "hts")):
        if st not in data:
            continue
        d = data[st]
        chunk = d["chunk"]
        packets = 1 + -(-(n - 1) // chunk)
        r0 = min(d["points"], key=lambda r: r["qp"])
        est_b = r0["estimated_bpp"] * 416 * 240 * n / 8
        container = V.SEQ_HEADER.size + packets * V.PACKET.size + uf_codec.HEADER.size
        rans = packets * flush
        remainder = r0["full"]["bytes"] - est_b - container - rans
        vals = [container, rans, remainder]
        y = ypos + (i - 0.5) * height
        bars = ax.barh(y, vals, height=height * 0.9, color=colour[st],
                       label=f"{st.upper()}: {packets} packets")
        for b, v in zip(bars, vals):
            ax.text(v + 6, b.get_y() + b.get_height() / 2, f"{v:.0f}", va="center",
                    fontsize=8.5, color=INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels(causes, fontsize=8.5, color=INK_2)
    ax.invert_yaxis()
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK)

    fig.text(0.008, 0.012,
             "Every frame decodes bit-exactly from the bytes alone; at skip_thres = 0 the reconstruction equals "
             "upstream's forward(). Once the container and the 4-byte rANS flush\nper packet are removed, "
             "arithmetic coding adds 0.1-0.4% to the estimate (LD). HTS keeps a steady ~1.6%: the last chunk "
             "codes a repeated padding frame the estimate pro-rates.\nLD writes 64 packets to HTS's 9, so at "
             "qp 0 it pays ~540 fixed bytes to HTS's ~100 - chunking amortises per-packet costs.",
             fontsize=8, color=INK_2, va="bottom")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(f"{FIG}/uf_video_real_bitstream.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {FIG}/uf_video_real_bitstream.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["image", "video", "bitstream", "video-bitstream", "all"], default="all", nargs="?")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    if a.which in ("image", "all"):
        do_image()
    if a.which in ("video", "all"):
        do_video()
    if a.which in ("bitstream", "all"):
        do_uf_bitstream()
    if a.which in ("video-bitstream", "all"):
        do_uf_video_bitstream()

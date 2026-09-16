"""DCVC-UF on a real YUV sequence: intra frame + inter frames, on CPU.

Forward path only -- no bitstream is written (see README.md). You get the
reconstructions, the entropy model's estimated bpp, and true YUV-PSNR.

  python run_video.py                        # LD model, 8 frames, qp 32
  python run_video.py --structure hts --frames 16

Why this script exists instead of upstream's test_video.py: that script drives
compress()/decompress(), which need the CUDA-only CUTLASS extension. Everything
below is the same model on the plain PyTorch path.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.models.image_model import DMCI
from src.utils.common import get_state_dict, ModelStructure
from src.utils.metrics import calc_psnr
from src.utils.transforms import ycbcr420_to_444_np, yuv_444_to_420
from src.utils.video_reader import YUV420Reader

import sys
# Rate estimation: use the quantised symbols, not the training-time noise proxy.
# forward_one_frame() adds uniform noise before estimating bits, which inflates the
# rate badly at low qp (8x at qp 0 on kodim19, checked against real bitstreams --
# see scripts/bitstream/uf_bitstream_demo.py). eval_rate() swaps the noise for
# rounding; the estimate then lands within 0.7% of the arithmetic-coded payload.
sys.path.insert(0, "/work/scripts/experiments")
from dcvc_yuv import eval_rate  # noqa: E402

WEIGHTS = "/work/weights/dcvc"


def pad_to(x, pad_r, pad_b):
    """Replicate-pad the right/bottom edges, the way video codecs extend a picture."""
    if not pad_r and not pad_b:
        return x
    return F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")


def crop(x, H, W):
    return x[:, :, :H, :W]


def real_bpp(out, H, W):
    """Rate over the ORIGINAL picture area.

    forward_one_frame normalises by the padded size; padded samples cost bits
    but are not pixels anyone sees, so charging them to the real area is the
    honest figure and the one comparable to test_video.py.
    """
    return (out["bits_y"].item() + out["bits_z"].item()) / (H * W)


def frame_psnr(x_hat_one, y_ref, u_ref, v_ref):
    """YUV-PSNR the way upstream reports it: (6*Y + U + V) / 8, in 4:2:0."""
    yuv = x_hat_one + 0.5
    y_rec, uv_rec = yuv_444_to_420(yuv)
    y_rec = torch.clamp(y_rec * 255, 0, 255).squeeze(0).numpy()[0]
    uv_rec = torch.clamp(uv_rec * 255, 0, 255).squeeze(0).numpy()
    p_y = calc_psnr(y_ref, y_rec)
    p_u = calc_psnr(u_ref, uv_rec[0])
    p_v = calc_psnr(v_ref, uv_rec[1])
    return (6 * p_y + p_u + p_v) / 8, p_y, p_u, p_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yuv", default="/work/Dataset/RaceHorses_416x240_30.yuv")
    ap.add_argument("--width", type=int, default=416)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--frames", type=int, default=8, help="total frames incl. the intra frame")
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument("--structure", choices=["ld", "hts", "htl"], default="ld")
    ap.add_argument("--out-dir", default="/work/outputs/02-video-compression")
    ap.add_argument("--pad-to", type=int, default=64,
                    help="pad the picture up to a multiple of this (64 is the minimum the model accepts)")
    args = ap.parse_args()

    torch.manual_seed(0)   # add_noise() in the rate estimate is stochastic
    os.makedirs(args.out_dir, exist_ok=True)

    structure = ModelStructure(args.structure)
    if structure is ModelStructure.LD:
        from src.models.video_model_ld import DMC, g_frame_delay
        p_net = DMC()
    else:
        from src.models.video_model_ht import DMC, g_frame_delay
        p_net = DMC(model_structure=structure)

    W, H = args.width, args.height
    # test_video.py pads to a multiple of 16, but that is only what it *reports*
    # -- the CUDA proxy pads further internally. On the plain PyTorch path the
    # latent sits at W/32 and is then pixel_unshuffled by 2, so the picture has
    # to be a multiple of 64. (416x240 -> 448x256, ~15% more samples; those are
    # coded but not displayed, so the bpp below reads slightly high.)
    pad_r, pad_b = DMCI.get_padding_size(H, W, args.pad_to)
    padded_W, padded_H = W + pad_r, H + pad_b
    if pad_r or pad_b:
        print(f"padding {W}x{H} -> {padded_W}x{padded_H} (multiple of {args.pad_to})")

    i_net = DMCI().eval()
    i_net.load_state_dict(get_state_dict(f"{WEIGHTS}/cvpr2026_image.pth.tar"))
    p_net = p_net.eval()
    p_net.load_state_dict(get_state_dict(f"{WEIGHTS}/cvpr2026_video_{args.structure}.pth.tar"))
    print(f"models: image + video_{args.structure}  |  chunk size (g_frame_delay) = {g_frame_delay}")
    print(f"sequence: {os.path.basename(args.yuv)}  {W}x{H}  qp={args.qp}  frames={args.frames}\n")

    reader = YUV420Reader(args.yuv, W, H)
    frames = []
    for _ in range(args.frames):
        y, uv = reader.read_one_frame()
        frames.append((ycbcr420_to_444_np(y, uv), y[0], uv[0], uv[1]))

    qp = torch.tensor([args.qp])
    rows = []
    recon_yuv = open(os.path.join(args.out_dir, f"rec_{args.structure}_qp{args.qp}.yuv"), "wb")

    def write_recon(x_hat_one):
        yuv = x_hat_one + 0.5
        y_rec, uv_rec = yuv_444_to_420(yuv)
        for t in (y_rec, uv_rec):
            a = torch.clamp(t * 255, 0, 255).squeeze(0).numpy().round().astype(np.uint8)
            recon_yuv.write(a.tobytes())

    if g_frame_delay > 1:
        print(f"note: this model codes {g_frame_delay} frames into ONE latent, so every")
        print(f"      frame in a chunk shows the same bpp -- the chunk's rate divided by")
        print(f"      {g_frame_delay}. There is no per-frame rate to report; that is the point")
        print(f"      of the chunk-based design.\n")

    print(f"{'frame':>6} {'type':>5} {'bpp':>9} {'YUV-PSNR':>10} {'Y':>8} {'U':>8} {'V':>8}")
    print("-" * 60)

    # --- intra frame: the image codec, exactly as in project 01 -------------
    yuv0, y0, u0, v0 = frames[0]
    x0 = torch.from_numpy(yuv0).unsqueeze(0).float() / 255.0 - 0.5
    with torch.no_grad(), eval_rate():
        out = i_net.forward_one_frame(pad_to(x0, pad_r, pad_b), qp)
    x_hat_i_padded = out["x_hat"]
    x_hat_i = crop(x_hat_i_padded, H, W)
    psnr, py, pu, pv = frame_psnr(x_hat_i, y0, u0, v0)
    bpp_i = real_bpp(out, H, W)
    print(f"{0:>6} {'I':>5} {bpp_i:>9.4f} {psnr:>9.2f}dB {py:>7.2f} {pu:>7.2f} {pv:>7.2f}")
    rows.append({"frame": 0, "type": "I", "bpp": bpp_i, "psnr": psnr})
    write_recon(x_hat_i)

    # Seed the decoded-picture buffer from the intra reconstruction.
    # add_ref_feature_from_frame() takes the CUDA proxy in eval mode; this is the
    # pure-PyTorch line that its training branch uses, which is all we need.
    p_net.clear_dpb()
    # The padded reconstruction, not the cropped one: the inter model keeps
    # running at the padded resolution, so its reference must match.
    p_net.ref_feature = F.pixel_unshuffle(x_hat_i_padded, 8)

    # --- inter frames, in chunks of g_frame_delay --------------------------
    idx = 1
    while idx < len(frames):
        chunk = frames[idx:idx + g_frame_delay]
        if len(chunk) < g_frame_delay:
            # The model always codes a full chunk; upstream repeats the last
            # frame to fill it, and drops the padding when reporting.
            chunk = chunk + [chunk[-1]] * (g_frame_delay - len(chunk))
            real_n = len(frames) - idx
        else:
            real_n = g_frame_delay

        x = torch.cat([torch.from_numpy(c[0]).unsqueeze(0) for c in chunk], dim=1)
        x = x.float() / 255.0 - 0.5

        with torch.no_grad(), eval_rate():
            out = p_net.forward_one_frame(pad_to(x, pad_r, pad_b), qp)

        # The chunk's bits cover g_frame_delay frames at once -- divide to get a
        # per-frame figure.
        bpp_per_frame = real_bpp(out, H, W) / g_frame_delay
        x_hats = out["x_hat"] if isinstance(out["x_hat"], (list, tuple)) else [out["x_hat"]]

        for k in range(real_n):
            _, y_r, u_r, v_r = chunk[k]
            psnr, py, pu, pv = frame_psnr(crop(x_hats[k], H, W), y_r, u_r, v_r)
            print(f"{idx + k:>6} {'P':>5} {bpp_per_frame:>9.4f} {psnr:>9.2f}dB "
                  f"{py:>7.2f} {pu:>7.2f} {pv:>7.2f}")
            rows.append({"frame": idx + k, "type": "P", "bpp": bpp_per_frame, "psnr": psnr})
            write_recon(crop(x_hats[k], H, W))

        idx += g_frame_delay

    recon_yuv.close()

    avg_bpp = sum(r["bpp"] for r in rows) / len(rows)
    avg_psnr = sum(r["psnr"] for r in rows) / len(rows)
    print("-" * 60)
    print(f"{'avg':>6} {'':>5} {avg_bpp:>9.4f} {avg_psnr:>9.2f}dB")
    kbps = avg_bpp * W * H * 30 / 1000
    print(f"\n{len(rows)} frames  |  {kbps:.1f} kbps at 30 fps  (ESTIMATED rate -- no bitstream on CPU)")

    with open(os.path.join(args.out_dir, f"rd_{args.structure}_qp{args.qp}.json"), "w") as f:
        json.dump({"sequence": args.yuv, "structure": args.structure, "qp": args.qp,
                   "avg_bpp": avg_bpp, "avg_psnr": avg_psnr, "frames": rows}, f, indent=2)
    print(f"wrote reconstruction + rd json to {args.out_dir}")


if __name__ == "__main__":
    main()

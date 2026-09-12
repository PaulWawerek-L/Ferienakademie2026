"""Task 3 -- DCVC-UF video RD curve on the first N frames of RaceHorses.

Frame 0 is coded with the image model; the rest go through the inter model in
chunks of g_frame_delay. Writes a reconstructed .yuv per qp so PSNR is computed
by the same tool the VTM experiments use.

  python exp3_dcvc_video_rd.py [--qps 0 15 30 45 63] [--frames 64] [--structures hts ld]
"""
import argparse
import json
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dcvc_yuv import (crop, estimated_bits, eval_rate, pad, padding_for, read_yuv420,
                      to_model_input, write_yuv420)

from src.models.image_model import DMCI
from src.utils.common import get_state_dict, ModelStructure

OUT = "/work/outputs/experiments/03-dcvc-video"
SEQ = "/work/Dataset/RaceHorses_416x240_30.yuv"
WEIGHTS = "/work/weights/dcvc"
W, H, FPS = 416, 240, 30


def build_p_net(structure):
    if structure is ModelStructure.LD:
        from src.models.video_model_ld import DMC, g_frame_delay
        return DMC(), g_frame_delay
    from src.models.video_model_ht import DMC, g_frame_delay
    return DMC(model_structure=structure), g_frame_delay


def code_sequence(i_net, p_net, g_frame_delay, frames, qp, rec_path, pad_r, pad_b):
    """Code every frame at one qp; returns total estimated bits."""
    qp_t = torch.tensor([qp])
    total_bits = 0.0
    with open(rec_path, "wb") as fh:
        # --- intra frame ---
        x0 = to_model_input(frames[0][0])
        with torch.no_grad(), eval_rate():
            out = i_net.forward_one_frame(pad(x0, pad_r, pad_b), qp_t)
        total_bits += estimated_bits(out)
        x_hat_padded = out["x_hat"]
        write_yuv420(fh, crop(x_hat_padded, H, W))

        # Seed the DPB. add_ref_feature_from_frame() takes the CUDA proxy in eval
        # mode; this is the one line its training branch uses.
        p_net.clear_dpb()
        p_net.ref_feature = F.pixel_unshuffle(x_hat_padded, 8)

        # --- inter frames, chunk by chunk ---
        idx = 1
        while idx < len(frames):
            chunk = frames[idx:idx + g_frame_delay]
            real_n = len(chunk)
            if real_n < g_frame_delay:
                # The model always codes a full chunk; upstream repeats the last
                # frame to fill it and drops the padding when reporting.
                chunk = chunk + [chunk[-1]] * (g_frame_delay - real_n)

            x = torch.cat([to_model_input(c[0]) for c in chunk], dim=1)
            with torch.no_grad(), eval_rate():
                out = p_net.forward_one_frame(pad(x, pad_r, pad_b), qp_t)

            # A partial chunk still costs a whole chunk's bits. Charging only the
            # real frames' share would flatter the codec, so bill it pro rata.
            total_bits += estimated_bits(out) * real_n / g_frame_delay

            x_hats = out["x_hat"] if isinstance(out["x_hat"], (list, tuple)) else [out["x_hat"]]
            for k in range(real_n):
                write_yuv420(fh, crop(x_hats[k], H, W))
            idx += g_frame_delay
    return total_bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 15, 30, 45, 63])
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--structures", nargs="+", default=["hts", "ld"],
                    choices=["ld", "hts", "htl"])
    ap.add_argument("--tag", default=None,
                    help="suffix for the output json, e.g. 8f -- lets a short run "
                         "matched to the VTM frame count sit beside the full one")
    args = ap.parse_args()

    torch.manual_seed(0)
    os.makedirs(OUT, exist_ok=True)

    # Trim the reference so PSNR compares exactly the coded frames.
    ref = f"{OUT}/ref_{args.frames}f.yuv"
    frame_bytes = W * H * 3 // 2
    with open(SEQ, "rb") as src, open(ref, "wb") as dst:
        dst.write(src.read(frame_bytes * args.frames))

    frames = read_yuv420(SEQ, W, H, args.frames)
    print(f"RaceHorses {W}x{H}, {len(frames)} frames | qps: {args.qps}")
    pad_r, pad_b = padding_for(H, W)
    if pad_r or pad_b:
        print(f"padding {W}x{H} -> {W + pad_r}x{H + pad_b} "
              f"(coded but not displayed; rate is billed to the real area)")

    i_net = DMCI().eval()
    i_net.load_state_dict(get_state_dict(f"{WEIGHTS}/cvpr2026_image.pth.tar"))

    for name in args.structures:
        structure = ModelStructure(name)
        p_net, g_frame_delay = build_p_net(structure)
        p_net = p_net.eval()
        p_net.load_state_dict(get_state_dict(f"{WEIGHTS}/cvpr2026_video_{name}.pth.tar"))

        print(f"\n=== {name.upper()} (chunk = {g_frame_delay}) ===")
        print(f"{'qp':>4} {'est. bpp':>10} {'YUV-PSNR':>10} {'kbps':>9}")
        print("-" * 38)
        points = []
        for qp in args.qps:
            rec = f"{OUT}/rh_{name}_qp{qp:02d}_rec.yuv"
            bits = code_sequence(i_net, p_net, g_frame_delay, frames, qp, rec, pad_r, pad_b)
            bpp = bits / (W * H * len(frames))
            pj = f"{OUT}/psnr_{name}_qp{qp:02d}.json"
            subprocess.run([sys.executable, "/work/scripts/experiments/yuv_psnr.py",
                            ref, rec, str(W), str(H), "--frames", str(len(frames)),
                            "--quiet", "--json", pj], check=True)
            p = json.load(open(pj))
            kbps = bpp * W * H * FPS / 1000
            print(f"{qp:>4} {bpp:>10.4f} {p['psnr_yuv']:>9.2f}dB {kbps:>9.1f}")
            points.append({"qp": qp, "bpp": bpp, "kbps": kbps,
                           "psnr_yuv": p["psnr_yuv"], "psnr_y": p["psnr_y"],
                           "psnr_u": p["psnr_u"], "psnr_v": p["psnr_v"],
                           "real_bitstream": False})

        suffix = f"_{args.tag}" if args.tag else ""
        json.dump({"label": f"DCVC-UF {name.upper()}", "sequence": "RaceHorses_416x240",
                   "frames": len(frames), "chunk": g_frame_delay,
                   "rate": "entropy-model estimate (no bitstream on CPU)",
                   "points": points},
                  open(f"{OUT}/rd_{name}{suffix}.json", "w"), indent=2)
        print(f"wrote {OUT}/rd_{name}{suffix}.json")


if __name__ == "__main__":
    main()

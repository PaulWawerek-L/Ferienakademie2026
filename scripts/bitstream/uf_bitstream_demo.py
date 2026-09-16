"""Real DCVC-UF-Intra bitstreams on CPU: actual bytes against the estimate.

For every qp this encodes kodim19 to a .bin file, decodes it back with a
separately constructed model, checks the reconstruction is bit-exact, and scores
it with the same PSNR tool the VTM experiments use. It reports three rates:

  estimated  -- the entropy model's figure from forward(), what task 1 plotted
  actual     -- bytes on disk with skip_thres = 0 (every latent coded)
  skipped    -- bytes on disk with upstream's skip_thres = 0.15

The gap between the first two is the real cost of arithmetic coding. The gap
between the last two is what skipping buys, paid for in PSNR.

  python uf_bitstream_demo.py [--qps 0 15 30 45 63]
"""
import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))

import uf_codec  # noqa: E402
from dcvc_yuv import crop, estimated_bits, eval_rate, read_yuv420, to_model_input, write_yuv420  # noqa: E402

from src.models.image_model import DMCI  # noqa: E402
from src.utils.common import get_state_dict  # noqa: E402

CKPT = "/work/weights/dcvc/cvpr2026_image.pth.tar"
SRC = "/work/Dataset/kodim19.png"
REF = "/work/outputs/experiments/kodim19_420.yuv"
OUT = "/work/outputs/experiments/09-uf-real-bitstream"
PSNR_TOOL = os.path.join(HERE, "..", "experiments", "yuv_psnr.py")


def load(skip_thres):
    net = DMCI().eval()
    net.load_state_dict(get_state_dict(CKPT))
    return uf_codec.prepare(net, skip_thres)


def yuv_psnr(rec_path, w, h, json_path):
    subprocess.run([sys.executable, PSNR_TOOL, REF, rec_path, str(w), str(h),
                    "--frames", "1", "--quiet", "--json", json_path], check=True)
    return json.load(open(json_path))["psnr_yuv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 15, 30, 45, 63])
    ap.add_argument("--skip-thres", type=float, default=0.15,
                    help="upstream's test_video.py default")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)

    from PIL import Image
    W, H = Image.open(SRC).size
    if not os.path.exists(REF):
        os.makedirs(os.path.dirname(REF), exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", SRC,
                        "-pix_fmt", "yuv420p", "-f", "rawvideo", REF], check=True)
    x = to_model_input(read_yuv420(REF, W, H, 1)[0][0])
    pixels = W * H

    settings = {"full": 0.0, "skipped": args.skip_thres}
    # Two model objects, not one per setting. Each holds ~170 MB of weights plus
    # an rANS coder that pre-allocates 8 x 10 MB of stream buffers and 8 threads;
    # four of them at once was enough to get the container OOM-killed on an 8 GB
    # Docker VM. The decoder is still a SEPARATE object from the encoder -- the
    # stream has to carry everything, nothing may leak from the encoder's state --
    # and both are re-prepared when the skip threshold changes.
    enc_net = load(0.0)
    dec_net = load(0.0)

    rows = {qp: {"qp": qp} for qp in args.qps}
    for qp in args.qps:
        with torch.inference_mode(), eval_rate():
            est = estimated_bits(enc_net.forward_one_frame(x, torch.tensor([qp])))
        rows[qp]["estimated_bpp"] = est / pixels
        gc.collect()

    for key, thres in settings.items():
        uf_codec.prepare(enc_net, thres)
        uf_codec.prepare(dec_net, thres)
        for qp in args.qps:
            t0 = time.perf_counter()
            bs, x_hat_enc = uf_codec.compress(enc_net, x, qp, skip_thres=thres)
            t_enc = time.perf_counter() - t0
            bin_path = f"{OUT}/kodim19_qp{qp:02d}_{key}.bin"
            with open(bin_path, "wb") as f:
                f.write(bs)

            t0 = time.perf_counter()
            x_hat_dec = uf_codec.decompress(dec_net, open(bin_path, "rb").read())
            t_dec = time.perf_counter() - t0
            exact = bool(torch.equal(x_hat_dec, x_hat_enc))

            rec = f"{OUT}/kodim19_qp{qp:02d}_{key}_rec.yuv"
            with open(rec, "wb") as fh:
                write_yuv420(fh, crop(x_hat_dec, H, W))
            psnr = yuv_psnr(rec, W, H, f"{OUT}/psnr_qp{qp:02d}_{key}.json")

            rows[qp][key] = {"bytes": len(bs), "bpp": len(bs) * 8 / pixels,
                             "psnr_yuv": psnr, "roundtrip_exact": exact,
                             "encode_s": t_enc, "decode_s": t_dec}
            del bs, x_hat_enc, x_hat_dec
            gc.collect()
            print(f"  done: qp {qp:>2} {key:<7} exact={exact}", flush=True)

    print()
    print(f"{'qp':>3} | {'estimated':>9} | {'actual':>8} {'PSNR':>7} | "
          f"{'skip ' + str(args.skip_thres):>11} {'PSNR':>7} | {'overhead':>8} | exact")
    print("-" * 82)
    rows = [rows[qp] for qp in args.qps]
    for row in rows:
        f, sk = row["full"], row["skipped"]
        row["coding_overhead"] = f["bpp"] / row["estimated_bpp"] - 1
        print(f"{row['qp']:>3} | {row['estimated_bpp']:>9.4f} | {f['bpp']:>8.4f} {f['psnr_yuv']:>6.2f} | "
              f"{sk['bpp']:>11.4f} {sk['psnr_yuv']:>6.2f} | {row['coding_overhead']:>+7.1%} | "
              f"{'yes' if f['roundtrip_exact'] and sk['roundtrip_exact'] else 'NO'}")

    # ru_maxrss is kilobytes on Linux
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"\npeak memory: {peak_mb:.0f} MB")

    json.dump({"image": "kodim19", "width": W, "height": H,
               "model": "DCVC-UF-Intra (cvpr2026_image)", "skip_thres": args.skip_thres,
               "header_bytes": uf_codec.HEADER.size, "peak_memory_mb": peak_mb,
               "points": rows},
              open(f"{OUT}/rd.json", "w"), indent=2)
    print(f"\nbitstreams, reconstructions and rd.json in {OUT}/")
    print("bpp includes this codec's 13-byte header; overhead = actual / estimated - 1.")


if __name__ == "__main__":
    main()

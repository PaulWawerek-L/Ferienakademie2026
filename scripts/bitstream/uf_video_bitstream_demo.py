"""Real DCVC-UF video bitstreams on CPU: RaceHorses, actual bytes against the estimate.

For each structure and qp this encodes the sequence (intra frame + inter chunks)
to one .bin, decodes it back with separately constructed models, checks every
frame is bit-exact, and scores the reconstruction with the shared PSNR tool.

The estimated rates are task 3's (results/data/task3_dcvc_video_*_64f.json):
same sequence, same frames, same qp for intra and inter.

  python uf_video_bitstream_demo.py [--structures hts ld] [--frames 64]
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
import uf_video_codec as V  # noqa: E402
from dcvc_yuv import read_yuv420, to_model_input, write_yuv420  # noqa: E402
from src.models.image_model import DMCI  # noqa: E402
from src.utils.common import get_state_dict  # noqa: E402

WEIGHTS = "/work/weights/dcvc"
SEQ = "/work/Dataset/RaceHorses_416x240_30.yuv"
OUT = "/work/outputs/experiments/10-uf-video-bitstream"
RESULTS = "/work/results/data"
PSNR_TOOL = os.path.join(HERE, "..", "experiments", "yuv_psnr.py")
W, H, FPS = 416, 240, 30


def i_model():
    n = DMCI().eval()
    n.load_state_dict(get_state_dict(f"{WEIGHTS}/cvpr2026_image.pth.tar"))
    return n


def p_model(structure):
    n, chunk = V.build_inter_model(structure)
    n = n.eval()
    n.load_state_dict(get_state_dict(f"{WEIGHTS}/cvpr2026_video_{structure}.pth.tar"))
    return n, chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures", nargs="+", default=["hts", "ld"], choices=["ld", "hts", "htl"])
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 15, 30, 45, 63])
    ap.add_argument("--skip-thres", type=float, default=0.15)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)

    n = args.frames
    ref = f"{OUT}/ref_{n}f.yuv"
    with open(SEQ, "rb") as src, open(ref, "wb") as dst:
        dst.write(src.read(W * H * 3 // 2 * n))
    frames = [to_model_input(f[0]) for f in read_yuv420(SEQ, W, H, n)]
    settings = {"full": 0.0, "skipped": args.skip_thres}

    for structure in args.structures:
        est = {}
        est_file = f"{RESULTS}/task3_dcvc_video_{structure}_{n}f.json"
        if os.path.exists(est_file):
            est = {p["qp"]: p["bpp"] for p in json.load(open(est_file))["points"]}

        # Encoder and decoder hold separate model objects: the inter models carry
        # a decoded-picture buffer, and it must be rebuilt from the bytes alone.
        i_enc, i_dec = i_model(), i_model()
        (p_enc, chunk), (p_dec, _) = p_model(structure), p_model(structure)
        rows = {qp: {"qp": qp, "estimated_bpp": est.get(qp)} for qp in args.qps}

        print(f"\n=== {structure.upper()} (chunk {chunk}), RaceHorses {W}x{H}, {n} frames ===", flush=True)
        for key, thres in settings.items():
            for net in (i_enc, i_dec):
                uf_codec.prepare(net, thres)
            for net in (p_enc, p_dec):
                V.prepare(net, thres)
            for qp in args.qps:
                t0 = time.perf_counter()
                bs, rec_enc = V.encode_sequence(i_enc, p_enc, structure, chunk, frames, qp, qp, thres)
                t_enc = time.perf_counter() - t0
                path = f"{OUT}/rh_{structure}_qp{qp:02d}_{key}.bin"
                with open(path, "wb") as f:
                    f.write(bs)

                t0 = time.perf_counter()
                _, rec_dec = V.decode_sequence(i_dec, p_dec, open(path, "rb").read())
                t_dec = time.perf_counter() - t0
                exact = len(rec_dec) == n and all(torch.equal(a, b) for a, b in zip(rec_enc, rec_dec))

                rec = f"{OUT}/rh_{structure}_qp{qp:02d}_{key}_rec.yuv"
                with open(rec, "wb") as fh:
                    for x in rec_dec:
                        write_yuv420(fh, x)
                pj = f"{OUT}/psnr_{structure}_qp{qp:02d}_{key}.json"
                subprocess.run([sys.executable, PSNR_TOOL, ref, rec, str(W), str(H),
                                "--frames", str(n), "--quiet", "--json", pj], check=True)
                psnr = json.load(open(pj))["psnr_yuv"]

                bpp = len(bs) * 8 / (W * H * n)
                rows[qp][key] = {"bytes": len(bs), "bpp": bpp, "kbps": bpp * W * H * FPS / 1000,
                                 "psnr_yuv": psnr, "roundtrip_exact": exact,
                                 "encode_s": t_enc, "decode_s": t_dec}
                print(f"  {key:<7} qp {qp:>2}: {len(bs):>7} bytes {bpp:.4f} bpp {psnr:6.2f} dB "
                      f"exact={exact} enc {t_enc:.1f}s dec {t_dec:.1f}s", flush=True)
                del bs, rec_enc, rec_dec
                gc.collect()

        rows = [rows[qp] for qp in args.qps]
        print(f"\n{'qp':>3} | {'estimated':>9} | {'actual':>8} {'PSNR':>7} | "
              f"{'skip ' + str(args.skip_thres):>11} {'PSNR':>7} | {'vs est.':>8} | exact")
        print("-" * 80)
        for r in rows:
            f, s = r["full"], r["skipped"]
            ovh = f"{f['bpp'] / r['estimated_bpp'] - 1:+.1%}" if r["estimated_bpp"] else "n/a"
            print(f"{r['qp']:>3} | {r['estimated_bpp'] or float('nan'):>9.4f} | {f['bpp']:>8.4f} "
                  f"{f['psnr_yuv']:>6.2f} | {s['bpp']:>11.4f} {s['psnr_yuv']:>6.2f} | {ovh:>8} | "
                  f"{'yes' if f['roundtrip_exact'] and s['roundtrip_exact'] else 'NO'}")

        json.dump({"sequence": "RaceHorses_416x240", "frames": n, "structure": structure,
                   "chunk": chunk, "skip_thres": args.skip_thres,
                   "rate_note": "actual = whole .bin incl. container headers; estimated from task 3",
                   "points": rows}, open(f"{OUT}/rd_{structure}.json", "w"), indent=2)
        del i_enc, i_dec, p_enc, p_dec
        gc.collect()

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"\npeak memory: {peak_mb:.0f} MB")


if __name__ == "__main__":
    main()

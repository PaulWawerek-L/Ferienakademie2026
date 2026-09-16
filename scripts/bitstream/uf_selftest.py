"""Fast guard for the DCVC-UF CPU bitstream path, run by project 01's smoke test.

Encodes kodim19 once, then checks the three properties that would each fail
silently otherwise:

  1. the decode is bit-exact, using a separately constructed model;
  2. the decoder actually reads the stream (one flipped byte changes the output);
  3. with skip_thres = 0 the reconstruction equals upstream's own forward().

(3) is the one that catches a bug shared by encoder and decoder, which (1) cannot.
Exits non-zero on any failure.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))

import uf_codec  # noqa: E402
from dcvc_yuv import read_yuv420, to_model_input  # noqa: E402
from src.models.image_model import DMCI  # noqa: E402
from src.utils.common import get_state_dict  # noqa: E402

CKPT = "/work/weights/dcvc/cvpr2026_image.pth.tar"
REF = "/work/outputs/experiments/kodim19_420.yuv"


def model():
    net = DMCI().eval()
    net.load_state_dict(get_state_dict(CKPT))
    return uf_codec.prepare(net, 0.0)


def main():
    if not os.path.exists(REF):
        import subprocess
        os.makedirs(os.path.dirname(REF), exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i",
                        "/work/Dataset/kodim19.png", "-pix_fmt", "yuv420p", "-f", "rawvideo", REF],
                       check=True)
    x = to_model_input(read_yuv420(REF, 512, 768, 1)[0][0])
    qp = 32
    enc, dec = model(), model()

    bs, x_hat = uf_codec.compress(enc, x, qp)
    ok = True

    same = torch.equal(uf_codec.decompress(dec, bs), x_hat)
    print(f"  bit-exact decode ({len(bs)} bytes, {len(bs) * 8 / (512 * 768):.4f} bpp): {same}")
    ok &= same

    bad = bytearray(bs)
    bad[uf_codec.HEADER.size + (len(bs) - uf_codec.HEADER.size) // 2] ^= 0xFF
    try:
        reads = not torch.equal(uf_codec.decompress(model(), bytes(bad)), x_hat)
    except Exception:
        reads = True  # a decoder that errors on corruption is reading the stream
    print(f"  decoder depends on the payload: {reads}")
    ok &= reads

    with torch.no_grad():
        ref = enc.forward_one_frame(x, torch.tensor([qp]), recon_only=True)
    matches = torch.equal(x_hat, ref)
    print(f"  equals upstream forward() at skip_thres=0: {matches}")
    ok &= matches

    print("UF BITSTREAM SELFTEST", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

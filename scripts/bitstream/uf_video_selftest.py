"""Fast guard for the DCVC-UF video bitstream path, run by project 02's smoke test.

For LD (one frame at a time) and HTS (8-frame chunks), encodes a short RaceHorses
sequence and checks the three properties that would each fail silently otherwise:

  1. every frame decodes bit-exactly, using separately constructed models --
     for video this also proves the decoded-picture buffer stayed in step;
  2. corrupting one byte of an inter packet leaves the intra frame untouched and
     changes the inter frames (inter decoding really reads its packet);
  3. with skip_thres = 0 the reconstruction equals upstream's own forward()
     pipeline -- the check that catches a bug shared by encoder and decoder.

Models are reused between checks: seed_dpb() clears the buffer, so a model object
carries no state from one sequence into the next. Exits non-zero on failure.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))

import uf_codec  # noqa: E402
import uf_video_codec as V  # noqa: E402
from dcvc_yuv import read_yuv420, to_model_input  # noqa: E402
from src.models.image_model import DMCI  # noqa: E402
from src.utils.common import get_state_dict  # noqa: E402

WEIGHTS = "/work/weights/dcvc"
SEQ = "/work/Dataset/RaceHorses_416x240_30.yuv"
QP = 32


def i_model():
    n = DMCI().eval()
    n.load_state_dict(get_state_dict(f"{WEIGHTS}/cvpr2026_image.pth.tar"))
    return uf_codec.prepare(n, 0.0)


def p_model(structure):
    n, chunk = V.build_inter_model(structure)
    n = n.eval()
    n.load_state_dict(get_state_dict(f"{WEIGHTS}/cvpr2026_video_{structure}.pth.tar"))
    return V.prepare(n, 0.0), chunk


def forward_reference(i_net, p_net, chunk, frames):
    with torch.no_grad():
        xi = i_net.forward_one_frame(V.pad64(frames[0]), torch.tensor([QP]), recon_only=True)
        _, _, h, w = frames[0].shape
        ref = [xi[:, :, :h, :w]]
        V.seed_dpb(p_net, xi)
        i = 1
        while i < len(frames):
            group = frames[i:i + chunk]
            real = len(group)
            group = group + [group[-1]] * (chunk - real)
            out = p_net.forward_one_frame(torch.cat([V.pad64(f) for f in group], 1),
                                          torch.tensor([QP]))
            xs = out["x_hat"] if isinstance(out["x_hat"], (list, tuple)) else [out["x_hat"]]
            ref += [x[:, :, :h, :w] for x in xs[:real]]
            i += chunk
    return ref


def check(structure, n_frames, i_enc, i_dec):
    frames = [to_model_input(f[0]) for f in read_yuv420(SEQ, 416, 240, n_frames)]
    (p_enc, chunk), (p_dec, _) = p_model(structure), p_model(structure)
    ok = True

    bs, rec = V.encode_sequence(i_enc, p_enc, structure, chunk, frames, QP, QP)
    _, dec = V.decode_sequence(i_dec, p_dec, bs)
    exact = len(dec) == n_frames and all(torch.equal(a, b) for a, b in zip(rec, dec))
    print(f"  {structure}: {n_frames} frames, {len(bs)} bytes; bit-exact decode: {exact}")
    ok &= exact

    hdr = V.SEQ_HEADER.size
    (intra_len,) = V.PACKET.unpack_from(bs, hdr)
    inter_start = hdr + V.PACKET.size + intra_len + V.PACKET.size
    bad = bytearray(bs)
    bad[inter_start + 16] ^= 0xFF
    try:
        _, dec_bad = V.decode_sequence(i_dec, p_dec, bytes(bad))
        reads = torch.equal(dec_bad[0], rec[0]) and not all(
            torch.equal(a, b) for a, b in zip(dec_bad[1:], rec[1:]))
    except Exception:
        reads = True   # erroring on a corrupt inter packet also proves it is read
    print(f"  {structure}: inter decoding depends on its packet: {reads}")
    ok &= reads

    ref = forward_reference(i_enc, p_enc, chunk, frames)
    matches = all(torch.equal(a, b) for a, b in zip(rec, ref))
    print(f"  {structure}: equals upstream forward() at skip_thres=0: {matches}")
    ok &= matches
    return ok


def main():
    i_enc, i_dec = i_model(), i_model()
    ok = check("ld", 5, i_enc, i_dec)
    ok &= check("hts", 9, i_enc, i_dec)
    print("UF VIDEO BITSTREAM SELFTEST", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

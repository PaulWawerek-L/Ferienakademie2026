"""Shared YUV plumbing for the DCVC-UF experiments.

Keeps the image and video experiments byte-compatible with the VTM ones: same
input file, same 4:2:0 reconstruction on disk, PSNR computed by the same tool.
"""
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from src.models.common_model import CompressionModel
from src.models.image_model import DMCI
from src.utils.transforms import ycbcr420_to_444_np, yuv_444_to_420

PAD_TO = 64  # see projects/02-video-compression/README.md for why 64, not 16


def read_yuv420(path, w, h, n_frames):
    """-> list of (yuv444 float array 3xHxW in 0..255, y, u, v planes)."""
    y_size, uv_size = w * h, (w // 2) * (h // 2)
    frame_size = y_size + 2 * uv_size
    out = []
    with open(path, "rb") as f:
        for _ in range(n_frames):
            buf = f.read(frame_size)
            if len(buf) < frame_size:
                break
            a = np.frombuffer(buf, dtype=np.uint8)
            y = a[:y_size].reshape(1, h, w).astype(np.float32)
            uv = np.stack([
                a[y_size:y_size + uv_size].reshape(h // 2, w // 2),
                a[y_size + uv_size:].reshape(h // 2, w // 2),
            ]).astype(np.float32)
            out.append((ycbcr420_to_444_np(y, uv), y[0], uv[0], uv[1]))
    return out


def to_model_input(yuv444):
    """0..255 YCbCr 4:4:4 -> the centred tensor the model expects."""
    return torch.from_numpy(yuv444).unsqueeze(0).float() / 255.0 - 0.5


def padding_for(h, w):
    return DMCI.get_padding_size(h, w, PAD_TO)


def pad(x, pad_r, pad_b):
    if not pad_r and not pad_b:
        return x
    return F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")


def crop(x, h, w):
    return x[:, :, :h, :w]


def write_yuv420(fh, x_hat_one):
    """Model output -> one 4:2:0 frame appended to an open binary file."""
    yuv = x_hat_one + 0.5
    y_rec, uv_rec = yuv_444_to_420(yuv)
    for t in (y_rec, uv_rec):
        a = torch.clamp(t * 255, 0, 255).squeeze(0).numpy().round().astype(np.uint8)
        fh.write(a.tobytes())


@contextmanager
def eval_rate():
    """Measure rate the way the entropy coder would, not the way training did.

    forward_one_frame estimates bits as get_prob_train(add_noise(y_res), scales),
    where add_noise adds uniform(-0.5, 0.5). That is the continuous relaxation of
    quantisation used to keep training differentiable -- a variational upper
    bound, not the coded rate.

    It costs little at high rate but is badly wrong at low rate: a latent that
    quantises to exactly 0 and costs almost nothing still carries entropy once
    noise is added, which puts a floor under the estimate. Left uncorrected it
    made DCVC-UF look several dB worse than VTM at the low end -- the opposite of
    the published result, and an artefact of the measurement, not the codec.

    get_prob_train(v, s) is the Gaussian mass over [v-0.5, v+0.5], i.e. exactly a
    quantisation bin's probability. Feeding it the ROUNDED value therefore gives
    the arithmetic coder's rate. And process_with_mask defines y_q as
    QuantFunc.apply(y_res) = torch.round(y_res), so swapping add_noise for round
    reproduces the real quantised symbols rather than approximating them.
    """
    original = CompressionModel.add_noise
    CompressionModel.add_noise = staticmethod(torch.round)
    try:
        yield
    finally:
        CompressionModel.add_noise = original


def estimated_bits(out):
    """Total estimated bits for whatever the model just coded.

    Still an estimate, not a bitstream length -- the real encoder needs the
    CUDA-only CUTLASS extension, and the entropy coder's own skip threshold and
    rANS overhead are not modelled here. Under eval_rate() it is the standard
    "estimated bpp" that learned-compression papers report, and it tracks the
    coded rate closely. The VTM numbers, by contrast, are real bytes on disk.
    """
    return out["bits_y"].item() + out["bits_z"].item()

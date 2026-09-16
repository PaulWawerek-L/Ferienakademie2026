"""A CPU encoder and decoder for DCVC-UF-Intra that write and read real bitstreams.

Upstream's DMCI.compress()/decompress() run entirely inside the CUDA-only CUTLASS
proxy (src/layers/extensions/inference/dmci_proxy.cpp). This module re-implements
that proxy's control flow in PyTorch on top of UF's own network modules and UF's
own rANS coder, so it needs no GPU. The weights, the CDF tables and the entropy
coder are all upstream's; only the orchestration is new.

Everything below mirrors a specific place in upstream's C++, and the comments say
which, because each of these was a way to produce a stream that encodes fine and
decodes to garbage:

* Call order. rANS is last-in-first-out, and the spatial prior is autoregressive,
  so the decoder needs z before step 0 before step 1. The proxy therefore encodes
  y steps 3, 2, 1, 0 and then z (DMCIProxy::worker). RT's Python encodes z first;
  RT's coder buffers, UF's writes immediately, so copying RT's order breaks UF.
* Flattening. The index kernels in elementwise/stream.cu write element
  (h*W*C + w*C + c): NHWC. For z this is not a convention but a requirement --
  encode_z picks the CDF as (i % ch) + cdf_offset, so channel must be innermost.
* Symbol packing. build_index_enc_kernel stores (symbol << 8) + scale_index as
  int16; encode_y_internal splits it back with >> 8 and & 0xff. Positions whose
  scale does not exceed skip_thres are not written at all.
* Scale index. scale_to_index clamps to [0.11, 16], takes log, subtracts
  LOG_SCALE_MIN and scales by 1/LOG_SCALE_STEP; to_uint8 rounds DOWN. The
  constants are upstream's rounded float literals, used as-is so the bins agree.
* Clamping. UF's Python process_with_mask does not clamp y_q, but a symbol has
  to fit in the int8 half of the packed int16. The reconstruction is computed
  from the clamped value so encoder and decoder stay identical.
* Decoder output. RansDecoder reuses a buffer allocated at twice the requested
  size, and the exposed accessor returns the whole buffer. Every read is sliced
  to the count just decoded; without that, stale symbols leak in.

The container around the rANS payload is this module's own small header, not
upstream's SPS/NAL layout, and the scale index is computed in float32 where the
proxy uses float16. Streams therefore round-trip exactly through this decoder but
are not expected to be readable by upstream's CUDA decoder.
"""
import math
import struct

import numpy as np
import torch

from src.models.image_model import DMCI

# Upstream constants (src/layers/extensions/inference/def_const.h), deliberately
# the rounded literals the kernels use rather than recomputed logarithms.
SCALE_MIN = 0.11
SCALE_MAX = 16.0
SCALE_LEVEL = 128
LOG_SCALE_MIN = -2.2073
LOG_SCALE_MAX = 2.7726
LOG_STEP_RECIP = 1.0 / ((LOG_SCALE_MAX - LOG_SCALE_MIN) / (SCALE_LEVEL - 1))

Z_CHANNELS = 128           # g_ch_z
Y_CHANNELS = 256           # g_ch_y
SYMBOL_MIN, SYMBOL_MAX = -128, 127

MAGIC = b"DUFI"            # DCVC-UF Intra
HEADER = struct.Struct("<4sHHBf")  # magic, height, width, qp, skip_thres


def _qp(qp):
    return torch.tensor([qp], dtype=torch.long)


def _pack4(x):
    """Collapse the four channel quarters of a masked tensor into one.

    Within one spatial-prior step each channel quarter is active on a different
    2x2 phase, so the sum keeps every coded value and loses nothing.
    """
    x0, x1, x2, x3 = x.chunk(4, 1)
    return (x0 + x1) + (x2 + x3)


def _nhwc(x):
    return x.permute(0, 2, 3, 1).reshape(-1)


def _from_nhwc(flat, c, h, w):
    # contiguous(): the permuted view has channels_last strides, which sends x86's
    # oneDNN down a different conv kernel than the encoder's NCHW tensors took.
    # The floats then differ by ~1e-6 and the decode is no longer bit-exact.
    return flat.reshape(1, h, w, c).permute(0, 3, 1, 2).contiguous()


def _scale_index(scales):
    s = scales.float().clamp(SCALE_MIN, SCALE_MAX)
    idx = torch.floor((torch.log(s) - LOG_SCALE_MIN) * LOG_STEP_RECIP)
    return idx.clamp(0, SCALE_LEVEL - 1)


def _step_prior(net, step, y_hat_so_far, common_params):
    adaptor = (None, net.y_spatial_prior_adaptor_1, net.y_spatial_prior_adaptor_2,
               net.y_spatial_prior_adaptor_3)[step]
    params = torch.cat((y_hat_so_far, common_params), dim=1)
    return net.y_spatial_prior(adaptor(params)).chunk(2, 1)


def _hyper_params(net, z_hat, y_h, y_w):
    params = net.y_prior_fusion(net.hyper_dec(z_hat))
    params = params[:, :, :y_h, :y_w]
    scales, means = net.separate_prior_image(params)
    return scales, means, net.y_spatial_prior_reduction(params)


def prepare(net: DMCI, skip_thres: float = 0.0):
    """Build the CDF tables and attach UF's rANS encoder/decoder to the model."""
    net.update(skip_thres)
    return net


@torch.no_grad()
def compress(net: DMCI, x: torch.Tensor, qp: int, skip_thres: float = 0.0):
    """x: (1, 3, H, W) YCbCr 4:4:4 in [-0.5, 0.5], H and W multiples of 64.

    Returns (bitstream bytes, x_hat). x_hat is what the decoder will reproduce.
    """
    _, _, H, W = x.shape
    if H % 64 or W % 64:
        raise ValueError(f"{W}x{H}: both sides must be multiples of 64")

    q_enc = net.index_select_dim0(net.q_scale_enc, _qp(qp))
    q_dec = net.index_select_dim0(net.q_scale_dec, _qp(qp))
    qy_enc = net.index_select_dim0(net.q_scale_y_enc, _qp(qp))
    qy_dec = net.index_select_dim0(net.q_scale_y_dec, _qp(qp))

    y = net.enc(x, q_enc)
    z_hat = torch.round(net.hyper_enc(y)).clamp(SYMBOL_MIN, SYMBOL_MAX)
    _, c, y_h, y_w = y.shape
    scales, means, common = _hyper_params(net, z_hat, y_h, y_w)
    y = y * qy_enc

    masks = net.get_mask_4x(1, c, y_h, y_w, y.device)
    y_hat_so_far = None
    step_symbols = []
    for step in range(4):
        if step:
            scales, means = _step_prior(net, step, y_hat_so_far, common)
        mask = masks[step]
        s_hat = scales * mask
        mu = means * mask
        coded = s_hat > skip_thres                 # stream.cu: _scale > skip_thres
        y_q = torch.round((y - mu) * mask) * coded
        y_q = y_q.clamp(SYMBOL_MIN, SYMBOL_MAX)
        y_hat_step = y_q + mu
        y_hat_so_far = y_hat_step if y_hat_so_far is None else y_hat_so_far + y_hat_step

        s_w = _pack4(s_hat)
        keep = _nhwc(s_w > skip_thres)
        sym = _nhwc(_pack4(y_q)).to(torch.int32)
        idx = _nhwc(_scale_index(s_w)).to(torch.int32)
        packed = (sym * 256 + idx)[keep].to(torch.int16).numpy()
        step_symbols.append(np.ascontiguousarray(packed))

    x_hat = net.dec(y_hat_so_far * qy_dec, q_dec)

    enc = net.entropy_coder.encoder
    enc.reset()
    for step in (3, 2, 1, 0):                      # LIFO: decoder reads 0 first
        enc.encode_y(step_symbols[step])
    z_sym = _nhwc(z_hat).to(torch.int8).numpy()
    enc.encode_z(np.ascontiguousarray(z_sym), qp * Z_CHANNELS, Z_CHANNELS)
    enc.flush()
    payload = np.asarray(enc.get_encoded_stream(), dtype=np.uint8).tobytes()

    header = HEADER.pack(MAGIC, H, W, qp, skip_thres)
    return header + payload, x_hat


@torch.no_grad()
def decompress(net: DMCI, bitstream: bytes):
    """Reconstruct x_hat from the bytes alone."""
    magic, H, W, qp, skip_thres = HEADER.unpack_from(bitstream, 0)
    if magic != MAGIC:
        raise ValueError("not a DCVC-UF intra bitstream from this codec")
    payload = np.frombuffer(bitstream, dtype=np.uint8, offset=HEADER.size)

    q_dec = net.index_select_dim0(net.q_scale_dec, _qp(qp))
    qy_dec = net.index_select_dim0(net.q_scale_y_dec, _qp(qp))
    y_h, y_w = H // 16, W // 16
    z_h, z_w = H // 64, W // 64

    dec = net.entropy_coder.decoder
    dec.set_stream(np.ascontiguousarray(payload))

    n_z = Z_CHANNELS * z_h * z_w
    dec.decode_z(n_z, qp * Z_CHANNELS, Z_CHANNELS)
    z_flat = np.asarray(dec.get_decoded_tensor())[:n_z]   # buffer is over-allocated
    z_hat = _from_nhwc(torch.from_numpy(z_flat.astype(np.float32)), Z_CHANNELS, z_h, z_w)

    scales, means, common = _hyper_params(net, z_hat, y_h, y_w)
    masks = net.get_mask_4x(1, Y_CHANNELS, y_h, y_w, z_hat.device)
    quarter = Y_CHANNELS // 4

    y_hat_so_far = None
    for step in range(4):
        if step:
            scales, means = _step_prior(net, step, y_hat_so_far, common)
        mask = masks[step]
        s_w = _pack4(scales * mask)
        keep = _nhwc(s_w > skip_thres)
        idx = _nhwc(_scale_index(s_w))[keep].to(torch.uint8).numpy()

        values = torch.zeros(keep.numel(), dtype=torch.float32)
        n = int(keep.sum())
        if n:
            dec.decode_y(np.ascontiguousarray(idx))
            decoded = np.asarray(dec.get_decoded_tensor())[:n]
            values[keep] = torch.from_numpy(decoded.astype(np.float32))
        y_q_w = _from_nhwc(values, quarter, y_h, y_w)

        # Inverse of _pack4: every quarter gets the packed values, the mask keeps
        # only the phase that quarter was coded on.
        y_hat_step = (torch.cat((y_q_w, y_q_w, y_q_w, y_q_w), dim=1) + means) * mask
        y_hat_so_far = y_hat_step if y_hat_so_far is None else y_hat_so_far + y_hat_step

    return net.dec(y_hat_so_far * qy_dec, q_dec)

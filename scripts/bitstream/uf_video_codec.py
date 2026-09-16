"""A CPU encoder and decoder for DCVC-UF video that write and read real bitstreams.

Upstream codes inter frames only inside CUDA proxies (dmc_ld_proxy.cpp,
dmc_hts_proxy.cpp, dmc_htl_proxy.cpp). This module re-implements their control
flow in PyTorch on UF's own modules, CDF tables and rANS coder. The intra frame
goes through uf_codec.py, exactly as test_video.py sends it through DMCI.

The three structures are not coded the same way, and each difference below is
taken from the proxy that implements it:

  structure  spatial prior     scales                   encode_y calls
  ld         2 steps, means    from the hyperprior      one, whole y (NHWC)
  hts        4 steps, means    from the hyperprior      one, whole y (NHWC)
  htl        4 steps, both     re-predicted each step   four, steps 3,2,1,0

When scales come from the hyperprior they are known before any y is decoded, so
LD and HTS gather every step's symbols into one array indexed by those scales
and decode it in one call -- part of why they are fast. HTL predicts scales per
step, so, like the image codec, it must write the steps in reverse (rANS is
last-in-first-out) and decode them one at a time.

State. Every inter model carries a decoded-picture buffer (ref_feature, memory,
ctx) that each frame's decoded feature advances. The encoder and decoder each hold
their own model object; they stay in step only because y_hat is reproduced bit for
bit, which is exactly what the round-trip check verifies. The proxies update the
buffer eagerly on the encoder side and lazily on the decoder side; the values are
the same, and this module follows Python's forward_one_frame (lazy on both) so it
can be compared against forward() directly.

Container: a small sequence header, then one length-prefixed packet per intra
frame / inter chunk. It is this module's own layout, not upstream's SPS/NAL, so
streams are not expected to be readable by upstream's CUDA decoder.
"""
import struct

import numpy as np
import torch
import torch.nn.functional as F

import uf_codec
from uf_codec import SYMBOL_MAX, SYMBOL_MIN, Z_CHANNELS, _nhwc, _from_nhwc, _pack4, _scale_index

from src.utils.common import ModelStructure

MAGIC = b"DUFV"
SEQ_HEADER = struct.Struct("<4sBHHHBBf")  # magic, structure, W, H, frames, qp_i, qp_p, skip
PACKET = struct.Struct("<I")
STRUCTURES = {"ld": 0, "hts": 1, "htl": 2}
STRUCTURE_NAMES = {v: k for k, v in STRUCTURES.items()}


def build_inter_model(structure):
    """-> (model, frames per chunk). Imports stay local: the two modules both
    define DMC and a g_frame_delay, and only one of them is wanted."""
    if structure == "ld":
        from src.models.video_model_ld import DMC, g_frame_delay
        return DMC(), g_frame_delay
    from src.models.video_model_ht import DMC, g_frame_delay
    ms = ModelStructure.HTS if structure == "hts" else ModelStructure.HTL
    return DMC(model_structure=ms), g_frame_delay


def pad64(x):
    _, _, h, w = x.shape
    pad_r, pad_b = (-w) % 64, (-h) % 64
    if pad_r or pad_b:
        x = F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")
    return x


def _qp_vectors(net, qp):
    idx = torch.tensor([qp], dtype=torch.long)
    return (net.index_select_dim0(net.q_encoder, idx),
            net.index_select_dim0(net.q_decoder, idx),
            net.index_select_dim0(net.q_feature, idx))


def _quantise(y, scales, means, mask, skip_thres):
    """One spatial-prior step. y_q is zeroed where the position is skipped and
    clamped to what fits the int8 half of a packed symbol; the reconstruction is
    derived from that final value so encoder and decoder agree."""
    s_hat = scales * mask
    mu = means * mask
    coded = s_hat > skip_thres
    y_q = (torch.round((y - mu) * mask) * coded).clamp(SYMBOL_MIN, SYMBOL_MAX)
    return y_q, y_q + mu, s_hat


def _whole_y_symbols(y_q, scales, skip_thres):
    """LD/HTS: every position of y indexed by the hyperprior scales, one array."""
    keep = _nhwc(scales > skip_thres)
    sym = _nhwc(y_q).to(torch.int32)
    idx = _nhwc(_scale_index(scales)).to(torch.int32)
    return [np.ascontiguousarray((sym * 256 + idx)[keep].to(torch.int16).numpy())]


# ---------------------------------------------------------------------------
# encoder: prior -> (y_hat before q_dec scaling, symbol arrays in write order)
# ---------------------------------------------------------------------------

def _encode_prior(net, structure, y, params, skip_thres):
    q_enc, q_dec, scales, means = net.separate_prior_video(params)
    y = y * q_enc
    b, c, h, w = y.shape

    if structure == "ld":                                   # forward_prior_2x
        m0, m1 = net.get_mask_2x(b, c, h, w, y.device)
        y_q0, y_hat, _ = _quantise(y, scales, means, m0, skip_thres)
        means = net.y_spatial_prior(y_hat, params)
        y_q1, y_hat1, _ = _quantise(y, scales, means, m1, skip_thres)
        return (y_hat + y_hat1) * q_dec, _whole_y_symbols(y_q0 + y_q1, scales, skip_thres)

    masks = net.get_mask_4x(b, c, h, w, y.device)            # forward_prior_4x
    common = net.y_spatial_prior_reduction(params)
    adaptors = (None, net.y_spatial_prior_adaptor_1, net.y_spatial_prior_adaptor_2,
                net.y_spatial_prior_adaptor_3)
    y_hat, y_q_all, steps = None, None, []
    for k in range(4):
        if k:
            if structure == "hts":
                means = net.y_spatial_prior(adaptors[k](y_hat, common))
            else:
                scales, means = net.y_spatial_prior(
                    adaptors[k](torch.cat((y_hat, common), dim=1))).chunk(2, 1)
        y_q, y_hat_k, s_hat = _quantise(y, scales, means, masks[k], skip_thres)
        y_hat = y_hat_k if y_hat is None else y_hat + y_hat_k
        if structure == "hts":
            y_q_all = y_q if y_q_all is None else y_q_all + y_q
        else:
            s_w = _pack4(s_hat)
            keep = _nhwc(s_w > skip_thres)
            sym = _nhwc(_pack4(y_q)).to(torch.int32)
            idx = _nhwc(_scale_index(s_w)).to(torch.int32)
            steps.append(np.ascontiguousarray((sym * 256 + idx)[keep].to(torch.int16).numpy()))

    if structure == "hts":
        return y_hat * q_dec, _whole_y_symbols(y_q_all, scales, skip_thres)
    return y_hat * q_dec, steps[::-1]                         # LIFO: 3, 2, 1, 0


# ---------------------------------------------------------------------------
# decoder
# ---------------------------------------------------------------------------

def _decode_symbols(dec, keep, idx):
    values = torch.zeros(keep.numel(), dtype=torch.float32)
    n = int(keep.sum())
    if n:
        dec.decode_y(np.ascontiguousarray(idx[keep].to(torch.uint8).numpy()))
        decoded = np.asarray(dec.get_decoded_tensor())[:n]   # buffer is over-allocated
        values[keep] = torch.from_numpy(decoded.astype(np.float32))
    return values


def _decode_prior(net, structure, params, skip_thres, dec):
    q_enc, q_dec, scales, means = net.separate_prior_video(params)
    b, c, h, w = scales.shape

    if structure in ("ld", "hts"):
        keep = _nhwc(scales > skip_thres)
        y_q = _from_nhwc(_decode_symbols(dec, keep, _nhwc(_scale_index(scales))), c, h, w)
        if structure == "ld":
            m0, m1 = net.get_mask_2x(b, c, h, w, scales.device)
            y_hat = (y_q + means) * m0
            means = net.y_spatial_prior(y_hat, params)
            return (y_hat + (y_q + means) * m1) * q_dec

        masks = net.get_mask_4x(b, c, h, w, scales.device)
        common = net.y_spatial_prior_reduction(params)
        adaptors = (None, net.y_spatial_prior_adaptor_1, net.y_spatial_prior_adaptor_2,
                    net.y_spatial_prior_adaptor_3)
        y_hat = (y_q + means) * masks[0]
        for k in range(1, 4):
            means = net.y_spatial_prior(adaptors[k](y_hat, common))
            y_hat = y_hat + (y_q + means) * masks[k]
        return y_hat * q_dec

    # htl: scales re-predicted per step, so each step is decoded on its own
    masks = net.get_mask_4x(b, c, h, w, scales.device)
    common = net.y_spatial_prior_reduction(params)
    adaptors = (None, net.y_spatial_prior_adaptor_1, net.y_spatial_prior_adaptor_2,
                net.y_spatial_prior_adaptor_3)
    quarter = c // 4
    y_hat = None
    for k in range(4):
        if k:
            scales, means = net.y_spatial_prior(
                adaptors[k](torch.cat((y_hat, common), dim=1))).chunk(2, 1)
        s_w = _pack4(scales * masks[k])
        keep = _nhwc(s_w > skip_thres)
        y_q_w = _from_nhwc(_decode_symbols(dec, keep, _nhwc(_scale_index(s_w))), quarter, h, w)
        step = (torch.cat((y_q_w, y_q_w, y_q_w, y_q_w), dim=1) + means) * masks[k]
        y_hat = step if y_hat is None else y_hat + step
    return y_hat * q_dec


# ---------------------------------------------------------------------------
# one inter chunk
# ---------------------------------------------------------------------------

def prepare(net, skip_thres=0.0):
    net.update(skip_thres)
    return net


def seed_dpb(net, x_hat_intra_padded):
    """What add_ref_feature_from_frame() does: pixel-unshuffle the intra
    reconstruction into the reference feature. Upstream passes the padded x_hat."""
    net.clear_dpb()
    net.ref_feature = F.pixel_unshuffle(x_hat_intra_padded, 8)


@torch.no_grad()
def compress_chunk(net, structure, x, qp, skip_thres=0.0, reset_feature_memory=False):
    """x: (1, 3 * chunk, H, W) padded to multiples of 64, in [-0.5, 0.5].
    Returns (payload bytes, list of x_hat per frame)."""
    q_encoder, q_decoder, q_feature = _qp_vectors(net, qp)
    net.apply_feature_adaptor()
    y = net.encoder(x, net.ctx, q_encoder)
    z_hat = torch.round(net.hyper_encoder(y)).clamp(SYMBOL_MIN, SYMBOL_MAX)
    _, _, h, w = x.shape
    assert z_hat.shape == (1, Z_CHANNELS, h // 64, w // 64), z_hat.shape
    params = net.res_prior_param_decoder(z_hat, net.memory, q_feature)
    y_hat, arrays = _encode_prior(net, structure, y, params, skip_thres)
    x_hat, feature = net.get_recon_and_feature(y_hat, net.ctx, q_decoder)
    net.set_ref_feature(feature, reset_feature_memory)

    enc = net.entropy_coder.encoder
    enc.reset()
    for a in arrays:
        enc.encode_y(a)
    enc.encode_z(np.ascontiguousarray(_nhwc(z_hat).to(torch.int8).numpy()),
                 qp * Z_CHANNELS, Z_CHANNELS)
    enc.flush()
    payload = np.asarray(enc.get_encoded_stream(), dtype=np.uint8).tobytes()
    return payload, (x_hat if isinstance(x_hat, (list, tuple)) else [x_hat])


@torch.no_grad()
def decompress_chunk(net, structure, payload, qp, h, w, skip_thres=0.0,
                     reset_feature_memory=False):
    """h, w: padded size. Returns list of x_hat per frame."""
    _, q_decoder, q_feature = _qp_vectors(net, qp)
    net.apply_feature_adaptor()
    dec = net.entropy_coder.decoder
    dec.set_stream(np.frombuffer(payload, dtype=np.uint8).copy())
    z_h, z_w = h // 64, w // 64
    n_z = Z_CHANNELS * z_h * z_w
    dec.decode_z(n_z, qp * Z_CHANNELS, Z_CHANNELS)
    z_flat = np.asarray(dec.get_decoded_tensor())[:n_z]
    z_hat = _from_nhwc(torch.from_numpy(z_flat.astype(np.float32)), Z_CHANNELS, z_h, z_w)
    params = net.res_prior_param_decoder(z_hat, net.memory, q_feature)
    y_hat = _decode_prior(net, structure, params, skip_thres, dec)
    x_hat, feature = net.get_recon_and_feature(y_hat, net.ctx, q_decoder)
    net.set_ref_feature(feature, reset_feature_memory)
    return x_hat if isinstance(x_hat, (list, tuple)) else [x_hat]


# ---------------------------------------------------------------------------
# whole sequence
# ---------------------------------------------------------------------------

def encode_sequence(i_net, p_net, structure, chunk, frames, qp_i, qp_p, skip_thres=0.0):
    """frames: list of (1, 3, H, W) tensors in [-0.5, 0.5], all the same size.
    Returns (bitstream bytes, list of cropped reconstructions)."""
    _, _, h, w = frames[0].shape
    out = [SEQ_HEADER.pack(MAGIC, STRUCTURES[structure], w, h, len(frames), qp_i, qp_p, skip_thres)]
    recon = []

    bs, x_hat = uf_codec.compress(i_net, pad64(frames[0]), qp_i, skip_thres)
    out += [PACKET.pack(len(bs)), bs]
    recon.append(x_hat[:, :, :h, :w])
    seed_dpb(p_net, x_hat)

    i = 1
    while i < len(frames):
        group = frames[i:i + chunk]
        real = len(group)
        group = group + [group[-1]] * (chunk - real)   # upstream repeats the last frame
        x = torch.cat([pad64(f) for f in group], dim=1)
        payload, x_hats = compress_chunk(p_net, structure, x, qp_p, skip_thres)
        out += [PACKET.pack(len(payload)), payload]
        recon += [xh[:, :, :h, :w] for xh in x_hats[:real]]
        i += chunk
    return b"".join(out), recon


def decode_sequence(i_net, p_net, bitstream):
    magic, s, w, h, n, qp_i, qp_p, skip_thres = SEQ_HEADER.unpack_from(bitstream, 0)
    if magic != MAGIC:
        raise ValueError("not a DCVC-UF video bitstream from this codec")
    structure = STRUCTURE_NAMES[s]
    chunk = 1 if structure == "ld" else 8
    pos = SEQ_HEADER.size
    ph, pw = h + (-h) % 64, w + (-w) % 64

    def packet():
        nonlocal pos
        (length,) = PACKET.unpack_from(bitstream, pos)
        pos += PACKET.size
        data = bitstream[pos:pos + length]
        pos += length
        return data

    x_hat = uf_codec.decompress(i_net, packet())
    recon = [x_hat[:, :, :h, :w]]
    seed_dpb(p_net, x_hat)
    while len(recon) < n:
        x_hats = decompress_chunk(p_net, structure, packet(), qp_p, ph, pw, skip_thres)
        recon += [xh[:, :, :h, :w] for xh in x_hats[:n - len(recon)]]
    return structure, recon

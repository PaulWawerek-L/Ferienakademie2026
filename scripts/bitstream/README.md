# Real bitstreams on CPU

Both commands are CPU-only and write bytes that decode back bit-exactly:

| | command | derivation |
|---|---|---|
| **DCVC-UF-Intra** | `make uf-bitstream` | a port of the CUDA image proxy — [`UF_PORT_NOTES.md`](UF_PORT_NOTES.md) |
| **DCVC-UF video** (LD, HTS, HTL) | `make uf-video-bitstream` | ports of the three inter proxies |

Upstream DCVC-UF cannot do this on its own: its `compress()` goes through
`inference_extensions_cuda` — CUTLASS fused kernels — and raises
`NotImplementedError` when that import fails, with the network *and* the entropy
coding both inside the CUDA code. What is re-implemented here is only that
orchestration; the weights, CDF tables and rANS coder are upstream's.

## DCVC-UF-Intra

[`uf_codec.py`](uf_codec.py) re-implements what UF's CUDA proxy does,
on UF's own modules, CDF tables and rANS coder. The derivation, read off the C++,
is in [`UF_PORT_NOTES.md`](UF_PORT_NOTES.md).

Measured on kodim19 (512×768, YUV420). "payload vs est." excludes the codec's own
13-byte header; every row decodes back bit-exactly from a separate model object.

| qp | estimated bpp | actual bpp | payload vs est. | PSNR | actual, skip 0.15 | PSNR | exact |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.0239 | 0.0243 | +0.68% | 30.89 dB | 0.0239 | 30.85 dB | yes |
| 15 | 0.0499 | 0.0503 | +0.42% | 33.00 dB | 0.0496 | 32.97 dB | yes |
| 30 | 0.1103 | 0.1109 | +0.25% | 35.17 dB | 0.1101 | 35.15 dB | yes |
| 45 | 0.2958 | 0.2967 | +0.20% | 37.90 dB | 0.2959 | 37.89 dB | yes |
| 63 | 0.7642 | 0.7678 | +0.44% | 41.76 dB | 0.7668 | 41.76 dB | yes |

At equal 33.00 dB that is 0.0503 bpp against VTM's 0.0729 — 31 % fewer bits,
both sides measured in bytes. Figure: `results/figures/uf_real_bitstream.png`.

## DCVC-UF video

[`uf_video_codec.py`](uf_video_codec.py) ports the three inter proxies
(`dmc_ld_proxy.cpp`, `dmc_hts_proxy.cpp`, `dmc_htl_proxy.cpp`); the intra frame goes
through `uf_codec.py`, as `test_video.py` sends it through `DMCI`. What differs
from the image codec, and between the structures, is in
[`UF_PORT_NOTES.md`](UF_PORT_NOTES.md#video).

**DCVC-UF video**, RaceHorses 416×240, 64 frames (intra frame + inter frames, same
qp for both). "total vs est." is the whole file against task 3's estimate —
see below for what the gap is made of.

HTS (8-frame chunks), ≈ 146 ms/frame to encode, 81 ms/frame to decode:

| qp | estimated bpp | actual bpp | total vs est. | PSNR | actual, skip 0.15 | PSNR | exact |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.0041 | 0.0043 | +4.6% | 25.05 dB | 0.0042 | 24.99 dB | yes |
| 15 | 0.0107 | 0.0110 | +2.8% | 27.27 dB | 0.0107 | 27.25 dB | yes |
| 30 | 0.0283 | 0.0289 | +1.9% | 29.45 dB | 0.0285 | 29.41 dB | yes |
| 45 | 0.0797 | 0.0810 | +1.6% | 31.63 dB | 0.0805 | 31.62 dB | yes |
| 63 | 0.1932 | 0.1964 | +1.7% | 33.29 dB | 0.1959 | 33.29 dB | yes |

LD (one frame at a time), ≈ 127 ms/frame to encode, 85 ms/frame to decode:

| qp | estimated bpp | actual bpp | total vs est. | PSNR | actual, skip 0.15 | PSNR | exact |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.0071 | 0.0078 | +10.0% | 26.27 dB | 0.0076 | 26.23 dB | yes |
| 15 | 0.0171 | 0.0178 | +4.2% | 28.58 dB | 0.0174 | 28.54 dB | yes |
| 30 | 0.0455 | 0.0463 | +1.7% | 31.11 dB | 0.0457 | 31.07 dB | yes |
| 45 | 0.1238 | 0.1247 | +0.7% | 33.94 dB | 0.1237 | 33.91 dB | yes |
| 63 | 0.3268 | 0.3278 | +0.3% | 36.89 dB | 0.3272 | 36.88 dB | yes |

**Where the extra bytes go.** At qp 0 LD is 10 % over the estimate, but almost
none of that is arithmetic coding. Removing this codec's container (286 B: header
plus a 4-byte length per packet) and the rANS coder's 4-byte flush per packet
(256 B) leaves **+0.1–0.4 %** — the same as the image codec. LD writes 64
packets, HTS 9, so HTS pays about 100 fixed bytes where LD pays about 540: the
chunk design amortises per-packet costs, and at very low rates that is visible in
real bytes. HTS keeps a steady +1.5–1.6 % after the same removal: 63 inter frames
are 7 full chunks plus 7 frames, so the last chunk codes a repeated padding frame
(1/64 of the inter rate ≈ 1.56 %) that the estimate only bills pro rata.
Figure: `results/figures/uf_video_real_bitstream.png`.

**Scope and limits**

- Streams are not expected to be readable by upstream's CUDA decoder: the header
  is this codec's own, and scale indices are computed in float32 where the proxy
  uses float16, so a scale on a bin boundary can land in a neighbouring bin.
- Requires the patched `MLCodec_extensions_cpp` from `docker/dcvc/` (it exposes the
  decoder's output to Python; upstream binds no accessor).

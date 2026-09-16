# 02 — Learned Video Compression (DCVC-UF)

**Concepts:** Motion · Temporal Prior · Video Coding

Same repo and same container image as project 01 — `make shell-video-compression`
drops you in the same place.

## Run

```bash
make smoke-video-compression        # LD on 5 frames, then HTS on 9
make shell-video-compression
python /work/projects/02-video-compression/run_video.py --structure hts --frames 17 --qp 40
```

Writes a reconstructed `.yuv` plus per-frame `rd_*.json` to
`outputs/02-video-compression/`. To look at the result:

```bash
ffmpeg -f rawvideo -pix_fmt yuv420p -s 416x240 \
  -i /work/outputs/02-video-compression/rec_ld_qp32.yuv frame_%02d.png
```

## The three model variants

| `--structure` | chunk | size | what it is |
|---|---:|---:|---|
| `ld` | 1 frame | 39 MB | low-delay, one frame at a time — the familiar design |
| `hts` | 8 frames | 325 MB | chunk-based, "small" |
| `htl` | 8 frames | 482 MB | chunk-based, "large" |

**The chunk is the whole idea of DCVC-UF.** `hts`/`htl` encode 8 frames into a
*single* latent and decode them together, which is where the speed comes from.
A consequence worth understanding before you misread the output: there is no
per-frame rate inside a chunk. `run_video.py` prints the chunk's bits divided by
8, so all 8 rows show the same bpp — that is not a bug, it is the design.

Compare the two on RaceHorses at qp 32 (measured):

| | avg bpp | avg YUV-PSNR |
|---|---:|---:|
| `ld`, 8 frames | 0.0914 | 31.52 dB |
| `hts`, 9 frames | 0.0621 | 29.64 dB |

Per-frame with `ld`: intra 0.2695 bpp / 33.15 dB, then inter frames at
0.034–0.133 bpp / ~31 dB. **On average an inter frame costs about a quarter of
the intra frame** — that gap is the temporal prior, and it is the thing to explain.

These are estimated rates, taken from the quantised symbols. An earlier version
used `forward()`'s noise-based `bpp`, which inflated them — most of all for the
cheap inter frames, where it had suggested "about half".

## Test data

`Dataset/RaceHorses_416x240_30.yuv` — 300 frames, YUV420 8-bit, 30 fps.
Raw YUV has no header, so every tool must be told `416x240 yuv420p` explicitly;
getting it wrong yields skewed images rather than an error.

## Two CPU details worth knowing

**Padding.** `test_video.py` pads to a multiple of 16, but that is only what it
reports — the CUDA proxy pads more internally. On the plain PyTorch path the
latent sits at W/32 and is then `pixel_unshuffle`d by 2, so the picture must be a
multiple of **64**. RaceHorses becomes 448×256 (~15 % more samples). Those samples
are coded but never displayed, so `run_video.py` normalises the rate by the
original 416×240 area — the honest figure, and the one comparable to upstream.

**Seeding the reference buffer.** `add_ref_feature_from_frame()` goes through the
CUDA proxy in eval mode. Its training branch is one line of plain PyTorch —
`F.pixel_unshuffle(frame, 8)` — and that is what `run_video.py` uses to hand the
intra reconstruction to the inter model. Without this you cannot start an inter
frame on CPU at all.

## Real bitstreams

Upstream codes inter frames only inside CUDA proxies. This repo ports all three
structures to the CPU:

```bash
make uf-video-bitstream
```

```python
import uf_video_codec as V                        # scripts/bitstream/
bs, recon = V.encode_sequence(i_net, p_net, "hts", 8, frames, qp_i, qp_p)
structure, recon = V.decode_sequence(i_net_dec, p_net_dec, bs)   # from the bytes alone
```

The structures are coded differently, and the port follows each proxy:

| structure | spatial prior | scales | written as |
|---|---|---|---|
| `ld` | 2 steps | from the hyperprior | one symbol array for the whole frame |
| `hts` | 4 steps, means only | from the hyperprior | one array for the whole chunk |
| `htl` | 4 steps, scales + means | re-predicted per step | four arrays, steps 3, 2, 1, 0 |

When the scales come from the hyperprior they are known before any latent is
decoded, so LD and HTS decode a whole frame or chunk in one call — part of why
they are fast. HTL must decode step by step.

On RaceHorses (64 frames) every frame decodes bit-exactly, and once fixed
per-packet bytes are removed, coding adds 0.1–0.4 % over the estimate. At the
lowest rate LD pays ~540 fixed bytes (64 packets) where HTS pays ~100 (9 packets):
**chunking amortises per-packet costs**, which shows up in real bytes. It does
not make HTS the better compressor on this sequence, though: at equal PSNR, HTS
needs about 30 % more bits than LD on average (roughly equal at the lowest rate,
about 2× at 33 dB). What chunking buys here is speed. Full numbers are in
`scripts/bitstream/README.md` and `scripts/experiments/README_EXPERIMENTS.md`.

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
| `ld`, 8 frames | 0.2297 | 31.52 dB |
| `hts`, 9 frames | 0.1124 | 29.64 dB |

Per-frame with `ld`: intra 0.4756 bpp / 33.15 dB, then inter frames at
0.16–0.26 bpp / ~31 dB. **The inter frames cost about half the intra frame** —
that gap is the temporal prior, and it is the thing to explain.

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

## Limitation

No real bitstream without CUDA — same as project 01. Speed is *not* the problem:
9 frames take about 6 s on a laptop CPU.

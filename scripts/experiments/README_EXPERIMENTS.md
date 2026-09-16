# Experiments (tasks 1–8)

Everything here runs from the repo root on the host; each script dispatches
itself into the right container.

```bash
bash scripts/experiments/run_all.sh          # tasks 1-8 + figures
bash scripts/experiments/run_all.sh 1 2      # just the image RD pair
```

| # | Script | Container | Cost |
|---|---|---|---|
| 1 | `exp1_dcvc_image_rd.py` | image-compression | seconds |
| 2 | `exp2_vtm_intra_rd.sh` | hybrid-vtm | ~4 min |
| 3 | `exp3_dcvc_video_rd.py` (64 frames, plus 8 matched to VTM) | video-compression | ~7 min |
| 4 | `exp4_vtm_inter_rd.sh` | hybrid-vtm | ~30 min (8 frames) |
| 5 | `exp5_adain_styles.sh` | style-transfer | ~1 min |
| 6 | `exp6_sr_compare.py` | super-resolution | ~20 s |
| 6b | `exp6b_perception_distortion.py` | super-resolution | ~30 s |
| 7 | `exp7_timing.py` via `run_timing.sh` | all five | ~4 min |
| 8 | `scripts/bitstream/uf_bitstream_demo.py` + `uf_video_bitstream_demo.py` | image- / video-compression | ~1 min + ~8 min |

Bulk intermediates (reconstructed `.yuv`, bitstreams) land in
`outputs/experiments/` and are git-ignored. **Figures and the RD numbers behind
them are committed to `results/`** — the full list of figures is in the root
README.

---

## Measurement rules

Four decisions make the curves comparable. Each one was a bug first.

**1. One source file, one colour space.** kodim19.png is converted to YUV420
once (`outputs/experiments/kodim19_420.yuv`) and *both* the DCVC and VTM image
experiments read that same file. Comparing an RGB-domain PSNR against a
YUV-domain one is not a comparison.

**2. One PSNR tool.** VTM prints its own YUV-PSNR; DCVC's `test_video.py` uses
`(6Y + U + V)/8`. They are different definitions and disagree by roughly half a
dB. Every experiment writes a reconstructed `.yuv` and calls
`yuv_psnr.py`, which implements the DCVC convention for everyone.

**3. Bit depth is detected, not assumed.** The VTM configs set
`InternalBitDepth: 10` and write the reconstruction at that depth — 10 bits
inside 16-bit words. Read as 8-bit it produces noise and a PSNR near 7 dB, which
looks like a broken codec rather than a broken reader. `yuv_psnr.py` detects the
container width from the file size and scales 16-bit samples down for the
comparison. The scripts also pass `--OutputBitDepth=8` so new runs are 8-bit
throughout.

**4. DCVC rate is measured the way the coder would, not the way training did.**
This one changes the conclusion. `forward_one_frame` estimates bits as
`get_prob_train(add_noise(y_res), scales)`, where `add_noise` adds
uniform(−0.5, 0.5) — the continuous relaxation that keeps training
differentiable. It is a variational upper bound, and at low rate it is badly
loose: a latent that quantises to exactly 0 and costs almost nothing still
carries entropy once noise is added, so the estimate has a floor.

`get_prob_train(v, s)` is the Gaussian mass over `[v−0.5, v+0.5]` — exactly a
quantisation bin — so feeding it the *rounded* value gives the arithmetic
coder's rate. `process_with_mask` defines `y_q = QuantFunc.apply(y_res) =
round(y_res)`, so `dcvc_yuv.eval_rate()` swaps `add_noise` for `round` and
reproduces the real symbols. Measured on kodim19:

| qp | train-mode bpp | eval-mode bpp | YUV-PSNR (unchanged) |
|---:|---:|---:|---:|
| 0 | 0.1898 | **0.0239** | 30.89 dB |
| 63 | 0.8981 | **0.7642** | 41.76 dB |

Uncorrected, DCVC-UF looked several dB *worse* than VTM at low rate — the
opposite of the published result, and an artefact of the measurement.

**Tasks 1 and 3 report estimated DCVC rates** (VTM's are always real bytes). Task 8
checks that estimate against real bitstreams written by the CPU port of upstream's
encoder: within 0.2–0.7 % for images, and within 0.1–0.4 % for video once fixed
per-packet bytes are set aside.

---

## Task 4 — two deviations from the brief, both deliberate

**Frame count: 8, not 64.** VTM is a reference encoder. Measured here on
416×240 random access, one 64-frame point costs 494 s at QP 42 and over an hour
at QP 0; the full sweep was heading past five hours for a curve whose shape is
already clear at 8 frames. The default is 8. `VTM_FRAMES=64 bash
scripts/experiments/run_all.sh 4` restores the brief if you want it overnight.

Task 3 (DCVC) still runs the full 64 frames — it is fast enough that there was
no reason to cut it.

**The frame counts are matched for the comparison figure.** Frames 0–7 and 0–63
of RaceHorses are different content, so a 64-frame neural curve plotted against
an 8-frame VTM curve would be comparing two sequences. `exp3` is therefore also
run with `--frames 8 --tag 8f`, and `plot_rd.py` picks the DCVC files whose
frame count matches VTM's. The 64-frame results remain the task-3 deliverable.

**QP list.** The brief asked for QP `0 15 30 45 63` for VTM inter, the same list
as task 3. VTM's QP is not DCVC's qp index: VTM QP 0 is near-lossless (measured
4.36 bpp on this sequence) and QP 63 falls far below any rate the neural codec
reaches, so that list yields a curve that barely shares an axis with anything
else. Both sets are run — `--tag requested` (`0 15 30 45 63`) and `--tag std`
(`22 27 32 37 42`, the JVET common-test QPs, which overlap the DCVC range) — and
`plot_rd.py` merges them into a single curve, because they are one codec sampled
at ten operating points, not two codecs.

QPs are encoded cheapest-first and `rd.json` is rewritten after every point, so
an interrupted run still yields a usable curve. `rebuild_rd.py` re-measures an
existing directory without re-encoding — useful when only the measurement
changed, which during this work it did twice.

---

## Results

### Task 1 + 2 — image, kodim19 (512×768)

| DCVC-UF-Intra qp | bpp | YUV-PSNR | | VTM intra QP | bpp | YUV-PSNR |
|---:|---:|---:|---|---:|---:|---:|
| 0 | 0.0239 | 30.89 | | 42 | 0.0729 | 33.00 |
| 15 | 0.0499 | 33.00 | | 37 | 0.1551 | 35.08 |
| 30 | 0.1103 | 35.17 | | 32 | 0.3506 | 37.70 |
| 45 | 0.2958 | 37.90 | | 27 | 0.6936 | 40.84 |
| 63 | 0.7642 | 41.76 | | 22 | 1.1735 | 44.03 |

At an equal 33.00 dB, DCVC-UF-Intra spends 0.0499 bpp against VTM's 0.0729 —
**about 31 % fewer bits**. The gap narrows to 5–8 % at the high-rate end.
(The paper reports 10.6 % averaged over all of Kodak; this is one image.)

### Task 3 — video, RaceHorses, first 64 frames

| qp | HTS bpp | HTS PSNR | LD bpp | LD PSNR |
|---:|---:|---:|---:|---:|
| 0 | 0.0041 | 25.05 | 0.0071 | 26.27 |
| 15 | 0.0107 | 27.27 | 0.0171 | 28.58 |
| 30 | 0.0283 | 29.45 | 0.0455 | 31.11 |
| 45 | 0.0797 | 31.63 | 0.1238 | 33.94 |
| 63 | 0.1932 | 33.29 | 0.3268 | 36.89 |

HTS codes 8 frames into one latent; LD codes one at a time. **On this sequence LD
is the more efficient of the two.** Interpolated to equal YUV-PSNR, HTS needs about
30 % more bits on average over the overlapping 26–33 dB range — roughly the same
at the lowest rate, growing to about 2× at 33 dB. Task 8's real bytes agree
(+30 %). What the chunk design buys here is speed (task 7) and amortised
per-packet costs (task 8), not compression. This is one sequence, 64 frames, with
intra and inter qp set equal: an observation, not a verdict on the models.

The 8-frame variants used for the VTM comparison (`rd_*_8f.json`) sit at higher
bpp across the board — the intra frame's cost is amortised over 8 frames instead
of 64, not a difference in the codec.

### Task 4 — VTM inter (RA), RaceHorses, first 8 frames

| QP | bpp | YUV-PSNR | set |
|---:|---:|---:|---|
| 0 | 4.3623 | 59.24 | requested |
| 15 | 1.1727 | 45.18 | requested |
| 22 | 0.5184 | 40.08 | std |
| 27 | 0.2828 | 36.57 | std |
| 30 | 0.2007 | 34.83 | requested |
| 32 | 0.1595 | 33.71 | std |
| 37 | 0.0867 | 31.19 | std |
| 42 | 0.0471 | 28.80 | std |
| 45 | 0.0339 | 27.50 | requested |
| 63 | 0.0059 | 20.73 | requested |

The two sets sample one curve. Note QP 0 at 4.36 bpp and QP 15 at 1.17 bpp: the
requested list reaches an order of magnitude past anything the neural codec is
asked for, which is why `plot_rd.py` uses a log rate axis.

**The 8-frame comparison understates DCVC-UF.** With 1 intra + 7 inter frames the
intra frame dominates the rate, so the temporal prior barely gets to pay for
itself. DCVC-UF's own numbers show the effect: HTS at qp 0 costs 0.0113 bpp over
8 frames and 0.0041 bpp over 64, purely from amortising the intra frame. The
published DCVC-UF-vs-VTM result uses long sequences. Treat the video figure as
"the pipeline works end to end and both codecs are measured identically", not as
a verdict on which codec is better.

### Task 5 — AdaIN across styles

Figures `results/figures/adain_style_grid.png` (eight styles, each applied to
kodim19 and to a RaceHorses frame) and `results/figures/adain_alpha_sweep.png`
(alpha 0 → 1). There is no metric: style transfer has no reference image to score
against, so the result is the picture.

### Task 6 — x4 super-resolution on kodim19

| method | PSNR | SSIM | time (single cold run) |
|---|---:|---:|---:|
| nearest | 23.17 dB | 0.655 | 0.001 s |
| bilinear | 23.44 dB | 0.659 | 0.002 s |
| bicubic | 23.71 dB | 0.677 | 0.002 s |
| lanczos | **23.79 dB** | **0.683** | 0.004 s |
| Real-ESRGAN | 23.14 dB | 0.662 | 4.48 s |

**Real-ESRGAN loses on both metrics and is about three orders of magnitude slower —
and still looks far better.** The times are single cold runs; task 7 measures
Real-ESRGAN warm at 2.9 s. Task 6b below takes the metric paradox apart.

### Task 6b — why fidelity metrics and the eye disagree

`exp6b_perception_distortion.py`, figure `results/figures/sr_perception_distortion.png`.
Four measurements, four separate reasons.

**A. A perceptual metric ranks it first.**

| method | PSNR | SSIM | LPIPS (lower better) | sharpness |
|---|---:|---:|---:|---:|
| nearest | 23.17 dB | 0.6552 | **0.4166** | 147 |
| bilinear | 23.44 dB | 0.6589 | **0.5310** | 30 |
| bicubic | 23.71 dB | 0.6766 | **0.5288** | 40 |
| lanczos | 23.79 dB | 0.6826 | **0.5362** | 44 |
| realesrgan | 23.14 dB | 0.6623 | **0.2412** | 245 |

LPIPS puts Real-ESRGAN **2.2× ahead** of lanczos
while PSNR puts it last. The metrics are not noisy versions of each other; they
ask different questions.

**B. PSNR is phase-sensitive.** Take the untouched original and shift it one
pixel: **22.86 dB**, *below* Real-ESRGAN's
23.14 dB. A pixel-perfect photograph, moved one pixel,
scores worse than the "bad" reconstruction. PSNR asks whether each pixel is
where it was, not whether the image looks like the scene. Texture that is
statistically right but half a pixel off is punished exactly like texture that
is wrong.

**C. The metric rewards destroying detail.** Gaussian-blur the Real-ESRGAN output:

| sigma | PSNR | SSIM | sharpness |
|---:|---:|---:|---:|
| 0.0 | 23.14 dB | 0.6623 | 245 |
| 0.3 | 23.24 dB | 0.6637 | 218 |
| 0.5 | 23.43 dB | 0.6699 | 167 |
| 0.8 | 23.64 dB | 0.6771 | 104 |
| 1.2 | 23.62 dB | 0.6740 | 61 |
| 1.6 | 23.42 dB | 0.6609 | 40 |

PSNR **peaks at sigma 0.8** — blurring gains
0.50 dB while removing
57% of the gradient energy.
SSIM peaks there too, so this is not an MSE quirk.

**D. Across methods, PSNR is ranking smoothness.**
corr(PSNR, sharpness) = **-0.83**.

**Why this happens.** Super-resolution is ill-posed: many high-res images
downsample to the same low-res one. The MSE-optimal estimator is the posterior
mean E[HR | LR] — an *average* over every plausible texture — and the average of
many plausible textures is a blur. So the PSNR-maximising answer is blurry by
construction. A GAN instead matches the *distribution* of natural images, which
necessarily moves it off the posterior mean; Blau & Michaeli (CVPR 2018) proved
this is a genuine tradeoff, not an engineering failure.

**One confound of our own.** Real-ESRGAN was trained on a second-order
real-world degradation model (blur + noise + JPEG + resize). This benchmark feeds
it a clean bicubic downsample — out of its training distribution. A
bicubic-trained model (EDSR, or ESRGAN's PSNR-oriented variant) would score
better here without looking better.

**For the course:** never show the metric table without the pictures — compare
the railing in `results/figures/sr_comparison.png`.

### Task 7 — inference time per project

Figure `results/figures/inference_time.png`. Measured on CPU with warm-up runs
discarded and the median of the timed runs reported; model construction,
checkpoint loading and file I/O are outside the timed region.

| project | input | median time | output Mpx/s |
|---|---|---:|---:|
| 01 Image (DCVC-UF-Intra) | kodim19 512x768 | **1.895 s** | 0.207 |
| 02 Video (DCVC-UF HTS) | RaceHorses 416x240, chunk of 8 | **0.141 s** | 0.708 |
| 03 Hybrid (VTM intra) | kodim19 512x768, QP 32 | **40.400 s** | 0.010 |
| 04 Style (AdaIN) | kodim19 512x768 | **2.639 s** | 0.149 |
| 05 Super-res (Real-ESRGAN) | 128x128 -> 512x512 | **2.906 s** | 0.090 |

Two things this measurement exists to avoid getting wrong:

**Warm-up.** The first pass through a PyTorch graph pays for oneDNN algorithm
selection and allocator growth. Task 6 measured Real-ESRGAN cold at 4.48 s; warm
it is 2.91 s — a 1.5x error from timing setup instead of inference.

**Seconds are not comparable on their own.** The projects do not share an input
size: 512×768 for the image work, 416×240 per frame for video, a 128×128 crop
for super-resolution. That is why the figure carries a throughput panel as well;
reading only the seconds would rank a small crop against a full image.

DCVC-UF video codes a 416×240 frame in **141 ms** — about 7 fps on a laptop CPU
for a neural video codec, which is what "ultra-fast" buys. VTM needs 40 s for one
intra frame: per pixel it is ~73× slower, because a reference encoder exists to
define correctness, not to run quickly.

### Task 8 — real DCVC-UF bitstreams (image and video)

Figure `results/figures/uf_real_bitstream.png`. Bytes on disk, decoded back
bit-exactly; see `scripts/bitstream/README.md` for how the CPU codec was derived.

| qp | estimated bpp | actual bpp | payload vs est. | PSNR | actual, skip 0.15 | PSNR | exact |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.0239 | 0.0243 | +0.68% | 30.89 dB | 0.0239 | 30.85 dB | yes |
| 15 | 0.0499 | 0.0503 | +0.42% | 33.00 dB | 0.0496 | 32.97 dB | yes |
| 30 | 0.1103 | 0.1109 | +0.25% | 35.17 dB | 0.1101 | 35.15 dB | yes |
| 45 | 0.2958 | 0.2967 | +0.20% | 37.90 dB | 0.2959 | 37.89 dB | yes |
| 63 | 0.7642 | 0.7678 | +0.44% | 41.76 dB | 0.7668 | 41.76 dB | yes |

This is the check the measurement rules above were missing: rule 4 (rate from
quantised symbols) is now confirmed against real bytes, within 0.7 %. The noise
proxy it replaced would have been 7.9× the real rate at qp 0. And the task 1 + 2
conclusion survives: at 33.00 dB DCVC-UF needs 31 % fewer bits than VTM, real
bytes on both sides.

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

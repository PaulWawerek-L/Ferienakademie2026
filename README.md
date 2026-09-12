# Ferienakademie 2026 — Learned Compression & Image Restoration

## Projects

1. **Learned Image Compression – DCVC-UF**
   Core concepts: Analysis/Synthesis Transform, Hyperprior, Entropy Model

2. **Learned Video Compression – DCVC-UF**
   Core concepts: Motion, Temporal Prior, Video Coding

3. **Hybrid Codecs – VTM + Learned Module**
   Core concepts: Traditional + Neural Coding

4. **Style Transfer – AdaIN**
   Core concepts: Feature Statistics, Representation, Normalization

5. **Super Resolution – Real-ESRGAN**
   Core concepts: GAN, Perceptual Loss, Image Restoration

---

## Quick start

**Only prerequisite: Docker** (Desktop, or Colima on macOS). No conda, no CUDA,
no Python on the host — everything runs in containers.

```bash
git clone https://github.com/ZongxieCHEN/Ferienakademie2026.git
cd Ferienakademie2026

make sources    # clone the 4 upstream repos into third_party/   ~1 min, 75 MB
make base       # shared PyTorch base image                      ~5 min, once
make build      # the four project images                        ~3 min (VTM compiles)
make weights    # pretrained weights -> ./weights/               ~2 min, 160 MB
make smoke      # every project's smoke test, pass/fail table    ~70 s
```

`make sources` is not optional on a fresh clone: `third_party/` is git-ignored,
and the containers bind-mount it. Without it the model code is missing.

After `make weights`, projects 01 and 02 still need the four DCVC-UF checkpoints,
which upstream hosts on OneDrive with no scriptable URL — `make weights` prints
the instructions. Until then those two projects fall back to a checkpoint-free
preflight, so `make smoke` stays green either way.

### Reproduce every figure and number

```bash
make experiments                            # all six tasks + every figure
bash scripts/experiments/run_all.sh 1 2      # or a subset, by task number
```

Roughly 45 minutes end to end, dominated by task 4 (VTM is a reference encoder).
It rewrites everything under `results/`. Details, the measurement rules, and the
full result tables: [`scripts/experiments/README_EXPERIMENTS.md`](scripts/experiments/README_EXPERIMENTS.md).

### Work inside a project

```bash
make shell-style-transfer     # interactive shell in the AdaIN container
make smoke-super-resolution   # just one smoke test
```

Figures and RD data land in `results/` (committed); bulk intermediates —
reconstructed `.yuv`, bitstreams, per-frame PNGs — land in `outputs/`
(git-ignored). Both are on the host, so results open straight from
Finder/Explorer with nothing to copy out of a container.

---

## Where the upstream source lives

`make sources` clones the four upstream codebases into `third_party/`:

```
third_party/DCVC              projects 01 + 02   ->  mounted at /opt/DCVC
third_party/AdaIN             project 04         ->  /opt/AdaIN
third_party/Real-ESRGAN       project 05         ->  /opt/Real-ESRGAN
third_party/VVCSoftware_VTM   project 03         ->  /opt/vtm/src  (read-only)
```

The Dockerfiles also clone them, so the images work on their own — but
`compose.yaml` bind-mounts these directories *over* the in-image copies. That
matters for a course:

- the model code is **readable and editable on the host**, in your own editor;
- `.vscode/settings.json` puts them on the analysis path, so go-to-definition on
  `from src.models.image_model import DMCI` actually lands somewhere;
- an edit takes effect on the next `docker compose run` — **no rebuild**.

`third_party/` is git-ignored: these are upstream checkouts pinned by
`scripts/fetch_sources.sh`, not our code.

---

## Results

Figures and the RD numbers behind them are committed under `results/`:

| Figure | What |
|---|---|
| `results/figures/task1_rd_dcvc_image.png` | DCVC-UF-Intra RD, kodim19 |
| `results/figures/task2_rd_vtm_intra.png` | VTM all-intra RD, kodim19 |
| `results/figures/task3_rd_dcvc_video.png` | DCVC-UF video RD, RaceHorses 64 frames |
| `results/figures/task4_rd_vtm_inter.png` | VTM inter RD, RaceHorses |
| `results/figures/compare_image_kodim19.png` | tasks 1 + 2 overlaid |
| `results/figures/compare_video_racehorses.png` | tasks 3 + 4 overlaid |
| `results/figures/adain_style_grid.png` | 8 styles x 2 contents |
| `results/figures/adain_alpha_sweep.png` | alpha 0 -> 1 |
| `results/figures/sr_comparison.png` | Real-ESRGAN vs classical resampling |
| `results/figures/sr_perception_distortion.png` | why PSNR/SSIM and the eye disagree |

`outputs/` holds the bulk intermediates (reconstructed `.yuv`, bitstreams,
per-frame PNGs) and **is** git-ignored — it runs to hundreds of MB.

Reproduce everything with `bash scripts/experiments/run_all.sh`; the methodology,
its three measurement traps, and the full result tables are in
[`scripts/experiments/README_EXPERIMENTS.md`](scripts/experiments/README_EXPERIMENTS.md).

---

## How the environments are organised

**One environment per project — but layered, not five independent stacks.**

The dependency conflicts are real, not stylistic:

| Project | Why it cannot share an environment |
|---|---|
| 01 + 02 DCVC-UF | Builds a pybind11 C++ entropy coder against a specific Python/torch |
| 03 VTM | Pure C++/CMake. No Python, no PyTorch at all |
| 04 AdaIN | Archived Jan 2024. Its pins (`torch==1.13.1` + `torchvision==0.4.0`) are an impossible pair, and torch 1.13.1 has no arm64 wheel |
| 05 Real-ESRGAN | `basicsr==1.4.2` imports `torchvision.transforms.functional_tensor`, deleted in torchvision 0.17 — needs a patch |

But ~1.5 GB of that is the same PyTorch install, so all Python projects derive
from one `ferienakademie/base:cpu` image and add only a thin layer:

```
ferienakademie/base:cpu  (python 3.12 + torch 2.12.1+cpu + torchvision + cv2/ffmpeg)
├── ferienakademie/dcvc:cpu         projects 01 + 02   (same repo, same model)
├── ferienakademie/adain:cpu        project 04
└── ferienakademie/realesrgan:cpu   project 05

ferienakademie/vtm:latest           project 03  (debian + VTM binaries, no torch)
```

Students download the base once instead of four times — the difference between
~2 GB and ~6 GB on conference wifi.

**Projects 01 and 02 deliberately share one image.** DCVC-UF-Intra (image
coding) is the intra-frame codec of the same DCVC-UF model in the same repo.
Five projects, four environments.

---

## What actually runs on a laptop (no GPU)

These images are CPU-only and multi-arch (`linux/amd64` + `linux/arm64`), because
students run them on their own machines. That has one hard consequence:

Measured on an M-series Mac (arm64, 10 CPUs, 8 GB to the Docker VM) via `make smoke`:

| Project | On CPU | Smoke test |
|---|---|---|
| 01 Image compression | ⚠️ **Forward pass only** — reconstruction, *estimated* bpp, PSNR. No real bitstream. | ✅ 11 s |
| 02 Video compression | ⚠️ Same. Surprisingly fast: 9 frames in ~6 s. | ✅ 9 s |
| 03 Hybrid / VTM | ✅ Fully works. VTM is CPU software anyway (just slow — it is a reference encoder). | ✅ 39 s |
| 04 Style transfer | ✅ Fully works, seconds per image. | ✅ 5 s |
| 05 Super resolution | ✅ Works; x4 on a full image takes minutes, so start with a crop. | ✅ 7 s |

Reference points from that run:

- **DCVC-UF-Intra** on kodim19 (512×768), a real rate–distortion sweep:
  0.195 bpp / 26.30 dB → 0.485 bpp / 33.25 dB → 1.009 bpp / 38.06 dB.
- **DCVC-UF video** on RaceHorses, qp 32: intra frame 0.476 bpp / 33.15 dB, then
  inter frames at 0.16–0.26 bpp / ~31 dB (LD) — the P-frames cost roughly half the
  intra frame, which is the temporal prior doing its job.
- **VTM** encoded 4 frames of RaceHorses at QP 37 to 7165 bytes (0.1435 bpp,
  YUV-PSNR 31.45 dB) in 40 s, and the decode was bit-exact against the encoder
  reconstruction.
- **Real-ESRGAN** took 7 s for 128×128 → 512×512.

Scale expectations from there: VTM is ~10 s per frame, so a 300-frame sequence is
an overnight job. DCVC-UF is the opposite — ~0.6 s per frame on a laptop CPU,
which is what the paper's "ultra-fast" claim buys you.

### The DCVC-UF limitation, precisely

Upstream builds two extensions. Only the first can exist without a GPU:

| Extension | What it is | CPU? |
|---|---|---|
| `MLCodec_extensions_cpp` | rANS entropy coder, plain C++/pybind11 | ✅ built in the image |
| `inference_extensions_cuda` | CUTLASS fused kernels | ❌ CUDA-only, skipped |

Without the second, `model.compress()` / `model.decompress()` raise
`NotImplementedError` — there is no CPU fallback in upstream. `model.forward()`
is ordinary PyTorch and runs fine.

**This is survivable for teaching.** `forward()` runs the analysis transform,
the hyperprior, and the entropy model, and returns the entropy model's estimated
bpp — which is exactly what concepts 1–3 of project 01 are about. What students
lose is the real bitstream, i.e. the gap between estimated and actual rate.

If you get access to a GPU machine, rebuild with a CUDA base and both extensions
and the full path works — nothing else in this repo needs to change.

---

## Pretrained weights

`make weights` downloads and **sha256-verifies** AdaIN and Real-ESRGAN.

DCVC-UF checkpoints must be fetched by hand: upstream hosts them on OneDrive,
which has no stable direct-download URL. `make weights` prints the instructions.
Put the four `.pth.tar` files in `weights/dcvc/`:

```
cvpr2026_image.pth.tar        169 MB   image codec / intra frames
cvpr2026_video_ld.pth.tar      39 MB   low-delay, codes 1 frame at a time
cvpr2026_video_hts.pth.tar    325 MB   chunk of 8, "small"
cvpr2026_video_htl.pth.tar    482 MB   chunk of 8, "large"
```

Projects 01 and 02 fall back to a checkpoint-free environment preflight until
these exist, so `make smoke` stays green either way.

Weights are not baked into the images — they carry their own licenses, would add
~160 MB to every layer, and change independently of the code. `weights/` is
bind-mounted, so one download serves every project and survives `docker compose down`.

---

## Layout

```
docker/          Dockerfiles — infrastructure, maintained by the TA
projects/        per-project READMEs, smoke tests, student code
scripts/         fetch_sources.sh, fetch_weights.sh, smoke_all.sh, yuv2png.sh
scripts/experiments/   the six RD / comparison experiments and their plots
third_party/     upstream checkouts, bind-mounted into the containers (git-ignored)
Dataset/         kodim19.png, RaceHorses_416x240_30.yuv (300 frames, YUV420 8-bit)
weights/         downloaded checkpoints (git-ignored)
results/         figures and RD data — committed
outputs/         bulk intermediates: .yuv, bitstreams (git-ignored)
```

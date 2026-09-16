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

**Only prerequisite: Docker** (Desktop + WSL2 on Windows, or Colima on macOS). No conda, no CUDA,
no Python on the host — everything runs in containers.

*Note (Windows): Setup in WSL2 is recommended. To activate docker in WSL2, enable "integration with my default WSL distro" (tested on Ubuntu 26.04) under* ``Docker Desktop>Settings>Resources>WSL integration``

```bash
git clone https://github.com/ZongxieCHEN/Ferienakademie2026.git
cd Ferienakademie2026

make sources    # clone the 4 upstream repos into third_party/   ~1 min, 75 MB
make base       # shared PyTorch base image                      ~5 min, once
make build      # the four project images                        ~3 min (VTM compiles)
make weights    # pretrained weights -> ./weights/               ~2 min, 160 MB
make smoke      # every project's smoke test, pass/fail table    ~100 s
```

`make sources` is not optional on a fresh clone: `third_party/` is git-ignored,
and the containers bind-mount it. Without it the model code is missing.

After `make weights`, projects 01 and 02 still need the four DCVC-UF checkpoints,
which upstream hosts on OneDrive with no scriptable URL — `make weights` prints
the instructions. Until then those two projects fall back to a checkpoint-free
preflight, so `make smoke` stays green either way.

### Reproduce every figure and number

```bash
make experiments                            # tasks 1-8 + every figure
bash scripts/experiments/run_all.sh 1 2      # or a subset, by task number
```

About 45 minutes end to end — measured at 44.7 min from a fresh clone on an
M-series Mac — dominated by task 4 (VTM is a reference encoder). It rewrites
everything under `results/`, keeps going if one step fails, and exits non-zero
with a list of the failed steps. Details, the measurement rules, and the
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

*Note (Linux): the containers run as root, so with Docker Engine on Linux the files
they write to `outputs/`, `results/` and `weights/` belong to root. To edit or delete
them on the host, take them back with* ``sudo chown -R $USER: outputs results weights``

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
| `results/figures/inference_time.png` | inference time + throughput per project |
| `results/figures/uf_real_bitstream.png` | real DCVC-UF bytes vs estimate vs VTM |
| `results/figures/uf_video_real_bitstream.png` | real DCVC-UF video bytes vs estimate, and where the extra bytes go |

`outputs/` holds the bulk intermediates (reconstructed `.yuv`, bitstreams,
per-frame PNGs) and **is** git-ignored — it runs to hundreds of MB.

Reproduce everything with `bash scripts/experiments/run_all.sh`; the methodology,
its four measurement rules, and the full result tables are in
[`scripts/experiments/README_EXPERIMENTS.md`](scripts/experiments/README_EXPERIMENTS.md).

---

## How the environments are organised

**One environment per project — but layered, not five independent stacks.**

The dependency conflicts are real, not stylistic:

| Project | Why it cannot share an environment |
|---|---|
| 01 + 02 DCVC-UF | Builds a pybind11 C++ entropy coder against a specific Python/torch |
| 03 VTM | Pure C++/CMake. No Python, no PyTorch at all |
| 04 AdaIN | Archived Jan 2024. Its pins (`torch==1.13.1` + `torchvision==0.4.0`) are an impossible pair, and torch 1.13.1 has no wheel for Python 3.12 (they stop at cp311) |
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

These images are CPU-only, because students run them on their own machines, and
nothing in them is architecture-specific: they are written to build on both
`linux/arm64` and `linux/amd64`. **So far they have been verified end to end only on
arm64** (an M-series Mac); Windows/WSL2 setup notes above come from a collaborator.

Measured on an M-series Mac (arm64, 10 CPUs, 8 GB to the Docker VM) via `make smoke`:

| Project | On CPU | Smoke test |
|---|---|---|
| 01 Image compression | ✅ **Real bitstreams** via a CPU port of upstream's CUDA-only encoder/decoder (see below). | ✅ 21 s |
| 02 Video compression | ✅ **Real bitstreams** (LD, HTS, HTL) via a CPU port of upstream's CUDA-only inter coding. ~0.14 s/frame to encode. | ✅ 24 s |
| 03 Hybrid / VTM | ✅ Fully works. VTM is CPU software anyway (just slow — it is a reference encoder). | ✅ 39 s |
| 04 Style transfer | ✅ Fully works, seconds per image. | ✅ 6 s |
| 05 Super resolution | ✅ Works; x4 on a full image takes minutes, so start with a crop. | ✅ 7 s |

Reference points from that run:

- **DCVC-UF-Intra** on kodim19 (512×768, YUV420), real bitstreams:
  0.0243 bpp / 30.89 dB → 0.1109 bpp / 35.17 dB → 0.7678 bpp / 41.76 dB, each
  decoded back bit-exactly.
- **DCVC-UF video** on RaceHorses, qp 32 (LD, estimated rate): intra frame
  0.270 bpp / 33.15 dB, then inter frames at 0.034–0.133 bpp / ~31 dB — on average
  about a quarter of the intra frame, which is the temporal prior doing its job.
- **VTM** encoded 4 frames of RaceHorses at QP 37 to 7165 bytes (0.1435 bpp,
  YUV-PSNR 31.45 dB) in 40 s, and the decode was bit-exact against the encoder
  reconstruction.
- **Real-ESRGAN**: 128×128 → 512×512 in ~2.9 s of warm inference (task 7); the
  7 s smoke test also loads the model.

Scale expectations from there: VTM is ~10 s per frame, so a 300-frame sequence is
an overnight job. DCVC-UF is the opposite — ~0.14 s per frame on a laptop CPU
(HTS, task 7), which is what the paper's "ultra-fast" claim buys you.

### Real bitstreams on CPU

Upstream's DCVC-UF writes a bitstream only through `inference_extensions_cuda`
(CUTLASS kernels): `model.compress()` / `decompress()` raise `NotImplementedError`
without a GPU. This repo ports that path to the CPU, for the image model **and
all three video structures** (LD, HTS, HTL):

```bash
make uf-bitstream          # kodim19 -> .bin -> decoded back, vs estimate and vs VTM
make uf-video-bitstream    # RaceHorses, 64 frames, HTS + LD -> .bin -> decoded back
```

The weights, CDF tables and rANS coder are upstream's; only the orchestration the
CUDA proxies perform is re-implemented, in
[`uf_codec.py`](scripts/bitstream/uf_codec.py) (image) and
[`uf_video_codec.py`](scripts/bitstream/uf_video_codec.py) (video). Each is
verified three ways: bit-exact decode from separate model objects (for video,
every frame — which also proves the reference buffer stayed in step), a flipped
byte changes the output, and at `skip_thres = 0` the reconstruction is identical
to upstream's own `forward()`. `make smoke` runs these checks for the image model,
LD and HTS; HTL passes them too but is left out of smoke for its 482 MB checkpoint.

What the real bytes showed:

- image: arithmetic coding adds only **0.2–0.7 %** to the entropy model's estimate;
- at equal 33.00 dB DCVC-UF needs **31 % fewer bits than VTM**, now measured in
  bytes on both sides;
- image: encode ≈ 1.8 s, decode ≈ 1.0 s, ~1.1 GB peak memory on a laptop CPU;
- video: after removing fixed per-packet bytes, coding again adds only 0.1–0.4 %;
  LD pays ~540 fixed bytes over 64 frames where HTS pays ~100 — chunking
  amortises per-packet costs, visible in real bytes at low rate;
- video: ≈ 0.14 s/frame to encode and 0.08 s/frame to decode, ~2.2 GB peak.

One limit: the streams round-trip through these decoders but are **not** expected
to be readable by upstream's CUDA decoder (different container; scale indices
computed in float32 where the proxies use float16).

Details, measured tables and the derivation:
[`scripts/bitstream/README.md`](scripts/bitstream/README.md).

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

Task 6b additionally downloads LPIPS's AlexNet (233 MB) into `weights/torch/` on
its first run; nothing to do by hand.

Weights are not baked into the images — they carry their own licenses, run to
hundreds of MB, and change independently of the code. `weights/` is
bind-mounted, so one download serves every project and survives `docker compose down`.

---

## Layout

```
docker/          Dockerfiles — infrastructure, maintained by the TA
projects/        per-project READMEs, smoke tests, student code
scripts/         fetch_sources.sh, fetch_weights.sh, smoke_all.sh, yuv2png.sh
scripts/experiments/   tasks 1-8: RD / comparison / timing experiments and their plots
scripts/bitstream/     CPU codecs that write real DCVC-UF bitstreams, and their self-tests
third_party/     upstream checkouts, bind-mounted into the containers (git-ignored)
Dataset/         kodim19.png, RaceHorses_416x240_30.yuv (300 frames, YUV420 8-bit)
weights/         downloaded checkpoints (git-ignored)
results/         figures and RD data — committed
outputs/         bulk intermediates: .yuv, bitstreams (git-ignored)
```

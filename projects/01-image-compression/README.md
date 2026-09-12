# 01 — Learned Image Compression (DCVC-UF-Intra)

**Concepts:** Analysis/Synthesis Transform · Hyperprior · Entropy Model

Upstream: [microsoft/DCVC](https://github.com/microsoft/DCVC) (DCVC-UF, CVPR 2026).
The image codec is `DMCI` — the intra-frame codec of the DCVC-UF video model,
so this project and project 02 share one repo and one container image.

## Run

```bash
make smoke-image-compression       # RD sweep over kodim19
make shell-image-compression
python /work/projects/01-image-compression/run_image.py --rate-num 12
```

`run_image.py` sweeps the qp range and writes one reconstruction per rate point
plus `rd.json` to `outputs/01-image-compression/`. Measured on kodim19 (512×768):

| qp | est. bpp | PSNR-RGB |
|---:|---------:|---------:|
| 0  | 0.1953   | 26.30 dB |
| 21 | 0.2545   | 29.33 dB |
| 42 | 0.4853   | 33.25 dB |
| 63 | 1.0090   | 38.06 dB |

`preflight.py` is the fallback: it verifies the environment with random weights
and needs no checkpoint, which is what the smoke test runs before you have done
the manual OneDrive download.

## Where the concepts live in the code

`src/models/image_model.py`, `DMCI.forward_one_frame()`:

| Concept | Code |
|---|---|
| Analysis transform | `self.enc(x, curr_q_enc)` → latent `y` |
| Hyperprior | `self.hyper_enc(y)` → `z`, quantise, `self.hyper_dec(z_hat)` → `params` |
| Entropy model | `self.forward_prior_4x(...)` → `scales_hat`; `get_y_bits` / `get_z_bits` → `bpp` |
| Synthesis transform | `self.dec(y_hat, curr_q_dec)` → `x_hat` |

`qp` is an integer index in `0 .. DMCI.qp_num()-1` — one model covers the whole
bitrate range, so a rate–distortion curve is a loop over `qp`, not four models.

## CPU limitation

`compress()` / `decompress()` need `inference_extensions_cuda` (CUTLASS, CUDA-only)
and raise `NotImplementedError` here. `forward()` gives you the reconstruction and
the entropy model's **estimated** bpp, which covers all three concepts above.
What you cannot measure without a GPU is the gap between estimated and actual rate.

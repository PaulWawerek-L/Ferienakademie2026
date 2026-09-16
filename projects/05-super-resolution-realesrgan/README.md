# 05 — Super Resolution (Real-ESRGAN)

**Concepts:** GAN · Perceptual Loss · Image Restoration

Upstream: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).

## Run

```bash
make smoke-super-resolution    # x4 on a 128x128 crop
make shell-super-resolution
```

```bash
cd /opt/Real-ESRGAN && python inference_realesrgan.py \
  -n RealESRGAN_x4plus \
  --model_path /work/weights/realesrgan/RealESRGAN_x4plus.pth \
  -i <input dir> -o /work/outputs/05-super-resolution \
  --outscale 4 --fp32
```

`--fp32` is the conservative choice, not a requirement. Measured on an arm64 Mac
with torch 2.12, half precision runs and is about 1.9× faster (0.39 s vs 0.73 s
for 64×64 → 256×256); fp16 support on CPUs varies by hardware and PyTorch build,
though, and it has not been checked on amd64. Try it without `--fp32` if speed
matters. Use `--tile 256` if you hit memory limits on a large image.

## CPU cost

kodim19 is 512×768 (portrait); x4 on the full image means a 2048×3072 output.
That is 24× the pixels of the 128×128 smoke-test crop, which takes 2.9 s warm, so
expect on the order of a minute and several GB of memory on a laptop CPU (an
extrapolation, not a measurement). Start with crops; scale up once the pipeline
works.

## Dependency note

`basicsr==1.4.2` (its last release) imports
`torchvision.transforms.functional_tensor`, a private module deleted in
torchvision 0.17. `rgb_to_grayscale` moved to the public
`torchvision.transforms.functional`, so the Dockerfile rewrites that one import
after install. Without the patch, every `import basicsr` fails.

`gfpgan` / `facexlib` are deliberately not installed — they are only needed for
`--face_enhance`, which this project does not use.

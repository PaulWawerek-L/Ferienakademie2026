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

`--fp32` matters: the default half-precision path has poor CPU kernel coverage.
Use `--tile 256` if you hit memory limits on a large image.

## CPU cost

x4 on the full 768×512 kodim19 means a 3072×2048 output and several minutes on a
laptop. Start with crops; scale up once the pipeline works.

## Dependency note

`basicsr==1.4.2` (its last release) imports
`torchvision.transforms.functional_tensor`, a private module deleted in
torchvision 0.17. `rgb_to_grayscale` moved to the public
`torchvision.transforms.functional`, so the Dockerfile rewrites that one import
after install. Without the patch, every `import basicsr` fails.

`gfpgan` / `facexlib` are deliberately not installed — they are only needed for
`--face_enhance`, which this project does not use.

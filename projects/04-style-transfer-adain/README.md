# 04 — Style Transfer (AdaIN)

**Concepts:** Feature Statistics · Representation · Normalization

Upstream: [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN)
(archived Jan 2024).

## Run

```bash
make smoke-style-transfer      # kodim19 (content) x RaceHorses frame (style)
make shell-style-transfer
```

```bash
cd /opt/AdaIN && python test.py \
  --content /work/Dataset/kodim19.png --style <style.png> \
  --vgg /work/weights/adain/vgg_normalised.pth \
  --decoder /work/weights/adain/decoder.pth \
  --output /work/outputs/04-style-transfer --alpha 1.0
```

`--alpha` interpolates between content and style — sweeping it is the cheapest
way to see what AdaIN actually does.

## The idea, in one function

`net.py`, `adaptive_instance_normalization()`: normalise the content feature to
zero mean / unit variance per channel, then re-scale by the **style** feature's
per-channel mean and std. No learned parameters — style transfer as a statistics
swap. That single function is the whole point of the project.

## Dependency note

Upstream `requirements.txt` is **not** installed and should not be: it pins
`torch==1.13.1` with `torchvision==0.4.0` (an impossible pair — 0.4.0 belongs to
torch 1.2), includes the bogus `pkg-resources==0.0.0`, and torch 1.13.1 has no
arm64 wheel. The model code itself is plain `nn.Sequential` and runs unmodified
on current torch; only the pins were rotten.

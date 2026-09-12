# 03 — Hybrid Codecs (VTM + learned module)

**Concepts:** Traditional + Neural Coding

VTM is the VVC reference software: [VVCSoftware_VTM](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM).
This is the only project with no PyTorch — it is pure C++, in its own image.

## Run

```bash
make smoke-hybrid-vtm          # encode + decode 4 frames, verify bit-exactness
make shell-hybrid-vtm
```

Binaries are on `PATH` as `EncoderApp` / `DecoderApp`; configs are in `$VTM_CFG`
(`encoder_intra_vtm.cfg`, `encoder_lowdelay_vtm.cfg`, `encoder_randomaccess_vtm.cfg`).

```bash
EncoderApp -c $VTM_CFG/encoder_randomaccess_vtm.cfg \
  -i /work/Dataset/RaceHorses_416x240_30.yuv \
  --SourceWidth=416 --SourceHeight=240 --InputBitDepth=8 --FrameRate=30 \
  --FramesToBeEncoded=4 --QP=37 \
  -b out.bin -o rec.yuv
```

VTM is a *reference* encoder — correctness first, speed never. Expect seconds
per frame. Use few frames and small QP sweeps.

## Build note (arm64)

VTM supports arm64, but both of its architecture guards test
`CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64"` — the macOS spelling. Linux reports
`aarch64`, so on an arm64 Linux container the guards miss, `-msse4.1` gets added,
and the build fails. The Dockerfile patches both guards to accept either spelling.
If you ever build VTM outside this image on an ARM Linux box, you will hit the
same thing.

## Comparison baseline

DCVC-UF's published BD-rate numbers are against **VTM-17.0**. This image defaults
to VTM-23.14; set `VTM_TAG=VTM-17.0` in the environment and rebuild if you need
numbers comparable to the paper.

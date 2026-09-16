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

VTM is a *reference* encoder — correctness first, speed never. Measured here:
about 40 s for one 512×768 intra frame at QP 32, and roughly 10 s per 416×240
random-access frame at QP 37 (up to a minute at very low QP). Use few frames and
small QP sweeps.

## Build note (arm64)

VTM supports arm64, but its architecture guards — five of them, across three
CMakeLists — test `CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64"`, the macOS spelling.
Linux reports `aarch64`, so on an arm64 Linux container the guards miss,
`-msse4.1` gets added and the x86 SIMD sources are compiled, and the build fails.
`docker/vtm/patch-arm64.sh` patches every guard, found by search.
If you ever build VTM outside this image on an ARM Linux box, you will hit the
same thing.

## Comparison baseline

DCVC-UF's published BD-rate numbers are against **VTM-17.0**. This image defaults
to VTM-23.14; set `VTM_TAG=VTM-17.0` in the environment and rebuild if you need
numbers comparable to the paper. **Not tested with VTM-17.0**: an older tree may
spell its architecture guards differently, in which case the arm64 patch reports
"nothing to patch" and the build fails the same way as above.

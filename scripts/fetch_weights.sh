#!/usr/bin/env bash
# Download the pretrained weights for Ferienakademie 2026 into ./weights/.
#
# Weights are NOT baked into the Docker images: they carry their own licenses,
# they would add ~160 MB to every layer, and they change independently of the
# code. weights/ is bind-mounted into every container, so one download serves
# all projects and survives `docker compose down`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="${ROOT}/weights"

# name | destination (relative to weights/) | url | sha256
# Checksums, not sizes: these files are fetched over the network by 20+ people,
# and a truncated or captive-portal-HTML download is the failure we actually
# need to catch before it turns into a confusing torch.load traceback.
DOWNLOADS=(
  "AdaIN decoder|adain/decoder.pth|https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth|379ca41d59f3a37eed3599bbbc2560c19da5c458870a5ffd3a9dd41aa88f9472"
  "AdaIN VGG (normalised)|adain/vgg_normalised.pth|https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth|804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a"
  "Real-ESRGAN x4plus|realesrgan/RealESRGAN_x4plus.pth|https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth|4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"
)

sha256_of() {
  # cut, not awk: keeps this free of nested-quote hazards, and both tools exist
  # everywhere (sha256sum on Linux/in-container, shasum on macOS).
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

ok=0; failed=0
for entry in "${DOWNLOADS[@]}"; do
  IFS='|' read -r name rel url want <<< "${entry}"
  dest="${W}/${rel}"
  mkdir -p "$(dirname "${dest}")"

  if [[ -f "${dest}" ]]; then
    if [[ "$(sha256_of "${dest}")" == "${want}" ]]; then
      printf '  [skip] %-28s already present and verified\n' "${name}"
      ok=$((ok+1)); continue
    fi
    printf '  [warn] %-28s checksum mismatch, re-downloading\n' "${name}"
  fi

  printf '  [get ] %-28s -> weights/%s\n' "${name}" "${rel}"
  if curl -fL --retry 3 --retry-delay 2 --progress-bar -o "${dest}.part" "${url}"; then
    have="$(sha256_of "${dest}.part")"
    if [[ "${have}" != "${want}" ]]; then
      printf '  [FAIL] %-28s sha256 mismatch\n           got      %s\n           expected %s\n' \
        "${name}" "${have}" "${want}"
      rm -f "${dest}.part"; failed=$((failed+1)); continue
    fi
    mv "${dest}.part" "${dest}"
    ok=$((ok+1))
  else
    printf '  [FAIL] %-28s download error\n' "${name}"
    rm -f "${dest}.part"; failed=$((failed+1))
  fi
done

echo
echo "Automatic downloads: ${ok} ok, ${failed} failed."
echo

# --- DCVC-UF: manual, and there is no way around it ------------------------
mkdir -p "${W}/dcvc"
if compgen -G "${W}/dcvc/*.pth.tar" > /dev/null; then
  echo "DCVC-UF checkpoints found:"
  ls -la "${W}/dcvc/"
else
  cat <<'MSG'
DCVC-UF checkpoints must be downloaded by hand -- upstream hosts them on
OneDrive, which has no stable direct-download URL that a script can use.

  1. Open the "Pretrained models" link in https://github.com/microsoft/DCVC
  2. Download all 4 checkpoints (1 image + 3 video: HT-L, HT-S, LD)
  3. Put the .pth.tar files into:   weights/dcvc/

Projects 01 and 02 cannot run until this is done. Everything else works now.
MSG
fi

# --- DCVC-RT: the model that writes a real bitstream on CPU -----------------
mkdir -p "${W}/dcvc-rt"
if compgen -G "${W}/dcvc-rt/*.pth.tar" > /dev/null; then
  echo
  echo "DCVC-RT checkpoints found:"
  ls -la "${W}/dcvc-rt/"
else
  cat <<'MSG'

DCVC-RT checkpoints (optional) are also a manual OneDrive download. They are
what `make bitstream` needs for meaningful rate numbers -- without them the
demo still writes a real, exactly-decodable bitstream, but with random weights,
so the bytes mean nothing.

  1. Open the "Pretrained models" link in
     third_party/DCVC/DCVC-family/DCVC-RT/README.md
  2. Put cvpr2025_image.pth.tar (and cvpr2025_video.pth.tar) into:
     weights/dcvc-rt/
MSG
fi

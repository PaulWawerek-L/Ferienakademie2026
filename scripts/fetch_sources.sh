#!/usr/bin/env bash
# Clone the four upstream codebases into third_party/ so they are readable and
# editable on the host.
#
# Why this exists: the Dockerfiles clone these repos too, so the images work
# standalone -- but code that only lives inside an image cannot be opened in an
# editor, cannot be navigated ("go to definition" on `from src.models.image_model
# import DMCI` goes nowhere), and cannot be modified without a rebuild. For a
# course whose entire point is reading and changing these algorithms, that is
# backwards. compose.yaml bind-mounts these directories over the in-image copies,
# so an edit here takes effect on the next `docker compose run` with no rebuild.
#
#   bash scripts/fetch_sources.sh            # all four
#   bash scripts/fetch_sources.sh --no-vtm   # skip VTM (it is the big one)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${ROOT}/third_party"
mkdir -p "${TP}"

WITH_VTM=1
[[ "${1:-}" == "--no-vtm" ]] && WITH_VTM=0

# name | url | ref     -- refs match the ones the Dockerfiles build from.
REPOS=(
  "DCVC|https://github.com/microsoft/DCVC.git|main"
  "AdaIN|https://github.com/naoto0804/pytorch-AdaIN.git|master"
  "Real-ESRGAN|https://github.com/xinntao/Real-ESRGAN.git|master"
)
[[ ${WITH_VTM} -eq 1 ]] && REPOS+=(
  "VVCSoftware_VTM|https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git|VTM-23.14"
)

for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url ref <<< "${entry}"
  dest="${TP}/${name}"
  if [[ -d "${dest}/.git" ]]; then
    echo "  [skip] ${name} already cloned ($(git -C "${dest}" rev-parse --short HEAD))"
  else
    echo "  [get ] ${name} @ ${ref}"
    git clone --depth 1 --branch "${ref}" "${url}" "${dest}" 2>&1 | sed 's/^/         /'
  fi
done

# Real-ESRGAN needs realesrgan/version.py, which its setup.py generates at
# install time. The image generated one, but the bind mount shadows it, so the
# host checkout needs its own or `import realesrgan` fails.
RE="${TP}/Real-ESRGAN"
if [[ -d "${RE}" && ! -f "${RE}/realesrgan/version.py" ]]; then
  python3 - "${RE}" <<'PY'
import sys, time, os
root = sys.argv[1]
short = open(os.path.join(root, "VERSION")).read().strip()
info = ", ".join(x if x.isdigit() else f'"{x}"' for x in short.split("."))
with open(os.path.join(root, "realesrgan", "version.py"), "w") as f:
    f.write(f"""# GENERATED VERSION FILE
# TIME: {time.asctime()}
__version__ = '{short}'
__gitsha__ = 'unknown'
version_info = ({info})
""")
print(f"  [gen ] Real-ESRGAN/realesrgan/version.py ({short})")
PY
fi

echo
echo "Sources in third_party/ (git-ignored -- they are upstream checkouts, not ours):"
for d in "${TP}"/*/; do
  [[ -d "${d}/.git" ]] || continue
  printf '  %-18s %s  %s\n' "$(basename "${d}")" \
    "$(git -C "${d}" rev-parse --short HEAD)" \
    "$(du -sh "${d}" 2>/dev/null | cut -f1)"
done
echo
echo "compose.yaml mounts these over the in-image copies -- edit here, no rebuild needed."

#!/usr/bin/env bash
# Task 7 -- measure inference time for every project, each in its own container.
#
#   bash scripts/experiments/run_timing.sh              # all five
#   bash scripts/experiments/run_timing.sh style-transfer
#
# Numbers are only comparable against each other if the machine is otherwise
# idle: these are CPU benchmarks on a laptop, and a background build will show up
# in them. The env block in each JSON records thread count and torch version.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH="/Applications/Docker.app/Contents/Resources/bin:${PATH}"
COMPOSE="${COMPOSE:-docker compose}"

ALL=(image-compression video-compression hybrid-vtm style-transfer super-resolution)
TARGETS=("$@"); [[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=("${ALL[@]}")

# The DCVC projects need PYTHONPATH to find the repo; the others already have it.
for proj in "${TARGETS[@]}"; do
  echo
  echo "=== ${proj} ==="
  case "${proj}" in
    image-compression|video-compression) cd_cmd="cd /opt/DCVC && " ;;
    style-transfer)                      cd_cmd="cd /opt/AdaIN && " ;;
    *)                                   cd_cmd="" ;;
  esac
  ${COMPOSE} run --rm --no-deps "${proj}" bash -lc \
    "${cd_cmd}python3 /work/scripts/experiments/exp7_timing.py --project ${proj}"
done

echo
echo "results:"
ls -1 outputs/experiments/07-timing/*.json 2>/dev/null | sed 's|^|  |'

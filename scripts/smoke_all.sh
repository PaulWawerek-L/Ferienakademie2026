#!/usr/bin/env bash
# Run each project's smoke test in its own container and print a pass/fail table.
#
# Usage:
#   scripts/smoke_all.sh              # all projects
#   scripts/smoke_all.sh style-transfer super-resolution
#
# Exit code is non-zero if any selected project failed, so this is usable in CI.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PATH="/Applications/Docker.app/Contents/Resources/bin:${PATH}"
COMPOSE="${COMPOSE:-docker compose}"

# compose service | what to run inside the container | needs weights?
TESTS=(
  "image-compression|bash /work/projects/01-image-compression/smoke.sh|yes"
  "video-compression|bash /work/projects/02-video-compression/smoke.sh|yes"
  "hybrid-vtm|bash /work/projects/03-hybrid-vtm/smoke.sh|no"
  "style-transfer|bash /work/projects/04-style-transfer-adain/smoke.sh|yes"
  "super-resolution|bash /work/projects/05-super-resolution-realesrgan/smoke.sh|yes"
)
# 01 and 02 share one image but now exercise different code (image codec vs the
# inter models + DPB), so both are worth running. Each falls back to the
# checkpoint-free preflight when weights/dcvc/ is empty.

WANT=("$@")
selected() {
  [[ ${#WANT[@]} -eq 0 ]] && return 0
  local s
  for s in "${WANT[@]}"; do [[ "$s" == "$1" ]] && return 0; done
  return 1
}

mkdir -p outputs logs
declare -a RESULTS=()
rc_all=0

for entry in "${TESTS[@]}"; do
  IFS='|' read -r svc cmd _needs <<< "${entry}"
  selected "${svc}" || continue

  log="logs/smoke_${svc}.log"
  printf '\n\033[1m=== %s ===\033[0m (log: %s)\n' "${svc}" "${log}"
  start=$(date +%s)
  if ${COMPOSE} run --rm --no-deps "${svc}" bash -lc "${cmd}" 2>&1 | tee "${log}"; then
    rc=${PIPESTATUS[0]}
  else
    rc=${PIPESTATUS[0]}
  fi
  dur=$(( $(date +%s) - start ))

  if [[ ${rc} -eq 0 ]]; then
    RESULTS+=("PASS|${svc}|${dur}")
  else
    RESULTS+=("FAIL|${svc}|${dur}")
    rc_all=1
  fi
done

printf '\n\033[1m%-6s %-22s %s\033[0m\n' "RESULT" "PROJECT" "TIME"
printf '%s\n' "------------------------------------------"
for r in "${RESULTS[@]}"; do
  IFS='|' read -r status svc dur <<< "${r}"
  if [[ "${status}" == "PASS" ]]; then colour='\033[32m'; else colour='\033[31m'; fi
  printf "${colour}%-6s\033[0m %-22s %ss\n" "${status}" "${svc}" "${dur}"
done
echo
exit ${rc_all}

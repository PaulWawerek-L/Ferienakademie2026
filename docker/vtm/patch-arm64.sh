#!/bin/bash
# Make VTM's architecture guards recognise Linux's spelling of ARM.
#
# VTM does support arm64 -- but every guard tests
#     CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64"
# which is what *macOS* reports. Linux reports "aarch64", so inside an arm64
# Linux container none of the guards fire: the top-level CMakeLists adds
# -msse4.1 and the library CMakeLists glob the x86/ SIMD sources, and the build
# dies with "c++: error: unrecognized command-line option '-msse4.1'".
#
# Patch by search, not by filename: the guard currently appears 5 times across
# 3 CMakeLists, and missing one still drags x86 sources into that target.
set -euo pipefail

NEEDLE='CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64"'
REPLACE='CMAKE_SYSTEM_PROCESSOR MATCHES "(arm64|aarch64)"'
# No regex anchors in the replacement: "x86_64" contains neither "arm64" nor
# "aarch64", so an unanchored match is already exact enough.

# while-read rather than mapfile: macOS still ships bash 3.2, and this script is
# useful to run on the host too when debugging a build.
FILES=()
while IFS= read -r line; do
  FILES+=("${line}")
done < <(grep -rl -F "${NEEDLE}" . --include=CMakeLists.txt || true)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No arm64 guards found -- upstream may have fixed this. Nothing to patch."
  exit 0
fi

echo "Patching ${#FILES[@]} file(s):"
printf '  %s\n' "${FILES[@]}"

# Write-and-move instead of sed -i: BSD sed (macOS) requires an argument to -i
# and GNU sed (the container) forbids one, so -i is the one flag that cannot be
# written portably.
for f in "${FILES[@]}"; do
  # '@' as the delimiter: the replacement text itself contains '|'
  # (the "(arm64|aarch64)" alternation), which would terminate an s|...| early.
  sed "s@${NEEDLE}@${REPLACE}@g" "${f}" > "${f}.patched"
  mv "${f}.patched" "${f}"
done

if grep -rq -F "${NEEDLE}" . --include=CMakeLists.txt; then
  echo "ERROR: guards still present after patching" >&2
  exit 1
fi

echo "Result:"
grep -rn "CMAKE_SYSTEM_PROCESSOR MATCHES" . --include=CMakeLists.txt

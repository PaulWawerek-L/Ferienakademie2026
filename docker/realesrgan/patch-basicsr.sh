#!/bin/bash
# Repair basicsr 1.4.2 against modern torchvision.
#
# basicsr/data/degradations.py imports torchvision.transforms.functional_tensor,
# a PRIVATE module that torchvision deleted in 0.17. The function it wants,
# rgb_to_grayscale, simply moved to the public torchvision.transforms.functional,
# so this is a rename rather than a reimplementation.
#
# basicsr 1.4.2 is the final release and upstream is unmaintained, so there is
# no version of it that works here without this patch.
set -euo pipefail

# find_spec, not `import basicsr`: importing it is exactly what is broken, so
# locating the package by importing it would fail before we could fix it.
BASICSR_DIR="$(python -c 'import importlib.util as u; print(u.find_spec("basicsr").submodule_search_locations[0])')"
echo "basicsr at: ${BASICSR_DIR}"

OLD='from torchvision.transforms.functional_tensor import rgb_to_grayscale'
NEW='from torchvision.transforms.functional import rgb_to_grayscale'

# Patch by search so a future release that moves the import is still covered.
found=0
while IFS= read -r f; do
  [[ "${f}" == *.pyc ]] && continue
  echo "  patching ${f}"
  sed "s@${OLD}@${NEW}@g" "${f}" > "${f}.patched"
  mv "${f}.patched" "${f}"
  found=$((found + 1))
done < <(grep -rl -F "${OLD}" "${BASICSR_DIR}" || true)

if [[ ${found} -eq 0 ]]; then
  echo "Nothing to patch -- basicsr may already be fixed."
else
  echo "Patched ${found} file(s)."
fi

# Stale bytecode from the pre-patch install would shadow the fix.
find "${BASICSR_DIR}" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

python -c "import basicsr.data.degradations; print('basicsr imports cleanly')"

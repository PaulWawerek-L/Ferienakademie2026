#!/bin/bash
# Expose RansDecoder's decoded output to Python.
#
# DCVC-UF's pybind module binds the decoder's set_stream/decode_y/decode_z/set_cdf
# but NOT any way to read the decoded symbols back. The only accessor,
# get_decoded_tensor_cpp(), is declared for C++ callers -- the CUDA inference
# proxy links against it directly -- and never reaches Python. So from Python you
# can run a decode and then cannot retrieve the result.
#
# That single omission is what stops a CPU decoder from existing. Encoding is
# already fully bound (encode_y, encode_z, flush, get_encoded_stream), so this
# adds the one missing half. It is a pure addition: a lambda in bind.cpp wrapping
# the existing C++ accessor as a numpy array. No header or implementation change.
set -euo pipefail

BIND="${1:-src/cpp/py_rans/bind.cpp}"
[[ -f "${BIND}" ]] || { echo "not found: ${BIND}" >&2; exit 1; }

if grep -q '"get_decoded_tensor"' "${BIND}"; then
  echo "already patched: ${BIND}"
  exit 0
fi

ANCHOR='        .def("decode_z", &RansDecoder::decode_z)'
grep -qF "${ANCHOR}" "${BIND}" || { echo "anchor line not found in ${BIND}" >&2; exit 1; }

python3 - "${BIND}" <<'PY'
import sys
path = sys.argv[1]
s = open(path).read()
anchor = '        .def("decode_z", &RansDecoder::decode_z)\n'
addition = anchor + '''        // Added for the CPU bitstream path: upstream binds the decode calls but
        // no accessor, so Python can decode and never see the symbols.
        .def("get_decoded_tensor",
             [](RansDecoder& self) {
                 auto v = self.get_decoded_tensor_cpp();
                 return py::array_t<int8_t>(static_cast<py::ssize_t>(v->size()), v->data());
             })
'''
assert anchor in s
open(path, "w").write(s.replace(anchor, addition, 1))
print(f"patched {path}: RansDecoder.get_decoded_tensor exposed")
PY

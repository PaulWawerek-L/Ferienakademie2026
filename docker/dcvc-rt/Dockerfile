# syntax=docker/dockerfile:1.7
# DCVC-RT (CVPR 2025) -- the one DCVC model that writes a REAL bitstream on CPU.
#
# Why a separate image rather than a layer on the DCVC one:
# both DCVC-UF (repo root, src/cpp) and DCVC-RT (DCVC-family/DCVC-RT/src/cpp)
# build a pybind11 module called `MLCodec_extensions_cpp`, and the two are not
# compatible -- UF's pmf_to_quantized_cdf takes one argument, RT's takes two
# (pmf, precision). Same name, different ABI, so they cannot share a Python
# environment. One image each is the only clean answer.
#
# What this buys: DCVC-UF's compress() needs the CUDA-only CUTLASS extension and
# has no CPU fallback. DCVC-RT's entropy coding is plain Python over the CPU rANS
# coder, and its network falls back to PyTorch when the CUDA extension is absent
# ("cannot import cuda implementation for inference, fallback to pytorch").
# Only its stream/event scheduling assumes CUDA, and scripts/bitstream/cpu_cuda_shim.py
# no-ops exactly that.
ARG BASE_IMAGE=ferienakademie/base:cpu
FROM ${BASE_IMAGE}

RUN git clone --depth 1 https://github.com/microsoft/DCVC.git /opt/DCVC

WORKDIR /opt/DCVC/DCVC-family/DCVC-RT

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 120 -r requirements.txt

# RT's own entropy coder. Build it from RT's tree, never the repo root's.
RUN cd src/cpp && pip install --no-build-isolation . \
 && python -c "import MLCodec_extensions_cpp as m; \
      assert m.pmf_to_quantized_cdf.__doc__.count('arg1'), 'wrong extension: UF variant installed'; \
      print('DCVC-RT rANS extension OK (2-arg pmf_to_quantized_cdf)')"

ENV PYTHONPATH=/opt/DCVC/DCVC-family/DCVC-RT:/work/scripts/bitstream
WORKDIR /work
CMD ["/bin/bash"]

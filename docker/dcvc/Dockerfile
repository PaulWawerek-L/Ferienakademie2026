# syntax=docker/dockerfile:1.7
# Projects 01 (image) + 02 (video): microsoft/DCVC -> DCVC-UF (CVPR 2026).
#
# NOTE: 01 and 02 share this ONE image. DCVC-UF-Intra is the intra-frame codec
# of the same model, so there is no second repo and no second environment.
#
# CPU limitation (important, see projects/01-image-compression/README.md):
# upstream builds two extensions. We build only the first:
#   1. MLCodec_extensions_cpp   -- rANS entropy coder, plain C++/pybind11  -> OK
#   2. inference_extensions_cuda -- CUTLASS fused kernels, CUDA-only       -> SKIPPED
# Without (2), model.compress()/decompress() raise NotImplementedError, i.e. no
# real bitstream. model.forward() is pure PyTorch and runs fine, giving the
# reconstruction, the estimated bpp from the entropy model, and PSNR -- which is
# exactly what the three core concepts of this project need.
ARG BASE_IMAGE=ferienakademie/base:cpu
FROM ${BASE_IMAGE}

ARG DCVC_REF=main
RUN git clone --depth 1 --branch ${DCVC_REF} \
        https://github.com/microsoft/DCVC.git /opt/DCVC

WORKDIR /opt/DCVC

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Upstream binds the rANS decoder's decode calls but no way to read the result
# back -- see the script. Without this there can be no CPU decoder.
COPY patch-expose-decoder.sh /tmp/patch-expose-decoder.sh
RUN bash /tmp/patch-expose-decoder.sh src/cpp/py_rans/bind.cpp

# CPU entropy coder only. Upstream compiles with -Werror, which is strict across
# compilers/architectures; if this ever fails on a new arch, relax it here
# rather than in the students' checkout.
RUN cd src/cpp && pip install --no-build-isolation . \
    && python -c "import MLCodec_extensions_cpp as m; \
         assert hasattr(m.RansDecoder(), 'get_decoded_tensor'), 'decoder accessor missing'; \
         print('rANS ext OK (encoder + decoder accessor):', m.__file__)"

ENV PYTHONPATH=/opt/DCVC
WORKDIR /work
CMD ["/bin/bash"]

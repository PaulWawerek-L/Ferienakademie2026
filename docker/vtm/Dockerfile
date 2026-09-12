# syntax=docker/dockerfile:1.7
# Project 03: VVC reference software (VTM) -- the "traditional" half of the
# hybrid codec. Pure C++/CMake, no Python, no PyTorch: this is the one project
# that genuinely cannot share the base image.
#
# Multi-stage: the build stage needs a full toolchain (~1 GB) but the students
# only need three binaries, so the final image stays small.
FROM debian:bookworm-slim AS build

ARG VTM_TAG=VTM-23.14
ARG BUILD_JOBS=

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${VTM_TAG} \
        https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git /src

WORKDIR /src

# VTM's arm64 guards only recognise the macOS spelling; see the script.
COPY patch-arm64.sh /tmp/patch-arm64.sh
RUN bash /tmp/patch-arm64.sh

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build ${BUILD_JOBS:+-j ${BUILD_JOBS}} \
    && ls -la bin/

# ---------------------------------------------------------------------------
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg python3 python3-numpy ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# VTM emits two flavours: dynamically linked ones under a nested
# bin/<gen>/<compiler>/<arch>/<config>/ path, and *Static variants directly in
# bin/. Take the static ones -- they need nothing from the build stage, which is
# the whole point of the runtime stage.
COPY --from=build /src/bin/EncoderAppStatic /usr/local/bin/EncoderApp
COPY --from=build /src/bin/DecoderAppStatic /usr/local/bin/DecoderApp
COPY --from=build /src/cfg /opt/vtm/cfg

ENV VTM_CFG=/opt/vtm/cfg
WORKDIR /work
CMD ["/bin/bash"]

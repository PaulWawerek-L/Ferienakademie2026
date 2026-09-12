# syntax=docker/dockerfile:1.7
# Project 04: Style Transfer -- AdaIN (naoto0804/pytorch-AdaIN).
ARG BASE_IMAGE=ferienakademie/base:cpu
FROM ${BASE_IMAGE}

RUN git clone --depth 1 https://github.com/naoto0804/pytorch-AdaIN.git /opt/AdaIN

# Upstream requirements.txt is deliberately NOT installed -- it is unusable:
#   * torch==1.13.1 + torchvision==0.4.0 is an impossible pair (0.4.0 <-> torch 1.2)
#   * pkg-resources==0.0.0 is a pip-freeze artifact that does not exist on PyPI
#   * torch 1.13.1 ships no linux/arm64 wheel, so it cannot install on Apple Silicon
# The repo was archived in Jan 2024. Its model code is plain nn.Sequential and
# runs unmodified on the base image's modern torch; only the pins were rotten.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install tensorboardX

ENV PYTHONPATH=/opt/AdaIN
WORKDIR /work
CMD ["/bin/bash"]

# syntax=docker/dockerfile:1.7
# Shared base for all Python projects of Ferienakademie 2026.
#
# CPU-only and multi-arch (linux/amd64 + linux/arm64) on purpose: students run
# this on their own laptops, which mostly have no NVIDIA GPU. We deliberately
# use the PyTorch "cpu" wheel index instead of PyPI -- on x86_64, plain PyPI
# torch drags in ~3 GB of nvidia-* CUDA wheels that would never be used here.
FROM python:3.12-slim-bookworm

ARG TORCH_VERSION=2.12.1
ARG TORCHVISION_VERSION=0.27.1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 \
    MPLBACKEND=Agg \
    TORCH_HOME=/work/weights/torch

# apt over HTTPS + retries. Not cosmetic: a lecture hall full of students
# building at once on conference wifi hits deb.debian.org hard, and a single
# dropped connection otherwise fails the whole build. Plain HTTP is also the
# first thing a captive portal or corporate proxy breaks.
RUN set -eux; \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' \
        /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list 2>/dev/null || true; \
    printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' \
        > /etc/apt/apt.conf.d/99-ferienakademie-retries; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build ca-certificates curl \
        libglib2.0-0 ffmpeg; \
    rm -rf /var/lib/apt/lists/*

# torch first, in its own layer: it is by far the biggest download, and pinning
# it separately means a project's requirements change does not re-pull it.
# --retries/--timeout for the same reason as the apt hardening above: torch is a
# ~250 MB download and pip's default 15 s timeout gives up on the first stall.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 120 \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}+cpu" "torchvision==${TORCHVISION_VERSION}+cpu"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 120 \
        numpy pillow opencv-python-headless tqdm scipy matplotlib pybind11

WORKDIR /work
CMD ["/bin/bash"]

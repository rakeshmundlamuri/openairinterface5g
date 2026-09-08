# SPDX-License-Identifier: LicenseRef-CSSL-1.0
#
# Self-contained image for the Deep-JSCC-over-real-PDSCH demo (doc/JSCC_DEMO.md).
# Builds OAI's rfsim-only gNB/UE binaries and sets up the Python/PyTorch environment
# needed to train and run the DeepJSCC encoder/decoder. rfsim/gNB/UE need no GPU and no
# RF hardware - pure software, proven throughout this project's testing - so this runs
# on any cloud VM with a few CPU cores. Only *training* benefits from a GPU.
#
# Build (CPU, default):  docker build -f scripts/jscc/Dockerfile -t oai-jscc-demo .
# Build (CUDA 12.4 training on a GPU VM):
#   docker build -f scripts/jscc/Dockerfile --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
#     -t oai-jscc-demo-gpu .
#   # base image still has no CUDA runtime - either switch FROM to an nvidia/cuda image,
#   # or run with a host that provides one (e.g. `docker run --gpus all --runtime=nvidia`
#   # against a matching nvidia/cuda base); train.py auto-selects cuda when available.
# Run:    docker run --rm -it oai-jscc-demo bash
#         (inside the container) see doc/JSCC_DEMO.md for the train/encode/demo commands

FROM ubuntu:noble

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

RUN apt-get update && apt-get install --yes --no-install-recommends \
        sudo git ca-certificates python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /oai
COPY . /oai

# build_oai's -I flag installs the RAN build/runtime dependencies itself, so we don't
# duplicate that package list here - one source of truth, see cmake_targets/build_oai.
RUN cd cmake_targets && \
    ./build_oai -I --gNB --nrUE --build-tool-opt "-j$(nproc)" && \
    ./build_oai --gNB --nrUE -P --build-tool-opt "-j$(nproc)"

RUN python3 -m venv /oai/scripts/jscc/.venv && \
    /oai/scripts/jscc/.venv/bin/pip install --upgrade pip && \
    /oai/scripts/jscc/.venv/bin/pip install --index-url "$TORCH_INDEX_URL" torch==2.14.0 torchvision==0.29.0 && \
    /oai/scripts/jscc/.venv/bin/pip install numpy pillow pyarrow

ENV PATH="/oai/scripts/jscc/.venv/bin:${PATH}"
WORKDIR /oai/scripts/jscc

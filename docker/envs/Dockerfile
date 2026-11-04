# syntax=docker/dockerfile:1
#
# Simulusion with the environments that cannot be pip-installed on every
# platform: DeepMind Lab (DMLab-30) and MineRL 0.4.4 (Minecraft Diamond), plus
# BSuite. Both need Linux and Python 3.11 (the patched MineRL wheel is cp311
# only), so they live in this image rather than in a pip extra.
#
# Build from the repository root:
#   docker build -f docker/envs/Dockerfile -t simulusion-envs .
#
# Train Dreamer on a DMLab-30 level / Minecraft Diamond / BSuite (GPU):
#   docker run --gpus all --rm -v "$PWD/runs:/data/simulusion" simulusion-envs \
#     simulusion train --config dreamer model.env_backend=dmlab \
#       model.env=explore_goal_locations_small model.action_repeat=1
#   ... model.env_backend=minecraft model.env=diamond model.action_repeat=1
#   ... model.env_backend=bsuite model.env=catch/0 model.action_repeat=1
#
# Minecraft runs a real game client, so commands start under xvfb-run (the
# entrypoint does this). CPU-only build: --build-arg PYTORCH_INDEX_URL=
#   https://download.pytorch.org/whl/cpu

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    SIMULUSION_HOME=/data/simulusion

# Python 3.11, OpenJDK 8 for Minecraft, a virtual display, and OpenGL.
# xvfb-run waits for a signal from Xvfb that never arrives when it runs as
# PID 1, so tini is the container's init.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gpg-agent ca-certificates curl wget git ffmpeg \
        xvfb x11-xserver-utils libglew-dev libgl1 libglib2.0-0 openjdk-8-jdk tini \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /venv --upgrade-deps
ENV PATH="/venv/bin:$PATH"
# deepmind_lab and MineRL 0.4.4 import pkg_resources, which setuptools 81
# removed. The constraint keeps every later pip install from upgrading it.
RUN echo "setuptools<81" > /etc/pip-constraints.txt \
    && pip install -c /etc/pip-constraints.txt "setuptools<81"
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt
# install_dmlab.sh's Bazel rule calls `python3`; make it the venv's.
RUN ln -sf /venv/bin/python /usr/local/bin/python3

# DeepMind Lab (built from source with Bazel).
COPY docker/envs/install_dmlab.sh /tmp/install_dmlab.sh
RUN PYTHON=/venv/bin/python bash /tmp/install_dmlab.sh && rm /tmp/install_dmlab.sh

# MineRL 0.4.4 with the block break speed patch DreamerV3 uses. MineRL writes
# into its package directory at runtime, so make it writable for any user.
ARG MINERL_WHEEL=https://github.com/danijar/minerl/releases/download/v0.4.4-patched/minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl
RUN pip install "${MINERL_WHEEL}" \
    && python -c "import minerl" \
    && chmod -R a+rwX /venv/lib/python3.11/site-packages/minerl

# Declared here, not at the top: an ARG is in the environment of every later
# RUN, so its value is part of their cache keys. Here, switching between CUDA
# and CPU torch reuses the DMLab and MineRL layers instead of rebuilding them.
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
# The version uv.lock resolves, so the image runs what the project tests. The
# constraint keeps the Simulusion install below from swapping torch for the
# version another torchvision release pins.
ARG TORCH_VERSION=2.13.0
RUN pip install --index-url "${PYTORCH_INDEX_URL}" "torch==${TORCH_VERSION}" torchvision \
    && echo "torch==${TORCH_VERSION}" >> /etc/pip-constraints.txt

WORKDIR /app
COPY pyproject.toml setup.py README.md ./
COPY simulusion ./simulusion
# No `dmc` extra: DeepMind Control's renderer segfaults in this image under
# WSL with both EGL and OSMesa, and the image exists for DMLab, MineRL and
# BSuite. Install it yourself if your host's GL works.
RUN pip install --editable ".[gym,bsuite,viz]"
COPY demos ./demos
COPY scripts ./scripts

ENTRYPOINT ["tini", "--", "xvfb-run", "-a", "-s", "-screen 0 1024x768x24"]
CMD ["python", "-c", "import deepmind_lab, minerl, bsuite, simulusion; print('simulusion-envs ready')"]

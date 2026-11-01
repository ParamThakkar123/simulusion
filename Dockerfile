# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG SIMULUSION_EXTRAS=

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SIMULUSION_HOME=/data/simulusion

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py README.md ./
COPY simulusion ./simulusion

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --index-url "${PYTORCH_INDEX_URL}" torch torchvision torchaudio && \
    if [ -n "${SIMULUSION_EXTRAS}" ]; then \
        python -m pip install --editable ".[${SIMULUSION_EXTRAS}]"; \
    else \
        python -m pip install --editable .; \
    fi && \
    simulusion version && \
    mkdir -p "${SIMULUSION_HOME}"

VOLUME ["/data/simulusion"]

ENTRYPOINT ["simulusion"]
CMD ["--help"]

# ==========================================
# Builder Stage
# ==========================================
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# System deps for building wheels (common requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    wget \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create venv (keeps pip installs isolated & copyable)
ENV VENV_PATH=/opt/venv
RUN python3 -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"
# Set environment variables for PyTorch build
ENV FORCE_CUDA="1"
# ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
# Optional but usually helpful for modern builds

WORKDIR /
COPY requirements.txt requirements.txt

# Install PyTorch (PINNED) + your deps
# Example versions shown directly on PyTorch's "previous versions" page
RUN pip install --no-cache-dir tensorboard && \
    pip install --upgrade --no-cache-dir pip setuptools wheel numpy Cython && \
    pip install --no-cache-dir \
      torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
      --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-build-isolation --no-cache-dir -r requirements.txt


# ==========================================
# Runtime Stage
# ==========================================
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Only runtime OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
ENV VENV_PATH=/opt/venv
COPY --from=builder ${VENV_PATH} ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

WORKDIR /

# Wandb Params
ENV WANDB_MODE="offline"
ENV WANDB_API_KEY=""
ENV WANDB_WATCH="false"
ENV HYDRA_FULL_ERROR=1
ENV PATH="/usr/local/bin:${PATH}"

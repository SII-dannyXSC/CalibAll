# CalibAll Docker Environment
# GPU required: NVIDIA GPU with CUDA 12.8+ support

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    ffmpeg libgl1-mesa-glx libglib2.0-0 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (Python 3.12)
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"
RUN conda create -n caliball python=3.12 -y
SHELL ["conda", "run", "-n", "caliball", "/bin/bash", "-c"]

# PyTorch (CUDA 12.8)
RUN pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Copy project
WORKDIR /workspace/CalibAll
COPY pyproject.toml requirements.txt ./
COPY src/ src/

# Base dependencies
RUN pip install -e . && pip install -r requirements.txt

# Special dependencies
RUN pip install setuptools wheel ninja \
    && pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
RUN pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
    pytorch3d==0.7.9+pt2.9.0cu128
RUN pip install git+https://github.com/microsoft/MoGe.git

# Third-party repos
RUN mkdir -p third_party && cd third_party \
    && git clone --depth 1 https://github.com/facebookresearch/co-tracker \
    && cd co-tracker && pip install -e . && cd .. \
    && git clone --depth 1 https://github.com/facebookresearch/sam3.git \
    && cd sam3 && pip install -e . && cd .. \
    && git clone --depth 1 https://github.com/facebookresearch/dinov2 \
    && git clone --depth 1 https://github.com/Daniella1/urdf_files_dataset.git urdf

# Copy remaining files
COPY scripts/ scripts/
COPY assets/ assets/

# Checkpoints volume mount point
VOLUME /workspace/CalibAll/ckpt
VOLUME /workspace/CalibAll/data

EXPOSE 8765

# Default: launch web UI
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "caliball"]
CMD ["python", "scripts/caliball_web.py", "--host", "0.0.0.0"]

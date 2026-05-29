# NOTE: base image pulled from nvcr.io (NVIDIA NGC) because docker.io is
# unreachable from this host. Switch back to `nvidia/cuda:...` if a Docker Hub
# mirror is configured in /etc/docker/daemon.json.
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    FORCE_CUDA=1 \
    TORCH_EXTENSIONS_DIR=/workspace/.torch_extensions \
    HF_HOME=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace/CalibAll

# ----------------------------------------------------------------------------
# 1. System dependencies (Python 3.12 + build toolchain + OpenGL/EGL runtime).
# ----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        git \
        git-lfs \
        wget \
        curl \
        ca-certificates \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libegl1 \
        libgles2 \
        libxrender1 \
        libxext6 \
        libsm6 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

# ----------------------------------------------------------------------------
# 2. Create an isolated venv (Ubuntu 24.04 marks the system Python as
#    PEP 668 externally-managed, so a venv is the cleanest path).
# ----------------------------------------------------------------------------
RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools==75.0.0 wheel ninja

# ----------------------------------------------------------------------------
# 3. PyTorch (CUDA 12.8) — install first so subsequent CUDA-extension builds
#    can link against it.
# ----------------------------------------------------------------------------
RUN pip install \
        torch==2.9.0 \
        torchvision==0.24.0 \
        torchaudio==2.9.0 \
        --index-url https://download.pytorch.org/whl/cu128

# ----------------------------------------------------------------------------
# 4. CalibAll package + Python requirements.
# ----------------------------------------------------------------------------
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install -e . \
    && pip install -r requirements.txt

# ----------------------------------------------------------------------------
# 5. Special dependencies from README §4.
#    - nvdiffrast: kernels are JIT-compiled at first runtime, only the
#      Python package needs to be installed here.
#    - pytorch3d: pre-built wheel that matches torch 2.9.0 + cu128.
#    - MoGe: monocular geometry estimator for intrinsics.
# ----------------------------------------------------------------------------
RUN pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
RUN pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
        pytorch3d==0.7.9+pt2.9.0cu128
RUN pip install git+https://github.com/microsoft/MoGe.git

# ----------------------------------------------------------------------------
# 6. Third-party repositories from README §5.
#    Cloned into the image so that `docker compose up` is enough — the
#    compose file uses an anonymous volume to keep this directory intact
#    even when the project root is bind-mounted on top.
# ----------------------------------------------------------------------------
RUN mkdir -p third_party \
    && git clone --depth 1 https://github.com/facebookresearch/co-tracker third_party/co-tracker \
    && git clone --depth 1 https://github.com/facebookresearch/sam3.git    third_party/sam3 \
    && git clone --depth 1 https://github.com/facebookresearch/dinov2      third_party/dinov2 \
    && pip install -e third_party/co-tracker \
    && pip install -e third_party/sam3

# ----------------------------------------------------------------------------
# 7. Project source (after dependency layers so source-only edits invalidate
#    only this layer).
# ----------------------------------------------------------------------------
COPY . .

EXPOSE 8765

CMD ["python", "scripts/caliball_web.py", "--host", "0.0.0.0", "--port", "8765"]

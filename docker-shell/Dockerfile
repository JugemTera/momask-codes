FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# Install PyTorch first (matching requirements.txt: torch==1.12.0 + CUDA 11.3)
RUN pip install --no-cache-dir \
    torch==1.12.0+cu113 \
    torchvision==0.13.0+cu113 \
    torchaudio==0.12.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Copy and install remaining dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# Default command
CMD ["/bin/bash"]

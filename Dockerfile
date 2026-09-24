FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Kolkata

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    curl \
    wget \
    rsync \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ML dependencies
COPY requirements-ml.txt /tmp/requirements-ml.txt

RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install \
      torch==2.1.0+cu118 \
      torchvision==0.16.0+cu118 \
      torchaudio==2.1.0+cu118 \
      --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install -r /tmp/requirements-ml.txt

# Copy the complete prepared MVSA environment
COPY . /app

ENV PYTHONPATH=/app:/app/scripts:/app/vehicle-counting:/app/historical-processor

CMD ["/bin/bash"]

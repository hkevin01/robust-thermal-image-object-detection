# Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Create virtual environment inside container
RUN python3.10 -m venv /workspace/venv

# Activate virtual environment
ENV PATH="/workspace/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt

# Copy project files
COPY . /workspace/

# Install project in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /workspace/data/raw \
    /workspace/data/processed \
    /workspace/data/annotations \
    /workspace/data/metadata \
    /workspace/assets/models \
    /workspace/logs

# Set up entrypoint
CMD ["/bin/bash"]

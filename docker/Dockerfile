# Syntax allowing for build arguments before FROM
ARG BASE_IMAGE=rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.8.0
ARG WAVE_REPO
ARG WAVE_BRANCH

# ============================================================================
# Stage 1: Base - System Dependencies
# ============================================================================
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV HIP_PLATFORM=amd

# Install essential build tools & python deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    ninja-build \
    gfortran \
    pkg-config \
    wget \
    curl \
    vim \
    ca-certificates \
    # Runtime deps for TheRock components (e.g. compression libs)
    zstd \
    libzstd-dev \
    # Common deps for hipBLASLt build
    libnuma1 \
    libmsgpack-dev \
    libboost-dev \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libdrm-dev \
    # Build tools
    make \
    build-essential

# ============================================================================
# Stage 2: Setup iree-kernel-benchmark
# ============================================================================
FROM base AS setup

# Copy the repository files
COPY . /workspace
WORKDIR /workspace

# Install Rust for wave compilation (if needed)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Run setup script with conditional arguments based on build args
RUN if [ -n "$WAVE_REPO" ] && [ -n "$WAVE_BRANCH" ]; then \
        ./setup.sh --wave-repo "$WAVE_REPO" --wave-branch "$WAVE_BRANCH" --no-install-torch; \
    else \
        ./setup.sh --no-install-torch; \
    fi

CMD ["/bin/bash"]

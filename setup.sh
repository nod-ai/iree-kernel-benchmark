#!/bin/bash

# Setup script for iree-kernel-benchmark
# Usage: ./setup.sh [--wave-repo REPO] [--wave-branch BRANCH] [--venv-path PATH]

set -e

# Default values
VENV_PATH=".venv"
WAVE_REPO=""
WAVE_BRANCH=""
INSTALL_FROM_SOURCE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --wave-repo)
            WAVE_REPO="$2"
            shift 2
            ;;
        --wave-branch)
            WAVE_BRANCH="$2"
            shift 2
            ;;
        --venv-path)
            VENV_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --wave-repo REPO     Wave repository to clone (e.g., iree-org/wave)"
            echo "  --wave-branch BRANCH Wave branch to checkout"
            echo "  --venv-path PATH     Path for Python virtual environment (default: .venv)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Note: If --wave-repo is provided, --wave-branch must also be provided."
            echo "      If neither are provided, wave-lang will be installed from PyPI."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Validate wave repo and branch arguments
if [[ -n "$WAVE_REPO" && -z "$WAVE_BRANCH" ]] || [[ -z "$WAVE_REPO" && -n "$WAVE_BRANCH" ]]; then
    echo "Error: Both --wave-repo and --wave-branch must be provided together."
    echo "Use --help for usage information."
    exit 1
fi

if [[ -n "$WAVE_REPO" && -n "$WAVE_BRANCH" ]]; then
    INSTALL_FROM_SOURCE=true
fi

echo "Setting up iree-kernel-benchmark environment..."
echo "Virtual environment path: $VENV_PATH"

if [[ "$INSTALL_FROM_SOURCE" == "true" ]]; then
    echo "Wave repository: $WAVE_REPO"
    echo "Wave branch: $WAVE_BRANCH"
    echo "Installing wave from source..."
else
    echo "Installing wave-lang from PyPI..."
fi

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

if [[ "$INSTALL_FROM_SOURCE" == "true" ]]; then
    # Check if Rust is installed, install if not
    echo "Checking for Rust installation..."
    if ! command -v rustc &> /dev/null; then
        echo "Rust not found. Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
        echo "Rust installed successfully."
    else
        echo "Rust is already installed."
    fi
    
    # Install wave from source
    echo "Cloning wave repository..."
    if [[ -d "wave" ]]; then
        echo "Removing existing wave directory..."
        rm -rf wave
    fi
    
    git clone "https://github.com/$WAVE_REPO.git"
    cd wave
    git checkout "$WAVE_BRANCH"
    
    echo "Installing wave dependencies..."
    pip install -r requirements-iree-pinned.txt
    pip install -r pytorch-rocm-requirements.txt
    pip install -r requirements.txt
    pip install -e .
    cd ..
else
    # Install wave from PyPI
    echo "Installing torch & IREE dependencies"
    pip install --pre --no-cache-dir --find-links https://iree.dev/pip-release-links.html iree-base-compiler iree-base-runtime --upgrade
    pip install -r pytorch-rocm-requirements.txt
    echo "Installing wave-lang from PyPI..."
    pip install wave-lang
fi

# Install project requirements
echo "Installing project requirements..."
pip install -r requirements.txt

# Install Triton
echo "Installing Triton..."
if [[ -d "aiter" ]]; then
    echo "Removing existing aiter directory..."
    rm -rf aiter
fi

git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python setup.py develop
cd ..

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "To run benchmarks, use:"
echo "  python3 -m kernel_bench.cli.bench --backend=all --kernel_type=all --max_kernels=50 --machine=mi325x"

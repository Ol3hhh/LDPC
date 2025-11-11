# LDPC GPU Decoder

This repository contains CPU-based and GPU-based implementations of an LDPC (Low-Density Parity-Check) decoder using CUDA. It supports iterative Min-Sum decoding and is optimized for high-performance execution on NVIDIA GPUs.

<img width="951" height="205" alt="image" src="https://github.com/user-attachments/assets/23ad86bf-f590-48e9-859b-d476b8795a40" />

## Requirements

- C++11 compatible compiler
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed

## Build Instructions

```bash
# Create build directory
mkdir build
cd build

# Configure project with CMake
cmake ..

# Build the executable
make

# Run the decoder
./ldpc_decoder

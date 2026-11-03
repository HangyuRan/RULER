#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

# Run inside the intended Python environment on a Linux CUDA machine.
# CairoSVG also needs the system Cairo library; FlashAttention needs a matching
# CUDA toolkit when a prebuilt wheel is unavailable.
"$PYTHON" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Use Python 3.10, 3.11, or 3.12 for the pinned RULER dependencies"'

# Extend an existing server environment. requirements.txt is the preserved
# historical snapshot and is deliberately not reinstalled here. pip may adjust
# direct/transitive versions to satisfy this requested graph; it does not prune
# packages merely because the project does not import them.
"$PYTHON" -m pip install \
  "verl[vllm] @ git+https://github.com/volcengine/verl.git@v0.7.1" \
  "torch==2.9.0" "vllm==0.12.0" "transformers==4.57.6" "numpy==1.26.4" \
  "huggingface-hub>=0.34.0,<1.0" "requests>=2.32.2,<3" \
  "CairoSVG>=2.7,<3" "scikit-image>=0.22,<0.26" "lpips==0.1.4" \
  "clip @ git+https://github.com/openai/CLIP.git" "hpsv2==1.2.0" \
  "pytorch-lightning>=2.2,<3" "tensorboardX>=2.6,<3" \
  "setuptools>=77.0.3,<81" wheel packaging ninja

# VERL's use_remove_padding=True training path requires FlashAttention.
# Install after PyTorch so its extension is built for the selected torch/CUDA.
# Preserve the original server's FlashAttention version. Compatibility with the
# target torch/CUDA environment must still be checked on that server.
"$PYTHON" -m pip install --no-build-isolation "flash-attn==2.7.4.post1"
"$PYTHON" -m pip check

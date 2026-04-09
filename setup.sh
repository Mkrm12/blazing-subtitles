#!/bin/bash
# =============================================================================
# setup.sh — Run ONCE in your Lightning AI terminal
# Fixes: NumPy 2.x clash, pkg_resources missing, all version conflicts
# Order is CRITICAL — do not rearrange
# =============================================================================
set -e

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         LIGHTNING AI PIPELINE ENVIRONMENT SETUP          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
# STEP 1: setuptools FIRST — fixes the 'No module named pkg_resources' crash
# lightning_fabric/__init__.py does __import__("pkg_resources") at boot time
# =============================================================================
echo ">>> [1/13] Fixing pkg_resources (setuptools)..."
pip install --upgrade "setuptools>=69.5.1" wheel pip --quiet

# =============================================================================
# STEP 2: Pin NumPy to 1.26.4 BEFORE anything else
# This is the #1 root cause. pyannote, matplotlib, easyocr were all compiled
# against NumPy 1.x. NumPy 2.x breaks their C extensions at import time.
# =============================================================================
echo ">>> [2/13] Pinning NumPy to 1.26.4 (the CRITICAL fix)..."
pip install "numpy==1.26.4" --quiet

# =============================================================================
# STEP 3: PyTorch for CUDA 12.1 (T4 compatible)
# =============================================================================
echo ">>> [3/13] Installing PyTorch 2.3.1 + CUDA 12.1..."
pip install \
    torch==2.3.1 \
    torchaudio==2.3.1 \
    torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet

# =============================================================================
# STEP 4: RE-PIN numpy immediately — torch install can silently upgrade it
# =============================================================================
echo ">>> [4/13] Re-pinning NumPy (torch may have pulled in 2.x)..."
pip install "numpy==1.26.4" --force-reinstall --no-deps --quiet

# =============================================================================
# STEP 5: Transformers ecosystem (pinned tight for stability)
# =============================================================================
echo ">>> [5/13] Installing Transformers ecosystem..."
pip install \
    "transformers==4.44.2" \
    "tokenizers>=0.19.1,<0.20.0" \
    "huggingface_hub>=0.24.0,<0.26.0" \
    "safetensors>=0.4.3" \
    "accelerate>=0.30.0,<0.34.0" \
    --quiet

pip install \
    "timm==1.0.9" \
    "einops==0.8.0" \
    --quiet

# =============================================================================
# STEP 6: Matplotlib — pin to a version compiled against NumPy 1.x
# 3.8.4 is the last stable pre-NumPy-2.x matplotlib
# =============================================================================
echo ">>> [6/13] Installing matplotlib 3.8.4 (NumPy 1.x compatible)..."
pip install "matplotlib==3.8.4" --quiet

# =============================================================================
# STEP 7: Lightning / PyTorch-Lightning (pyannote dependency)
# Must be installed BEFORE pyannote so it doesn't get overwritten
# =============================================================================
echo ">>> [7/13] Installing Lightning stack..."
pip install \
    "lightning==2.3.3" \
    "pytorch-lightning==2.3.3" \
    --quiet

# =============================================================================
# STEP 8: pyannote — install in the right order, manually control deps
# pyannote.audio --no-deps avoids it pulling in incompatible versions
# =============================================================================
echo ">>> [8/13] Installing pyannote (carefully)..."

# Core pyannote libraries first
pip install \
    "pyannote.core==5.0.0" \
    "pyannote.database==5.1.0" \
    "pyannote.pipeline==3.0.1" \
    "pyannote.metrics==3.2.1" \
    --quiet

# SpeechBrain — pyannote's neural backbone (use --no-deps, manual control)
pip install "speechbrain==1.0.0" --no-deps --quiet
pip install "hyperpyyaml>=1.2.2" "packaging>=23.0" --quiet

# Supporting audio libs
pip install \
    "asteroid-filterbanks==0.4.0" \
    "pytorch-metric-learning==2.6.0" \
    --quiet

# Finally install pyannote.audio itself — no deps to prevent cascade conflicts
pip install "pyannote.audio==3.3.1" --no-deps --quiet

# =============================================================================
# STEP 9: Whisper stack
# ctranslate2 must be pinned — 4.3.x is the last version that ships
# the bundled libcudnn without needing system CUDA to be perfectly configured
# =============================================================================
echo ">>> [9/13] Installing faster-whisper..."
pip install "ctranslate2==4.3.1" --quiet
pip install "faster-whisper==1.0.3" --quiet

# =============================================================================
# STEP 10: Vision stack
# =============================================================================
echo ">>> [10/13] Installing vision stack..."
pip install "easyocr==1.7.1" --quiet
pip install "opencv-python-headless==4.10.0.84" --quiet
pip install "Pillow==10.4.0" --quiet

# =============================================================================
# STEP 11: Utilities
# =============================================================================
echo ">>> [11/13] Installing utilities..."
pip install \
    "tqdm==4.66.5" \
    "ffmpeg-python==0.2.0" \
    "soundfile>=0.12.1" \
    "librosa>=0.10.2" \
    --quiet

# =============================================================================
# STEP 12: FINAL numpy re-pin — the last safeguard
# Some packages in steps above may have silently upgraded numpy. Nail it down.
# =============================================================================
echo ">>> [12/13] Final NumPy re-pin (last safeguard)..."
pip install "numpy==1.26.4" --force-reinstall --no-deps --quiet

# =============================================================================
# STEP 13: Verification — if this passes, the run will not crash on imports
# =============================================================================
echo ">>> [13/13] Running environment verification..."
python - << 'PYEOF'
import sys
print(f"Python: {sys.version}")

# NumPy — must be 1.x
import numpy as np
assert np.__version__.startswith("1."), f"FATAL: NumPy is {np.__version__}, must be 1.x"
print(f"✓ numpy {np.__version__}")

# pkg_resources — must exist
import pkg_resources
print(f"✓ pkg_resources OK")

# PyTorch + CUDA
import torch
assert torch.cuda.is_available(), "FATAL: CUDA not available"
print(f"✓ torch {torch.__version__}")
print(f"✓ CUDA {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")

# Transformers
import transformers
print(f"✓ transformers {transformers.__version__}")

# matplotlib
import matplotlib
print(f"✓ matplotlib {matplotlib.__version__}")

# faster-whisper
import faster_whisper
print(f"✓ faster-whisper OK")

# pyannote
from pyannote.audio import Pipeline
print(f"✓ pyannote.audio OK")

# EasyOCR
import easyocr
print(f"✓ easyocr OK")

# OpenCV
import cv2
print(f"✓ opencv {cv2.__version__}")

# Pillow
from PIL import Image
import PIL
print(f"✓ Pillow {PIL.__version__}")

print("")
print("╔══════════════════════════════════════╗")
print("║  ALL CHECKS PASSED — READY TO RUN   ║")
print("╚══════════════════════════════════════╝")
PYEOF

echo ""
echo ">>> Setup complete. Run: bash run.sh"
#!/bin/bash
# ============================================================
# FR-01 IMPLEMENTATION — STEP 1: ENVIRONMENT SETUP
# Energy-Efficient Edge-Based DL Framework for Smart Agriculture
# ============================================================
# Run this first. Creates a virtual environment and installs
# all required dependencies for CPU-only training.
# Usage: bash step1_setup.sh

set -e  # Exit on any error

echo "=============================================="
echo " FR-01 Setup: Smart Agriculture Detection"
echo "=============================================="

# 1. Create virtual environment
echo "[1/5] Creating Python virtual environment..."
python -m venv fr01_env
echo "      Done."

# 2. Activate it
echo "[2/5] Activating environment..."
source fr01_env/bin/activate

# 3. Upgrade pip
echo "[3/5] Upgrading pip..."
pip install --upgrade pip --quiet

# 4. Install dependencies
echo "[4/5] Installing dependencies (this may take 3-5 minutes)..."
pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    ultralytics==8.2.0 \
    kaggle \
    Pillow \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    scikit-learn \
    pyyaml \
    tqdm \
    --quiet

echo "      Done."

# 5. Verify installs
echo "[5/5] Verifying installation..."
python -c "
import torch, ultralytics, sklearn, pandas, matplotlib
print(f'  torch      : {torch.__version__}')
print(f'  ultralytics: {ultralytics.__version__}')
print(f'  CPU cores  : {torch.get_num_threads()} threads available')
print('  All packages OK.')
"

echo ""
echo "=============================================="
echo " Setup complete!"
echo " Next step: Run  python step2_dataset.py"
echo "=============================================="

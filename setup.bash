#!/bin/bash
echo "=== GNR Project 2 Setup ==="

# Step 1: Clone the repository
echo ">>> Cloning repository..."
git clone https://github.com/Dibyanshu-2005/GNR-Project-2.git

# Step 2: Copy inference.py to current directory
cp GNR-Project-2/inference.py .

# Step 3: Create conda environment
echo ">>> Creating conda environment (Python 3.11)..."
conda create -n gnr_project_env python=3.11 -y

# Step 4: Install PyTorch with CUDA 12.6 support
echo ">>> Installing PyTorch..."
conda run -n gnr_project_env pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Step 5: Install all dependencies (autoawq required for 72B AWQ model)
echo ">>> Installing dependencies..."
conda run -n gnr_project_env pip install \
    "transformers==4.51.3" \
    "accelerate==1.6.0" \
    "qwen-vl-utils==0.0.10" \
    "pillow==11.2.1" \
    "pandas==2.2.3" \
    "huggingface_hub==0.30.2"

# Step 6: Download model weights via snapshot_download
# (avoids loading into RAM — target system only has 16GB RAM)
echo ">>> Downloading Qwen2.5-VL-7B-Instruct weights (~44GB, this will take a while)..."
conda run -n gnr_project_env python -c "
from huggingface_hub import snapshot_download
print('Downloading Qwen2.5-VL-7B-Instruct...')
snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct')
print('Download complete!')
"

echo "=== Setup complete! ==="
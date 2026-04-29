#!/bin/bash
set -e

echo "=== GNR Project 2 Setup ==="

# Step 1: Clone the repository
echo ">>> Cloning repository..."
git clone https://github.com/Dibyanshu-2005/GNR-Project-2.git

# Step 2: Copy inference.py to current directory
cp GNR-Project-2/inference.py .

# Step 3: Create conda environment
echo ">>> Creating conda environment (Python 3.11)..."
conda create -n gnr_project_env python=3.11 -y

# Step 4: Install PyTorch with CUDA support
echo ">>> Installing PyTorch..."
conda run -n gnr_project_env pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Step 5: Install all other dependencies
echo ">>> Installing dependencies..."
conda run -n gnr_project_env pip install transformers accelerate qwen-vl-utils pillow pandas

# Step 6: Download model weights (saved to HuggingFace cache)
echo ">>> Downloading model weights (this will take a few minutes)..."
conda run -n gnr_project_env python -c "
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
print('Downloading Qwen2.5-VL-7B-Instruct...')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
del model
print('Model downloaded successfully!')
"

echo "=== Setup complete! ==="
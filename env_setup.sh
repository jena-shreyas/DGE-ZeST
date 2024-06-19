#!/bin/zsh
conda create -n DGE python=3.8 -y 
conda activate DGE

# Install torch
# CUDA 11.7
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
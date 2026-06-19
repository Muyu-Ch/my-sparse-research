#!/bin/bash
# ============================================================
# RTX 5060 科研环境一键安装脚本
# 用法: bash setup_5060.sh
# ============================================================
set -e

echo "=== 创建虚拟环境 ==="
python3 -m venv .venv --upgrade-deps
source .venv/bin/activate

echo ""
echo "=== 安装 PyTorch (CUDA 12.6) ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo ""
echo "=== 安装科研依赖 ==="
pip install transformers matplotlib numpy datasets accelerate bitsandbytes

echo ""
echo "=== 验证环境 ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('WARNING: CUDA 不可用，请检查 nvidia-smi')
"

echo ""
echo "=== 环境搭建完成 ==="
echo "激活环境: source .venv/bin/activate"
echo "跑第一个实验: cd experiments && python 01_activation_sparsity.py"

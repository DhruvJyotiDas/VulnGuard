#!/usr/bin/env bash
# MI300X / ROCm environment setup. Run on the GPU box, not in CI.
set -euo pipefail

# 1) sanity: is ROCm visible?
rocm-smi || echo "rocm-smi not found — check ROCm install"

# 2) python env
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 3) PyTorch for ROCm (adjust rocm6.2 to your installed ROCm version)
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch

# 4) this package + extras
pip install -e ".[explain,adversarial]"

# 5) verify the GPU is actually used (ROCm reuses the cuda namespace)
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip/cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
PY

# Notes:
#  * MI300X has 192GB HBM3 — raise TrainConfig.batch_size (64-128) and set
#    grad_accum=1. The paper's batch_size=4 + accum=4 was a small-GPU workaround.
#  * Keep prefer_bf16=True. Do NOT use fp16 with w1=30 (overflow -> collapse).
#  * If you hit MIOpen/flash-attn issues, the models default to eager/SDPA
#    attention; you do not need flash-attn for these encoders.

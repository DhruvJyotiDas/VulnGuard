"""Shared utilities: reproducibility, device selection (CUDA *and* ROCm), logging.

ROCm note: PyTorch's ROCm build reuses the `torch.cuda` namespace, so
`torch.cuda.is_available()` returns True on an MI300X. We therefore do NOT
special-case ROCm for device selection; we only adjust the autocast dtype
(bf16 by default, see train/trainer.py) because MI300X (CDNA3) has native,
stable bf16 and FP16 with extreme loss weights is a known source of training
collapse.
"""
from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np


def set_seed(seed: int) -> None:
    """Seed every RNG we touch. Does NOT force deterministic cuDNN/MIOpen by
    default because that cripples throughput; pass strict=True only when you
    need bit-exact reproduction for the rejoinder."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def set_strict_determinism(seed: int) -> None:
    set_seed(seed)
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover
        pass


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_dtype(prefer_bf16: bool = True):
    """Return the autocast dtype. bf16 on any device that supports it
    (MI300X always does), else fp16, else None (cpu)."""
    import torch

    if not torch.cuda.is_available():
        return None
    if prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_logger(name: str = "vulnguard") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger

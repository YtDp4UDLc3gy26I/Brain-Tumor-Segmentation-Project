# src/utils/device.py
from __future__ import annotations
import os
import torch

def get_device(preferred: str | None = "auto") -> torch.device:
    """
    preferred: one of {"auto","cuda","mps","cpu"} (case-insensitive) or None
    """
    if preferred and preferred.lower() != "auto":
        return torch.device(preferred.lower())

    if torch.cuda.is_available():
        return torch.device("cuda")

    # MPS (Apple Silicon / macOS 12.3+)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Allow silent CPU fallback for ops not implemented on MPS
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")

    return torch.device("cpu")


def non_blocking_for(device: torch.device) -> bool:
    # non_blocking transfers only help on CUDA with pinned memory
    return device.type == "cuda"


def dataloader_kwargs_for(device: torch.device) -> dict:
    # pin_memory is only beneficial for CUDA
    return dict(pin_memory=(device.type == "cuda"))
"""Pick a torch device that actually runs (handles incompatible CUDA builds)."""

from __future__ import annotations

import torch


def resolve_torch_device(requested: str, log_warning=print) -> str:
    """
    `torch.cuda.is_available()` can be True while this PyTorch wheel has no
    kernels for the GPU (cudaErrorNoKernelImageForDevice). Probe before use.
    """
    req = (requested or "cpu").lower()
    if req != "cuda":
        return req
    if not torch.cuda.is_available():
        log_warning("CUDA requested but not available; using CPU.")
        return "cpu"
    try:
        x = torch.zeros(1, device="cuda")
        x.add_(1)
        torch.cuda.synchronize()
        return "cuda"
    except Exception as e:
        log_warning(
            f"CUDA requested but failed ({e}); using CPU. "
            "Install a PyTorch build matching your GPU, or pass --device cpu."
        )
        return "cpu"

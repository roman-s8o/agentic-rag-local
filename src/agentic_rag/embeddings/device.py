from __future__ import annotations


def resolve_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

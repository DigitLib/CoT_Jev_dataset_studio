"""
backend.py - the few tensor operations the v4 engine needs, for either backend.

The torch model (modeling_nanogentzen.GentzenPolicyValueModel) takes a LongTensor; the NumPy model
(numpy_model.NumpyGentzenModel) takes an int array. Both return probabilities from predict_epistemic().
Search and System 1 only ever read those probabilities as NumPy arrays.
"""
from typing import Any, List

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def as_input(rows: List[List[int]], device: str, model: Any):
    """Token id rows -> the input type `model` expects."""
    if getattr(model, "accepts_numpy", False) or torch is None:
        return np.asarray(rows, dtype=np.int64)
    return torch.tensor(rows, dtype=torch.long, device=device)


def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def no_grad(fn):
    return torch.no_grad()(fn) if torch is not None else fn

"""Signed block-Hadamard rotation, the same contract as Prism's PTQ1_0 runtime.

Row vectors use R = D H with D = diag(signs) and H = blockdiag(H_block), the
normalized Sylvester matrix in natural order. Stored weights are W_rot = W R;
the runtime computes x' = x R (signs first, then H) and y = W_rot x'.

Two implementations are kept:

- ``rotate``/``inverse_rotate``/``rotate_hessian``: a fast Walsh-Hadamard
  transform that never builds R. FP16/BF16 inputs accumulate in FP32 and are
  cast back once.
- ``hadamard_matrix``/``rotation_matrix``: the dense construction used by the
  proven experiment scripts. The quantize and endpoint stages use it by
  default (``rotation.impl: dense``) so their codes match those runs bitwise.

Signs are one int8 vector per input width, shared by every tensor of that
width, and are always stored explicitly (Prism keys them by width).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def validate_rotation(signs: Tensor, block_size: int, width: int):
    if type(block_size) is not int or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Hadamard block_size must be a positive power of two")
    if type(width) is not int or width <= 0 or width % block_size:
        raise ValueError("input width must be a positive multiple of block_size")
    if not isinstance(signs, Tensor) or signs.ndim != 1 or signs.numel() != width:
        raise ValueError("one rotation sign is required per input coordinate")
    if signs.is_complex() or not bool(((signs == -1) | (signs == 1)).all()):
        raise ValueError("rotation signs must be exactly -1 or +1")


def normalized_hadamard(x: Tensor, block_size: int) -> Tensor:
    """Differentiable block FWHT with orthonormal scaling and bounded scratch."""
    if not isinstance(x, Tensor) or x.ndim == 0 or not x.is_floating_point():
        raise ValueError("Hadamard input must be a floating tensor with a last axis")
    if type(block_size) is not int or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Hadamard block_size must be a positive power of two")
    if x.shape[-1] == 0 or x.shape[-1] % block_size:
        raise ValueError("input width must be a positive multiple of block_size")
    original_shape, original_dtype = x.shape, x.dtype
    if x.numel() == 0:
        return x.clone()
    y = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    y = y.reshape(-1, block_size)
    width = 1
    while width < block_size:
        groups = y.reshape(-1, block_size // (2 * width), 2, width)
        left, right = groups[..., 0, :], groups[..., 1, :]
        y = torch.stack((left + right, left - right), dim=-2).reshape(-1, block_size)
        width *= 2
    return (y / math.sqrt(block_size)).reshape(original_shape).to(original_dtype)


def rotate(x: Tensor, signs: Tensor, block_size: int) -> Tensor:
    """Apply row-vector R = D H (signs then normalized Hadamard)."""
    if not isinstance(x, Tensor) or x.ndim == 0:
        raise ValueError("rotation input must have a last axis")
    validate_rotation(signs, block_size, x.shape[-1])
    return normalized_hadamard(x * signs.to(device=x.device, dtype=x.dtype), block_size)


def inverse_rotate(x: Tensor, signs: Tensor, block_size: int) -> Tensor:
    """Apply row-vector R^T = H D (normalized Hadamard then signs)."""
    if not isinstance(x, Tensor) or x.ndim == 0:
        raise ValueError("rotation input must have a last axis")
    validate_rotation(signs, block_size, x.shape[-1])
    return normalized_hadamard(x, block_size) * signs.to(device=x.device, dtype=x.dtype)


def rotate_hessian(h: Tensor, signs: Tensor, block_size: int) -> Tensor:
    """R^T H R for a symmetric [width, width] H without forming R."""
    return rotate(rotate(h, signs, block_size).transpose(0, 1).contiguous(), signs, block_size)


# Dense construction (parity path with OpenBonsai research/experiments/gptq_ternary.py).

def hadamard_matrix(n: int, device=None) -> Tensor:
    """Dense normalized Sylvester H_n in FP32 (natural order)."""
    if type(n) is not int or n <= 0 or n & (n - 1):
        raise ValueError("Hadamard size must be a positive power of two")
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def rotation_matrix(signs: Tensor, block_size: int, device=None) -> Tensor:
    """Dense FP32 R = diag(signs) @ blockdiag(H_block); W_rot = W R, x_rot = x R."""
    width = signs.numel()
    validate_rotation(signs, block_size, width)
    s = signs.to(device=device, dtype=torch.float32)
    h = hadamard_matrix(block_size, device)
    r = torch.block_diag(*([h] * (width // block_size)))
    return s[:, None] * r


def seeded_signs(seed: int, width: int) -> Tensor:
    """int8 +/-1 from torch's CPU generator; seed == width reproduces the
    experiments' shared-sign mode (``--shared-signs``)."""
    if type(width) is not int or width <= 0:
        raise ValueError("width must be a positive integer")
    g = torch.Generator().manual_seed(int(seed))
    return (torch.randint(0, 2, (width,), generator=g) * 2 - 1).to(torch.int8)

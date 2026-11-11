"""Portable 2-bit ternary payload used for state codes.

Four codes occupy a byte, least significant code first: 0 -> 00,
+1 -> 01, -1 -> 10. 11 is invalid. Final bytes are zero padded.
This is the run-state storage codec, not Prism's PTQ1_0 (see ptq1_0.py).
BITCOS and base-3 payloads are legacy formats (tag legacy-staged-pipeline).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def _ternary_flat(codes: Tensor) -> Tensor:
    if not isinstance(codes, Tensor) or codes.is_complex():
        raise ValueError("codes must be a real tensor")
    flat = codes.detach().reshape(-1)
    if not bool(((flat == -1) | (flat == 0) | (flat == 1)).all()):
        raise ValueError("ternary codes must be -1, 0, or +1")
    return flat


def pack_ternary_codes(codes: Tensor) -> Tensor:
    """Pack a tensor of {-1, 0, +1} values into a flat uint8 tensor."""
    flat = _ternary_flat(codes)
    encoded = torch.where(flat < 0, 2, flat).to(torch.uint8)
    if encoded.numel() % 4:
        encoded = torch.cat((encoded, encoded.new_zeros(4 - encoded.numel() % 4)))
    slots = encoded.reshape(-1, 4)
    return slots[:, 0] | (slots[:, 1] << 2) | (slots[:, 2] << 4) | (slots[:, 3] << 6)


def unpack_ternary_codes(packed: Tensor, numel: int) -> Tensor:
    """Unpack codes, rejecting reserved codes and nonzero final-byte padding."""
    if isinstance(numel, bool) or not isinstance(numel, int) or numel < 0:
        raise ValueError("numel must be a nonnegative integer")
    if not isinstance(packed, Tensor) or packed.dtype != torch.uint8 or packed.ndim != 1:
        raise ValueError("packed codes must be a flat uint8 tensor")
    if packed.numel() != math.ceil(numel / 4):
        raise ValueError("packed byte count does not match numel")
    slots = torch.stack(tuple((packed >> shift) & 3 for shift in (0, 2, 4, 6)), dim=1).flatten()
    if bool((slots == 3).any()):
        raise ValueError("reserved 2-bit code 11 is invalid")
    if bool((slots[numel:] != 0).any()):
        raise ValueError("final-byte padding must be zero")
    slots = slots[:numel].to(torch.int8)
    return torch.where(slots == 2, -1, slots)

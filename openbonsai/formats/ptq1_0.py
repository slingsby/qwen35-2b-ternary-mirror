"""Prism PTQ1_0 block codec (encode and decode), numpy only.

Layout checked against Prism llama.cpp c9c34bc (ggml-common.h, ggml-quants.c):

- type id 143; one block = 128 consecutive rotated input weights of one row
  in 28 bytes: ``uint8 qs[24]; uint8 qh[2]; fp16 d`` (little-endian);
- digit = code + 1 in {0, 1, 2}; a byte packs base-3 digits most significant
  first, q = sum(d_n * 3**(4 - n)), and stores ceil(q * 256 / 243);
- ``qs[m]``, m < 16, holds weights m + 16n (n = 0..4);
  ``qs[16 + m]``, m < 8, holds weights 80 + m + 8n;
  ``qh[h]``, h < 2, holds four weights 120 + h + 2m (m = 0..3), q *= 3 first;
- decode digit n of byte b: ((uint8)(b * 3**n) * 3) >> 8; weight = (digit - 1) * d.

Canonical form (``canonicalize``), the bytes OpenBonsai research/experiments/export_gguf.py
writes and ``quantize_row_ptq1_0_ref`` would write for the same function: a
scale with the fp16 sign bit set (including -0.0) negates its group's codes
and clears the bit; then a zero scale zeroes its group's codes (they are
dead). An all-zero group with a nonzero scale keeps the caller's scale and
encodes as qs = 128, qh = 127 (Prism's own quantizer writes d = 0 there).
"""
from __future__ import annotations

import numpy as np

GGML_TYPE_PTQ1_0 = 143
LLAMA_FTYPE_MOSTLY_PTQ1_0 = 143   # include/llama.h (gguf-py's LlamaFileType says 129; the C++ side uses 143)
QK = 128
BLOCK_BYTES = 28

_POW3_DEC = np.array([1, 3, 9, 27, 81], dtype=np.uint16)
# C assigns the multiplication to uint8 BEFORE multiplying by three.
DECODE_LUT = ((((np.arange(256, dtype=np.uint16)[:, None] * _POW3_DEC) & 255) * 3) >> 8).astype(np.int8) - 1


def _as_numpy(x, dtype=None):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:  # pragma: no cover
        pass
    x = np.asarray(x)
    return x.astype(dtype) if dtype is not None and x.dtype != dtype else x


def canonicalize(codes: np.ndarray, sbits: np.ndarray) -> tuple[int, int]:
    """Canonicalize groups in place; same function, canonical bytes.

    ``codes``: contiguous int8 [rows, cols]; ``sbits``: uint16 fp16 bit patterns
    [rows, cols/128]. A scale with the sign bit set (incl. -0.0) negates the
    group's codes and clears the bit; then a zero scale zeroes the group's
    codes. Returns (negative-scale groups, zero-scale groups whose nonzero
    codes were cleared)."""
    rows, g = sbits.shape
    cv = codes.reshape(rows, g, QK)
    if not np.shares_memory(cv, codes):
        raise ValueError("codes must be contiguous for in-place canonicalization")
    neg = (sbits & 0x8000) != 0
    n_neg = int(neg.sum())
    if n_neg:
        cv[neg] = -cv[neg]
        sbits &= 0x7FFF
    dead = (sbits == 0) & cv.any(-1)
    n_dead = int(dead.sum())
    if n_dead:
        cv[dead] = 0
    return n_neg, n_dead


def pack(codes: np.ndarray, sbits: np.ndarray) -> np.ndarray:
    """Pack canonical codes int8 [rows, cols] and fp16 scale bits uint16 [rows, cols/128]
    into uint8 [rows, cols/128*28]. Arithmetic form of the spec's verified einsum
    packer (tests check byte equality with it)."""
    rows, width = codes.shape
    if width % QK or sbits.shape != (rows, width // QK):
        raise ValueError(f"pack needs [rows, k*128] codes and [rows, k] scale bits, got {codes.shape}, {sbits.shape}")
    nb = rows * (width // QK)
    d = (codes.reshape(nb, QK) + np.int8(1)).astype(np.uint16)
    out = np.empty((nb, BLOCK_BYTES), np.uint8)
    q = 81 * d[:, 0:16] + 27 * d[:, 16:32] + 9 * d[:, 32:48] + 3 * d[:, 48:64] + d[:, 64:80]
    out[:, 0:16] = (q * 256 + 242) // 243
    q = 81 * d[:, 80:88] + 27 * d[:, 88:96] + 9 * d[:, 96:104] + 3 * d[:, 104:112] + d[:, 112:120]
    out[:, 16:24] = (q * 256 + 242) // 243
    q = (27 * d[:, 120:122] + 9 * d[:, 122:124] + 3 * d[:, 124:126] + d[:, 126:128]) * 3
    out[:, 24:26] = (q * 256 + 242) // 243
    out[:, 26:28] = np.ascontiguousarray(sbits, dtype="<u2").reshape(nb, 1).view(np.uint8)
    return out.reshape(rows, (width // QK) * BLOCK_BYTES)


def check(codes: np.ndarray, sbits: np.ndarray) -> None:
    """Codes in {-1, 0, 1} and finite fp16 scales, else ValueError."""
    if codes.size and (codes.min() < -1 or codes.max() > 1):
        raise ValueError("PTQ1_0 codes must be -1, 0 or +1")
    if ((sbits & 0x7C00) == 0x7C00).any():
        raise ValueError("PTQ1_0 scales must be finite")


def normalize_scales(codes, scales):
    """Return canonical (codes int8, scales fp16) copies (see ``canonicalize``)."""
    codes = np.ascontiguousarray(_as_numpy(codes), dtype=np.int8).copy()
    sbits = np.ascontiguousarray(_as_numpy(scales).astype(np.float16)).view(np.uint16).copy()
    check(codes, sbits)
    canonicalize(codes, sbits)
    return codes, sbits.view(np.float16)


def encode(codes, scales) -> np.ndarray:
    """codes int8 {-1,0,1} [rows, cols], scales fp16 [rows, cols/128] -> uint8 [rows, cols/128*28]."""
    codes = _as_numpy(codes)
    scales = _as_numpy(scales)
    if codes.ndim != 2 or codes.shape[1] % QK:
        raise ValueError("PTQ1_0 needs 2-D codes whose width is a multiple of 128")
    rows, cols = codes.shape
    if scales.shape != (rows, cols // QK):
        raise ValueError(f"scales must be [rows, cols/128] = {(rows, cols // QK)}, got {scales.shape}")
    if not np.isin(codes, (-1, 0, 1)).all():
        raise ValueError("PTQ1_0 codes must be -1, 0 or +1")
    codes, scales = normalize_scales(codes, scales)
    return pack(codes, scales.view(np.uint16))


def decode(blocks, cols: int | None = None):
    """uint8 [rows, nb*28] (or [nb, 28]) -> (codes int8 [rows, nb*128], scales fp16 [rows, nb])."""
    blocks = _as_numpy(blocks, np.uint8)
    if blocks.ndim == 1:
        blocks = blocks.reshape(1, -1)
    rows = blocks.shape[0]
    if blocks.shape[1] % BLOCK_BYTES:
        raise ValueError("PTQ1_0 rows must be a whole number of 28-byte blocks")
    nb = blocks.shape[1] // BLOCK_BYTES
    if cols is not None and cols != nb * QK:
        raise ValueError(f"expected {cols // QK} blocks per row, found {nb}")
    b = np.ascontiguousarray(blocks).reshape(-1, BLOCK_BYTES)
    codes = np.concatenate((DECODE_LUT[b[:, :16]].transpose(0, 2, 1).reshape(-1, 80),
                            DECODE_LUT[b[:, 16:24]].transpose(0, 2, 1).reshape(-1, 40),
                            DECODE_LUT[b[:, 24:26], :4].transpose(0, 2, 1).reshape(-1, 8)), axis=1)
    scales = np.ascontiguousarray(b[:, 26:28]).view("<f2").reshape(rows, nb).astype(np.float16)
    return codes.reshape(rows, nb * QK), scales


def dequantize(blocks) -> np.ndarray:
    """FP32 weights codes * d, exactly what ``dequantize_row_ptq1_0`` returns."""
    codes, scales = decode(blocks)
    rows, cols = codes.shape
    return (codes.reshape(rows, -1, QK).astype(np.float32)
            * scales.astype(np.float32)[..., None]).reshape(rows, cols)


def nbytes(rows: int, cols: int) -> int:
    if cols % QK:
        raise ValueError("width must be a multiple of 128")
    return rows * cols // QK * BLOCK_BYTES

"""GPTQ error feedback with ternary group codes.

``fit_group`` and ``gptq_ternary`` are copied unchanged from
research/experiments/gptq_ternary.py (the run that measured dev NLL 3.30);
tests/test_gptq_parity.py checks them bitwise against an embedded oracle.
The old ``refit_scales`` option is removed: refitting the scales against the
original rotated weights undid GPTQ's error correction (NLL 5.15 vs 3.32).

Format: codes {-1, 0, 1} in groups of ``group`` consecutive rotated input
columns, one scale per (row, group). The dequantized rotated weight is
``codes * scales`` and the original-basis weight is that times R^T.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .hadamard import inverse_rotate, rotate, rotate_hessian, rotation_matrix

DEFAULT_RATIOS = tuple(0.3 + 0.05 * i for i in range(19))


def fit_group(w, ratios):
    """Per-row ternary (alpha, delta) for one [rows, group] block, least squares over a threshold grid."""
    a = w.abs()
    m = a.mean(1, keepdim=True)
    best_err, best_alpha, best_delta = None, None, None
    for r in ratios:
        delta = r * m
        mask = a > delta
        alpha = (a * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)
        err = (a - alpha * mask).pow(2).sum(1, keepdim=True)
        if best_err is None:
            best_err, best_alpha, best_delta = err, alpha, delta
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_alpha = torch.where(better, alpha, best_alpha)
            best_delta = torch.where(better, delta, best_delta)
    return best_alpha, best_delta


@torch.no_grad()
def gptq_ternary(w, h, group=128, damp=0.01, ratios=None):
    """GPTQ error feedback with ternary group codes. w: [out, in] FP32 (rotated), h: [in, in]."""
    ratios = ratios or [0.3 + 0.05 * i for i in range(19)]
    w = w.clone()
    out_f, in_f = w.shape
    dead = torch.diag(h) == 0
    h = h.clone()
    h[dead, dead] = 1
    w[:, dead] = 0
    h += damp * torch.mean(torch.diag(h)) * torch.eye(in_f, device=h.device)
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    codes = torch.zeros_like(w, dtype=torch.int8)
    scales = torch.zeros(out_f, in_f // group, device=w.device)
    for i1 in range(0, in_f, group):
        i2 = i1 + group
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        alpha, delta = fit_group(w1, ratios)
        for j in range(group):
            col = w1[:, j]
            c = torch.sign(col) * (col.abs() > delta[:, 0])
            q = c * alpha[:, 0]
            codes[:, i1 + j] = c.to(torch.int8)
            e = (col - q) / hinv1[j, j]
            w1[:, j:] -= e[:, None] * hinv1[j, j:][None, :]
            err1[:, j] = e
        scales[:, i1 // group] = alpha[:, 0]
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return codes, scales


class Hessian:
    """Running X^T X over every token row seen by one linear's input."""

    def __init__(self):
        self.sum: Tensor | None = None
        self.count = 0

    def add(self, x: Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        hh = x.T @ x
        self.sum = hh if self.sum is None else self.sum + hh
        self.count += x.shape[0]

    def mean(self) -> Tensor:
        if self.sum is None or not self.count:
            raise RuntimeError("Hessian saw no calibration rows")
        return self.sum / self.count


def dequantize(codes: Tensor, scales: Tensor) -> Tensor:
    """codes * scales in FP32, in the rotated basis."""
    rows, cols = codes.shape
    group = cols // scales.shape[1]
    return (codes.float().view(rows, -1, group) * scales.float()[..., None]).view(rows, cols)


@torch.no_grad()
def quantize_linear(weight: Tensor, h_mean: Tensor, signs: Tensor, block: int, *, group: int = 128,
                    damp: float = 0.01, impl: str = "dense"):
    """GPTQ-ternary one weight [out, in] given its input Hessian (original basis).

    Returns codes int8, scales fp16, the dequantized weight in the original
    basis (FP32, what the runtime computes) and error statistics. The dense
    path is the experiments' arithmetic, operation for operation.
    """
    w = weight.float()
    if impl == "dense":
        rot = rotation_matrix(signs, block, w.device)
        h = rot.T @ h_mean @ rot
        w_rot = w @ rot
    elif impl == "fwht":
        rot = None
        h = rotate_hessian(h_mean, signs, block)
        w_rot = rotate(w, signs, block)
    else:
        raise ValueError(f"unknown rotation impl {impl!r}")
    codes, scales = gptq_ternary(w_rot, h, group=group, damp=damp)
    scales = scales.half().float()  # what gets stored and exported
    deq = (codes.float().view(w.shape[0], -1, group) * scales[..., None]).view_as(w)
    err = deq - w_rot
    stats = {"rel_mse": (err.pow(2).sum() / w_rot.pow(2).sum()).item(),
             "h_rel_err": ((err @ h) * err).sum().item() / ((w_rot @ h) * w_rot).sum().item()}
    w_new = deq @ rot.T if rot is not None else inverse_rotate(deq, signs, block)
    return codes, scales.half(), w_new, stats


@torch.no_grad()
def quantize_rows(weight: Tensor, h: Tensor, signs: Tensor, block: int, *, group: int = 128,
                  damp: float = 0.01, chunk: int = 16384, impl: str = "dense"):
    """GPTQ-ternary a large [rows, in] matrix (head, embedding) in row chunks on h.device.

    Rows are independent given H. Returns CPU codes, fp16 scales and the
    dequantized original-basis weight in ``weight.dtype``.
    """
    rows, in_f = weight.shape
    dev = h.device
    if impl == "dense":
        rot = rotation_matrix(signs, block, dev)
        h_rot = rot.T @ h @ rot
    elif impl == "fwht":
        rot = None
        h_rot = rotate_hessian(h, signs, block)
    else:
        raise ValueError(f"unknown rotation impl {impl!r}")
    codes = torch.empty(rows, in_f, dtype=torch.int8)
    scales = torch.empty(rows, in_f // group, dtype=torch.float16)
    deq = torch.empty(rows, in_f, dtype=weight.dtype)
    for r0 in range(0, rows, chunk):
        wf = weight[r0:r0 + chunk].to(dev).float()
        w = wf @ rot if rot is not None else rotate(wf, signs, block)
        c, s = gptq_ternary(w, h_rot, group=group, damp=damp)
        codes[r0:r0 + chunk] = c.cpu()
        scales[r0:r0 + chunk] = s.half().cpu()
        d = (c.float().view(c.shape[0], -1, group) * s.half().float()[..., None]).view_as(w)
        back = d @ rot.T if rot is not None else inverse_rotate(d, signs, block)
        deq[r0:r0 + chunk] = back.to(weight.dtype).cpu()
    return codes, scales, deq

"""Ternary modules with trainable group scales (codes frozen).

``TernaryLinear`` and ``TernaryEmbedding`` are the modules of
research/experiments/e2e_kd.py, unchanged in arithmetic: BF16 signs, a shared
BF16 dense Hadamard block, and scales ``s0 * exp(log_mult)``. The forward is
what Prism's runtime computes: y = (C * s) (R x) with R = blockdiag(H) diag(signs)
in column-vector form (signs first, then H; core/hadamard.py writes the same
rotation for row vectors as x D H). The one addition: a Linear's bias, if it
has one, is kept as a frozen buffer and added (the scripts' Qwen3.5 has none).

``dense_weight`` is the experiments' ``apply_state`` arithmetic: it writes
the dequantized weight (C * s) R^T back into an ordinary Linear, which is
how the decoder-only and endpoint results were scored.

BF16 remains the default. A float16 Hadamard selects FP16 arithmetic for
hardware such as T4 that lacks native BF16 support.

``LatentTernaryLinear`` (the refine stage, docs/refine.md) also trains the
codes: they come from an FP32 latent weight through a straight-through
estimator, and its forward is bit for bit a ``TernaryLinear``'s.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .hadamard import hadamard_matrix, rotation_matrix


class TernaryLinear(torch.nn.Module):
    """y = (C * s) (R x) + b with R = blockdiag(H) diag(signs); s = s0 * exp(u)."""

    def __init__(self, codes, scales, signs, block, h, bias=None):
        super().__init__()
        self.out_features, self.in_features = codes.shape
        self.group = self.in_features // scales.shape[1]
        self.block = block
        self.register_buffer("codes", codes.to(torch.int8), persistent=False)
        self.register_buffer("s0", scales.float(), persistent=False)
        self.register_buffer("signs", signs.to(h.dtype), persistent=False)
        self.h = h  # shared [block, block] BF16 Hadamard, not a parameter
        self.register_buffer("bias", None if bias is None else bias.detach().clone(), persistent=False)
        self.log_mult = torch.nn.Parameter(torch.zeros_like(self.s0))

    def scales(self):
        return self.s0 * torch.exp(self.log_mult)

    def weight_deq(self):
        s = self.scales().to(self.h.dtype)
        c = self.codes.view(self.out_features, -1, self.group).to(self.h.dtype)
        return (c * s[..., None]).view(self.out_features, self.in_features)

    def forward(self, x):
        shape = x.shape
        xr = (x * self.signs).view(*shape[:-1], self.in_features // self.block, self.block) @ self.h
        return F.linear(xr.view(shape), self.weight_deq(), self.bias)


ASSIGNS = ("nearest", "quantile")
INITS = ("center", "original")
LATENT_CLAMP = 1.5


def _group_ranks(values: Tensor, group: int) -> Tensor:
    """Rank (0 = smallest) of each entry within its group of ``group`` consecutive columns, ties by
    position; uint8 [rows, cols] (group <= 256)."""
    rows, cols = values.shape
    v = values.reshape(-1, group)
    order = torch.argsort(v, dim=1, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(group, device=v.device).expand_as(order).contiguous())
    return ranks.to(torch.uint8).view(rows, cols)


class LatentTernaryLinear(TernaryLinear):
    """A TernaryLinear whose codes are trained through an FP32 latent weight (docs/refine.md).

    ``latent`` V [out, in] is in units of each group's input scale S0 (the
    latent weight is S0 * V), so one learning rate moves every group's
    thresholds at the same rate. The scale is s = S0 * exp(log_mult), as in
    distillation. Codes:

    - ``nearest``: C = clamp(round(V * exp(-log_mult)), -1, 1), the threshold
      at s/2 in weight units (the least-squares code for a given scale);
    - ``quantile``: in each (row, group) the ``k`` smallest |V| are 0 and the
      rest sign(V); ``k`` moves from the group's input zero count to ``zeros``
      as ``set_anneal`` goes from 0 to 1. Ties in |V| go by the magnitude of
      the original rotated weight, and a zero latent that must become nonzero
      takes that weight's sign.

    The weight is q = s * (C + (V - V.detach())) in FP32, cast to BF16: the
    added term is exactly zero, so the forward is a TernaryLinear's bit for
    bit, and the backward is the identity straight-through estimator
    (dL/dV = s * dL/dq). ``init="center"`` starts at V = C0 (the input codes),
    so the module computes exactly what its TernaryLinear did;
    ``init="original"`` adds the clipped residual of the original rotated
    weight, (W_rot - C0 * S0) / S0 within +/-0.25, which leaves the codes at
    C0. ``freeze()`` fixes the codes (only the scales train after it);
    ``hard()`` returns the int8 codes and fp16 scales the state stores.
    """

    def __init__(self, codes, scales, signs, block, h, bias=None, *, assign: str = "nearest", init: str = "center",
                 w_rot: Tensor | None = None, zeros: int = 42):
        super().__init__(codes, scales, signs, block, h, bias=bias)
        if assign not in ASSIGNS:
            raise ValueError(f"assign must be one of {ASSIGNS}, not {assign!r}")
        if init not in INITS:
            raise ValueError(f"init must be one of {INITS}, not {init!r}")
        if (init == "original" or assign == "quantile") and w_rot is None:
            raise ValueError(f"init={init!r}, assign={assign!r} needs the original rotated weight (w_rot)")
        self.assign = assign
        self.frozen = False
        # own copies: the codes change in place, and an int8 input is not copied by TernaryLinear
        self.register_buffer("codes", self.codes.clone(), persistent=False)
        self.register_buffer("codes0", self.codes.clone(), persistent=False)
        c0 = self.codes.float()
        v = c0.clone()
        if init == "original":
            s0 = self._expand(self.s0)
            safe = torch.where(s0 > 0, s0, torch.ones_like(s0))
            residual = torch.where(s0 > 0, (w_rot.float() - c0 * s0) / safe, torch.zeros_like(s0))
            v = c0 + residual.clamp(-0.25, 0.25)
        self.latent = torch.nn.Parameter(v)
        if assign == "quantile":
            if not 0 <= int(zeros) <= self.group or self.group > 256:
                raise ValueError(f"zeros must be in [0, {self.group}] (group <= 256)")
            zeros0 = (self.codes == 0).view(self.out_features, -1, self.group).sum(-1)
            self.register_buffer("zeros0", zeros0.to(torch.int16), persistent=False)
            self.register_buffer("k", zeros0.to(torch.int16), persistent=False)
            self.zeros_target = int(zeros)
            self.register_buffer("tie", _group_ranks(w_rot.float().abs(), self.group), persistent=False)
            self.register_buffer("fallback", torch.where(w_rot < 0, -1, 1).to(torch.int8), persistent=False)

    def _expand(self, per_group: Tensor) -> Tensor:
        """[out, in/group] -> [out, in] (each group's value on its columns)."""
        return per_group[..., None].expand(self.out_features, -1, self.group).reshape(self.out_features,
                                                                                       self.in_features)

    @torch.no_grad()
    def set_anneal(self, frac: float) -> None:
        """``quantile``: zeros per group = round(input count + (zeros - input count) * frac)."""
        if self.assign != "quantile":
            return
        frac = min(max(float(frac), 0.0), 1.0)
        z0 = self.zeros0.float()
        self.k.copy_(torch.round(z0 + (self.zeros_target - z0) * frac).to(torch.int16))

    @torch.no_grad()
    def assign_codes(self) -> Tensor:
        """Codes the latent and scales give now (int8 [out, in]); the frozen codes once frozen."""
        if self.frozen:
            return self.codes
        v = self.latent.detach()
        if self.assign == "nearest":
            scaled = v * self._expand(torch.exp(-self.log_mult.detach()))
            return torch.round(scaled).clamp_(-1, 1).to(torch.int8)
        # quantile: exact lexicographic order (|V| first, then the original weight's rank) via the
        # order-preserving int32 bits of a nonnegative float
        bits = v.abs().view(torch.int32).to(torch.int64)
        key = ((bits << 8) | self.tie.to(torch.int64)).view(-1, self.group)
        k = self.k.view(-1, 1).to(torch.int64)
        kth = key.sort(dim=1).values.gather(1, (k - 1).clamp_min(0))
        zero = (key <= kth) & (k > 0)
        sign = torch.where(v > 0, 1, torch.where(v < 0, -1, self.fallback.to(torch.int64))).to(torch.int8)
        return torch.where(zero.view_as(v), torch.zeros_like(sign), sign)

    @torch.no_grad()
    def freeze(self) -> None:
        """Fix the codes at their current values; the latent gets no more gradient."""
        if not self.frozen:
            self.codes.copy_(self.assign_codes())
            self.frozen = True
        self.latent.requires_grad_(False)
        self.latent.grad = None

    def weight_deq(self):
        s = self.scales()
        if self.frozen:
            c = self.codes.float()
        else:
            codes = self.assign_codes()
            self.codes.copy_(codes)
            c = codes.float() + (self.latent - self.latent.detach())
        q = c.view(self.out_features, -1, self.group) * s[..., None]
        return q.view(self.out_features, self.in_features).to(torch.bfloat16)

    @torch.no_grad()
    def clamp_latent(self) -> None:
        self.latent.clamp_(-LATENT_CLAMP, LATENT_CLAMP)

    @torch.no_grad()
    def hard(self) -> tuple[Tensor, Tensor]:
        """(int8 codes, fp16 scales) on the CPU: what the state stores."""
        return self.assign_codes().cpu().clone(), self.scales().detach().to(torch.float16).cpu()


class TernaryEmbedding(torch.nn.Module):
    """e = (C[ids] * s[ids]) R^T, i.e. stored rows live in the rotated basis."""

    def __init__(self, codes, scales, signs, block, h):
        super().__init__()
        self.num, self.dim = codes.shape
        self.group = self.dim // scales.shape[1]
        self.block = block
        self.register_buffer("codes", codes.to(torch.int8), persistent=False)
        self.register_buffer("s0", scales.float(), persistent=False)
        self.register_buffer("signs", signs.to(h.dtype), persistent=False)
        self.h = h
        self.log_mult = torch.nn.Parameter(torch.zeros_like(self.s0))

    def scales(self):
        return self.s0 * torch.exp(self.log_mult)

    def forward(self, ids):
        c = self.codes[ids].view(*ids.shape, -1, self.group).to(self.h.dtype)
        s = (self.s0[ids] * torch.exp(self.log_mult[ids])).to(self.h.dtype)
        rows = (c * s[..., None]).view(*ids.shape, self.dim // self.block, self.block)
        return (rows @ self.h).view(*ids.shape, self.dim) * self.signs


def shared_hadamard(block: int, device, dtype=torch.bfloat16) -> Tensor:
    """The BF16 Hadamard block shared by every ternary module."""
    return hadamard_matrix(block, device).to(dtype)


@torch.no_grad()
def dense_weight(codes: Tensor, scales: Tensor, signs: Tensor, block: int, *, device, dtype) -> Tensor:
    """(C * s) R^T in ``dtype``: the experiments' ``apply_state`` arithmetic."""
    codes = codes.to(device).float()
    scales = scales.to(device).float()
    out_f, in_f = codes.shape
    rot = rotation_matrix(signs, block, codes.device)
    deq = (codes.view(out_f, -1, in_f // scales.shape[1]) * scales[..., None]).view(out_f, in_f)
    return (deq @ rot.T).to(dtype)

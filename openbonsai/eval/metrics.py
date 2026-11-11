"""Teacher-vs-student next-token metrics.

``p`` is the teacher (the original BF16 model), ``q`` the student (the
compressed model), both next-token distributions at one position. Every
metric is computed per position in FP64 and averaged over positions.

Full vocabulary (needs the teacher's logits at every position):

- ``ear_full``: expected acceptance rate, ``sum_v min(p(v), q(v)) = 1 - TV(p, q)``:
  the probability that speculative sampling accepts a token drafted from
  ``q`` for the target ``p`` (Leviathan et al., 2023). 1 for identical models.
- ``kl_full``: ``KL(p || q)``.

From the teacher's top-k only (stored top-k log-probabilities):

- ``ear_top{k}_lb``: ``sum`` over the teacher top-k of ``min(p_k, q_k)``. A lower
  bound on ``ear_full`` (it drops the non-negative terms outside the top-k).
- ``ear_top{k}_ub``: the lower bound plus ``min(p_tail, q_tail)`` for the mass
  outside the top-k. An upper bound on ``ear_full`` (the EAR of the
  distributions with the tail merged into one bucket).
- ``kl_top{k}``: KL on the top-k plus that tail bucket (``core.losses.topk_kl``).
  A lower bound on ``kl_full`` (merging outcomes never increases KL).

The bounds are exact when the stored ``p_k`` are exact: the ``compare``
reference keeps FP64 log-probabilities. Distillation's teacher shards store
fp16 log-probabilities, so there they hold up to that rounding.
"""
from __future__ import annotations

import torch

from ..core.losses import topk_kl

# Positions per FP64 [chunk, vocab] slab: ~127 MB per tensor at Qwen3.5's 248,320 tokens.
CHUNK = 64

__all__ = ["CHUNK", "full_vocab", "topk_ear", "topk_kl"]


def full_vocab(ref_logits: torch.Tensor, logq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position ``(ear, kl)`` over the full vocabulary.

    ``ref_logits``: [n, V] teacher logits (any float dtype); ``logq``: [n, V]
    student log-probabilities (FP64). Returns two [n] FP64 tensors.
    """
    lp = ref_logits.double().log_softmax(-1)
    p = lp.exp()
    ear = torch.minimum(p, logq.exp()).sum(-1)
    kl = torch.where(p > 0, p * (lp - logq), 0.0).sum(-1)
    return ear, kl


def topk_ear(tidx: torch.Tensor, tlogp: torch.Tensor, logq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position ``(lower, upper)`` bounds on the full EAR from the teacher top-k.

    ``tidx``: [n, k] teacher token ids; ``tlogp``: [n, k] teacher log-probabilities;
    ``logq``: [n, V] student log-probabilities (FP64). Returns two [n] FP64 tensors.
    """
    p = tlogp.double().exp()
    lqk = logq.gather(-1, tidx).double()
    lower = torch.minimum(p, lqk.exp()).sum(-1)
    tail_p = (1 - p.sum(-1)).clamp_min(0)
    tail_q = (-torch.expm1(lqk.logsumexp(-1).clamp_max(0))).clamp_min(0)
    return lower, lower + torch.minimum(tail_p, tail_q)

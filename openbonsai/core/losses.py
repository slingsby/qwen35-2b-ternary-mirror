"""Distillation losses against a teacher's stored top-k log-probabilities."""
from __future__ import annotations

import torch


def kd_chunk_loss(h, head, tidx, tlogp):
    """Cross-entropy to the teacher's top-k plus one tail bucket for its remaining mass,
    so the optimum is q = p on the top-k (not a sharpened, tail-free distribution).

    Unchanged from OpenBonsai research/experiments/e2e_kd.py. ``h``: [T, hidden] student
    hidden states; ``tidx``: [T, k] teacher token ids; ``tlogp``: [T, k]
    teacher log-probabilities (fp16). Returns the summed (not mean) loss.
    """
    logq = head(h).float().log_softmax(-1)
    lqk = logq.gather(-1, tidx)
    p = tlogp.float().exp()
    tail_p = (1 - p.sum(-1)).clamp_min(0)
    log_tail_q = torch.log(-torch.expm1(lqk.logsumexp(-1).clamp_max(-1e-6)))
    return -(p * lqk).sum() - (tail_p * log_tail_q).sum()


def topk_kl(tidx, tlogp, logq):
    """KL(p || q) per row restricted to the teacher top-k plus the tail bucket.

    ``logq``: [T, V] student log-probabilities (FP64 recommended). Returns [T].
    """
    p_log = tlogp.double()
    p = p_log.exp()
    lqk = logq.gather(-1, tidx).double()
    tail_p = (1 - p.sum(-1)).clamp_min(0)
    tail_q = (-torch.expm1(lqk.logsumexp(-1).clamp_max(-1e-12))).clamp_min(1e-300)
    kl = (p * (p_log - lqk)).sum(-1)
    return kl + torch.where(tail_p > 0, tail_p * (tail_p.clamp_min(1e-300).log() - tail_q.log()),
                            torch.zeros_like(tail_p))

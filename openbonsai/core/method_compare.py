"""Matched full-model STE training modules (TrainTable, TrainLinear, TrainEmbedding) used by recipe/train.py.

Not used by the production pipeline. One shared table owns endpoint codes/scales/master.
The PV proposal uses the actual hard-forward gradient, a factored preconditioner,
nearest ternary assignment and explicit physical-weight trust bounds.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .ternary import TernaryLinear, TernaryEmbedding


class _HardWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, master, log_scale, base_scale, dtype):
        scales = base_scale * log_scale.exp()
        group = master.shape[-1] // scales.shape[-1]
        flat = master.detach().reshape(-1, master.shape[-1])
        codes = torch.empty_like(master, dtype=torch.int8)
        result = torch.empty_like(master, dtype=dtype)
        cf, rf, sf = codes.reshape_as(flat), result.reshape_as(flat), scales.reshape(-1, scales.shape[-1])
        for lo in range(0, len(flat), 4096):
            hi = lo + 4096
            cf[lo:hi] = flat[lo:hi].round().clamp_(-1, 1).to(torch.int8)
            rf[lo:hi] = (cf[lo:hi].reshape(-1, sf.shape[-1], group).to(dtype)
                         * sf[lo:hi].to(dtype)[..., None]).reshape_as(rf[lo:hi])
        ctx.save_for_backward(codes, scales)
        ctx.group = group
        return result

    @staticmethod
    def backward(ctx, grad):
        codes, scales = ctx.saved_tensors
        # dV=s*dq and dlog(s)=sum(q*dq); finite, explicit, shared by both uses.
        master_grad = torch.empty_like(grad, dtype=torch.float32)
        scale_grad = torch.empty_like(scales)
        gf, mf, cf = (a.reshape(-1, grad.shape[-1]) for a in (grad, master_grad, codes))
        sf, uf = scales.reshape(-1, scales.shape[-1]), scale_grad.reshape(-1, scales.shape[-1])
        for lo in range(0, len(gf), 4096):
            hi = lo + 4096
            weighted = gf[lo:hi].float().reshape(-1, sf.shape[-1], ctx.group) * sf[lo:hi, :, None]
            mf[lo:hi] = weighted.reshape_as(mf[lo:hi])
            uf[lo:hi] = (weighted * cf[lo:hi].reshape_as(weighted)).sum(-1)
        return master_grad, scale_grad, None, None


class TrainTable(nn.Module):
    def __init__(self, old, *, residual=None):
        super().__init__()
        self.register_buffer('s0', old.s0.detach().clone())
        # Provenance only: retaining every frozen reference on CUDA wastes ~0.75GB.
        self.register_buffer('initial_codes', old.codes.detach().cpu().clone())
        self.log_mult = old.log_mult
        self.group = old.group
        self.dtype = old.h.dtype
        v = old.codes.float()
        if residual is not None:
            v.add_(residual.clamp(-.49, .49))
        self.master = nn.Parameter(v)
        self.hard_only = False

    def codes(self):
        result = torch.empty_like(self.master, dtype=torch.int8)
        for lo in range(0, len(result), 4096):
            result[lo:lo+4096] = self.master.detach()[lo:lo+4096].round().clamp_(-1, 1).to(torch.int8)
        return result

    def scales(self):
        return self.s0 * self.log_mult.exp()

    def weight(self, ids=None):
        v, u, s = self.master, self.log_mult, self.s0
        if ids is not None:
            v, u, s = v[ids], u[ids], s[ids]
        if self.hard_only:
            return (v.detach().round().clamp(-1, 1).reshape(*s.shape, self.group).to(self.dtype)
                    * (s * u.exp()).to(self.dtype)[..., None]).reshape_as(v)
        return _HardWeight.apply(v, u, s, self.dtype)


class TrainLinear(TernaryLinear):
    def __init__(self, old, table):
        nn.Module.__init__(self)
        self.table = table
        self.in_features, self.out_features = old.in_features, old.out_features
        self.block, self.h = old.block, old.h
        self.register_buffer('signs', old.signs)
        self.register_buffer('bias', old.bias)

    def forward(self, x):
        shape = x.shape
        xr = (x * self.signs).reshape(*shape[:-1], self.in_features // self.block, self.block) @ self.h
        return F.linear(xr.reshape(shape), self.table.weight(), self.bias)


class TrainEmbedding(TernaryEmbedding):
    def __init__(self, old, table):
        nn.Module.__init__(self)
        self.table = table
        self.num, self.dim, self.block, self.h = old.num, old.dim, old.block, old.h
        self.register_buffer('signs', old.signs)

    def forward(self, ids):
        rows = self.table.weight(ids).reshape(*ids.shape, self.dim // self.block, self.block)
        return (rows @ self.h).reshape(*ids.shape, self.dim) * self.signs


@torch.no_grad()
def sparse_project(proposal, old_codes, scales, *, fraction=.01, trust=.01):
    """Nearest-code projection with incumbent option, fraction and physical-L2 caps.

    Returns int8 codes plus actual accounting. Bounds apply per table, hence
    also bound their disjoint global concatenation. Stable tie order is explicit.
    """
    if not 0 < fraction <= 1 or not 0 < trust <= 1:
        raise ValueError('fraction and trust must be in (0,1]')
    group = proposal.shape[-1] // scales.shape[-1]
    old = old_codes.flatten()
    candidate = proposal.round().clamp(-1, 1).to(torch.int8).flatten()
    changed = torch.nonzero(candidate != old, as_tuple=False).flatten()
    if not len(changed):
        return old_codes.clone(), {'changed': 0, 'relative_l2': 0., 'fraction': 0.}
    # Physical squared distance to the preconditioned continuous proposal.
    p = proposal.flatten()[changed]
    s = scales.flatten()[changed // group]
    gain = ((old[changed].float() - p).square() - (candidate[changed].float() - p).square()) * s.square()
    valid = (gain > 0) & (s > 0)
    changed, gain = changed[valid], gain[valid]
    k = min(len(changed), max(0, int(old.numel() * fraction)))
    # Selection is a top-k subspace, not a complete sort of a vocabulary table.
    # Same-runtime interrupted/resumed probes check its actual replay behavior.
    order = torch.topk(gain, k, sorted=True).indices
    changed = changed[order]
    delta2 = ((candidate[changed].float() - old[changed].float()) * scales.flatten()[changed // group]).square()
    norm2 = (old_codes.reshape(*scales.shape, group).float().square().sum(-1) * scales.square()).sum()
    accepted = changed[delta2.cumsum(0) <= trust * trust * norm2]
    result = old.clone()
    result[accepted] = candidate[accepted]
    actual2 = ((candidate[accepted].float() - old[accepted].float()) * scales.flatten()[accepted // group]).square().sum()
    rel = math.sqrt(float(actual2 / norm2)) if norm2 > 0 else 0.
    return result.reshape_as(old_codes), {'changed': len(accepted), 'relative_l2': rel,
                                          'fraction': len(accepted) / old.numel()}


def full_kd_ce(student_logits, teacher_logits, targets=None, *, alpha=0., temperature=1.):
    """Token-mean full-vocabulary forward KL plus optional next-token CE."""
    if not 0 <= alpha <= 1 or temperature <= 0:
        raise ValueError('invalid loss weights')
    student = (student_logits.float() / temperature).log_softmax(-1)
    teacher = (teacher_logits.detach().float() / temperature).log_softmax(-1)
    kl = F.kl_div(student, teacher, log_target=True, reduction='none').sum(-1).mean() * temperature ** 2
    if alpha:
        if targets is None:
            raise ValueError('CE requires targets')
        return (1-alpha)*kl + alpha*F.cross_entropy(student_logits.float(), targets)
    return kl

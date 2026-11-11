"""Mirror-descent maths for ternary code training (used by recipe/train.py).

Equation numbers refer to patent application US20260220467A1 (Eqs. 9-10, 18-31);
a few printed equations are corrected here, as noted at each one.

Implementation: **two-step SMD** (Eq. 10): a persistent FP32 dual ``xi`` per
parameter, ``xi <- xi - lr * grad`` and ``w = grad_psi^-1(xi)``.  Potentials act on
the tensor they are given (in training: the scale-normalized code-unit master of
``method_compare.TrainTable``), never on physical weights implicitly.
"""
import torch


class Potential:
    """Strictly convex potential: ``value``, one declared (sub)gradient and the inverse map."""

    def value(self, w):
        raise NotImplementedError

    def grad(self, w):
        raise NotImplementedError

    def inverse(self, x):
        raise NotImplementedError


def _spow(x, e):
    """sign(x)|x|^e, keeping exact zeros and signs (copysign also keeps -0.0)."""
    return torch.copysign(x.abs().pow(e), x)


class QNorm(Potential):
    """psi(w) = (1/q) sum |w|^q (the paper's normalization; q=2 is exactly Euclidean).

    The patent's Eq. 18 omits 1/q; its Eq. 19-20 gradient/inverse then carry factors
    of q, which at q=2 halve the effective step relative to SGD.
    """

    def __init__(self, q):
        if not q > 1:
            raise ValueError('q-norm potential needs q > 1')
        self.q = float(q)

    def value(self, w):
        return w.abs().pow(self.q).sum() / self.q

    def grad(self, w):
        return w.clone() if self.q == 2 else _spow(w, self.q - 1)

    def inverse(self, x):
        return x.clone() if self.q == 2 else _spow(x, 1 / (self.q - 1))

    def fp32_min_magnitude(self):
        """Smallest |w| whose dual |w|^(q-1) is still a normal FP32 number."""
        return float(torch.finfo(torch.float32).tiny) ** (1 / (self.q - 1))


class PruneQuantize(Potential):
    """Patent Eq. 23: psi(w) = sum max(|w|^(1+eps), beta |w|^q); a two-well map.

    Strictly convex (max of strictly convex terms).  The kink at |w| = T with
    T = beta^(-1/(q-1-eps)) makes the gradient an interval
    [(1+eps) T^eps, q T^eps]; ``kink`` declares which element ``grad`` returns there
    ('mid' by default; the patent gives no rule).  The inverse (Eq. 25) is
    single-valued: small duals map near 0, the whole interval maps to exactly T.
    """

    def __init__(self, eps=.05, q=8., beta=1., kink='mid'):
        if not (0 < eps < 1 and q > 1 + eps and beta > 0 and kink in ('low', 'mid', 'high')):
            raise ValueError('need 0<eps<1, q>1+eps, beta>0, kink in low/mid/high')
        self.eps, self.q, self.beta, self.kink = float(eps), float(q), float(beta), kink
        self.T = beta ** (-1 / (q - 1 - eps))
        self.low = (1 + eps) * self.T ** eps
        self.high = q * self.T ** eps  # equals beta*q*T^(q-1)

    def value(self, w):
        a = w.abs()
        return torch.maximum(a.pow(1 + self.eps), self.beta * a.pow(self.q)).sum()

    def grad(self, w):
        a = w.abs()
        g = torch.where(a < self.T, (1 + self.eps) * a.pow(self.eps), self.beta * self.q * a.pow(self.q - 1))
        mid = {'low': self.low, 'mid': (self.low + self.high) / 2, 'high': self.high}[self.kink]
        g = torch.where(a == self.T, torch.full_like(g, mid), g)
        return torch.copysign(g, w) * (a > 0)

    def inverse(self, x):
        a = x.abs()
        small = (a / (1 + self.eps)).pow(1 / self.eps)
        large = (a / (self.beta * self.q)).pow(1 / (self.q - 1))
        mag = torch.where(a < self.low, small, torch.where(a <= self.high, torch.full_like(a, self.T), large))
        return torch.copysign(mag, x) * (a > 0)


class RigidCage(Potential):
    """Patent Eq. 26 per contiguous group of the last dimension:
    psi(w) = sum_groups ( ||w_g||_inf + eps/2 ||w_g||_2^2 ).

    Declared subgradient (Eq. 27): s + eps*w with s splitting unit l1 mass equally,
    with signs, over the tied maxima; s = 0 for an all-zero group.  The patent allows
    any such s ("may be chosen uniformly at random"); a fixed rule keeps replay exact.

    Inverse, derived from x - eps*w in the subdifferential (Eq. 28 corrected):
    sort |x| descending; take the SMALLEST k with sum_{j<=k}|x_j| - 1 > k |x_{k+1}|
    (|x_{p+1}| = 0); alpha = (sum_{j<=k}|x_j| - 1)/k.  Then the top k map to
    sign(x) alpha/eps and the rest to x/eps.  If ||x_g||_1 <= 1 the group maps to 0.
    The printed Eq. 28 says "largest k" and leaves the unsaturated branch unscaled.
    """

    def __init__(self, eps=.1, group=128):
        if not eps > 0 or group < 1:
            raise ValueError('need eps > 0 and group >= 1')
        self.eps, self.group = float(eps), int(group)

    def _groups(self, t):
        if t.shape[-1] % self.group:
            raise ValueError('last dimension must be a multiple of the group size')
        return t.reshape(-1, self.group)

    def value(self, w):
        g = self._groups(w)
        return g.abs().amax(-1).sum() + self.eps / 2 * g.square().sum()

    def grad(self, w):
        g = self._groups(w)
        a = g.abs()
        top = (a == a.amax(-1, keepdim=True)) & (a > 0)
        s = torch.sign(g) * top / top.sum(-1, keepdim=True).clamp_min(1)
        return (s + self.eps * g).reshape_as(w)

    def inverse(self, x):
        g = self._groups(x)
        a = g.abs()
        srt, idx = a.sort(dim=-1, descending=True, stable=True)
        cs = srt.cumsum(-1)
        k = torch.arange(1, self.group + 1, device=x.device, dtype=x.dtype)
        nxt = torch.cat([srt[:, 1:], torch.zeros_like(srt[:, :1])], -1)
        cond = cs - 1 > k * nxt                      # nondecreasing in k
        nonzero = cond[:, -1]                        # equals ||x_g||_1 > 1
        first = cond.to(torch.int8).argmax(-1, keepdim=True)
        alpha = (cs.gather(-1, first) - 1) / (first + 1).to(x.dtype)
        rank = torch.empty_like(idx)
        rank.scatter_(-1, idx, torch.arange(self.group, device=x.device).expand_as(idx))
        out = torch.where(rank <= first, torch.copysign(alpha.expand_as(g), g), g) / self.eps
        return (out * nonzero[:, None]).reshape_as(x)


def kurtosis_loss(tensors):
    """Patent Eq. 29: sum_layers sum_j ( |w_j|^4 - mean(|w|)^4 )^2, mean per tensor."""
    total = 0.
    for w in tensors:
        a = w.abs()
        total = total + (a.pow(4) - a.mean().pow(4)).square().sum()
    return total


def kurtosis_grad(w):
    """Exact gradient of one tensor's Eq. 29 term.

    d/dw_j = 8 sign(w_j) [ (a_j^4 - b^4) a_j^3 - (b^3/p) sum_m (a_m^4 - b^4) ],
    a = |w|, b = mean(a), p = numel.  The printed Eq. 31 omits the cross-term sum
    over m contributed through the shared mean.  (sign(0) = 0 picks the zero
    subgradient at w_j = 0.)
    """
    a = w.abs()
    b = a.mean()
    d = a.pow(4) - b.pow(4)
    return 8 * torch.sign(w) * (d * a.pow(3) - b.pow(3) / w.numel() * d.sum())


class MirrorDescent(torch.optim.Optimizer):
    """Two-step stochastic mirror descent with a persistent FP32 dual (patent Eq. 10).

    state['dual'] is initialized lazily as potential.grad(p) on the first step and
    saved by ``state_dict`` so interrupted/resumed runs replay exactly.  Gradient
    clipping, schedules and trust caps stay outside, as for the other arms.
    """

    def __init__(self, params, potential, lr):
        if not lr > 0:
            raise ValueError('lr must be positive')
        super().__init__(params, {'lr': lr})
        self.potential = potential

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.float()
                if not torch.isfinite(g).all():
                    raise RuntimeError('nonfinite gradient in mirror step')
                state = self.state[p]
                if 'dual' not in state:
                    state['dual'] = self.potential.grad(p.detach().float())
                state['dual'].add_(g, alpha=-group['lr'])
                w = self.potential.inverse(state['dual'])
                if not torch.isfinite(w).all():
                    raise RuntimeError('nonfinite primal after inverse mirror map')
                p.copy_(w.to(p.dtype))
        return loss


class _MirrorHardWeight(torch.autograd.Function):
    """Hard ternary forward from a dual tensor: codes = round(clamp(grad_psi^-1(dual))).

    Backward returns the STE primal-space gradient (s * dV) in the dual's slot, so an
    optimizer stepping this tensor performs mirror descent with that optimizer's
    direction.  The primal is rebuilt per 4096-row chunk and never stored.
    """

    @staticmethod
    def forward(ctx, dual, log_scale, base_scale, dtype, potential):
        scales = base_scale * log_scale.exp()
        group = dual.shape[-1] // scales.shape[-1]
        flat = dual.detach().reshape(-1, dual.shape[-1])
        codes = torch.empty(flat.shape, dtype=torch.int8, device=dual.device)
        result = torch.empty(flat.shape, dtype=dtype, device=dual.device)
        sf = scales.reshape(-1, scales.shape[-1])
        for lo in range(0, len(flat), 4096):
            hi = lo + 4096
            codes[lo:hi] = potential.inverse(flat[lo:hi].float()).round().clamp_(-1, 1).to(torch.int8)
            result[lo:hi] = (codes[lo:hi].reshape(-1, sf.shape[-1], group).to(dtype)
                             * sf[lo:hi].to(dtype)[..., None]).reshape_as(result[lo:hi])
        ctx.save_for_backward(codes, scales)
        ctx.group = group
        return result.reshape_as(dual)

    @staticmethod
    def backward(ctx, grad):
        codes, scales = ctx.saved_tensors
        dual_grad = torch.empty(grad.shape, dtype=torch.float32, device=grad.device)
        scale_grad = torch.empty_like(scales)
        gf, df = grad.reshape(-1, grad.shape[-1]), dual_grad.reshape(-1, grad.shape[-1])
        sf, uf = scales.reshape(-1, scales.shape[-1]), scale_grad.reshape(-1, scales.shape[-1])
        for lo in range(0, len(gf), 4096):
            hi = lo + 4096
            weighted = gf[lo:hi].float().reshape(-1, sf.shape[-1], ctx.group) * sf[lo:hi, :, None]
            df[lo:hi] = weighted.reshape_as(df[lo:hi])
            uf[lo:hi] = (weighted * codes[lo:hi].reshape_as(weighted)).sum(-1)
        return dual_grad, scale_grad, None, None, None


def mirror_table(old, potential):
    """A ``method_compare.TrainTable`` whose ``master`` parameter holds the dual.

    Initialized at grad_psi(codes) (the GPTQ codes exactly, no BF16 residual), so the
    hard forward is identical before any update.  Same shared-table/tied-endpoint
    ownership, scales and export path as the STE arm.
    """
    from .method_compare import TrainTable

    class MirrorTable(TrainTable):
        def codes(self):
            result = torch.empty_like(self.master, dtype=torch.int8)
            for lo in range(0, len(result), 4096):
                result[lo:lo+4096] = self.potential.inverse(self.master.detach()[lo:lo+4096].float()).round().clamp_(-1, 1).to(torch.int8)
            return result

        def weight(self, ids=None):
            v, u, s = self.master, self.log_mult, self.s0
            if ids is not None:
                v, u, s = v[ids], u[ids], s[ids]
            return _MirrorHardWeight.apply(v, u, s, self.dtype, self.potential)

    table = MirrorTable(old)
    table.potential = potential
    with torch.no_grad():
        for lo in range(0, len(table.master), 4096):
            table.master[lo:lo+4096] = potential.grad(table.master[lo:lo+4096])
    return table


def validations_since_best(history, key='combined_kl'):
    """Validations recorded after the best (lowest) one; ties keep the earliest best.

    Derived from the saved history alone, so a resumed run reaches the same patience
    decision as an uninterrupted one.
    """
    if not history:
        return 0
    best = min(range(len(history)), key=lambda i: (history[i][key], i))
    return len(history) - 1 - best


def patience_exhausted(history, patience, key='combined_kl'):
    """True when ``patience`` > 0 consecutive validations have failed to beat the best."""
    if patience < 0:
        raise ValueError('patience must be >= 0 (0 disables early stopping)')
    return patience > 0 and validations_since_best(history, key) >= patience

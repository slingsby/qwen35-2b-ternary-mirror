"""Zero-GPU contract tests for the mirror-descent maths (openbonsai/core/mirror.py)."""
import io

import numpy as np
import pytest
import torch

from openbonsai.core.method_compare import TrainTable, TrainLinear, TrainEmbedding
from openbonsai.core.mirror import (QNorm, PruneQuantize, RigidCage, MirrorDescent,
                                    kurtosis_loss, kurtosis_grad)
from openbonsai.core.ternary import TernaryLinear, TernaryEmbedding, shared_hadamard
from openbonsai.formats import ptq1_0 as P


def magnitudes(dtype):
    torch.manual_seed(0)
    w = torch.randn(4096, dtype=dtype) * torch.logspace(-2, 0.2, 4096, dtype=dtype)
    return torch.cat([w, torch.tensor([0., -0., 1., -1., 1.49, -1.49], dtype=dtype)])


# ------------------------------------------------------------------ q-norm ----
@pytest.mark.parametrize('q', [1.5, 2., 3., 8., 16.])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_qnorm_inverse_round_trip_signs_and_zeros(q, dtype):
    p = QNorm(q)
    w = magnitudes(dtype)
    w = w[(w.abs() >= p.fp32_min_magnitude()) | (w == 0)]
    back = p.inverse(p.grad(w))
    torch.testing.assert_close(back, w, rtol=1e-5 if dtype == torch.float32 else 1e-12, atol=0)
    assert torch.equal(torch.signbit(back), torch.signbit(w))
    assert torch.equal(back == 0, w == 0)


@pytest.mark.parametrize('q', [1.5, 3., 8., 16.])
def test_qnorm_gradient_matches_autograd(q):
    w = magnitudes(torch.float64)
    w = w[w != 0].requires_grad_()
    auto = torch.autograd.grad(QNorm(q).value(w), w)[0]
    torch.testing.assert_close(QNorm(q).grad(w.detach()), auto, rtol=1e-12, atol=1e-300)


def test_q2_mirror_descent_is_bitwise_sgd():
    torch.manual_seed(1)
    a = torch.nn.Parameter(torch.randn(64, 128))
    b = torch.nn.Parameter(a.detach().clone())
    md, sgd = MirrorDescent([a], QNorm(2), lr=.03), torch.optim.SGD([b], lr=.03)
    for step in range(5):
        g = torch.randn(64, 128)
        a.grad, b.grad = g.clone(), g.clone()
        md.step(); sgd.step()
        assert torch.equal(a, b), step


def test_qnorm_fp32_dual_floor_and_zero_mobility_are_declared():
    """Characterization, not a pass criterion for the method.

    High q stores |w|^(q-1): below fp32_min_magnitude the dual is subnormal/zero.
    The inverse is steep at 0: a tiny dual step moves a zero latent far, which is the
    measured way a one-well map drains the zero code (forensics 'zero shelf').
    """
    p = QNorm(16)
    assert 0.002 < p.fp32_min_magnitude() < 0.004
    moved = p.inverse(torch.tensor([1e-6, 1e-4]))
    assert 0.39 < float(moved[0]) < 0.41 and float(moved[1]) > .5  # second flips code 0 -> 1


# --------------------------------------------------------- prune-quantize ----
@pytest.mark.parametrize('kink', ['low', 'mid', 'high'])
def test_prune_quantize_round_trip_plateau_and_monotone(kink):
    p = PruneQuantize(eps=.05, q=8., beta=1., kink=kink)
    assert p.T == 1.
    w = magnitudes(torch.float64)
    w = w[(w.abs() > .05) | (w == 0)]  # below this (|w|^.05 dual) 1/eps powers lose digits
    torch.testing.assert_close(p.inverse(p.grad(w)), w, rtol=1e-9, atol=0)
    x = torch.linspace(-12, 12, 20001, dtype=torch.float64)
    y = p.inverse(x)
    assert (y.diff() >= 0).all()                                   # inverse of strictly convex: monotone
    plateau = (x.abs() >= p.low) & (x.abs() <= p.high)
    torch.testing.assert_close(y[plateau].abs(), torch.ones_like(y[plateau]))
    # Continuity at both thresholds.
    for t in (p.low, p.high):
        lo, hi = p.inverse(torch.tensor([t - 1e-9, t + 1e-9], dtype=torch.float64))
        assert abs(float(hi - lo)) < 1e-6


def test_prune_quantize_gradient_away_from_kink_and_fp32_stability():
    p = PruneQuantize(eps=.05, q=8.)
    w = magnitudes(torch.float64)
    w = w[(w != 0) & ((w.abs() - 1).abs() > 1e-6)].requires_grad_()
    torch.testing.assert_close(p.grad(w.detach()), torch.autograd.grad(p.value(w), w)[0], rtol=1e-10, atol=1e-300)
    x = torch.linspace(-50, 50, 100001)
    y = p.inverse(x)
    assert torch.isfinite(y).all() and float(y.abs().max()) < 2  # |w| = (50/8)^(1/7) ~ 1.30
    assert float(p.inverse(torch.tensor(.5))) < 1e-6              # (0.5/1.05)^20: small duals sit in the zero well


# -------------------------------------------------------------- rigid cage ----
def assert_cage_optimality(p, x, w, tol):
    """x - eps*w must lie in the subdifferential of ||.||_inf at w, per group."""
    for xg, wg in zip(x.reshape(-1, p.group), w.reshape(-1, p.group)):
        s = xg - p.eps * wg
        m = wg.abs().max()
        if m == 0:
            assert float(xg.abs().sum()) <= 1 + tol
            continue
        at = (wg.abs() - m).abs() <= tol * m
        assert float(s[~at].abs().sum()) <= tol * p.group
        assert bool((s[at] * torch.sign(wg[at]) >= -tol).all())
        assert abs(float(s.abs().sum()) - 1) <= tol


@pytest.mark.parametrize('scale', [.002, .01, .05, .3, 2.])
def test_rigid_cage_inverse_satisfies_convex_optimality(scale):
    torch.manual_seed(2)
    p = RigidCage(eps=.1, group=128)
    x = torch.randn(32, 256, dtype=torch.float64) * scale
    assert_cage_optimality(p, x, p.inverse(x), 1e-9)


def test_rigid_cage_ties_zero_region_and_round_trip():
    p = RigidCage(eps=.1, group=8)
    w = torch.tensor([[3., -3., 3., 1., 0., -2., .5, 3.],      # four tied maxima
                      [0.] * 8,                                # zero group
                      [1e-3, -1e-3, 0, 0, 0, 0, 0, 1e-3]], dtype=torch.float64)
    torch.testing.assert_close(p.inverse(p.grad(w)), w, rtol=1e-12, atol=1e-15)
    inside = torch.full((1, 8), 1 / 8, dtype=torch.float64)    # ||x||_1 == 1 exactly
    assert torch.equal(p.inverse(inside), torch.zeros_like(inside))
    assert torch.equal(p.inverse(inside * .5), torch.zeros_like(inside))
    tied = torch.tensor([[2., -2., 2., .1, 0, 0, 0, 0]], dtype=torch.float64)
    out = p.inverse(tied)
    assert torch.equal(out[0, :3].abs(), out[0, :3].abs()[:1].expand(3))  # ties never split
    assert_cage_optimality(p, tied, out, 1e-12)
    torch.manual_seed(3)
    r = torch.randn(64, 8, dtype=torch.float64)
    torch.testing.assert_close(p.inverse(p.grad(r)), r, rtol=1e-12, atol=1e-15)


def printed_eq28(x, eps):
    """Eq. 28 as transcribed: 'largest k', unsaturated branch left unscaled."""
    a = x.abs()
    srt, idx = a.sort(descending=True)
    cs = srt.cumsum(0)
    nxt = torch.cat([srt[1:], srt.new_zeros(1)])
    k = int(torch.nonzero(cs > torch.arange(1, len(x) + 1) * nxt + 1).max()) + 1
    alpha = (cs[k-1] - 1) / k
    out = x.clone()
    out[idx[:k]] = torch.sign(x[idx[:k]]) * alpha / eps
    return out


def test_printed_eq28_violates_optimality_and_corrected_map_does_not():
    torch.manual_seed(4)
    p = RigidCage(eps=.1, group=16)
    x = torch.randn(16, dtype=torch.float64) * .3
    with pytest.raises(AssertionError):
        assert_cage_optimality(p, x, printed_eq28(x, p.eps), 1e-9)
    assert_cage_optimality(p, x, p.inverse(x[None]).flatten(), 1e-9)


# ---------------------------------------------------------------- kurtosis ----
def test_kurtosis_gradient_analytic_autograd_finite_difference():
    torch.manual_seed(5)
    w = (torch.randn(37, dtype=torch.float64) * .4).requires_grad_()
    auto = torch.autograd.grad(kurtosis_loss([w]), w)[0]
    analytic = kurtosis_grad(w.detach())
    torch.testing.assert_close(analytic, auto, rtol=1e-10, atol=1e-14)
    h, e = 1e-6, torch.eye(37, dtype=torch.float64)
    fd = torch.stack([(kurtosis_loss([w.detach() + h*e[i]]) - kurtosis_loss([w.detach() - h*e[i]])) / (2*h)
                      for i in range(37)])
    torch.testing.assert_close(analytic, fd, rtol=1e-6, atol=1e-10)
    # Printed Eq. 31 (second factor |w|^3 - ||w||_1^3 |w| / p^4, no cross-term) disagrees.
    a, p = w.detach().abs(), w.numel()
    printed = 8 * (a**4 - (a.sum()/p)**4) * (a**3 - a.sum()**3 / p**4 * a) * torch.sign(w.detach())
    assert not torch.allclose(printed, auto, rtol=1e-3)
    # Per-layer means: two tensors are independent terms.
    v = torch.randn(9, dtype=torch.float64)
    torch.testing.assert_close(kurtosis_loss([w.detach(), v]), kurtosis_loss([w.detach()]) + kurtosis_loss([v]))


# ----------------------------------------------- optimizer and deployment ----
POTENTIALS = [QNorm(2), QNorm(8), PruneQuantize(eps=.05, q=8.), RigidCage(eps=.5, group=8)]


@pytest.mark.parametrize('pot', POTENTIALS, ids=lambda p: type(p).__name__ + str(getattr(p, 'q', '')))
def test_mirror_state_dict_replay_one_plus_one_equals_two(pot):
    torch.manual_seed(6)
    init, grads = torch.randn(16, 16) * .5, [torch.randn(16, 16) for _ in range(2)]
    full = torch.nn.Parameter(init.clone()); opt = MirrorDescent([full], pot, lr=.01)
    for g in grads:
        full.grad = g.clone(); opt.step()
    first = torch.nn.Parameter(init.clone()); o1 = MirrorDescent([first], pot, lr=.01)
    first.grad = grads[0].clone(); o1.step()
    buf = io.BytesIO(); torch.save({'p': first.detach(), 'opt': o1.state_dict()}, buf); buf.seek(0)
    saved = torch.load(buf)
    resumed = torch.nn.Parameter(saved['p'].clone()); o2 = MirrorDescent([resumed], pot, lr=.01)
    o2.load_state_dict(saved['opt'])
    resumed.grad = grads[1].clone(); o2.step()
    assert torch.equal(resumed, full)
    assert torch.equal(o2.state[resumed]['dual'], opt.state[full]['dual'])


def test_nonfinite_gradient_refused():
    p = torch.nn.Parameter(torch.ones(4))
    p.grad = torch.tensor([1., float('nan'), 0., 0.])
    with pytest.raises(RuntimeError, match='nonfinite'):
        MirrorDescent([p], QNorm(8), lr=.1).step()


def tied_pair():
    torch.manual_seed(42)
    c = torch.randint(-1, 2, (16, 256), dtype=torch.int8)
    s = torch.rand(16, 2).half() + .1
    h = shared_hadamard(128, 'cpu', dtype=torch.float32)
    old, emb = TernaryLinear(c, s, torch.ones(256), 128, h), TernaryEmbedding(c, s, torch.ones(256), 128, h)
    t = TrainTable(old)
    return old, emb, t, TrainLinear(old, t), TrainEmbedding(emb, t)


@pytest.mark.parametrize('pot', [PruneQuantize(eps=.05, q=8.), QNorm(8)], ids=['prunequant', 'q8'])
def test_tied_endpoint_one_dual_hard_forward_and_ptq1_0_export_unchanged(pot):
    old, emb, t, head, lookup = tied_pair()
    opt = MirrorDescent([t.master], pot, lr=.002)
    # Zero gradient: dual initialization must not move any code (identity before updates).
    t.master.grad = torch.zeros_like(t.master); opt.step()
    assert torch.equal(t.codes(), old.codes)
    x, ids = torch.randn(3, 256), torch.tensor([1, 2, 1])
    torch.testing.assert_close(head(x), old(x), rtol=0, atol=0)
    # Both tied uses accumulate into the one master and its one dual state.
    (head(x).square().sum() + lookup(ids).square().sum()).backward()
    assert len(opt.state) == 1 and head.table.master is lookup.table.master
    opt.step()
    codes = t.codes()
    assert codes.dtype == torch.int8 and set(codes.unique().tolist()) <= {-1, 0, 1}
    torch.testing.assert_close(t.weight(), (codes.float().reshape(16, 2, 128) * t.scales()[..., None]).reshape(16, 256))
    scales = t.scales().detach().half()
    blocks = P.encode(codes.numpy(), scales.numpy())
    expected = (codes.float().reshape(16, 2, 128) * scales.float()[..., None]).reshape(16, 256).numpy()
    np.testing.assert_array_equal(P.dequantize(blocks), expected)


@pytest.mark.parametrize('pot', [QNorm(2), PruneQuantize(eps=.05, q=8.)], ids=['euclid', 'prunequant'])
def test_mirror_table_dual_parameter_contract(pot):
    """Dual-as-parameter table: identity before updates, STE primal gradient in the dual
    slot, shared tied endpoint, and Adafactor stepping the dual."""
    from transformers import Adafactor
    from openbonsai.core.mirror import mirror_table
    from openbonsai.core.method_compare import TrainTable
    old, emb, ref, _, _ = tied_pair()
    t = mirror_table(old, pot)
    head, lookup = TrainLinear(old, t), TrainEmbedding(emb, t)
    assert torch.equal(t.codes(), old.codes)
    x, ids = torch.randn(3, 256), torch.tensor([1, 2, 1])
    torch.testing.assert_close(head(x), old(x), rtol=0, atol=0)
    torch.testing.assert_close(lookup(ids), emb(ids), rtol=0, atol=0)
    # Gradient in the dual slot equals the STE table's primal gradient (same codes/scales).
    (head(x).square().sum() + lookup(ids).square().sum()).backward()
    rh, rl = TrainLinear(old, ref), TrainEmbedding(emb, ref)
    (rh(x).square().sum() + rl(ids).square().sum()).backward()
    torch.testing.assert_close(t.master.grad, ref.master.grad)
    torch.testing.assert_close(t.log_mult.grad, ref.log_mult.grad)
    assert head.table.master is lookup.table.master
    opt = Adafactor([t.master], lr=.02, scale_parameter=False, relative_step=False, warmup_init=False,
                    beta1=None, decay_rate=-.8, weight_decay=0)
    before = t.master.detach().clone(); opt.step()
    assert not torch.equal(t.master, before)
    assert set(t.codes().unique().tolist()) <= {-1, 0, 1}


def test_prunequant_dual_init_sits_in_well_centres():
    from openbonsai.core.mirror import mirror_table
    old, _, _, _, _ = tied_pair()
    p = PruneQuantize(eps=.05, q=8.)
    t = mirror_table(old, p)
    d = t.master.detach()
    nz = old.codes != 0
    torch.testing.assert_close(d[nz].abs(), torch.full_like(d[nz], (p.low + p.high) / 2))
    assert torch.equal(d[~nz], torch.zeros_like(d[~nz]))


def test_patience_rule_on_a_recorded_trajectory_and_edge_cases():
    from openbonsai.core.mirror import validations_since_best, patience_exhausted
    recorded = [1.8898, .6644, .6236, .6178, .6456, .6372, .6385, .6403, .6303, .6366, .6372]
    hist = [{'step': 100 * i, 'combined_kl': v} for i, v in enumerate(recorded)]
    # Default patience 3 stops at the first point three validations past the update-300 best.
    first_stop = next(h['step'] for i, h in enumerate(hist) if patience_exhausted(hist[:i + 1], 3))
    assert first_stop == 600
    assert validations_since_best(hist) == 7
    assert not any(patience_exhausted(hist[:i + 1], 0) for i in range(len(hist)))   # 0 disables
    tie = [{'combined_kl': 1.}, {'combined_kl': .5}, {'combined_kl': .5}, {'combined_kl': .5}]
    assert validations_since_best(tie) == 2                                        # earliest best kept
    assert validations_since_best([]) == 0 and not patience_exhausted([], 3)
    with pytest.raises(ValueError):
        patience_exhausted(hist, -1)

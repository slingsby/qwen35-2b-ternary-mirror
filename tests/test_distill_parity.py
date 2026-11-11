"""Distillation parity with the proven experiment script.

The oracle below is a verbatim copy of OpenBonsai research/experiments/e2e_kd.py
(TernaryLinear, TernaryEmbedding, signs_for, install_ternary,
install_endpoints, kd_chunk_loss, _CastTo, make_small_params_trainable and
the ``cmd_train`` loop, including ``--train-small``). The only edits: the
text model path (``model.model`` for Qwen3_5ForCausalLM instead of
``model.model.language_model``), the endpoint Hadamard size (the tiny
model's hidden width instead of 4096), CPU instead of ``.cuda()``, and the
loop returns its per-step losses (and trained small masters) instead of
logging and evaluating. Tests never import research/.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from openbonsai.cli import execute
from openbonsai.stages import distill as D
from openbonsai.stages.common import install_ternary


# ---------------------------------------------------------------- oracle ----
def hadamard(n, device):
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


class TernaryLinear(torch.nn.Module):
    """y = (C * s) (R x) with R = blockdiag(H) diag(signs); s = s0 * exp(u)."""

    def __init__(self, codes, scales, signs, block, h):
        super().__init__()
        self.out_features, self.in_features = codes.shape
        self.group = self.in_features // scales.shape[1]
        self.block = block
        self.register_buffer("codes", codes.to(torch.int8), persistent=False)
        self.register_buffer("s0", scales.float(), persistent=False)
        self.register_buffer("signs", signs.to(torch.bfloat16), persistent=False)
        self.h = h  # shared [block, block] BF16 Hadamard, not a parameter
        self.log_mult = torch.nn.Parameter(torch.zeros_like(self.s0))

    def scales(self):
        return self.s0 * torch.exp(self.log_mult)

    def weight_deq(self):
        s = self.scales().to(torch.bfloat16)
        c = self.codes.view(self.out_features, -1, self.group).to(torch.bfloat16)
        return (c * s[..., None]).view(self.out_features, self.in_features)

    def forward(self, x):
        shape = x.shape
        xr = (x * self.signs).view(*shape[:-1], self.in_features // self.block, self.block) @ self.h
        return F.linear(xr.view(shape), self.weight_deq())


class TernaryEmbedding(torch.nn.Module):
    """e = (C[ids] * s[ids]) R^T, i.e. stored rows live in the rotated basis."""

    def __init__(self, codes, scales, signs, block, h):
        super().__init__()
        self.num, self.dim = codes.shape
        self.group = self.dim // scales.shape[1]
        self.block = block
        self.register_buffer("codes", codes.to(torch.int8), persistent=False)
        self.register_buffer("s0", scales.float(), persistent=False)
        self.register_buffer("signs", signs.to(torch.bfloat16), persistent=False)
        self.h = h
        self.log_mult = torch.nn.Parameter(torch.zeros_like(self.s0))

    def scales(self):
        return self.s0 * torch.exp(self.log_mult)

    def forward(self, ids):
        c = self.codes[ids].view(*ids.shape, -1, self.group).to(torch.bfloat16)
        s = (self.s0[ids] * torch.exp(self.log_mult[ids])).to(torch.bfloat16)
        rows = (c * s[..., None]).view(*ids.shape, self.dim // self.block, self.block)
        return (rows @ self.h).view(*ids.shape, self.dim) * self.signs


def install_endpoints(model, ep, hidden):
    dev = next(model.parameters()).device
    lm = model.model
    h = hadamard(hidden, dev).to(torch.bfloat16)
    if "lm_head" in ep:
        st = ep["lm_head"]
        model.lm_head = TernaryLinear(st["codes"].to(dev), st["scales"].to(dev), signs_for(st["seed"], hidden).to(dev), st["block"], h)
    if "embed_tokens" in ep:
        st = ep["embed_tokens"]
        lm.embed_tokens = TernaryEmbedding(st["codes"].to(dev), st["scales"].to(dev), signs_for(st["seed"], hidden).to(dev), st["block"], h)


def signs_for(seed, in_features):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (in_features,), generator=g) * 2 - 1).float()


def install_ternary_oracle(model, state):
    layers = model.model.layers
    dev = next(model.parameters()).device
    hs = {}
    for name, st in state["modules"].items():
        _, li, sub = name.split(".", 2)
        parent_name, attr = sub.rsplit(".", 1)
        parent = layers[int(li)].get_submodule(parent_name)
        lin = getattr(parent, attr)
        block = st["block"]
        if block not in hs:
            hs[block] = hadamard(block, dev).to(torch.bfloat16)
        tl = TernaryLinear(st["codes"].to(dev), st["scales"].to(dev), signs_for(st["seed"], lin.in_features).to(dev), block, hs[block])
        setattr(parent, attr, tl)
        del lin


class _CastTo(torch.nn.Module):
    """Parametrization: FP32 master -> the tensor's original dtype on every access."""

    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return x.to(self.dtype)


def make_small_params_trainable(lm):
    """Give every non-ternary decoder tensor (norms, conv1d, in_proj_a/b, A_log, dt_bias, final
    norm) an FP32 master; returns {name: master} split into (vector-like, matrix-like) dicts."""
    import torch.nn.utils.parametrize as parametrize
    vec, mat = {}, {}
    targets = []  # collect first: registering parametrizations adds submodules
    for mname, mod in lm.named_modules():
        if isinstance(mod, (TernaryLinear, TernaryEmbedding)) or "parametrizations" in mname:
            continue
        for pname, p in mod.named_parameters(recurse=False):
            full = f"{mname}.{pname}" if mname else pname
            if full.startswith("layers.") or full == "norm.weight":
                targets.append((mod, pname, full, p))
    for mod, pname, full, p in targets:
            dtype = p.dtype
            mod._parameters[pname] = torch.nn.Parameter(p.detach().float())
            parametrize.register_parametrization(mod, pname, _CastTo(dtype), unsafe=True)
            master = mod.parametrizations[pname].original
            master.requires_grad_(True)
            (mat if master.dim() >= 2 else vec)[full] = master
    return vec, mat


def kd_chunk_loss(h, head, tidx, tlogp):
    logq = head(h).float().log_softmax(-1)
    lqk = logq.gather(-1, tidx)
    p = tlogp.float().exp()
    tail_p = (1 - p.sum(-1)).clamp_min(0)
    log_tail_q = torch.log(-torch.expm1(lqk.logsumexp(-1).clamp_max(-1e-6)))
    return -(p * lqk).sum() - (tail_p * log_tail_q).sum()


def cmd_train(model, ids, tidx_all, tlogp_all, *, lr, epochs, accum, warmup, chunk, train_small=False,
              lr_small=1e-4, lr_small_mat=2e-5):
    torch.manual_seed(0)
    for p in model.parameters():
        p.requires_grad_(False)
    scale_params = [m.log_mult for m in model.modules() if isinstance(m, (TernaryLinear, TernaryEmbedding))]
    for p in scale_params:
        p.requires_grad_(True)
    small_vec, small_mat = {}, {}
    if train_small:
        small_vec, small_mat = make_small_params_trainable(model.model)
    small = list(small_vec.values()) + list(small_mat.values())
    model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    n = ids.shape[0]
    groups = [{"params": scale_params, "lr": lr}]
    if small_vec:
        groups.append({"params": list(small_vec.values()), "lr": lr_small})
    if small_mat:
        groups.append({"params": list(small_mat.values()), "lr": lr_small_mat})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-12)
    total_steps = epochs * n // accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    head = model.lm_head
    lm = model.model
    model.train()
    step, run_loss, run_tok = 0, 0.0, 0
    order = []
    for ep in range(epochs):
        order += torch.randperm(n, generator=torch.Generator().manual_seed(ep)).tolist()
    losses = []
    for it, i in enumerate(order):
        x = torch.from_numpy(ids[i:i + 1].astype(np.int64))
        tidx = torch.from_numpy(tidx_all[i].astype(np.int64))
        tlogp = torch.from_numpy(np.asarray(tlogp_all[i]))
        h = lm(input_ids=x, use_cache=False).last_hidden_state[0]
        loss = 0.0
        tlen = h.shape[0] - 1
        for c0 in range(0, tlen, chunk):
            c1 = min(c0 + chunk, tlen)
            loss = loss + checkpoint(kd_chunk_loss, h[c0:c1], head, tidx[c0:c1], tlogp[c0:c1], use_reentrant=False)
        (loss / (tlen * accum)).backward()
        run_loss += loss.item()
        run_tok += tlen
        if (it + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(scale_params + small, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(run_loss / run_tok)   # --log-every 1
            run_loss, run_tok = 0.0, 0
    mods = {}
    for name, m in model.model.layers.named_modules():
        if isinstance(m, TernaryLinear):
            mods[f"layers.{name}"] = m.scales().detach().half().cpu()
    for name, m in [("lm_head", model.lm_head), ("embed_tokens", model.model.embed_tokens)]:
        if isinstance(m, (TernaryLinear, TernaryEmbedding)):
            mods[name] = m.scales().detach().half().cpu()
    trained_small = {n: p.detach().float().cpu() for n, p in {**small_vec, **small_mat}.items()}
    return losses, mods, trained_small
# ------------------------------------------------------------ end oracle ----


@pytest.fixture
def endpointed(calibrated):
    assert execute(calibrated, "quantize") == "ran"
    assert execute(calibrated, "endpoints") == "ran"
    return calibrated


def _legacy_layout(state):
    """The run state in the scripts' layout: shared signs are seed = width."""
    return {key: {"codes": t.codes, "scales": t.scales, "seed": t.shape[1], "block": state.block}
            for key, t in state.tensors.items()}


def test_three_distillation_steps_match_e2e_kd(endpointed):
    ctx = endpointed
    adapter = ctx.adapter()
    D.run_teacher(ctx)
    ids, tidx, tlogp = D.load_teacher(ctx.run.stage_dir("distill") / "teacher")
    hp = dict(lr=0.05, epochs=3, accum=4, warmup=1, chunk=32)
    assert hp["epochs"] * len(ids) // hp["accum"] == 3

    decoder, endpoints = ctx.quantize_state(), ctx.endpoints_state()
    oracle, _ = ctx.load_model()
    install_ternary_oracle(oracle, {"modules": _legacy_layout(decoder)})
    install_endpoints(oracle, _legacy_layout(endpoints), adapter.hidden_size)
    want_losses, want_scales, _ = cmd_train(oracle, ids, tidx, tlogp, **hp)

    student, _ = ctx.load_model()
    install_ternary(adapter, student, ctx.state_through("endpoints").records(), torch.device("cpu"))
    history = D.train(student, adapter, ids, tidx, tlogp, ctx.run.stage_dir("distill"), log_every=1, ckpt_every=100,
                      device=torch.device("cpu"), log=lambda m: None, **hp)
    got = D.trained_scales(student, adapter)

    assert [h["step"] for h in history] == [1, 2, 3]
    assert [h["kd_ce"] for h in history] == want_losses   # bitwise; the plan asked for 1e-6
    assert sorted(got) == sorted(want_scales) and len(got) == 27
    assert all(torch.equal(got[k], want_scales[k]) for k in got)
    assert any(not torch.equal(got[k], t.scales) for k, t in ctx.state_through("endpoints").tensors.items())


def test_three_steps_with_train_small_match_e2e_kd(endpointed, tmp_path):
    """--train-small: FP32 masters behind a cast parametrization, two extra LR groups."""
    ctx = endpointed
    adapter = ctx.adapter()
    D.run_teacher(ctx)
    ids, tidx, tlogp = D.load_teacher(ctx.run.stage_dir("distill") / "teacher")
    hp = dict(lr=0.05, epochs=3, accum=4, warmup=1, chunk=32)
    small_lr = dict(lr_small=0.02, lr_small_mat=0.01)

    decoder, endpoints = ctx.quantize_state(), ctx.endpoints_state()
    oracle, _ = ctx.load_model()
    install_ternary_oracle(oracle, {"modules": _legacy_layout(decoder)})
    install_endpoints(oracle, _legacy_layout(endpoints), adapter.hidden_size)
    want_losses, want_scales, want_small = cmd_train(oracle, ids, tidx, tlogp, train_small=True, **small_lr, **hp)

    student, _ = ctx.load_model()
    install_ternary(adapter, student, ctx.state_through("endpoints").records(), torch.device("cpu"))
    history = D.train(student, adapter, ids, tidx, tlogp, tmp_path, log_every=1, ckpt_every=100,
                      device=torch.device("cpu"), log=lambda m: None, small=True, **small_lr, **hp)
    got_scales = D.trained_scales(student, adapter)
    got_small, dtypes = D.trained_small(student, adapter)

    assert [h["kd_ce"] for h in history] == want_losses
    assert all(torch.equal(got_scales[k], want_scales[k]) for k in want_scales)
    assert sorted(got_small) == sorted(want_small) and len(got_small) == 29
    assert all(torch.equal(got_small[k], want_small[k]) for k in want_small)
    assert set(dtypes.values()) == {"bfloat16"}
    base, _ = ctx.load_model()
    lm = adapter.text_model(base)
    moved = [k for k, v in got_small.items() if not torch.equal(v.to(torch.bfloat16), lm.get_parameter(k))]
    assert len(moved) > 20


def test_resumed_train_small_equals_uninterrupted(endpointed, tmp_path):
    ctx = endpointed
    D.run_teacher(ctx)
    data = D.load_teacher(ctx.run.stage_dir("distill") / "teacher")
    kw = dict(lr=0.02, epochs=2, accum=2, warmup=1, chunk=32, log_every=10, ckpt_every=2, small=True,
              lr_small=0.01, lr_small_mat=0.01, device=torch.device("cpu"), log=lambda m: None)
    adapter = ctx.adapter()

    def student():
        m, _ = ctx.load_model()
        install_ternary(adapter, m, ctx.state_through("endpoints").records(), torch.device("cpu"))
        return m

    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir(), b_dir.mkdir()
    a = student()
    hist_a = D.train(a, adapter, *data, a_dir, **kw)
    b = student()
    with pytest.raises(KeyboardInterrupt):
        D.train(b, adapter, *data, b_dir, stop_after=3, **kw)
    b = student()
    hist_b = D.train(b, adapter, *data, b_dir, **kw)
    assert [h["kd_ce"] for h in hist_b] == [h["kd_ce"] for h in hist_a[2:]]
    sa, sb = D.trained_small(a, adapter)[0], D.trained_small(b, adapter)[0]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    # A checkpoint from a run without small tensors does not resume a small run.
    c_dir = tmp_path / "c"
    c_dir.mkdir()
    D.train(student(), adapter, *data, c_dir, **{**kw, "small": False, "epochs": 1})
    with pytest.raises(ValueError, match="holds"):
        D.train(student(), adapter, *data, c_dir, **kw)


def test_distill_stage_saves_small_tensors_and_eval_installs_them(endpointed):
    from openbonsai.formats.state import read_scales, read_small
    from openbonsai.stages import evaluate as EV
    ctx = endpointed
    ctx.cfg["distill"]["train"] = {"scales": True, "small": True}
    ctx.cfg["distill"].update(lr_small=0.05, lr_small_mat=0.05)
    assert execute(ctx, "distill") == "ran"
    path = ctx.run.stage_dir("distill") / "scales.safetensors"
    assert len(read_scales(path)[0]) == 27
    small = read_small(path)
    assert len(small) == 29 and ctx.distill_small().keys() == small.keys()
    model, _ = ctx.load_model()
    EV.install(ctx, model, "distill")
    lm = ctx.adapter().text_model(model)
    assert all(torch.equal(lm.get_parameter(k), v.to(torch.bfloat16)) for k, v in small.items())

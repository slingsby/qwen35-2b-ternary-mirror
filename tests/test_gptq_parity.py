"""Bitwise parity with the proven experiment scripts.

The oracle below is a verbatim copy of OpenBonsai research/experiments/gptq_ternary.py
(fit_group, gptq_ternary, hadamard, signs_from_seed, rotation_matrix,
capture_layer0_inputs, layer_forward, quantize_model) and
research/experiments/endpoints.py (head_hessian, quantize_rows). The only
edits: the text model path (``model.model`` for Qwen3_5ForCausalLM instead of
``model.model.language_model``) and ``model.config.layer_types``. Tests never
import research/.
"""
import math
from types import SimpleNamespace

import pytest
import torch

from openbonsai.core import gptq as G
from openbonsai.core.hadamard import rotation_matrix, seeded_signs
from openbonsai.formats.state import load_state

# ---------------------------------------------------------------- oracle ----
ORDER = {
    "linear_attention": [["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"], ["linear_attn.out_proj"],
                         ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]],
    "full_attention": [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], ["self_attn.o_proj"],
                       ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]],
}


def hadamard(n, device):
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def signs_from_seed(seed, in_features):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (in_features,), generator=g) * 2 - 1).float()


def oracle_rotation_matrix(in_features, block, seed, device):
    signs = signs_from_seed(seed, in_features).to(device)
    h = hadamard(block, device)
    r = torch.block_diag(*([h] * (in_features // block)))
    return signs[:, None] * r


def fit_group(w, ratios):
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
def oracle_gptq_ternary(w, h, group=128, damp=0.01, ratios=None, refit=True):
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


def capture_layer0_inputs(model, calib, batch):
    layers = model.model.layers
    store = {"h": [], "kwargs": None}

    class Stop(Exception):
        pass

    def hook(mod, args, kwargs):
        store["h"].append(args[0].detach() if args else kwargs["hidden_states"].detach())
        if store["kwargs"] is None:
            store["kwargs"] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        raise Stop

    handle = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    dev = next(model.parameters()).device
    for b in range(0, len(calib), batch):
        try:
            model(input_ids=calib[b:b + batch].to(dev), use_cache=False)
        except Stop:
            pass
    handle.remove()
    return store["h"], store["kwargs"]


def layer_forward(layer, x, kwargs):
    out = layer(x, **kwargs)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def oracle_quantize_model(model, calib, args):
    layers = model.model.layers
    types = model.config.layer_types
    assert len(calib) % args.batch == 0
    inps, kwargs = capture_layer0_inputs(model, calib, args.batch)
    assert model.config._attn_implementation == "sdpa"
    assert kwargs.get("attention_mask") is None
    state = {}
    for li, layer in enumerate(layers):
        mods = dict(layer.named_modules())
        for group_names in ORDER[types[li]]:
            hs = {n: None for n in group_names}
            counts = {n: 0 for n in group_names}

            def make_hook(n):
                def hook(mod, args_):
                    x = args_[0].reshape(-1, args_[0].shape[-1]).float()
                    hh = x.T @ x
                    hs[n] = hh if hs[n] is None else hs[n] + hh
                    counts[n] += x.shape[0]
                return hook

            handles = [mods[n].register_forward_pre_hook(make_hook(n)) for n in group_names]
            for x in inps:
                layer_forward(layer, x, kwargs)
            for hd in handles:
                hd.remove()
            for n in group_names:
                lin = mods[n]
                w = lin.weight.float()
                in_f = w.shape[1]
                seed = in_f if args.shared_signs else 5090 + 1000 * li + sum(map(ord, n))
                rot = oracle_rotation_matrix(in_f, math.gcd(in_f, args.block), seed, w.device)
                h = rot.T @ (hs[n] / counts[n]) @ rot
                w_rot = w @ rot
                codes, scales = oracle_gptq_ternary(w_rot, h, group=args.group, damp=args.damp)
                scales = scales.half().float()
                deq = (codes.float().view(w.shape[0], -1, args.group) * scales[..., None]).view_as(w)
                lin.weight.copy_((deq @ rot.T).to(lin.weight.dtype))
                state[f"layers.{li}.{n}"] = {"codes": codes.cpu(), "scales": scales.half().cpu(), "seed": seed,
                                             "signs": signs_from_seed(seed, in_f).to(torch.int8),
                                             "block": math.gcd(in_f, args.block)}
        inps = [layer_forward(layer, x, kwargs) for x in inps]
    return state


@torch.no_grad()
def head_hessian(model, calib, batch):
    lm = model.model
    dev = next(model.parameters()).device
    h, n = None, 0
    for b in range(0, len(calib), batch):
        x = lm(input_ids=calib[b:b + batch].to(dev), use_cache=False).last_hidden_state
        x = x.reshape(-1, x.shape[-1]).float()
        h = x.T @ x if h is None else h + x.T @ x
        n += x.shape[0]
    return h / n


@torch.no_grad()
def oracle_quantize_rows(weight, h, seed, block, group, damp, chunk):
    rows, in_f = weight.shape
    dev = h.device
    rot = oracle_rotation_matrix(in_f, math.gcd(in_f, block), seed, dev)
    h_rot = rot.T @ h @ rot
    codes = torch.empty(rows, in_f, dtype=torch.int8)
    scales = torch.empty(rows, in_f // group, dtype=torch.float16)
    deq = torch.empty(rows, in_f, dtype=weight.dtype)
    for r0 in range(0, rows, chunk):
        w = weight[r0:r0 + chunk].to(dev).float() @ rot
        c, s = oracle_gptq_ternary(w, h_rot, group=group, damp=damp)
        codes[r0:r0 + chunk] = c.cpu()
        scales[r0:r0 + chunk] = s.half().cpu()
        d = (c.float().view(c.shape[0], -1, group) * s.half().float()[..., None]).view_as(w)
        deq[r0:r0 + chunk] = (d @ rot.T).to(weight.dtype).cpu()
    return codes, scales, deq
# ------------------------------------------------------------ end oracle ----


def _random_problem(rows, cols, seed, dead=()):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g)
    x = torch.randn(4 * cols, cols, generator=g) @ (torch.eye(cols) + 0.3 * torch.randn(cols, cols, generator=g))
    x[:, list(dead)] = 0
    return w, (x.T @ x) / x.shape[0]


@pytest.mark.parametrize("rows,cols,damp,dead", [(16, 256, 0.01, ()), (7, 384, 0.1, ()), (5, 128, 0.01, (3, 77))])
def test_gptq_ternary_bitwise_on_random_matrices(rows, cols, damp, dead):
    w, h = _random_problem(rows, cols, rows * cols, dead)
    ref_codes, ref_scales = oracle_gptq_ternary(w, h, group=128, damp=damp)
    codes, scales = G.gptq_ternary(w, h, group=128, damp=damp)
    assert torch.equal(codes, ref_codes) and torch.equal(scales, ref_scales)
    assert set(codes.unique().tolist()) <= {-1, 0, 1}
    if dead:
        assert not codes[:, list(dead)].any()


def test_fit_group_and_dense_rotation_match_the_oracle():
    w = torch.randn(9, 128, generator=torch.Generator().manual_seed(3))
    ratios = list(G.DEFAULT_RATIOS)
    for a, b in zip(G.fit_group(w, ratios), fit_group(w, [0.3 + 0.05 * i for i in range(19)])):
        assert torch.equal(a, b)
    for width, block in ((128, 128), (256, 128), (384, 128), (512, 256)):
        signs = seeded_signs(width, width)
        assert torch.equal(signs.float(), signs_from_seed(width, width))
        assert torch.equal(rotation_matrix(signs, block), oracle_rotation_matrix(width, block, width, "cpu"))


def test_quantize_linear_matches_the_oracle_arithmetic():
    w, h = _random_problem(12, 256, 11)
    signs = seeded_signs(256, 256)
    codes, scales, w_new, stats = G.quantize_linear(w.to(torch.bfloat16), h, signs, 128, group=128, damp=0.01)
    rot = oracle_rotation_matrix(256, 128, 256, "cpu")
    ref_c, ref_s = oracle_gptq_ternary(w.to(torch.bfloat16).float() @ rot, rot.T @ h @ rot, group=128, damp=0.01)
    ref_s = ref_s.half()
    deq = (ref_c.float().view(12, -1, 128) * ref_s.float()[..., None]).view(12, 256)
    assert torch.equal(codes, ref_c) and torch.equal(scales, ref_s)
    assert torch.equal(w_new, deq @ rot.T)
    assert 0 < stats["h_rel_err"] < 1


def test_quantize_and_endpoint_stages_are_bitwise_equal_to_the_scripts(calibrated, tiny_dir):
    from transformers import Qwen3_5ForCausalLM
    from openbonsai.cli import execute
    ctx = calibrated
    assert execute(ctx, "quantize") == "ran" and execute(ctx, "endpoints") == "ran"
    calib = ctx.calib_tokens()
    model = Qwen3_5ForCausalLM.from_pretrained(tiny_dir, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.eval()
    args = SimpleNamespace(batch=4, group=128, block=128, damp=0.01, shared_signs=True)
    ref = oracle_quantize_model(model, calib, args)
    state = load_state(ctx.run.stage_dir("quantize") / "state.safetensors")
    assert sorted(state.tensors) == sorted(ref)
    for key, st in ref.items():
        assert torch.equal(state.tensors[key].codes, st["codes"]), key
        assert torch.equal(state.tensors[key].scales, st["scales"]), key
        assert torch.equal(state.signs[st["codes"].shape[1]], st["signs"]), key
    # Endpoints: head from the ternary decoder's final hidden states, embedding from (1 + w)^2.
    h = head_hessian(model, calib, 4)
    hc, hs, _ = oracle_quantize_rows(model.lm_head.weight.detach().cpu().clone(), h, 128, 128, 128, 0.01, 128)
    g = 1.0 + model.model.layers[0].input_layernorm.weight.float()
    ec, es, _ = oracle_quantize_rows(model.model.embed_tokens.weight.detach().cpu().clone(), torch.diag(g * g),
                                     128, 128, 128, 0.01, 128)
    ep = load_state(ctx.run.stage_dir("endpoints") / "state.safetensors")
    assert torch.equal(ep.tensors["lm_head"].codes, hc) and torch.equal(ep.tensors["lm_head"].scales, hs)
    assert torch.equal(ep.tensors["embed_tokens"].codes, ec) and torch.equal(ep.tensors["embed_tokens"].scales, es)
    assert not torch.all(g == 1), "fixture norms must be nontrivial so the gain rule is exercised"


def test_quantize_rows_chunking_does_not_change_codes():
    w, h = _random_problem(37, 128, 5)
    signs = seeded_signs(128, 128)
    a = G.quantize_rows(w, h, signs, 128, chunk=37)
    b = G.quantize_rows(w, h, signs, 128, chunk=10)
    ref = oracle_quantize_rows(w, h, 128, 128, 128, 0.01, 10)
    for x, y, z in zip(a, b, ref):
        assert torch.equal(y, z)
        assert torch.allclose(x.float(), y.float(), atol=1e-6)

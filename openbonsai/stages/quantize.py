"""Sequential GPTQ ternary rounding of every planned decoder linear.

Port of OpenBonsai research/experiments/gptq_ternary.py ``quantize_model`` with the same
arithmetic. Layers are processed in order; each layer's inputs come from the
already-quantized prefix and are pushed forward in memory. Within a layer the
adapter's plan gives the sequential groups, so later Hessians see quantized
inputs. The model keeps W_q = (C * s) R^T in its dtype, which is numerically
what the runtime computes as (C * s)(R x).

Changes from the script, none of which alter numerics on its tested setup:
the layer order and keep list come from the adapter; signs are one shared
vector per width (the script's ``--shared-signs``) and the block is the global
model block; every layer's forward kwargs (masks, rotary tables) are captured
in one BF16 forward instead of reusing layer 0's, which is also correct for
eager attention and per-layer-type masks; each finished layer is written
atomically to ``quantize/layers/NNNN.safetensors`` and a rerun resumes after
the last finished layer.
"""
from __future__ import annotations

import time

import torch

from ..core.geometry import block_note
from ..core.gptq import Hessian, quantize_linear
from ..core.ternary import dense_weight
from ..formats.atomic import atomic_write_json
from ..formats.state import TernaryState, TernaryTensor, load_state, save_state
from .common import free


class _Stop(Exception):
    pass


@torch.no_grad()
def capture_layer0_inputs(model, adapter, calib, batch, device):
    """Hidden states entering layer 0 for every calibration batch."""
    layers = adapter.layers(model)
    store = []

    def hook(mod, args, kwargs):
        store.append(args[0].detach() if args else kwargs["hidden_states"].detach())
        raise _Stop

    handle = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for b in range(0, len(calib), batch):
            try:
                model(input_ids=calib[b:b + batch].to(device), use_cache=False)
            except _Stop:
                pass
    finally:
        handle.remove()
    return store


@torch.no_grad()
def capture_layer_kwargs(model, adapter, ids, device):
    """Keyword arguments every decoder layer receives, from one forward of ``ids``."""
    layers = adapter.layers(model)
    kwargs = [None] * len(layers)
    handles = []

    def make(i):
        def hook(mod, args, kw):
            kwargs[i] = {k: v for k, v in kw.items() if k != "hidden_states"}
            if i == len(layers) - 1:
                raise _Stop
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make(i), with_kwargs=True))
    try:
        model(input_ids=ids.to(device), use_cache=False)
    except _Stop:
        pass
    finally:
        for h in handles:
            h.remove()
    if any(k is None for k in kwargs):
        raise RuntimeError("could not capture every decoder layer's kwargs")
    return kwargs


def layer_forward(layer, x, kwargs):
    out = layer(x, **kwargs)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def quantize_layer(adapter, model, li, inps, kwargs, signs, block, *, group, damp, impl):
    """GPTQ every planned group of layer ``li`` in order; returns {key: (TernaryTensor, stats)}."""
    layer = adapter.layers(model)[li]
    mods = adapter.planned_linears(model, li)
    plan = adapter.layer_plan(li)
    out = {}
    for group_names in plan.groups:
        hs = {n: Hessian() for n in group_names}

        def make_hook(n):
            def hook(mod, args_):
                hs[n].add(args_[0])
            return hook

        handles = [mods[n].register_forward_pre_hook(make_hook(n)) for n in group_names]
        try:
            for x in inps:
                layer_forward(layer, x, kwargs)
        finally:
            for hd in handles:
                hd.remove()
        for n in group_names:
            lin = mods[n]
            h_mean = hs[n].mean()
            codes, scales, w_new, stats = quantize_linear(lin.weight, h_mean, signs[lin.in_features], block,
                                                          group=group, damp=damp, impl=impl)
            lin.weight.copy_(w_new.to(lin.weight.dtype))
            out[f"layers.{li}.{n}"] = (TernaryTensor(codes.cpu(), scales.cpu()), stats)
            del hs[n], h_mean, w_new
    return out


@torch.no_grad()
def apply_layer(adapter, model, li, state: TernaryState):
    """Re-apply a finished layer's weights exactly as ``quantize_layer`` wrote them."""
    mods = adapter.planned_linears(model, li)
    for key, t, signs, block in state.records():
        lin = mods[key.split(".", 2)[2]]
        lin.weight.copy_(dense_weight(t.codes, t.scales, signs, block, device=lin.weight.device,
                                      dtype=lin.weight.dtype))


def quantize_model(model, adapter, calib, signs, geometry, *, batch=4, damp=0.01, impl="dense",
                   layers_dir=None, log=print, stop_after: int | None = None):
    """Quantize all decoder layers; with ``layers_dir`` each layer is saved and reused on rerun."""
    if len(calib) % batch:
        raise ValueError(f"calibration blocks ({len(calib)}) must be a multiple of quantize.batch ({batch}): "
                         "captured rotary/mask tensors are tied to one batch size")
    device = next(model.parameters()).device
    inps = capture_layer0_inputs(model, adapter, calib, batch, device)
    kwargs = capture_layer_kwargs(model, adapter, calib[:batch], device)
    log(f"captured {len(inps)} batches of {tuple(inps[0].shape)}; layer kwargs {sorted(kwargs[0])}")
    states, report = [], {}
    meta = {"geometry": geometry.to_json(), "damp": damp, "impl": impl}
    for li, layer in enumerate(adapter.layers(model)):
        t0 = time.time()
        path = layers_dir / f"{li:04d}.safetensors" if layers_dir is not None else None
        if path is not None and path.exists():
            st = load_state(path)
            apply_layer(adapter, model, li, st)
            report.update(st.meta.get("stats", {}))
            log(f"layer {li} reused from {path.name}")
        else:
            result = quantize_layer(adapter, model, li, inps, kwargs[li], signs, geometry.block,
                                    group=geometry.group, damp=damp, impl=impl)
            stats = {k: v[1] for k, v in result.items()}
            st = TernaryState({k: v[0] for k, v in result.items()},
                              {w: signs[w] for w in {v[0].shape[1] for v in result.values()}},
                              geometry.block, geometry.group, {**meta, "stats": stats})
            if path is not None:
                save_state(path, st)
            report.update(stats)
            rels = {k.split(".", 2)[2]: (round(v["rel_mse"], 3), round(v["h_rel_err"], 4)) for k, v in stats.items()}
            log(f"layer {li} ({adapter.layer_plan(li).kind}) {time.time() - t0:.1f}s "
                f"(weight rel-mse, H-weighted rel err) {rels}")
        states.append(st)
        inps = [layer_forward(layer, x, kwargs[li]) for x in inps]
        free(device)
        if stop_after is not None and li >= stop_after:
            raise KeyboardInterrupt(f"stopped after layer {li}")
    tensors = {k: t for st in states for k, t in st.tensors.items()}
    used = {t.shape[1] for t in tensors.values()}
    state = TernaryState(tensors, {w: signs[w] for w in sorted(used)}, geometry.block, geometry.group,
                         {"stage": "quantize", **meta})
    return state, report


def run(ctx) -> dict:
    q = ctx.cfg["quantize"]
    log = ctx.log("quantize")
    out = ctx.run.stage_dir("quantize")
    layers_dir = out / "layers"
    layers_dir.mkdir(exist_ok=True)
    model, _ = ctx.load_model()
    adapter = ctx.adapter()
    signs, geo = ctx.signs(model)
    log(f"geometry: widths {list(geo.widths)}, block {geo.block}, group {geo.group}")
    if block_note(geo.block):
        log(f"note: {block_note(geo.block)}")
    calib = ctx.calib_tokens()
    t0 = time.time()
    state, report = quantize_model(model, adapter, calib, signs, geo, batch=int(q["batch"]),
                                   damp=float(q["damp"]), impl=ctx.cfg["rotation"]["impl"],
                                   layers_dir=layers_dir, log=log)
    save_state(out / "state.safetensors", state)
    atomic_write_json(out / "report.json", {"seconds": time.time() - t0, "geometry": geo.to_json(),
                                            "notes": [n for n in [block_note(geo.block)] if n], "modules": report})
    log(f"quantized {len(state.tensors)} linears in {time.time() - t0:.0f}s")
    return {"modules": len(state.tensors)}

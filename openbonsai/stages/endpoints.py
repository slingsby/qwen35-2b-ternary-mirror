"""Ternary token embedding and LM head in the same rotated group format.

Port of OpenBonsai research/experiments/endpoints.py with the same arithmetic.

LM head: GPTQ with the Hessian of its actual inputs (final-norm hidden states
of the already-ternary decoder) on the run's calibration blocks (all of
them: 256 x 2048 in the 9B example; the measured endpoints.py runs built their
own 128 blocks of 64 WikiText-2, 48 chat and 16 trace blocks). Rows are
independent, so the head is processed in row chunks.

Embedding: no input Hessian exists (inputs are one-hot), but an embedding
row reaches the network only through layer 0's entry norm, whose output is
proportional to gain * e (gain = 1 + w for zero-centered norms, w otherwise;
the adapter decides). GPTQ with H = R^T diag(gain^2) R minimizes the error
the first layer actually sees, up to the per-token RMS factor.

``endpoints.head`` / ``endpoints.embedding``: ``ternary``, ``bf16`` (kept
high precision) or ``auto`` (ternary unless the model ties them). Tied
embedding/head tables (Qwen3 0.6B-4B, Llama 3.2 1B/3B) stay high precision in
the default mode. Qwen3.5 can explicitly opt into experimental joint GPTQ
with endpoints.tied=ternary: one canonical table serves both uses. Its
combined reconstruction objective is experimental, not a quality guarantee.
"""
from __future__ import annotations

import time

import torch

from ..config import ConfigError
from ..core.gptq import quantize_rows
from ..formats.state import TernaryState, TernaryTensor, save_state
from .common import apply_dense, free


MODES = ("auto", "ternary", "bf16")
TIED_TERNARY = ("this model has a tied embedding and head (one shared table): independent ternary endpoints "
                "are refused. Opt into joint GPTQ with endpoints.tied=ternary, or keep endpoints.head and "
                "endpoints.embedding auto/bf16 (or endpoints.enabled=false)")


def ternary_endpoints(cfg: dict, adapter) -> tuple[bool, bool]:
    """(head ternary, embedding ternary) under the config: ``auto`` is ternary unless the model ties them."""
    e = cfg["endpoints"]
    tied_mode = e.get("tied", "bf16")
    if tied_mode not in ("bf16", "ternary"):
        raise ConfigError("endpoints.tied: use bf16 or ternary")
    if adapter.tied and tied_mode == "ternary":
        if adapter.name != "qwen3_5":
            raise NotImplementedError("joint tied-table GPTQ is currently validated only for qwen3_5")
        if any(e.get(k) == "bf16" for k in ("head", "embedding")):
            raise ConfigError("a tied ternary table must serve both embedding and head")
        return True, True
    out = []
    for key in ("head", "embedding"):
        mode = e.get(key) or "auto"
        if mode not in MODES:
            raise ConfigError(f"endpoints.{key}: {mode!r}; use one of {MODES}")
        if mode == "ternary" and adapter.tied:
            raise NotImplementedError(TIED_TERNARY)
        out.append(mode == "ternary" or (mode == "auto" and not adapter.tied))
    return out[0], out[1]


@torch.no_grad()
def head_hessian(text_model, calib, batch, device):
    """Mean x^T x of the final hidden states over the calibration blocks (FP32)."""
    h, n = None, 0
    for b in range(0, len(calib), batch):
        x = text_model(input_ids=calib[b:b + batch].to(device), use_cache=False).last_hidden_state
        x = x.reshape(-1, x.shape[-1]).float()
        h = x.T @ x if h is None else h + x.T @ x
        n += x.shape[0]
    return h / n


def run(ctx) -> dict:
    e = ctx.cfg["endpoints"]
    log = ctx.log("endpoints")
    out = ctx.run.stage_dir("endpoints")
    adapter = ctx.adapter()
    want_head, want_emb = ternary_endpoints(ctx.cfg, adapter)
    if adapter.tied and not want_head:
        log("tied embedding/head: the shared table stays high precision (no GPTQ)")
    model, _ = ctx.load_model()
    if adapter.tied and want_head:
        emb, head = adapter.embedding(model).weight, adapter.head(model).weight
        if emb is not head and (emb.data_ptr() != head.data_ptr() or emb.shape != head.shape
                                or emb.stride() != head.stride()):
            raise ValueError("tie_word_embeddings is true but embedding/head do not share the same storage")
    signs, geo = ctx.signs(model)
    hidden = adapter.hidden_size
    tensors = {}
    damp, chunk, impl = float(e["damp"]), int(e["chunk"]), ctx.cfg["rotation"]["impl"]
    if want_head or want_emb:
        decoder = ctx.quantize_state()
        apply_dense(adapter, model, decoder.records())
        log(f"applied {len(decoder.tensors)} ternary decoder linears")
    if adapter.tied and want_head:
        calib = ctx.calib_tokens()
        h = head_hessian(adapter.text_model(model), calib, int(e["batch"]), ctx.device)
        gain = adapter.embedding_gain(model).to(h.device)
        metric = joint_hessian(h, gain)
        emb = adapter.embedding(model)
        c, s, w = quantize_rows(emb.weight.detach().cpu(), metric, signs[hidden], geo.block,
                                group=geo.group, damp=damp, chunk=chunk, impl=impl)
        emb.weight.copy_(w.to(emb.weight.device))
        tensors["embed_tokens"] = TernaryTensor(c, s)
        want_head = want_emb = False  # the canonical table has been quantized exactly once
        log("joint tied-table GPTQ: one embed_tokens record, both uses retained")
        del h, metric, w
    if want_head:
        t0 = time.time()
        calib = ctx.calib_tokens()
        h = head_hessian(adapter.text_model(model), calib, int(e["batch"]), ctx.device)
        head = adapter.head(model)
        hc, hs, hd = quantize_rows(head.weight.detach().cpu().clone(), h, signs[hidden], geo.block,
                                   group=geo.group, damp=damp, chunk=chunk, impl=impl)
        head.weight.copy_(hd.to(head.weight.device))
        tensors["lm_head"] = TernaryTensor(hc, hs)
        del h
        free(ctx.device)
        log(f"head {tuple(hc.shape)} quantized in {time.time() - t0:.0f}s")
    if want_emb:
        t0 = time.time()
        emb = adapter.embedding(model)
        g = adapter.embedding_gain(model)
        ec, es, ed = quantize_rows(emb.weight.detach().cpu().clone(), torch.diag(g * g), signs[hidden], geo.block,
                                   group=geo.group, damp=damp, chunk=chunk, impl=impl)
        emb.weight.copy_(ed.to(emb.weight.device))
        tensors["embed_tokens"] = TernaryTensor(ec, es)
        log(f"embedding {tuple(ec.shape)} quantized in {time.time() - t0:.0f}s")
    state = TernaryState(tensors, {hidden: signs[hidden]} if tensors else {}, geo.block, geo.group,
                         {"stage": "endpoints", "head": "ternary" if want_head or (adapter.tied and tensors) else "bf16",
                          "embedding": "ternary" if want_emb or (adapter.tied and tensors) else "bf16",
                          "damp": damp, "tied": adapter.tied})
    save_state(out / "state.safetensors", state)
    return {"tensors": sorted(tensors)}


def joint_hessian(head_h, gain):
    """Equal-weight, unit-mean-diagonal head and embedding reconstruction metrics."""
    h = head_h.float()
    g2 = gain.float().square()
    tiny = torch.finfo(h.dtype).eps
    return h / h.diag().mean().clamp_min(tiny) + torch.diag(g2 / g2.mean().clamp_min(tiny))

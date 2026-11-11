"""Endpoints and distillation stages on the tiny model."""
import json

import numpy as np
import pytest
import torch

from openbonsai.cli import execute
from openbonsai.core.ternary import TernaryEmbedding, TernaryLinear
from openbonsai.formats.state import read_scales
from openbonsai.stages import distill as D
from openbonsai.stages.common import apply_dense, install_ternary


@pytest.fixture
def endpointed(calibrated):
    assert execute(calibrated, "quantize") == "ran"
    assert execute(calibrated, "endpoints") == "ran"
    return calibrated


def test_endpoint_state_and_dense_and_ternary_installs_agree(endpointed):
    ctx = endpointed
    state = ctx.state_through("endpoints")
    assert {"lm_head", "embed_tokens"} <= set(state.tensors) and len(state.tensors) == 27
    assert state.tensors["lm_head"].shape == (320, 128)
    ids = ctx.calib_tokens()[:2]
    dense, _ = ctx.load_model()
    apply_dense(ctx.adapter(), dense, state.records())
    tern, _ = ctx.load_model()
    installed = install_ternary(ctx.adapter(), tern, state.records(), torch.device("cpu"))
    assert isinstance(tern.lm_head, TernaryLinear) and isinstance(tern.model.embed_tokens, TernaryEmbedding)
    assert len(installed) == 27
    with torch.no_grad():
        a = dense(input_ids=ids, use_cache=False).logits.float()
        b = tern(input_ids=ids, use_cache=False).logits.float()
    # Same ternary weights; BF16 rounding of the two arithmetic orders differs slightly.
    assert torch.allclose(a, b, atol=0.15, rtol=0.05)
    assert (a.argmax(-1) == b.argmax(-1)).float().mean() > 0.9


def test_tied_models_keep_endpoints_high_precision(endpointed, monkeypatch):
    ctx = endpointed
    monkeypatch.setattr(type(ctx.adapter()), "tied", property(lambda self: True))
    with pytest.raises(NotImplementedError, match="tied"):
        execute(ctx, "endpoints", force=True)
    ctx.cfg["endpoints"].update(head="bf16", embedding="bf16")
    assert execute(ctx, "endpoints") == "ran"
    assert ctx.endpoints_state().tensors == {}


def test_teacher_shards_resume_and_must_be_contiguous(endpointed):
    ctx = endpointed
    D.run_teacher(ctx)
    t = ctx.run.stage_dir("distill") / "teacher"
    shards = sorted(t.glob("topk_*.npz"))
    assert [s.name for s in shards] == ["topk_000000.npz", "topk_000002.npz"]
    ids, idx, lp = D.load_teacher(t)
    assert ids.shape == (4, 64) and idx.shape == (4, 64, 16) and lp.dtype == np.float16
    assert (np.diff(lp.astype(np.float32), axis=-1) <= 0).all()  # top-k sorted
    first = shards[1].read_bytes()
    shards[1].unlink()
    before = shards[0].stat().st_mtime_ns
    D.run_teacher(ctx)
    assert shards[1].read_bytes() == first and shards[0].stat().st_mtime_ns == before
    shards[0].unlink()
    with pytest.raises(ValueError, match="contiguous"):
        D.load_teacher(t)


def _kd_objective(model, adapter, ids, tidx, tlogp):
    from openbonsai.core.losses import kd_chunk_loss
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(len(ids)):
            h = adapter.text_model(model)(input_ids=torch.from_numpy(ids[i:i + 1].astype(np.int64)),
                                          use_cache=False).last_hidden_state[0][:-1]
            total += float(kd_chunk_loss(h, adapter.head(model), torch.from_numpy(tidx[i][:-1].astype(np.int64)),
                                         torch.from_numpy(tlogp[i][:-1])))
            n += h.shape[0]
    return total / n


def _student(ctx):
    model, _ = ctx.load_model()
    install_ternary(ctx.adapter(), model, ctx.state_through("endpoints").records(), torch.device("cpu"))
    return model


def test_three_distillation_steps_reduce_the_kd_loss(endpointed, tmp_path):
    ctx = endpointed
    D.run_teacher(ctx)
    ids, tidx, tlogp = D.load_teacher(ctx.run.stage_dir("distill") / "teacher")
    model = _student(ctx)
    before = _kd_objective(model, ctx.adapter(), ids, tidx, tlogp)
    out = tmp_path / "train"
    out.mkdir()
    history = D.train(model, ctx.adapter(), ids, tidx, tlogp, out, lr=0.05, epochs=3, accum=4, warmup=1,
                      chunk=32, log_every=1, ckpt_every=1, device=torch.device("cpu"), log=lambda m: None)
    after = _kd_objective(model, ctx.adapter(), ids, tidx, tlogp)
    assert len(history) == 3 and after < before
    scales = D.trained_scales(model, ctx.adapter())
    assert len(scales) == 27 and all(s.dtype == torch.float16 for s in scales.values())


def test_resumed_training_equals_uninterrupted_training(endpointed, tmp_path):
    ctx = endpointed
    D.run_teacher(ctx)
    data = D.load_teacher(ctx.run.stage_dir("distill") / "teacher")
    kw = dict(lr=0.02, epochs=2, accum=2, warmup=1, chunk=32, log_every=10, ckpt_every=2,
              device=torch.device("cpu"), log=lambda m: None)
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir(), b_dir.mkdir()
    a = _student(ctx)
    hist_a = D.train(a, ctx.adapter(), *data, a_dir, **kw)
    assert [h["step"] for h in hist_a] == [1, 2, 3, 4]
    b = _student(ctx)
    with pytest.raises(KeyboardInterrupt):   # crash after step 3; the last checkpoint is step 2
        D.train(b, ctx.adapter(), *data, b_dir, stop_after=3, **kw)
    b = _student(ctx)  # fresh process state: only the checkpoint carries over
    hist_b = D.train(b, ctx.adapter(), *data, b_dir, **kw)
    assert [h["step"] for h in hist_b] == [3, 4]
    assert [h["kd_ce"] for h in hist_b] == [h["kd_ce"] for h in hist_a[2:]]
    sa, sb = D.trained_scales(a, ctx.adapter()), D.trained_scales(b, ctx.adapter())
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    # The step-3 record written before the crash is replaced, not duplicated.
    logged = [json.loads(line) for line in (b_dir / "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in logged] == [1, 2, 3, 4] and logged[2]["kd_ce"] == hist_a[2]["kd_ce"]


def test_max_steps_ends_the_cosine_schedule(endpointed, tmp_path):
    ctx = endpointed
    D.run_teacher(ctx)
    data = D.load_teacher(ctx.run.stage_dir("distill") / "teacher")
    history = D.train(_student(ctx), ctx.adapter(), *data, tmp_path, lr=0.02, epochs=4, accum=2, warmup=1,
                      chunk=32, max_steps=3, device=torch.device("cpu"), log=lambda m: None)
    assert [h["step"] for h in history] == [1, 2, 3] and history[-1]["lr"] == 0.0
    assert history[0]["lr"] > history[1]["lr"] > 0.0


def test_distill_stage_writes_scales_and_eval_uses_them(endpointed):
    ctx = endpointed
    assert execute(ctx, "distill") == "ran"
    out = ctx.run.stage_dir("distill")
    scales, meta = read_scales(out / "scales.safetensors")
    assert meta["base"] == ctx.distill_base() and meta["base"]["endpoints"] is not None
    state = ctx.state_through("distill")
    assert all(torch.equal(state.tensors[k].scales, v) for k, v in scales.items())
    steps = [json.loads(line)["step"] for line in (out / "train.jsonl").read_text().splitlines()]
    assert steps == [1, 2]
    assert execute(ctx, "distill") == "skipped"
    # Logging cadence is not part of the result: changing it neither reruns nor drops the checkpoint.
    ctx.cfg["distill"].update(log_every=7, ckpt_every=3)
    assert execute(ctx, "distill") == "skipped"
    # A second teacher-complete run does not load the teacher model.
    loads = []
    real = ctx.load_model
    ctx.load_model = lambda: (loads.append(1), real())[1]
    assert D.run_teacher(ctx)["rows"] == 4 and loads == []

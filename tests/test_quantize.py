"""Quantize stage: per-layer resume is byte-identical; kwargs are captured per layer."""
import pytest
import torch

from openbonsai.cli import execute
from openbonsai.stages import quantize as Q


def _state_bytes(ctx):
    return (ctx.run.stage_dir("quantize") / "state.safetensors").read_bytes()


def test_interrupted_quantize_resumes_to_identical_bytes(calibrated, make_ctx, tmp_path, monkeypatch):
    full = calibrated
    assert execute(full, "quantize") == "ran"
    reference = _state_bytes(full)

    # Second run directory with the same calibration: stop after layer 1, then rerun.
    ctx = make_ctx(tmp_path / "resumed")
    for stage in ("traces", "calibrate"):
        execute(ctx, stage)
    assert (ctx.run.stage_dir("calibrate") / "tokens.npy").read_bytes() == \
        (full.run.stage_dir("calibrate") / "tokens.npy").read_bytes()
    real = Q.quantize_model
    monkeypatch.setattr(Q, "quantize_model", lambda *a, **k: real(*a, **{**k, "stop_after": 1}))
    with pytest.raises(KeyboardInterrupt):
        execute(ctx, "quantize")
    layers = sorted(p.name for p in (ctx.run.stage_dir("quantize") / "layers").glob("*.safetensors"))
    assert layers == ["0000.safetensors", "0001.safetensors"]
    assert ctx.run.done_info("quantize") is None
    monkeypatch.setattr(Q, "quantize_model", real)
    reused = []
    log = ctx.log
    monkeypatch.setattr(ctx, "log", lambda stage: (lambda msg: (reused.append(msg), log(stage)(msg))))
    assert execute(ctx, "quantize") == "ran"
    assert sum("reused" in m for m in reused) == 2
    assert _state_bytes(ctx) == reference
    for name in layers + ["0002.safetensors", "0003.safetensors"]:
        assert (ctx.run.stage_dir("quantize") / "layers" / name).read_bytes() == \
            (full.run.stage_dir("quantize") / "layers" / name).read_bytes()
    assert execute(ctx, "quantize") == "skipped"


def test_changed_config_discards_stale_layers(calibrated):
    ctx = calibrated
    execute(ctx, "quantize")
    before = _state_bytes(ctx)
    ctx.cfg["quantize"]["damp"] = 0.05
    assert execute(ctx, "quantize") == "ran"
    assert _state_bytes(ctx) != before


def test_layer_kwargs_are_captured_per_layer_type(tiny_dir):
    from openbonsai.adapters import load_adapter
    adapter = load_adapter(tiny_dir)
    model, _ = adapter.load(tiny_dir, device="cpu", dtype=torch.bfloat16, attn="eager")
    ids = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(0))
    kwargs = Q.capture_layer_kwargs(model, adapter, ids, torch.device("cpu"))
    assert len(kwargs) == 4
    # Eager full attention gets a 4-D causal mask; Gated DeltaNet layers get none.
    assert kwargs[3]["attention_mask"] is not None and kwargs[3]["attention_mask"].ndim == 4
    assert all(kwargs[i]["attention_mask"] is None for i in range(3))
    with pytest.raises(ValueError, match="multiple of quantize.batch"):
        Q.quantize_model(model, adapter, ids[:1].repeat(3, 1), {}, None, batch=2)

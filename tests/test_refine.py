"""Refine stage (docs/refine.md) on tiny random models, CPU only.

- The latent module computes exactly what its TernaryLinear computes until it trains.
- Step 0 is the identity: no steps (or zero learning rates) give quantize's codes and scales bit for bit.
- Training lowers the held-out error; rollback keeps a window's input codes when it did not.
- Codes change only when allowed (freeze_codes_after 0 trains the scales alone); the flip statistics
  match a direct comparison; assign quantile ends with exactly ``zeros`` zeros per group.
- An interrupted run resumes per layer to a byte-identical state (also with window 2 and disk streams).
- Downstream: distill trains on refine's state and records its digest, the GGUF export verifies, eval
  scores ``--from refine``, and a changed refine setting makes refine, distill and export stale.
- The dense adapter (Llama with biases) runs the same stage end to end.
"""
import json
import math
from pathlib import Path
import re

import pytest
import torch

from openbonsai import cli
from openbonsai.cli import execute
from openbonsai.config import ConfigError, load_config, stage_inputs, upstream
from openbonsai.core.hadamard import rotate, seeded_signs
from openbonsai.core.ternary import LatentTernaryLinear, TernaryLinear, shared_hadamard
from openbonsai.formats.state import load_state, read_scales
from openbonsai.rundir import StageNotReady
from openbonsai.stages import common as C
from openbonsai.stages import refine as R

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "examples" / "tiny-qwen3_5.yaml"
# 8 calibration blocks of 64 tokens in the tiny config: 6 to train, 2 held out
BASE = ["refine.enabled=true", "refine.blocks=6", "refine.val_blocks=2", "refine.epochs=4"]
# large enough for the tiny fixture to flip codes in 24 steps (the 9B default is 5e-3)
FLIPS = ["refine.lr_codes=0.05", "refine.lr_scales=5.0e-3"]


def _sets(ctx, *assignments):
    from openbonsai.config import apply_set
    for a in assignments:
        apply_set(ctx.cfg, a)
    return ctx


@pytest.fixture
def endpointed(calibrated):
    assert execute(calibrated, "quantize") == "ran"
    assert execute(calibrated, "endpoints") == "ran"
    return calibrated


def _report(ctx):
    return json.loads((ctx.run.stage_dir("refine") / "report.json").read_text(encoding="utf-8"))


def _state_bytes(ctx):
    return (ctx.run.stage_dir("refine") / "state.safetensors").read_bytes()


def _same(a, b, field):
    return all(torch.equal(getattr(a.tensors[k], field), getattr(b.tensors[k], field)) for k in a.tensors)


# -- the module ----------------------------------------------------------------------------------------------

def _module_args(seed=0, out=48, width=256, block=128, group=128):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(-1, 2, (out, width), generator=g).to(torch.int8)
    scales = (0.01 + 0.02 * torch.rand(out, width // group, generator=g)).half()
    signs = seeded_signs(width, width)
    return codes, scales, signs, block, shared_hadamard(block, "cpu")


@pytest.mark.parametrize("assign,init", [("nearest", "center"), ("nearest", "original"), ("quantile", "center"),
                                         ("quantile", "original")])
def test_latent_module_is_its_ternary_linear_until_it_trains(assign, init):
    codes, scales, signs, block, h = _module_args()
    bias = torch.randn(codes.shape[0]).to(torch.bfloat16)
    w = torch.randn(codes.shape, generator=torch.Generator().manual_seed(1)) * 0.02
    w_rot = rotate(w, signs, block)
    ref = TernaryLinear(codes, scales, signs, block, h, bias=bias)
    lat = LatentTernaryLinear(codes, scales, signs, block, h, bias=bias, assign=assign, init=init, w_rot=w_rot,
                              zeros=42)
    x = torch.randn(2, 5, codes.shape[1]).to(torch.bfloat16)
    assert torch.equal(lat.weight_deq(), ref.weight_deq())
    assert torch.equal(lat(x), ref(x))
    assert torch.equal(lat.assign_codes(), codes)
    got_codes, got_scales = lat.hard()
    assert torch.equal(got_codes, codes) and torch.equal(got_scales, scales)
    assert torch.equal(codes, ref.codes)          # the input codes are never written through
    # The straight-through gradient reaches the latent and the log multipliers.
    lat(x).float().pow(2).sum().backward()
    assert lat.latent.grad is not None and lat.latent.grad.abs().sum() > 0
    assert lat.log_mult.grad is not None and lat.log_mult.grad.abs().sum() > 0


def test_latent_codes_follow_the_assignment_rules():
    codes, scales, signs, block, h = _module_args(out=8)
    lat = LatentTernaryLinear(codes, scales, signs, block, h)
    with torch.no_grad():
        lat.latent.copy_(torch.tensor([0.49, 0.51, -0.51, -1.4]).repeat(8, 64))
    assert lat.assign_codes()[0, :4].tolist() == [0, 1, -1, -1]
    with torch.no_grad():
        lat.log_mult.fill_(math.log(2.0))         # threshold at s/2 in weight units: V e^-u = 0.245 ... -0.7
    assert lat.assign_codes()[0, :4].tolist() == [0, 0, 0, -1]
    lat.freeze()
    with torch.no_grad():
        lat.latent.fill_(1.0)
        lat.log_mult.zero_()
    assert lat.assign_codes()[0, :4].tolist() == [0, 0, 0, -1] and not lat.latent.requires_grad
    # quantile: exactly k zeros per group, ties by the original weight's magnitude, signs from it at 0
    w_rot = torch.linspace(-1, 1, 8 * 256).view(8, 256)
    q = LatentTernaryLinear(codes, scales, signs, block, h, assign="quantile", w_rot=w_rot, zeros=42)
    q.set_anneal(1.0)
    with torch.no_grad():
        q.latent.zero_()                          # every |V| ties: the 42 smallest |w_rot| become 0
    c = q.assign_codes()
    assert ((c == 0).view(8, 2, 128).sum(-1) == 42).all()
    a = w_rot.abs().view(-1, 128)
    zero = (c == 0).view(-1, 128)
    assert (a[zero].view(-1, 42).max(1).values <= a[~zero].view(-1, 86).min(1).values).all()
    assert torch.equal(c[~(c == 0)], torch.where(w_rot < 0, -1, 1).to(torch.int8)[~(c == 0)])


# -- configuration -------------------------------------------------------------------------------------------

def test_refine_is_off_by_default_and_leaves_the_other_stages_hashes_alone():
    cfg = load_config(TINY)
    assert cfg["refine"]["enabled"] is False
    assert upstream(cfg, "distill") == ("quantize", "endpoints", "traces")
    assert upstream(cfg, "export") == ("distill",)
    on = load_config(TINY, ["refine.enabled=true"])
    assert upstream(on, "distill") == ("quantize", "refine", "endpoints", "traces")
    assert upstream(on, "refine") == ("quantize", "calibrate", "endpoints")
    assert upstream(load_config(TINY, ["refine.enabled=true", "distill.enabled=false"]), "export") == ("refine",)
    for stage in ("traces", "calibrate", "quantize", "endpoints", "distill", "export", "eval"):
        assert stage_inputs(cfg, stage) == stage_inputs(on, stage)
    # Where the streams live is not part of the result; mix/zeros only count when used.
    a = stage_inputs(on, "refine")
    assert a == stage_inputs(load_config(TINY, ["refine.enabled=true", "refine.cache=disk", "refine.mix=0.1",
                                                "refine.zeros=40"]), "refine")
    assert a != stage_inputs(load_config(TINY, ["refine.enabled=true", "refine.target=mix"]), "refine")
    assert a != stage_inputs(load_config(TINY, ["refine.enabled=true", "refine.lr_codes=0.01"]), "refine")


@pytest.mark.parametrize("sets,match", [
    (["refine.target=teacher"], "refine.target"),
    (["refine.blocks=5", "refine.batch=2"], "multiples of refine.batch"),
    (["refine.blocks=7", "refine.val_blocks=2"], "exceeds the 8 calibration blocks"),
    (["refine.val_blocks=0"], "rollback needs held-out blocks"),
    (["refine.layers=[2, 9]"], r"0 <= start < stop <= 4"),
    (["refine.assign=quantile", "refine.zero_anneal=0.9"], "zero_anneal"),
    (["refine.zeros=200", "refine.assign=quantile"], "refine.zeros"),
    (["refine.lr_codes=fast"], "refine.lr_codes must be a number"),
    (["refine.checkpoint=sometimes"], "refine.checkpoint"),
])
def test_bad_refine_settings_fail_before_any_stage(sets, match, tiny_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "run"
    args = ["run", "--model", str(tiny_dir), "--config", str(TINY), "--out", str(out), "--set", "refine.enabled=true",
            "--set", "refine.blocks=6", "--set", "refine.val_blocks=2"]
    assert cli.main(args + [a for s in sets for a in ("--set", s)]) == 2
    assert re.search(match, capsys.readouterr().err)
    assert not (out / "traces").exists()
    with pytest.raises(ConfigError, match=match):
        R.check_config(load_config(TINY, BASE + sets), 4)


def test_training_blocks_are_stratified_by_source_and_the_held_out_set_is_stable(calibrated):
    labels = R.source_labels(calibrated.run.stage_dir("calibrate"), 8)
    assert sorted(labels) == [0, 0, 0, 0, 1, 1, 2, 2]           # 4 text, 2 chat, 2 trace blocks, permuted
    tokens = calibrated.calib_tokens()
    sources = json.loads((calibrated.run.stage_dir("calibrate") / "sources.json").read_text())
    assert [s["blocks"] for s in sources["sources"]] == [4, 2, 2] and tokens.shape == (8, 64)
    train, val = R.select_blocks(labels, 4, 4, seed=0)
    assert sorted(labels[i] for i in val) == [0, 0, 1, 2] and sorted(labels[i] for i in train) == [0, 0, 1, 2]
    assert not set(train) & set(val)
    assert R.select_blocks(labels, 2, 4, seed=0)[1] == val     # more or fewer training blocks: same held-out set
    assert R.select_blocks(labels, 4, 4, seed=1) != (train, val)
    many = [k for k in range(4) for _ in range(25)]
    t, _ = R.select_blocks(many, 40, 0, seed=3)
    assert [sum(many[i] == k for i in t) for k in range(4)] == [10, 10, 10, 10]


# -- acceptance on the tiny Qwen3.5 ------------------------------------------------------------------------------

def test_step_zero_reproduces_the_quantize_state_bit_for_bit(endpointed):
    ctx = _sets(endpointed, *BASE, "refine.epochs=0")
    assert execute(ctx, "refine") == "ran"
    q, r = ctx.quantize_state(), ctx.refine_state()
    assert set(r.tensors) == set(q.tensors) and _same(q, r, "codes") and _same(q, r, "scales")
    assert r.block == q.block and r.group == q.group and all(torch.equal(q.signs[w], r.signs[w]) for w in q.signs)
    windows = _report(ctx)["windows"]
    assert len(windows) == 4 and all(w["val_before"] == w["val_after"] and not w["rollback"] for w in windows)
    # Zero learning rates: the steps run, nothing moves.
    _sets(ctx, "refine.epochs=2", "refine.lr_codes=0", "refine.lr_scales=0")
    assert execute(ctx, "refine") == "ran"
    r = ctx.refine_state()
    assert _same(q, r, "codes") and _same(q, r, "scales")
    assert all(w["steps"] == 12 and w["flips"] == 0 and not w["changed"] for w in _report(ctx)["windows"])


def test_training_lowers_the_held_out_error_and_rollback_guards_each_layer(endpointed):
    ctx = _sets(endpointed, *BASE, *FLIPS)
    assert execute(ctx, "refine") == "ran"
    windows = _report(ctx)["windows"]
    before = [w["objective_before"] for w in windows]
    after = [w["objective_after"] for w in windows]
    assert sum(after) / len(after) < sum(before) / len(before)
    kept = [w for w in windows if not w["rollback"]]
    assert kept and all(w["objective_after"] < w["objective_before"] for w in kept)
    assert all(w["train_loss_last"] < w["train_loss_first"] for w in kept)
    q, r = ctx.quantize_state(), ctx.refine_state()
    for w in windows:
        layer = {k: t for k, t in r.tensors.items() if k.startswith(f"layers.{w['layers'][0]}.")}
        same = all(torch.equal(t.codes, q.tensors[k].codes) and torch.equal(t.scales, q.tensors[k].scales)
                   for k, t in layer.items())
        assert same == w["rollback"]                 # a rolled-back layer is exactly its input
        assert 0 <= w["eps_damped"] <= 0.01          # Adam's eps does not damp the latent updates
        assert w["latent_grad_rms"] > 100 * float(ctx.cfg["refine"]["eps"])
    assert r.meta["rolled_back"] == [w["layers"][0] for w in windows if w["rollback"]]
    log = (ctx.run.root / "logs" / "refine.log").read_text(encoding="utf-8")
    assert "held-out rel err vs target" in log and "code flips" in log and "scale drift" in log


def test_codes_change_only_when_allowed_and_the_flip_statistics_are_exact(endpointed):
    ctx = _sets(endpointed, *BASE, *FLIPS, "refine.rollback=false", "refine.val_blocks=2")
    assert execute(ctx, "refine") == "ran"
    q, r = ctx.quantize_state(), ctx.refine_state()
    assert not _same(q, r, "codes")
    for w in _report(ctx)["windows"]:
        for key, stats in w["modules"].items():
            c0, c = q.tensors[key].codes, r.tensors[key].codes
            n = c.numel()
            assert stats["flips"] == int((c != c0).sum()) / n
            assert stats["sign_flips"] == int((c.int() * c0.int() == -1).sum()) / n
            assert stats["zero_flips"] == int(((c == 0) != (c0 == 0)).sum()) / n
            assert stats["zeros_after"] == int((c == 0).sum()) / n
        assert w["flips"] > 0
    # Codes frozen from the start: only the scales train.
    _sets(ctx, "refine.freeze_codes_after=0")
    assert execute(ctx, "refine") == "ran"
    r = ctx.refine_state()
    assert _same(q, r, "codes") and not _same(q, r, "scales")
    assert all(w["code_steps"] == 0 and w["eps_damped"] is None and w["latent_grad_rms"] is None
               for w in _report(ctx)["windows"])
    # Bonsai-like: every group ends with exactly `zeros` zeros.
    _sets(ctx, "refine.freeze_codes_after=0.8", "refine.assign=quantile", "refine.zeros=42")
    assert execute(ctx, "refine") == "ran"
    r = ctx.refine_state()
    for key, t in r.tensors.items():
        assert ((t.codes == 0).view(t.codes.shape[0], -1, 128).sum(-1) == 42).all(), key


def test_interrupted_refine_resumes_to_identical_bytes(endpointed, make_ctx, tmp_path, monkeypatch):
    sets = [*BASE, *FLIPS, "refine.epochs=2"]
    full = _sets(endpointed, *sets)
    assert execute(full, "refine") == "ran"
    reference = _state_bytes(full)
    ctx = make_ctx(tmp_path / "resumed", *sets)
    for stage in ("traces", "calibrate", "quantize", "endpoints"):
        execute(ctx, stage)
    real = R.refine_model
    monkeypatch.setattr(R, "refine_model", lambda *a, **k: real(*a, **{**k, "stop_after": 1}))
    with pytest.raises(KeyboardInterrupt):
        execute(ctx, "refine")
    layers = sorted(p.name for p in (ctx.run.stage_dir("refine") / "layers").glob("*.safetensors"))
    assert layers == ["0000.safetensors", "0001.safetensors"] and ctx.run.done_info("refine") is None
    monkeypatch.setattr(R, "refine_model", real)
    assert execute(ctx, "refine") == "ran"
    assert _state_bytes(ctx) == reference
    log = (ctx.run.root / "logs" / "refine.log").read_text(encoding="utf-8")
    assert "layers [0] reused" in log and "layers [1] reused" in log and "layers [2] reused" not in log
    assert execute(ctx, "refine") == "skipped"
    # A changed setting discards the saved layers (they were made under other inputs).
    _sets(ctx, "refine.lr_codes=0.02")
    assert execute(ctx, "refine") == "ran"
    assert _state_bytes(ctx) != reference


def test_window_two_with_disk_streams_resumes_and_matches_memory_streams(endpointed, make_ctx, tmp_path, monkeypatch):
    sets = [*BASE, *FLIPS, "refine.epochs=2", "refine.window=2", "refine.target=mix", "refine.rollback=false"]
    mem = _sets(endpointed, *sets, "refine.cache=cpu")
    assert execute(mem, "refine") == "ran"
    assert [w["layers"] for w in _report(mem)["windows"]] == [[0, 1], [2, 3]]
    reference = _state_bytes(mem)
    ctx = make_ctx(tmp_path / "disk", *sets, "refine.cache=disk")
    for stage in ("traces", "calibrate", "quantize", "endpoints"):
        execute(ctx, stage)
    real = R.refine_model
    monkeypatch.setattr(R, "refine_model", lambda *a, **k: real(*a, **{**k, "stop_after": 0}))
    with pytest.raises(KeyboardInterrupt):
        execute(ctx, "refine")
    assert sorted(p.name for p in (ctx.run.stage_dir("refine") / "layers").glob("*.safetensors")) == [
        "0000.safetensors", "0001.safetensors"]
    monkeypatch.setattr(R, "refine_model", real)
    assert execute(ctx, "refine") == "ran"
    assert _report(ctx)["streams"] == "disk" and not (ctx.run.stage_dir("refine") / "streams").exists()
    assert _state_bytes(ctx) == reference


def test_gpu_streams_are_captured_in_cpu_memory_and_moved_once_the_model_has_left(endpointed, make_ctx, tmp_path,
                                                                                 monkeypatch):
    """Setup holds the whole BF16 model on the device; GPU streams wait in CPU memory until it has gone, so
    the model and up to GPU_STREAM_BUDGET of streams are never on the GPU together. Same bytes as cpu."""
    sets = [*BASE, *FLIPS, "refine.epochs=1", "refine.rollback=false"]
    cpu = _sets(endpointed, *sets, "refine.cache=cpu")
    assert execute(cpu, "refine") == "ran"
    reference = _state_bytes(cpu)
    ctx = make_ctx(tmp_path / "gpu", *sets)
    for stage in ("traces", "calibrate", "quantize", "endpoints"):
        execute(ctx, stage)
    events = []
    real_append, real_relocate, real_to = R.Stream.append, R.Stream.relocate, torch.nn.Module.to

    def append(self, x):
        events.append(("append", self.where))
        real_append(self, x)

    def relocate(self, where, device):
        events.append(("relocate", where))
        real_relocate(self, where, device)

    def to(self, *args, **kwargs):
        if args and args[0] == "cpu" and hasattr(self, "lm_head"):
            events.append(("model to cpu", None))
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(R, "stream_placement", lambda *a, **k: "cuda")     # 'cuda' streams, CPU device
    monkeypatch.setattr(R.Stream, "append", append)
    monkeypatch.setattr(R.Stream, "relocate", relocate)
    monkeypatch.setattr(torch.nn.Module, "to", to)
    assert execute(ctx, "refine") == "ran"
    first = events.index(("model to cpu", None))
    assert events[:first] and set(events[:first]) == {("append", "cpu")}
    assert events[first + 1:first + 5] == [("relocate", "cuda")] * 4          # xq, xfp, vq, vfp
    assert _report(ctx)["streams"] == "cuda" and _state_bytes(ctx) == reference


def test_vram_cap_turns_a_spill_into_out_of_memory(monkeypatch):
    fractions = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (20 * 2**30, 32 * 2**30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 2**30)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f, device=None: fractions.append(f))
    logs = []
    cap = C.cap_vram("cuda:0", logs.append)
    assert fractions == [pytest.approx((21 * 2**30 - C.VRAM_MARGIN) / (32 * 2**30))]
    assert cap == pytest.approx(21 - C.VRAM_MARGIN / 2**30) and "capped" in logs[0]
    assert C.cap_vram("cpu", logs.append) is None and len(fractions) == 1
    # The context manager (traces, eval generation) lifts the cap however its block ends.
    with pytest.raises(KeyboardInterrupt), C.vram_cap("cuda:0", logs.append) as held:
        assert held == pytest.approx(cap) and len(fractions) == 2
        raise KeyboardInterrupt
    assert fractions[2] == 1.0
    with C.vram_cap("cpu", logs.append) as held:
        assert held is None
    assert len(fractions) == 3


def test_layers_outside_the_range_keep_their_codes_and_quantized_targets_run(endpointed):
    ctx = _sets(endpointed, *BASE, *FLIPS, "refine.layers=[1, 3]", "refine.target=quantized", "refine.init=original",
                "refine.rollback=false")
    assert execute(ctx, "refine") == "ran"
    q, r = ctx.quantize_state(), ctx.refine_state()
    assert [w["layers"] for w in _report(ctx)["windows"]] == [[1], [2]]
    for k in q.tensors:
        layer = int(k.split(".")[1])
        unchanged = torch.equal(q.tensors[k].codes, r.tensors[k].codes) and \
            torch.equal(q.tensors[k].scales, r.tensors[k].scales)
        assert unchanged == (layer not in (1, 2)), k
    assert r.meta["layers"] == [1, 3] and r.meta["settings"]["target"] == "quantized"


def test_distill_export_and_eval_use_the_refine_state(endpointed, monkeypatch):
    monkeypatch.chdir(ROOT)
    ctx = _sets(endpointed, *BASE, *FLIPS, "refine.rollback=false")
    assert execute(ctx, "refine") == "ran"
    assert execute(ctx, "distill") == "ran"
    scales, meta = read_scales(ctx.run.stage_dir("distill") / "scales.safetensors")
    assert meta["base"]["refine"] == ctx.run.done_key("refine")[1] and meta["base"] == ctx.distill_base()
    state = ctx.state_through("distill")
    r = ctx.refine_state()
    assert all(torch.equal(state.tensors[k].codes, r.tensors[k].codes) for k in r.tensors)   # refine's codes
    assert not _same(ctx.quantize_state(), r, "codes")
    through = ctx.state_through("refine")
    assert {"lm_head", "embed_tokens"} <= set(through.tensors) and len(through.tensors) == 27
    assert execute(ctx, "export") == "ran"
    done = ctx.run.done_info("export")["result"]
    assert done["format"] == "ptq1_0-gguf" and done["verified"] is True and done["source"] == "distill"
    for source in ("refine", "distill"):
        assert execute(ctx, "eval", source=source, suites=["compare"]) == "ran"
        assert (ctx.run.stage_dir("eval") / f"{source}.compare.json").exists()
    from openbonsai.stages.evaluate import resolve_source
    assert resolve_source(ctx, "auto") == "distill"
    # A changed refine setting makes refine, distill and export stale; where the streams live does not.
    _sets(ctx, "refine.cache=disk")
    assert ctx.stale_reason("export") is None
    _sets(ctx, "refine.lr_codes=0.02")
    for stage in ("refine", "distill", "export"):
        assert "refine output is stale" in ctx.stale_reason(stage)
    with pytest.raises(StageNotReady):
        ctx.state_through("distill")
    assert resolve_source(ctx, "auto") == "endpoints"
    # Without distill the export writes the refine state itself.
    _sets(ctx, "refine.lr_codes=0.05", "distill.enabled=false")
    assert execute(ctx, "export", force=True) == "ran"
    done = ctx.run.done_info("export")["result"]
    assert done["source"] == "refine" and done["verified"] is True


def test_openbonsai_run_with_refine_on_a_dense_model(tmp_path, monkeypatch):
    """Llama (q/k/v/o and MLP biases, untied): traces through eval with refine on, GGUF export verified."""
    from tiny_models import save_tiny_dense
    monkeypatch.chdir(ROOT)
    model = save_tiny_dense(tmp_path / "llama", "llama")
    out = tmp_path / "run"
    sets = BASE + FLIPS + ["refine.rollback=false"]
    assert cli.main(["run", "--model", str(model), "--config", "examples/tiny-dense.yaml", "--out", str(out)]
                    + [a for s in sets for a in ("--set", s)]) == 0
    for stage in ("refine", "distill", "export"):
        assert (out / stage / "DONE.json").exists(), stage
    q, r = load_state(out / "quantize" / "state.safetensors"), load_state(out / "refine" / "state.safetensors")
    assert set(q.tensors) == set(r.tensors) and len(r.tensors) == 14
    assert not _same(q, r, "codes")
    assert json.loads((out / "export" / "DONE.json").read_text())["result"]["verified"] is True
    assert (out / "eval" / "distill.compare.json").exists()
    assert cli.main(["eval", "--run", str(out), "--from", "refine", "--suite", "compare"]) == 0
    assert cli.main(["refine", "--run", str(out)]) == 0          # up to date
    assert json.loads((out / "refine" / "DONE.json").read_text())["result"]["layers"] == [0, 2]

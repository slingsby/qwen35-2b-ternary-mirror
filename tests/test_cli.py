"""Config resolution and the ``openbonsai`` command line end to end on the tiny model."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
import yaml

from openbonsai import cli
from openbonsai.config import ConfigError, DEFAULTS, apply_set, load_config, stage_inputs
from openbonsai.rundir import RunDir, StageNotReady

ROOT = Path(__file__).resolve().parents[1]


def test_config_layers_defaults_file_and_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"quantize": {"damp": 0.02}, "model": {"path": "x"}}))
    cfg = load_config(path, ["quantize.batch=8", "distill.teacher.n_seq=16", "eval.suites=[dev]"], model="y")
    assert cfg["quantize"] == {"damp": 0.02, "batch": 8} and cfg["model"]["path"] == "y"
    assert cfg["distill"]["teacher"]["n_seq"] == 16 and cfg["distill"]["teacher"]["topk"] == 64
    assert cfg["eval"]["suites"] == ["dev"] and DEFAULTS["quantize"]["damp"] == 0.01
    for bad in ("quantize.dmap=1", "nosuch.key=1", "quantize=1"):
        with pytest.raises(ConfigError):
            apply_set(load_config(), bad)
    path.write_text(yaml.safe_dump({"quantise": {"damp": 0.02}}))
    with pytest.raises(ConfigError, match="quantise"):
        load_config(path)


def test_set_indexes_lists_and_rejects_nested_typos_and_scalar_parents():
    cfg = load_config(None, ["calibrate.sources.2.path=old/ayot128.jsonl", "traces.prompts.rows=[0, 6000]",
                             "distill.teacher.sources=[{kind: jsonl, path: x.jsonl, n: 3}]",
                             "distill.teacher.sources.0.n=5"])
    assert isinstance(cfg["calibrate"]["sources"], list) and len(cfg["calibrate"]["sources"]) == 3
    assert cfg["calibrate"]["sources"][2] == {"kind": "traces", "n": 32, "path": "old/ayot128.jsonl"}
    assert cfg["traces"]["prompts"]["rows"] == [0, 6000] and cfg["distill"]["teacher"]["sources"][0]["n"] == 5
    bad = {"calibrate.sources.7.path=x": "list index", "calibrate.sources.x.path=x": "list index",
           "quantize.damp.x=1": "not a mapping", "distill.teacher.nseq=4": "nseq",
           "rotation.signs.sed=3": "sed", "eval.compare.sources=[{kind: jsonl, pth: x}]": "pth",
           "calibrate.sources=[{n: 3}]": "kind", "rotation.signs=3": "mapping", "quantize.damp={a: 1}": "value",
           "traces.prompts.kinds=x": "kinds"}
    for assignment, match in bad.items():
        with pytest.raises(ConfigError, match=match):
            apply_set(load_config(), assignment)


def test_stage_hash_ignores_device_but_not_numerics():
    a, b = load_config(), load_config(None, ["model.device=cuda"])
    assert stage_inputs(a, "quantize") == stage_inputs(b, "quantize")
    c = load_config(None, ["rotation.impl=fwht"])
    assert stage_inputs(a, "quantize") != stage_inputs(c, "quantize")
    assert stage_inputs(a, "calibrate") == stage_inputs(c, "calibrate")




def test_distill_train_accepts_the_list_form_and_hashes_small_lrs_only_when_used():
    cfg = load_config(None, ["distill.train=[scales, small]"])
    assert cfg["distill"]["train"] == {"scales": True, "small": True}
    with pytest.raises(ConfigError, match="norms"):
        load_config(None, ["distill.train=[scales, norms]"])
    with pytest.raises(ConfigError, match="smal"):
        load_config(None, ["distill.train.smal=true"])
    off = load_config(None, ["distill.lr_small=0.5"])
    assert stage_inputs(off, "distill") == stage_inputs(load_config(), "distill")
    on = load_config(None, ["distill.train.small=true"])
    assert stage_inputs(on, "distill") != stage_inputs(load_config(None, ["distill.train.small=true",
                                                                         "distill.lr_small=0.5"]), "distill")
    assert stage_inputs(load_config(None, ["export.chunk_rows=7"]), "export") == stage_inputs(load_config(), "export")


def test_end_to_end_run_on_cpu_then_skip_and_rerun(tiny_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "run"
    t0 = time.time()
    assert cli.main(["run", "--model", str(tiny_dir), "--config", "examples/tiny-qwen3_5.yaml", "--out", str(out)]) == 0
    elapsed = time.time() - t0
    assert elapsed < 60, f"tiny end-to-end run took {elapsed:.1f}s"
    for f in ("run.yaml", "signs.safetensors", "traces/traces.jsonl", "calib/tokens.npy", "quantize/state.safetensors",
              "endpoints/state.safetensors", "distill/scales.safetensors", "eval/distill.compare.json",
              f"export/OpenBonsai-{tiny_dir.name}-PTQ1_0.gguf"):
        assert (out / f).exists(), f
    assert json.loads((out / "export" / "DONE.json").read_text())["result"]["verified"] is True
    result = json.loads((out / "eval" / "distill.compare.json").read_text())
    assert result["tokens"] == 126 and 0 <= result["top1_agreement"] <= 1
    saved = yaml.safe_load((out / "run.yaml").read_text())
    assert saved["model"]["path"] == str(tiny_dir)
    # Second run: everything up to date.
    capsys.readouterr()
    assert cli.main(["run", "--out", str(out), "--until", "distill"]) == 0
    assert "\"quantize\": \"skipped\"" in capsys.readouterr().out
    for stage in ("traces", "calibrate", "quantize", "endpoints", "distill"):
        assert (out / "logs" / f"{stage}.log").read_text().rstrip().endswith("skipped")
    # A numeric override reruns quantize and everything after it, not the data stages.
    before = (out / "quantize" / "state.safetensors").read_bytes()
    assert cli.main(["run", "--out", str(out), "--set", "quantize.damp=0.05", "--until", "endpoints"]) == 0
    assert (out / "quantize" / "state.safetensors").read_bytes() != before
    assert (out / "calib" / "DONE.json").exists()
    assert "skipped" in (out / "logs" / "calibrate.log").read_text().splitlines()[-1]
    # Per-stage commands and inspect.
    assert cli.main(["eval", "--run", str(out), "--from", "quantize", "--suite", "compare"]) == 0
    assert (out / "eval" / "quantize.compare.json").exists()
    assert cli.main(["inspect", str(out / "quantize" / "state.safetensors")]) == 0
    assert '"block": 128' in capsys.readouterr().out
    assert cli.main(["export", "--run", str(out)]) == 2      # distill is stale after the quantize rerun
    assert "distill output is stale" in capsys.readouterr().err


# -- staleness, --force, disabled stages, sign specs and input files ----------

@pytest.fixture(scope="module")
def finished_run(tiny_dir, tmp_path_factory):
    """One tiny run through distill, copied per test."""
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        out = tmp_path_factory.mktemp("finished") / "run"
        assert cli.main(["run", "--model", str(tiny_dir), "--config", "examples/tiny-qwen3_5.yaml", "--out", str(out),
                         "--until", "distill"]) == 0
    finally:
        os.chdir(cwd)
    return out


@pytest.fixture
def run_copy(finished_run, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "run"
    shutil.copytree(finished_run, out)
    return out


def _ctx(out, *sets):
    from openbonsai.config import apply_set as set_, check_keys, deep_merge
    from openbonsai.stages.common import StageContext
    run = RunDir(out)
    cfg = check_keys(deep_merge(DEFAULTS, run.load_config()))
    for a in sets:
        set_(cfg, a)
    return StageContext(cfg, run)


def test_eval_never_combines_stale_distilled_scales_with_new_codes(run_copy, capsys):
    from openbonsai.formats.state import load_scales
    from openbonsai.stages.evaluate import resolve_source
    out = run_copy
    old = load_scales(out / "distill" / "scales.safetensors")
    assert resolve_source(_ctx(out), "auto") == "distill"
    assert cli.main(["run", "--out", str(out), "--set", "quantize.damp=0.05", "--until", "endpoints"]) == 0
    ctx = _ctx(out)
    assert "distill output is stale" in ctx.stale_reason("distill")
    assert resolve_source(ctx, "auto") == "endpoints"
    with pytest.raises(StageNotReady, match="stale"):
        ctx.state_through("distill")
    capsys.readouterr()
    assert cli.main(["eval", "--run", str(out), "--from", "distill", "--suite", "compare"]) == 2
    assert "distill output is stale" in capsys.readouterr().err
    assert not (out / "eval" / "distill.compare.json").exists()
    # Rerunning distill trains on the new codes; then the pair is consistent again.
    assert cli.main(["distill", "--run", str(out)]) == 0
    ctx = _ctx(out)
    state = ctx.state_through("distill")
    k = "layers.0.mlp.down_proj"
    assert torch.equal(state.tensors[k].codes, ctx.quantize_state().tensors[k].codes)
    assert not torch.equal(state.tensors[k].scales, old[k])
    # Scales whose recorded base does not match the current states are refused even if DONE looks current.
    from openbonsai.formats.state import read_scales, save_scales
    scales, meta = read_scales(out / "distill" / "scales.safetensors")
    save_scales(out / "distill" / "scales.safetensors", scales, {**meta, "base": {"quantize": "0" * 64}})
    with pytest.raises(StageNotReady, match="different quantize"):
        ctx.state_through("distill")


def test_endpoints_from_a_run_where_endpoints_never_ran_is_refused(run_copy):
    shutil.rmtree(run_copy / "endpoints")
    ctx = _ctx(run_copy)
    with pytest.raises(StageNotReady, match="endpoints has not completed"):
        ctx.state_through("endpoints")
    assert cli.main(["eval", "--run", str(run_copy), "--from", "endpoints", "--suite", "compare"]) == 2


def test_disabled_endpoints_clear_their_output_from_later_stages(run_copy):
    from openbonsai.cli import execute
    ctx = _ctx(run_copy, "endpoints.enabled=false")
    before = ctx.run.done_info("distill")["config_hash"]
    assert execute(ctx, "endpoints") == "disabled" and ctx.run.done_info("endpoints") is None
    assert ctx.stage_hash("distill") != before            # distill no longer reads endpoints
    with pytest.raises(StageNotReady, match="disabled"):
        ctx.state_through("endpoints")
    assert execute(ctx, "distill") == "ran"
    state = ctx.state_through("distill")
    assert "lm_head" not in state.tensors and "embed_tokens" not in state.tensors
    meta = __import__("openbonsai.formats.state", fromlist=["x"]).read_scales(
        run_copy / "distill" / "scales.safetensors")[1]
    assert meta["base"]["endpoints"] is None


def test_force_recomputes_resumable_work_and_later_stages_follow_changed_outputs(run_copy, monkeypatch):
    from openbonsai.cli import execute
    from openbonsai.stages import quantize as Q
    ctx = _ctx(run_copy)
    msgs = []
    log = ctx.log
    ctx.log = lambda stage: (lambda m: (msgs.append(m), log(stage)(m)))
    before = (run_copy / "quantize" / "state.safetensors").read_bytes()
    assert execute(ctx, "quantize", force=True) == "ran"
    assert not any("reused" in m for m in msgs)
    assert (run_copy / "quantize" / "state.safetensors").read_bytes() == before
    assert execute(ctx, "endpoints") == "skipped"         # same output digest: later stages stay current
    assert (run_copy / "distill" / "checkpoint.pt").exists()
    msgs.clear()
    assert execute(ctx, "distill", force="downstream") == "ran"   # run --force <earlier stage>
    assert not any("resumed_at_step" in m for m in msgs) and any("model not loaded" in m for m in msgs)
    msgs.clear()
    assert execute(ctx, "distill", force=True) == "ran"            # distill --force
    assert not any("resumed_at_step" in m for m in msgs) and any("teacher 2/4" in m for m in msgs)
    # A forced rerun whose result differs (same config) makes every later stage stale.
    real = Q.quantize_model
    monkeypatch.setattr(Q, "quantize_model", lambda *a, **k: real(*a, **{**k, "damp": 0.2}))
    assert execute(ctx, "quantize", force=True) == "ran"
    assert (run_copy / "quantize" / "state.safetensors").read_bytes() != before
    assert "endpoints output is stale" in ctx.stale_reason("endpoints")
    assert "stale" in ctx.stale_reason("distill")


def test_changed_sign_seed_rewrites_the_run_signs(run_copy):
    from openbonsai.cli import execute
    from openbonsai.core.hadamard import seeded_signs
    from openbonsai.formats.state import load_state, read_signs
    ctx = _ctx(run_copy, "rotation.signs.seed=7")
    assert execute(ctx, "quantize") == "ran"
    st = load_state(run_copy / "quantize" / "state.safetensors")
    assert torch.equal(st.signs[128], seeded_signs(7 + 128, 128))
    signs, _, meta = read_signs(run_copy / "signs.safetensors")
    assert meta["spec"]["seed"] == 7 and torch.equal(signs[256], seeded_signs(7 + 256, 256))
    assert execute(ctx, "endpoints") == "ran"
    assert torch.equal(ctx.endpoints_state().signs[128], seeded_signs(7 + 128, 128))


def test_edited_input_files_make_their_stages_stale(run_copy, tmp_path):
    data = tmp_path / "text.jsonl"
    shutil.copy(ROOT / "examples" / "tiny-data" / "text.jsonl", data)
    ctx = _ctx(run_copy, f"calibrate.sources.0.path={data.as_posix()}")
    assert ctx.stale_reason("calibrate") is not None      # a different path string
    from openbonsai.cli import execute
    assert execute(ctx, "calibrate") == "ran"
    assert ctx.stale_reason("calibrate") is None
    with open(data, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": "an edited line"}) + "\n")
    ctx = _ctx(run_copy, f"calibrate.sources.0.path={data.as_posix()}")
    assert "calibrate output is stale" in ctx.stale_reason("calibrate")


def test_tied_models_fail_before_any_stage_runs(tiny_dir, tmp_path, monkeypatch):
    from openbonsai.adapters.qwen3_5 import Qwen35Adapter
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(Qwen35Adapter, "tied", property(lambda self: True))
    out = tmp_path / "run"
    assert cli.main(["run", "--model", str(tiny_dir), "--config", "examples/tiny-qwen3_5.yaml", "--out", str(out)]) == 2
    assert not (out / "traces").exists()


def test_module_entry_point_and_missing_model(tmp_path):
    r = subprocess.run([sys.executable, "-m", "openbonsai", "--help"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "quantize" in r.stdout and "distill" in r.stdout
    assert cli.main(["calibrate", "--run", str(tmp_path / "empty")]) == 2

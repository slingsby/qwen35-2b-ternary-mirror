"""Checks ``openbonsai run`` makes before the first stage, the CLI's one-line errors, and the guard that keeps
the extended suite's chat-NLL prompts and answers out of training data."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from openbonsai import cli
from openbonsai.config import ConfigError, load_config
from openbonsai.eval import heldout as H

ROOT = Path(__file__).resolve().parents[1]
TINY = ["--config", "examples/tiny-qwen3_5.yaml"]


def _run(model, out, *sets):
    return cli.main(["run", "--model", str(model), *TINY, "--out", str(out)] + [a for s in sets for a in ("--set", s)])


def test_calibration_blocks_must_fill_quantize_batches(tiny_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "run"
    assert _run(tiny_dir, out, "calibrate.sources.0.n=3") == 2          # 3 + 2 + 2 = 7 blocks, batch 4
    err = capsys.readouterr().err
    assert "7 blocks" in err and "quantize.batch (4)" in err and "calibrate.sources.2.n=3" in err
    assert not (out / "traces").exists()
    cfg = load_config("examples/tiny-qwen3_5.yaml", ["calibrate.sources.0.n=3", "calibrate.sources.2.n=3"])
    cli.check_calibration_batch(cfg)                                      # the suggested fix: 8 blocks
    cli.check_calibration_batch(load_config("examples/qwen3_5-9b.yaml"))  # 256 blocks
    from openbonsai.config import STAGES
    from openbonsai.rundir import RunDir
    from openbonsai.stages.common import StageContext
    ctx = StageContext(load_config("examples/tiny-qwen3_5.yaml", ["calibrate.sources.0.n=3"], model=str(tiny_dir)),
                       RunDir(tmp_path / "r2"))
    cli.preflight(ctx, STAGES[:STAGES.index("calibrate") + 1])           # --until calibrate never quantizes
    with pytest.raises(ConfigError, match="quantize.batch"):
        cli.preflight(ctx, ["quantize"])


def test_a_stage_reading_disabled_traces_is_refused_before_any_stage(tiny_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "run"
    assert _run(tiny_dir, out, "traces.enabled=false") == 2
    assert "calibrate reads the traces stage's output, but traces is disabled" in capsys.readouterr().err
    assert not (out / "calibrate").exists()


def _without_chat_template(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    (dst / "chat_template.jinja").unlink(missing_ok=True)
    cfg_path = dst / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.pop("chat_template", None)
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return dst


def test_traces_need_a_chat_template(tiny_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    model = _without_chat_template(tiny_dir, tmp_path / "base")
    out = tmp_path / "run"
    assert _run(model, out) == 2
    err = capsys.readouterr().err
    assert "no chat template" in err and cli.NO_TEMPLATE_SETS in err and err.count("\n") == 2   # error + hint
    assert not (out / "traces").exists()
    # The fix the message shows (with the tiny config's own text sources) gets past the preflight.
    from openbonsai.rundir import RunDir
    from openbonsai.stages.common import StageContext
    sets = ["traces.enabled=false", "calibrate.sources=[{kind: jsonl, path: examples/tiny-data/text.jsonl, n: 4}]",
            "distill.teacher.sources=[{kind: jsonl, path: examples/tiny-data/text.jsonl, n: 2}]"]
    cli.preflight(StageContext(load_config("examples/tiny-qwen3_5.yaml", sets, model=str(model)), RunDir(out)))




def _bin_only(src: Path, dst: Path) -> Path:
    import torch
    from safetensors.torch import load_file
    shutil.copytree(src, dst)
    weights = dst / "model.safetensors"
    torch.save(load_file(str(weights)), dst / "pytorch_model.bin")
    weights.unlink()
    return dst


def test_the_gguf_export_needs_safetensors_weights(tiny_dir, tmp_path, monkeypatch, capsys):
    from openbonsai.adapters import load_adapter
    from openbonsai.stages.export import resolve_format, weights_problem
    monkeypatch.chdir(ROOT)
    model = _bin_only(tiny_dir, tmp_path / "bin")
    assert weights_problem(tiny_dir) is None and weights_problem(tmp_path) is None   # no weight files to tell by
    assert "pytorch_model.bin" in weights_problem(model)
    out = tmp_path / "run"
    assert _run(model, out, "export.format=ptq1_0-gguf") == 2
    assert "model.safetensors" in capsys.readouterr().err
    assert not (out / "traces").exists()
    adapter = load_adapter(str(model))
    cfg = load_config("examples/tiny-qwen3_5.yaml", ["export.format=auto"])
    fmt, reason = resolve_format(cfg, adapter, model)                     # auto: a latent checkpoint instead
    assert fmt == "latent" and "safetensors" in reason
    with pytest.raises(ConfigError, match="safetensors"):
        resolve_format(load_config("examples/tiny-qwen3_5.yaml", ["export.format=ptq1_0-gguf"]), adapter, model)


def test_errors_print_one_line_and_debug_shows_the_traceback(tiny_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("OPENBONSAI_DEBUG", raising=False)
    args = ["calibrate", "--run", str(tmp_path / "run"), "--model", str(tiny_dir), *TINY,
            "--set", "calibrate.sources=[{kind: bogus, n: 4}]"]
    assert cli.main(args) == 2                                            # a plain ValueError from the stage
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 2 and "unknown calibration source kind" in err[0] and "(ValueError)" in err[0]
    assert "--debug" in err[1]
    log = (tmp_path / "run" / "logs" / "calibrate.log").read_text(encoding="utf-8")
    assert "Traceback" in log and "unknown calibration source kind" in log   # the stage log keeps the traceback
    with pytest.raises(ValueError, match="unknown calibration source kind"):
        cli.main(args + ["--debug"])
    with pytest.raises(ValueError, match="unknown calibration source kind"):
        cli.main(["--debug"] + args)
    monkeypatch.setenv("OPENBONSAI_DEBUG", "1")
    with pytest.raises(ValueError, match="unknown calibration source kind"):
        cli.main(args)


# -- the extended suite's chat-NLL traces stay out of training data ----------------------------------

class _QwenLike:
    """Decodes every chat trace as one fixed Qwen3.5-rendered conversation."""

    def decode(self, ids, skip_special_tokens=False):
        return ("<|im_start|>user\nName three   rivers.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                "The Nile, the Amazon and the Danube.�")


def test_chat_needles_are_the_traces_prompts_and_answers(monkeypatch):
    from openbonsai.eval import suites
    from tiny_models import byte_tokenizer
    assert H.chat_needles(byte_tokenizer()) == ()      # not the suite's tokenizer: the suite refuses the model
    assert H.chat_needles(None) == ()
    monkeypatch.setattr(suites, "check_tokenizer", lambda tok, manifest, model=None: None)
    needles = dict((pid, text) for text, pid in H.chat_needles(_QwenLike()))
    assert len(needles) == 32
    assert needles["chat-nll/row-859/prompt"] == "Name three rivers."
    assert needles["chat-nll/row-859/answer"] == "The Nile, the Amazon and the Danube."
    with pytest.raises(H.ContaminationError, match=r"chat-nll/row-\d+/prompt.*traces\.prompts\.rows"):
        H.check_texts(["Please:  Name three\nrivers. Thanks", "unrelated"], "traces.prompts", tuple(
            (t, p) for p, t in needles.items()))


def test_calibration_and_trace_prompts_refuse_chat_nll_items(make_ctx, tmp_path, monkeypatch):
    from openbonsai.stages import calibrate, traces
    from tiny_models import byte_tokenizer
    prompt = json.loads((ROOT / "examples" / "prompts.jsonl").read_text(encoding="utf-8").splitlines()[1])["prompt"]
    text = json.loads((ROOT / "examples" / "tiny-data" / "text.jsonl").read_text(encoding="utf-8").splitlines()[0])
    sentence = text["text"].split(". ")[1][:30]          # inside the first 2 x 64 byte tokens
    fake = ((H.normalize(prompt), "chat-nll/row-859/prompt"), (H.normalize(sentence), "chat-nll/row-859/answer"))
    monkeypatch.setattr(H, "chat_needles", lambda tok: fake)
    ctx = make_ctx(tmp_path / "run")
    with pytest.raises(H.ContaminationError, match="chat-nll/row-859/prompt"):
        traces.load_prompts(ctx)
    src = [{"kind": "jsonl", "path": str(ROOT / "examples" / "tiny-data" / "text.jsonl"), "n": 2}]
    with pytest.raises(H.ContaminationError, match="chat-nll/row-859/answer"):
        calibrate.build(byte_tokenizer(), src, seq_len=64)
    blocks, _ = calibrate.build(byte_tokenizer(), src, seq_len=64, chat_guard=False)   # eval.compare data
    assert len(blocks) == 2


def test_trace_prompts_default_to_rows_without_chat_nll_prompts():
    import inspect
    from openbonsai.stages.traces import ultrachat_prompts
    example = yaml.safe_load((ROOT / "examples" / "qwen3_5-9b.yaml").read_text(encoding="utf-8"))
    assert example["traces"]["prompts"]["rows"] == [5000, 6000]
    assert inspect.signature(ultrachat_prompts).parameters["rows"].default == (5000, 6000)
    rows = [json.loads(line)["source_row_index"]
            for line in (ROOT / "openbonsai" / "eval" / "assets" / "bonsai-dev" / "chat-traces16.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 16 and max(rows) < 5000

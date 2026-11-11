"""PTQ1_0 GGUF export, the GGUF -> torch loader and distilled small tensors on the tiny Qwen3.5.

The tiny model has both layer types and grouped != tiled value heads
(2 key heads, 4 value heads), so every permutation of the qwen35 layout is
exercised. ``src_index`` is an independent statement of the grouped -> tiled
rule (OpenBonsai research/experiments/export_gguf.py's verifier): GGUF row i of a
permuted tensor holds HF row src_index(i).
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from openbonsai import cli
from openbonsai.adapters import adapter_for_config
from openbonsai.adapters.gguf_common import reference_kv_equal
from openbonsai.adapters.qwen3_5_gguf import Qwen35GGUF
from openbonsai.cli import execute
from openbonsai.config import DEFAULTS, apply_set, check_keys, deep_merge
from openbonsai.digest import file_sha256
from openbonsai.formats import gguf as G
from openbonsai.formats import ptq1_0
from openbonsai.formats.checkpoint import Checkpoint
from openbonsai.formats.state import LegacyTensor, read_small
from openbonsai.rundir import RunDir
from openbonsai.stages import export as E
from openbonsai.stages.common import StageContext, apply_small, install_ternary
from tiny_models import tiny_text_config

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


def src_index(kind, n, n_k=2, n_v=4, dv=32, qk=128):
    i = np.arange(n)
    if kind is None:
        return i
    hd, base = {"head": (1, 0), "v": (dv, 0), "qkv": (dv, qk)}[kind]
    out = i.copy()
    j = i[base:] - base
    t, e = j // hd, j % hd
    r, k = t // n_k, t % n_k
    out[base:] = base + (k * (n_v // n_k) + r) * hd + e
    return out


KIND = {"attn_qkv": "qkv", "attn_gate": "v", "ssm_conv1d": "qkv", "ssm_a": "head", "ssm_alpha": "head",
        "ssm_beta": "head", "ssm_dt": "head"}


def kind_of(name):
    stem = name.split(".")[2] if name.startswith("blk.") else ""
    return KIND.get(stem)


@pytest.fixture(scope="module")
def distilled(tiny_dir, tmp_path_factory):
    """A tiny run through distill with train.small (large small-tensor LRs so BF16 values move)."""
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        out = tmp_path_factory.mktemp("export") / "run"
        assert cli.main(["run", "--model", str(tiny_dir), "--config", "examples/tiny-qwen3_5.yaml", "--out", str(out),
                         "--until", "distill", "--set", "distill.train.small=true",
                         "--set", "distill.lr_small=0.05", "--set", "distill.lr_small_mat=0.05"]) == 0
    finally:
        os.chdir(cwd)
    return out


@pytest.fixture
def run_copy(distilled, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "run"
    shutil.copytree(distilled, out)
    return out


def _ctx(out, *sets):
    run = RunDir(out)
    cfg = check_keys(deep_merge(DEFAULTS, run.load_config()))
    for a in sets:
        apply_set(cfg, a)
    return StageContext(cfg, run)


def _export(ctx) -> Path:
    assert execute(ctx, "export") == "ran"
    return Path(ctx.run.done_info("export")["result"]["output"])


def test_export_then_read_decodes_the_state_exactly(run_copy):
    ctx = _ctx(run_copy)
    path = _export(ctx)
    adapter = ctx.adapter()
    state = ctx.state_through("distill")
    assert path == run_copy / "export" / f"OpenBonsai-{Path(ctx.model_path()).name}-PTQ1_0.gguf"
    assert not path.with_name(path.name + ".tmp").exists()
    report = json.loads(path.with_name(path.name + ".export.json").read_text(encoding="utf-8"))
    assert report["verify"] and all(x["ok"] for x in report["verify"])
    assert ctx.run.done_info("export")["output_sha256"] == file_sha256(path)
    n_moved = 0
    with G.GGUFReader(path) as r:
        assert r.value("general.architecture") == "qwen35" and r.value("prism.hadamard.block_size") == 128
        assert r.value("general.name") == path.stem and r.value("general.file_type") == 143
        assert r.value("prism.hadamard.sign_widths") == [128, 256]
        assert r.value("prism.hadamard.sign_values") == state.signs[128].tolist() + state.signs[256].tolist()
        assert r.value("prism.hadamard.inverse_weight_names") == ["token_embd.weight"]
        wn = r.value("prism.hadamard.weight_names")
        assert wn[0] == "output.weight" and len(wn) == 26
        assert wn[19:26] == ["blk.3.ffn_down.weight", "blk.3.ffn_gate.weight", "blk.3.ffn_up.weight",
                             "blk.3.attn_k.weight", "blk.3.attn_output.weight", "blk.3.attn_q.weight",
                             "blk.3.attn_v.weight"]
        assert len(r.value("tokenizer.ggml.tokens")) == 320 and r.value("tokenizer.ggml.eos_token_id") == 256
        names = [t.name for t in r.tensors]
        assert names[:3] == ["output.weight", "output_norm.weight", "token_embd.weight"]
        assert names[3:17] == sorted(names[3:17]) and all(n.startswith("blk.0.") for n in names[3:17])
        by_source = {t.name: t.source for t in adapter.gguf_tensors()}
        for t in r.tensors:
            if t.gtype != G.GGML_PTQ1_0:
                continue
            idx = src_index(kind_of(t.name), t.ne[1])
            n_moved += int((idx != np.arange(t.ne[1])).sum())
            src = state.tensors[by_source[t.name]]
            want_c, want_s = ptq1_0.normalize_scales(src.codes.numpy()[idx], src.scales.numpy()[idx])
            got_c, got_s = r.ptq1_0(t.name)
            assert np.array_equal(got_c, want_c), t.name
            assert np.array_equal(got_s.view(np.uint16), want_s.view(np.uint16)), t.name
    assert n_moved > 0
    # The reusable loader gives the state back (canonical) with the F32 tensors in HF conventions.
    back, floats, _ = adapter.read_gguf(path)
    assert back.block == state.block and set(back.tensors) == set(state.tensors)
    for k, t in state.tensors.items():
        c, s = ptq1_0.normalize_scales(t.codes, t.scales)
        assert np.array_equal(back.tensors[k].codes.numpy(), c) and np.array_equal(back.tensors[k].scales.numpy(), s)
    small = ctx.distill_small()
    with Checkpoint(ctx.model_dir()) as ck:
        for k, v in floats.items():
            ref = small[k] if k in small else ck.get(adapter.checkpoint_name(k)).float()
            assert torch.allclose(v, ref, rtol=0, atol=1e-6), k


def test_distilled_small_tensors_land_in_their_gguf_tensors(run_copy):
    ctx = _ctx(run_copy)
    path = _export(ctx)
    adapter = ctx.adapter()
    small = ctx.distill_small()
    masters = read_small(run_copy / "distill" / "scales.safetensors", cast=False)
    assert len(small) == 29 and set(masters) == set(small)
    assert all(torch.equal(small[k], masters[k].to(torch.bfloat16).float()) for k in small)
    with Checkpoint(ctx.model_dir()) as ck:
        orig = {k: ck.get(adapter.checkpoint_name(k)).to(torch.bfloat16).float() for k in small}
    assert sum(not torch.equal(small[k], orig[k]) for k in small) >= 25   # training moved them
    expect = {
        "output_norm.weight": small["norm.weight"] + 1,
        "blk.0.attn_norm.weight": small["layers.0.input_layernorm.weight"] + 1,
        "blk.2.post_attention_norm.weight": small["layers.2.post_attention_layernorm.weight"] + 1,
        "blk.3.attn_q_norm.weight": small["layers.3.self_attn.q_norm.weight"] + 1,
        "blk.3.attn_k_norm.weight": small["layers.3.self_attn.k_norm.weight"] + 1,
        "blk.1.ssm_norm.weight": small["layers.1.linear_attn.norm.weight"],                 # no +1
        "blk.0.ssm_a": -torch.exp(small["layers.0.linear_attn.A_log"])[src_index("head", 4)],
        "blk.0.ssm_dt.bias": small["layers.0.linear_attn.dt_bias"][src_index("head", 4)],
        "blk.1.ssm_alpha.weight": small["layers.1.linear_attn.in_proj_a.weight"][src_index("head", 4)],
        "blk.2.ssm_beta.weight": small["layers.2.linear_attn.in_proj_b.weight"][src_index("head", 4)],
        "blk.0.ssm_conv1d.weight": small["layers.0.linear_attn.conv1d.weight"][:, 0, :][src_index("qkv", 256)],
    }
    with G.GGUFReader(path) as r:
        for name, want in expect.items():
            got = torch.from_numpy(r.array(name).copy())
            assert got.shape == want.shape and torch.equal(got, want.float()), name


def test_gguf_model_reproduces_the_fake_quant_model_and_eval(run_copy):
    ctx = _ctx(run_copy)
    path = _export(ctx)
    adapter = ctx.adapter()
    ref, _ = ctx.load_model()
    install_ternary(adapter, ref, ctx.state_through("distill").records(), CPU)
    apply_small(adapter, ref, ctx.distill_small())
    ids = ctx.calib_tokens()[:2]
    with torch.no_grad():
        want = ref(input_ids=ids, use_cache=False).logits.float()
        model = adapter.from_gguf(path, model_dir=ctx.model_dir(), dtype=ctx.dtype)
        got = model(input_ids=ids, use_cache=False).logits.float()
        assert (got - want).abs().max().item() <= 1e-5
        # Prism keeps 1-D tensors in F32; then RMSNormGated multiplies in FP32, so BF16 logits move slightly.
        m32 = adapter.from_gguf(path, model_dir=ctx.model_dir(), dtype=ctx.dtype, small_fp32=True)
        assert m32.model.layers[0].linear_attn.norm.weight.dtype == torch.float32
        assert m32.model.layers[0].linear_attn.conv1d.weight.dtype == ctx.dtype
        got32 = m32(input_ids=ids, use_cache=False).logits.float()
        assert (got32 - want).abs().max().item() < 0.05 and (got32.argmax(-1) == want.argmax(-1)).float().mean() > 0.9
    assert model.lm_head.codes.dtype == torch.int8 and not any(p.is_meta for p in model.parameters())
    assert cli.main(["eval", "--run", str(run_copy), "--from", "distill", "--suite", "compare"]) == 0
    assert cli.main(["eval", "--run", str(run_copy), "--from", str(path), "--suite", "compare"]) == 0
    a = json.loads((run_copy / "eval" / "distill.compare.json").read_text())
    b = json.loads((run_copy / "eval" / f"{path.stem}.compare.json").read_text())
    for key in ("nll", "top1_agreement", "kl_top16", "ear_top16_lb", "ear_full", "kl_full"):
        assert a[key] == b[key], key
    # Without --run, a GGUF evaluation gets its own run directory next to the file.
    assert cli.main(["eval", "--from", str(path), "--model", str(ctx.model_path()), "--suite", "compare",
                     "--config", "examples/tiny-qwen3_5.yaml"]) == 0
    c = json.loads((Path(str(path) + ".eval") / "eval" / f"{path.stem}.compare.json").read_text())
    assert c["nll"] == a["nll"]
    assert cli.main(["eval", "--run", str(run_copy), "--from", str(path), "--install", "dense",
                     "--suite", "compare"]) == 2


def test_multimodal_checkpoint_names_and_conditional_generation_model(run_copy, tiny_mm_dir):
    ctx = _ctx(run_copy)
    state = ctx.state_through("endpoints")
    mm = _ctx(run_copy, f"model.path={tiny_mm_dir.as_posix()}")
    adapter = mm.adapter()
    assert adapter.checkpoint_name("norm.weight") == "model.language_model.norm.weight"
    out = run_copy / "mm.gguf"
    E.export_gguf(adapter, state, tiny_mm_dir, out, name="mm", log=lambda m: None)
    ref, _ = mm.load_model()
    install_ternary(adapter, ref, state.records(), CPU)
    model = adapter.from_gguf(out, model_dir=tiny_mm_dir)
    assert type(model).__name__ == "Qwen3_5ForConditionalGeneration" and model.model.visual is None
    ids = ctx.calib_tokens()[:1]
    with torch.no_grad():
        a = ref(input_ids=ids, use_cache=False).logits.float()
        b = model(input_ids=ids, use_cache=False).logits.float()
    assert (a - b).abs().max().item() <= 1e-5


def test_states_the_layout_cannot_hold_are_refused(run_copy, monkeypatch):
    ctx = _ctx(run_copy)
    adapter, model_dir = ctx.adapter(), ctx.model_dir()
    out = run_copy / "x.gguf"
    with pytest.raises(E.ExportError, match="no ternary lm_head and embed_tokens"):
        E.export_gguf(adapter, ctx.state_through("quantize"), model_dir, out, name="x")
    state = ctx.state_through("endpoints")
    legacy = {k: LegacyTensor(t.codes, t.scales, state.signs[t.shape[1]], state.block) for k, t in state.tensors.items()}
    legacy["layers.0.mlp.gate_proj"] = LegacyTensor(state.tensors["layers.0.mlp.gate_proj"].codes,
                                                    state.tensors["layers.0.mlp.gate_proj"].scales,
                                                    -state.signs[128], state.block)
    with pytest.raises(E.ExportError, match="one sign vector per input width"):
        E.export_gguf(adapter, legacy, model_dir, out, name="x")
    with pytest.raises(E.ExportError, match="no F32 slot"):
        E.export_gguf(adapter, state, model_dir, out, name="x", small={"layers.0.mlp.gate_proj.weight": torch.zeros(1)})
    bad = state.with_scales({"lm_head": state.tensors["lm_head"].scales.clone().fill_(float("inf"))})
    with pytest.raises(E.ExportError, match="finite"):
        E.export_gguf(adapter, bad, model_dir, out, name="x", verify=False)
    assert not out.exists() and not out.with_name("x.gguf.tmp").exists()   # no partial file left behind
    monkeypatch.setattr(type(adapter), "tied", property(lambda self: True))
    plan = {t.name: t for t in adapter.gguf_tensors()}
    assert "output.weight" not in plan
    assert not plan["token_embd.weight"].ternary


def test_export_preflight_needs_ternary_endpoints(tiny_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "run"
    assert cli.main(["run", "--model", str(tiny_dir), "--config", "examples/tiny-qwen3_5.yaml", "--out", str(out),
                     "--set", "endpoints.embedding=bf16"]) == 2
    assert not (out / "traces").exists()


def test_export_preflight_needs_distilled_endpoints_and_follows_until_and_skip(tiny_dir, tmp_path, monkeypatch):
    from openbonsai.config import STAGES, ConfigError, load_config
    monkeypatch.chdir(ROOT)
    out = tmp_path / "run"
    assert cli.main(["run", "--model", str(tiny_dir), "--config", "examples/tiny-qwen3_5.yaml", "--out", str(out),
                     "--set", "distill.endpoints=false"]) == 2
    assert not (out / "traces").exists()
    ctx = StageContext(load_config("examples/tiny-qwen3_5.yaml", ["distill.endpoints=false"], model=str(tiny_dir)),
                       RunDir(out))
    with pytest.raises(ConfigError, match="distill.endpoints"):
        cli.preflight(ctx)
    cli.preflight(ctx, STAGES[:STAGES.index("distill") + 1])              # --until distill: no export
    cli.preflight(ctx, [s for s in STAGES if s != "export"])              # --skip export
    bf16 = StageContext(load_config("examples/tiny-qwen3_5.yaml", ["endpoints.head=bf16"], model=str(tiny_dir)),
                        RunDir(out))
    with pytest.raises(ConfigError, match="ternary"):
        cli.preflight(bf16)
    cli.preflight(bf16, STAGES[:STAGES.index("distill") + 1])


def test_a_wrong_layout_permutation_fails_verification(run_copy, monkeypatch):
    """The content checks take rows from the adapter's independent rule, not from the layout."""
    ctx = _ctx(run_copy)
    adapter = ctx.adapter()
    real = type(adapter).gguf_tensors

    def broken(self):
        plan = real(self)
        for t in plan:
            if t.name in ("blk.0.attn_gate.weight", "blk.0.ssm_a"):
                t.perm = None                 # forget the grouped -> tiled value heads
        return plan

    monkeypatch.setattr(type(adapter), "gguf_tensors", broken)
    out = run_copy / "wrong.gguf"
    with pytest.raises(E.ExportError, match="verification failed"):
        E.export_gguf(adapter, ctx.state_through("distill"), ctx.model_dir(), out, name="wrong",
                      small=ctx.distill_small(), log=lambda m: None)
    report = json.loads(out.with_name("wrong.gguf.export.json").read_text())
    failed = [x for x in report["verify"] if not x["ok"]]
    assert {x["check"] for x in failed} >= {
        "layout permutations and F32 transforms == the adapter's independent rule",
        "decoded PTQ1_0 codes and fp16 scale bits == permuted, canonical state (exact)",
        "float tensors (F32, and BF16/F16 tables) bit-identical to the transformed sources"}
    assert "blk.0.attn_gate.weight rows" in failed[0]["detail"] and "blk.0.ssm_a rows" in failed[0]["detail"]


def test_checkpoint_tensors_without_a_slot_are_refused(run_copy):
    from types import SimpleNamespace
    ctx = _ctx(run_copy)
    adapter = ctx.adapter()
    plan = adapter.gguf_tensors()
    with Checkpoint(ctx.model_dir()) as ckpt:
        E.check_coverage(adapter, plan, ckpt)                            # the tiny checkpoint is covered
        names = dict(ckpt.map)
    for extra in ("model.layers.3.self_attn.q_proj.bias", "lm_head.bias"):
        with pytest.raises(E.ExportError, match="would be dropped"):
            E.check_coverage(adapter, plan, SimpleNamespace(map={**names, extra: "x"}))
    E.check_coverage(adapter, plan, SimpleNamespace(map={**names, "mtp.fc.weight": "x"}))   # not the text model
    with pytest.raises(G.GGUFError, match="attention_bias"):
        adapter_for_config(tiny_text_config(attention_bias=True)).gguf_tensors()


def test_verification_failure_keeps_the_final_path_clean(run_copy, monkeypatch):
    ctx = _ctx(run_copy)
    real = E.write_gguf

    def corrupting(path, kvs, infos, produce, **kw):
        info = real(path, kvs, infos, produce, **kw)
        t = next(t for t in infos if t.name == "blk.1.attn_qkv.weight")
        with open(path, "r+b") as f:
            f.seek(info["data_start"] + t.offset + 5)
            byte = f.read(1)[0]
            f.seek(-1, 1)
            f.write(bytes([byte ^ 0x40]))
        return info

    monkeypatch.setattr(E, "write_gguf", corrupting)
    out = run_copy / "bad.gguf"
    with pytest.raises(E.ExportError, match="verification failed"):
        E.export_gguf(ctx.adapter(), ctx.state_through("distill"), ctx.model_dir(), out, name="bad",
                      small=ctx.distill_small(), log=lambda m: None)
    assert not out.exists() and out.with_name("bad.gguf.tmp").exists()
    report = json.loads(out.with_name("bad.gguf.export.json").read_text())
    assert [x["check"] for x in report["verify"] if not x["ok"]] == [
        "decoded PTQ1_0 codes and fp16 scale bits == permuted, canonical state (exact)"]


def test_export_command_out_and_verify_flags(run_copy, capsys):
    target = run_copy / "elsewhere" / "m.gguf"
    assert cli.main(["export", "--run", str(run_copy), "--out", str(target), "--no-verify", "--name", "m"]) == 0
    info = RunDir(run_copy).done_info("export")
    assert target.exists() and info["result"]["verified"] is False and info["output_sha256"] == file_sha256(target)
    assert not json.loads(target.with_name("m.gguf.export.json").read_text()).get("verify")
    capsys.readouterr()
    assert cli.main(["export", "--run", str(run_copy)]) == 0          # flags persist in run.yaml
    assert '"status": "skipped"' in capsys.readouterr().out
    assert cli.main(["export", "--run", str(run_copy), "--verify"]) == 0
    info = RunDir(run_copy).done_info("export")
    assert info["result"]["verified"] is True and info["result"]["checks_passed"] >= 9
    with G.GGUFReader(target) as r:
        assert r.value("general.name") == "m"
    assert cli.main(["inspect", str(target)]) == 0
    assert '"general.name": "m"' in capsys.readouterr().out
    target.unlink()                                                   # a missing output is not "up to date"
    assert cli.main(["export", "--run", str(run_copy)]) == 0
    assert '"status": "ran"' in capsys.readouterr().out and target.exists()


def test_export_out_is_saved_absolute_and_an_edited_output_reruns(run_copy, capsys):
    args = cli.build_parser().parse_args(["export", "--run", str(run_copy), "--out", "rel/m.gguf"])
    assert cli._overrides(args) == [(("export", "out"), str(Path("rel/m.gguf").resolve()))]   # run.yaml outlives cwd
    target = run_copy / "m.gguf"
    assert cli.main(["export", "--run", str(run_copy), "--out", str(target), "--no-verify"]) == 0
    capsys.readouterr()
    assert cli.main(["export", "--run", str(run_copy)]) == 0
    assert '"status": "skipped"' in capsys.readouterr().out
    with open(target, "ab") as f:                                     # an edited output is not current
        f.write(b"\0")
    assert cli.main(["export", "--run", str(run_copy)]) == 0
    assert '"status": "ran"' in capsys.readouterr().out


def test_read_gguf_refuses_another_config(run_copy):
    ctx = _ctx(run_copy)
    path = _export(ctx)
    other = adapter_for_config(tiny_text_config(vocab_size=384))
    with pytest.raises(G.GGUFError, match="do not match"):
        other.read_gguf(path)


# -- layout derivation without weights ---------------------------------------------

def _qwen35_9b_text_config():
    from transformers import Qwen3_5TextConfig
    return Qwen3_5TextConfig(
        vocab_size=248320, hidden_size=4096, intermediate_size=12288, num_hidden_layers=32,
        num_attention_heads=16, num_key_value_heads=4, head_dim=256, linear_num_key_heads=16,
        linear_num_value_heads=32, linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 8, full_attention_interval=4,
        max_position_embeddings=262144, rms_norm_eps=1e-6, attn_output_gate=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                         "mrope_section": [11, 11, 10], "mrope_interleaved": True})


def test_9b_config_gives_the_official_layout_and_weight_names():
    adapter = adapter_for_config(_qwen35_9b_text_config())
    ref = adapter.gguf_reference()
    assert ref is not None
    plan = adapter.gguf_tensors()
    infos = [t.info() for t in plan]
    G.layout(infos)
    assert [(t.name, t.gtype, list(t.ne), t.offset) for t in infos] == \
        [(h["name"], h["type"], list(h["ne"]), h["offset"]) for h in ref["tensors"]]
    assert Qwen35GGUF.gguf_weight_names(plan) == ref["prism_arrays"]["prism.hadamard.weight_names"]
    theirs = {h["key"]: h for h in ref["kv"]}
    hparams = adapter._hparam_kvs()
    assert len(hparams) == 18
    for key, typ, val in hparams:
        assert theirs[key]["type"] == typ and reference_kv_equal(theirs[key], typ, val)[0], key
    # another size: its own layout, no reference
    small = adapter_for_config(tiny_text_config())
    assert small.gguf_reference() is None and len(small.gguf_tensors()) == 56


def test_layout_permutations_match_the_independent_rule():
    adapter = adapter_for_config(tiny_text_config())
    plan = {t.name: t for t in adapter.gguf_tensors()}
    for name, t in plan.items():
        n = t.shape[0] if t.shape else t.ne[-1]
        want = src_index(kind_of(name), n)
        got = np.arange(n) if t.perm is None else t.perm
        assert np.array_equal(got, want), name
        assert np.array_equal(adapter.gguf_source_rows(name, n), want), name     # the verifier's rule
        assert adapter.gguf_f32_op(name) == (None if t.ternary else t.op), name
    assert plan["blk.0.ssm_out.weight"].perm is None          # columns stay grouped; rows untouched


def test_mrope_sections_default_like_transformers():
    def sections(**rope):
        rp = {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25, **rope}
        kvs = {k: v for k, _, v in adapter_for_config(tiny_text_config(rope_parameters=rp))._hparam_kvs()}
        return kvs["qwen35.rope.dimension_sections"]

    assert sections(mrope_section=[2, 1, 1]) == [2, 1, 1, 0]
    assert sections() == [11, 11, 10, 0]                      # transformers' default, not [0, 0, 0, 0]
    with pytest.raises(G.GGUFError, match="mrope_section"):
        sections(mrope_section=[0, 0, 0])


def test_reference_kvs_compare_runtime_keys_by_key():
    from openbonsai.adapters.gguf_common import sha_items

    def h(key, typ, value):
        if typ.startswith("arr["):
            return {"key": key, "type": typ, "len": len(value), "sha256": sha_items(value)}
        return {"key": key, "type": typ, "value": value}

    ref = {"kv": [h("general.architecture", "str", "qwen35"), h("general.name", "str", "Official"),
                  h("qwen35.block_count", "u32", 32), h("prism.hadamard.block_size", "u32", 4096),
                  h("prism.hadamard.sign_values", "arr[i32]", [1, -1]),
                  h("tokenizer.ggml.add_bos_token", "bool", False), h("general.file_type", "u32", 143)]}
    ours = [("general.architecture", "str", "qwen35"), ("general.name", "str", "Mine"),
            ("qwen35.block_count", "u32", 32), ("prism.hadamard.block_size", "u32", 4096),
            ("prism.hadamard.sign_values", "arr[i32]", [-1, 1]), ("general.file_type", "u32", 143),
            ("tokenizer.ggml.add_eos_token", "bool", True), ("tokenizer.ggml.add_bos_token", "bool", True)]
    hard, soft = E.reference_kv_diff(ours, ref, "qwen35")     # a fine-tune: other tokenizer flags, KV order
    assert hard == [] and soft == ["general.name", "tokenizer.ggml.add_bos_token",
                                   "tokenizer.ggml.add_eos_token not in the reference"]
    same_keys = [kv for kv in ours if not kv[0].startswith(("general.name", "tokenizer."))]
    same_keys += [("general.name", "str", "Official"), ("tokenizer.ggml.add_bos_token", "bool", False)]
    assert E.reference_kv_diff(same_keys, ref, "qwen35") == ([], ["KV order"])
    hard, _ = E.reference_kv_diff([kv for kv in ours if kv[0] not in ("general.file_type", "prism.hadamard.block_size")]
                                  + [("qwen35.extra", "u32", 1), ("prism.hadamard.block_size", "u32", 128)],
                                  ref, "qwen35")
    assert "general.file_type missing" in hard and "prism.hadamard.block_size (128 vs 4096)" in hard \
        and "qwen35.extra not in the reference" in hard


def test_chat_templates_in_the_list_form_are_written_like_gguf_py():
    from openbonsai.adapters.gguf_common import chat_template_kvs
    assert chat_template_kvs(None) == [] and chat_template_kvs("t") == [("tokenizer.chat_template", "str", "t")]
    assert chat_template_kvs([{"name": "tool-use", "template": "T"}, {"name": "default", "template": "D"},
                              {"name": "", "template": "skipped"}, {"name": "rag", "template": "R"}]) == [
        ("tokenizer.chat_template.tool_use", "str", "T"), ("tokenizer.chat_template.rag", "str", "R"),
        ("tokenizer.chat_templates", "arr[str]", ["tool_use", "rag"]), ("tokenizer.chat_template", "str", "D")]
    with pytest.raises(G.GGUFError, match="chat_template"):
        chat_template_kvs({"default": "D"})


SNAPSHOT_9B = Path(os.environ.get("OPENBONSAI_PARITY_SNAPSHOT", "D:/openbonsai/runs/hf-cache/hub/"
                                  "models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"))


@pytest.mark.skipif(not (SNAPSHOT_9B / "tokenizer.json").exists(), reason="the Qwen3.5-9B snapshot is not here")
def test_a_9b_fine_tune_with_its_own_tokenizer_settings_keeps_the_reference_layout(tmp_path):
    """A 9B config is held to the official runtime KVs, not to the official tokenizer KV list."""
    for f in ("tokenizer.json", "config.json", "chat_template.jinja"):
        shutil.copy(SNAPSHOT_9B / f, tmp_path / f)
    tcfg = json.loads((SNAPSHOT_9B / "tokenizer_config.json").read_text(encoding="utf-8"))
    tcfg.update(bos_token="<|endoftext|>", add_eos_token=False)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tcfg), encoding="utf-8")
    adapter = adapter_for_config(_qwen35_9b_text_config())
    plan = adapter.gguf_tensors()
    signs = {w: torch.ones(w, dtype=torch.int8) for w in (4096, 12288)}
    kvs, _ = adapter.gguf_kvs(tmp_path, name="mine", signs=signs, block=4096, tensors=plan)
    hard, soft = E.reference_kv_diff(kvs, adapter.gguf_reference(), "qwen35")
    assert hard == [] and "tokenizer.ggml.add_eos_token not in the reference" in soft


def test_unsupported_layer_schedules_are_refused():
    with pytest.raises(G.GGUFError, match="every k-th"):
        adapter_for_config(tiny_text_config(layer_types=["full_attention"] + ["linear_attention"] * 2
                                            + ["full_attention"])).gguf_tensors()

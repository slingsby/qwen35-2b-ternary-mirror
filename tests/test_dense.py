"""The dense adapter (Llama, Mistral, Qwen2, Qwen3) and every registered adapter's plan.

- Conformance: for every registered model type, every Linear of every layer is
  planned or kept, and geometry derives from the plan.
- The family table, the tied-table rules and the GGUF layout rules (llama q/k
  rope permutation as Prism's converter writes it, llama3 ``rope_freqs``, the
  refusals that send a model to the latent export).
- ``openbonsai run`` end to end on tiny random models: Llama (untied, biases,
  llama3 rope) -> PTQ1_0 GGUF; Qwen3 (tied) -> GGUF with a BF16 shared table;
  Qwen2 -> latent checkpoint (Prism refuses qwen2); Qwen3.5 regression.
- Optional: Prism's own runtime scores the exported GGUFs like the torch model
  (set OPENBONSAI_PRISM_BIN to a Prism llama.cpp build's bin directory).
"""
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest
import torch
from torch import nn

from openbonsai import cli
from openbonsai.adapters import (FAMILIES, DenseAdapter, Qwen35Adapter, adapter_for_config, load_adapter,
                                 registered_adapters, supported_model_types)
from openbonsai.adapters.dense import PLAN, llama3_rope_factors, llama_rope_perm
from openbonsai.config import ConfigError, load_config
from openbonsai.core.geometry import block_note, derive_geometry
from openbonsai.formats.gguf import GGML_BF16, GGML_PTQ1_0, GGUFError, GGUFReader
from openbonsai.formats.state import load_state
from openbonsai.rundir import RunDir
from openbonsai.stages.common import StageContext, apply_dense
from openbonsai.stages.endpoints import ternary_endpoints
from openbonsai.stages.export import resolve_format
from tiny_models import save_tiny_dense, tiny_dense, tiny_dense_config, tiny_qwen3_5, tiny_text_config

ROOT = Path(__file__).resolve().parents[1]

# One tiny config per registered model type (a new registration must add one here).
CONFIGS = {
    "qwen3_5": lambda: tiny_qwen3_5(multimodal=True).config,
    "qwen3_5_text": lambda: tiny_text_config(),
    "llama": lambda: tiny_dense_config("llama"),
    "mistral": lambda: tiny_dense_config("mistral"),
    "qwen2": lambda: tiny_dense_config("qwen2"),
    "qwen3": lambda: tiny_dense_config("qwen3", tied=True),
}


def test_every_registered_model_type_has_a_conformance_config():
    assert set(CONFIGS) == set(supported_model_types())
    assert {c.name for c in registered_adapters()} == {"qwen3_5", "dense"}
    assert set(FAMILIES) == set(DenseAdapter.model_types)


@pytest.mark.parametrize("model_type", sorted(CONFIGS))
def test_every_linear_of_every_registered_adapter_is_planned_or_kept(model_type):
    config = CONFIGS[model_type]()
    adapter = adapter_for_config(config)
    model = adapter.empty_model(attn="eager")
    for i in range(adapter.n_layers):
        plan = adapter.layer_plan(i)
        planned = adapter.planned_linears(model, i)          # PlanError if a Linear is unplanned or missing
        linears = {n for n, m in adapter.layers(model)[i].named_modules() if isinstance(m, nn.Linear)}
        assert linears == set(plan.quantized) | set(plan.keep) and list(planned) == list(plan.quantized)
        assert not set(plan.quantized) & set(plan.keep) and len(plan.groups) == 4
    outside = [n for n, m in adapter.text_model(model).named_modules()
               if isinstance(m, nn.Linear) and not n.startswith("layers.")]
    assert outside == []                                        # only the head is left, and it is an endpoint
    assert isinstance(adapter.head(model), nn.Linear) and isinstance(adapter.embedding(model), nn.Embedding)
    geo = derive_geometry(adapter.quantized_widths(model), adapter.hidden_size)
    assert geo.block == 128 and set(geo.widths) <= {128, 256}
    if isinstance(adapter, DenseAdapter):
        assert all(adapter.layer_plan(i).groups == PLAN and adapter.layer_plan(i).keep == ()
                   for i in range(adapter.n_layers))
        biased = {n for n, m in adapter.layers(model)[0].named_modules()
                  if isinstance(m, nn.Linear) and m.bias is not None}
        assert biased == set(adapter.bias_modules())
    else:
        assert isinstance(adapter, Qwen35Adapter)


def test_family_table():
    assert {m: (f.gguf_arch, f.prism_hadamard) for m, f in FAMILIES.items()} == {
        "llama": ("llama", True), "mistral": ("llama", True), "qwen2": ("qwen2", False), "qwen3": ("qwen3", True)}
    assert not any(f.zero_centered_norm for f in FAMILIES.values())
    llama = adapter_for_config(tiny_dense_config("llama"))
    assert llama.add_bos and llama.chat_kwargs(thinking=False) == {} and llama.think_end is None
    with pytest.raises(ValueError, match="thinking"):
        llama.chat_kwargs(thinking=True)
    qwen3 = adapter_for_config(tiny_dense_config("qwen3", tied=True))
    assert qwen3.tied and qwen3.bos_id(None) is None and qwen3.think_end == "</think>"
    assert qwen3.chat_kwargs(thinking=True) == {"enable_thinking": True}
    assert adapter_for_config(tiny_dense_config("qwen2")).gguf_arch is None
    model = tiny_dense("llama")
    adapter = adapter_for_config(model.config)
    assert torch.equal(adapter.embedding_gain(model), model.model.layers[0].input_layernorm.weight.float())


def test_tied_tables_stay_high_precision_and_ternary_is_refused():
    tied = adapter_for_config(tiny_dense_config("qwen3", tied=True))
    untied = adapter_for_config(tiny_dense_config("llama"))
    cfg = load_config()
    assert cfg["endpoints"]["head"] == cfg["endpoints"]["embedding"] == "auto"
    assert ternary_endpoints(cfg, tied) == (False, False) and ternary_endpoints(cfg, untied) == (True, True)
    for key in ("head", "embedding"):
        with pytest.raises(NotImplementedError, match="joint GPTQ"):
            ternary_endpoints(load_config(None, [f"endpoints.{key}=ternary"]), tied)
    with pytest.raises(ConfigError, match="endpoints.head"):
        ternary_endpoints(load_config(None, ["endpoints.head=int4"]), untied)
    plan = {t.name: t for t in tied.gguf_tensors()}
    assert "output.weight" not in plan and plan["token_embd.weight"].gtype == GGML_BF16
    assert plan["token_embd.weight"].role == "" and plan["token_embd.weight"].source == "embed_tokens.weight"
    with pytest.raises(GGUFError, match="tied"):
        tied.gguf_tensors(endpoints=("embed_tokens",))
    hp = {t.name: t for t in untied.gguf_tensors(endpoints=())}           # untied, both kept high precision
    assert hp["output.weight"].gtype == hp["token_embd.weight"].gtype == GGML_BF16




def test_llama_rope_permutation_matches_prisms_converter_and_the_independent_rule():
    def permute(w, n_head, n_head_kv):   # Prism conversion/llama.py LlamaModel.permute, transcribed
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)

    llama = adapter_for_config(tiny_dense_config("llama"))
    plan = {t.name: t for t in llama.gguf_tensors()}
    q, k = torch.randn(128, 128), torch.randn(64, 128)
    assert torch.equal(q[plan["blk.1.attn_q.weight"].perm], permute(q, 4, 4))
    assert torch.equal(k[plan["blk.1.attn_k.weight"].perm], permute(k, 4, 2))
    assert torch.equal(k[:, 0][plan["blk.1.attn_k.bias"].perm], permute(k[:, 0], 4, 2))
    assert plan["blk.1.attn_v.weight"].perm is None and plan["blk.1.attn_output.bias"].perm is None
    for adapter in (llama, adapter_for_config(tiny_dense_config("qwen3", tied=True))):
        for t in adapter.gguf_tensors():
            n = t.ne[1] if t.ternary else t.shape[0]
            want = np.arange(n) if t.perm is None else t.perm
            assert np.array_equal(adapter.gguf_source_rows(t.name, n), want), t.name
    assert all(t.perm is None for t in adapter_for_config(tiny_dense_config("qwen3", tied=True)).gguf_tensors())
    assert np.array_equal(llama_rope_perm(8, 2), [0, 2, 1, 3, 4, 6, 5, 7])


def test_llama3_rope_freqs_reproduce_transformers_rotary():
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    cfg = tiny_dense_config("llama")
    adapter = adapter_for_config(cfg)
    factors = {t.name: t for t in adapter.gguf_tensors()}["rope_freqs.weight"]
    assert factors.op == "const" and factors.ne == (16,) and factors.data is not None
    assert np.array_equal(factors.data, llama3_rope_factors(adapter._rope(), 32))
    plain = 1.0 / (500000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
    torch.testing.assert_close(LlamaRotaryEmbedding(cfg).inv_freq.float(), plain / torch.from_numpy(factors.data),
                               rtol=1e-6, atol=0)
    assert all(t.name != "rope_freqs.weight" for t in adapter_for_config(tiny_dense_config("mistral")).gguf_tensors())


def test_gguf_refusals_name_the_reason_and_auto_falls_back_to_latent(tmp_path):
    qwen2 = adapter_for_config(tiny_dense_config("qwen2"))
    assert "'qwen2'" in qwen2.gguf_unsupported()
    yarn = tiny_dense_config("qwen3", rope_parameters={"rope_type": "yarn", "rope_theta": 1e6, "factor": 4.0,
                                                       "original_max_position_embeddings": 128})
    assert "yarn" in adapter_for_config(yarn).gguf_unsupported()
    assert "o_proj" in adapter_for_config(tiny_dense_config("qwen3", attention_bias=True)).gguf_unsupported()
    sliding = tiny_dense_config("qwen3", use_sliding_window=True, sliding_window=64,
                                layer_types=["full_attention", "sliding_attention"])
    assert "sliding" in adapter_for_config(sliding).gguf_unsupported()
    with pytest.raises(GGUFError, match="qwen2"):
        qwen2.gguf_tensors()
    model_dir = save_tiny_dense(tmp_path / "llama", "llama")
    llama = load_adapter(model_dir)
    assert llama.gguf_unsupported(model_dir) is None
    tj = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tj["model"]["byte_fallback"] = True                          # a SentencePiece-style vocabulary
    (model_dir / "tokenizer.json").write_text(json.dumps(tj), encoding="utf-8")
    assert "byte-level" in llama.gguf_unsupported(model_dir)
    assert resolve_format(load_config(), llama, model_dir)[0] == "latent"
    assert resolve_format(load_config(), llama) == ("ptq1_0-gguf", None)
    assert resolve_format(load_config(), qwen2)[0] == "latent"
    assert resolve_format(load_config(None, ["export.format=latent"]), llama) == ("latent", None)
    with pytest.raises(ConfigError, match="qwen2"):
        resolve_format(load_config(None, ["export.format=ptq1_0-gguf"]), qwen2)
    with pytest.raises(ConfigError, match="export.format"):
        resolve_format(load_config(None, ["export.format=onnx"]), llama)
    assert block_note(4096) is None and "4096" in block_note(128)


# -- end to end -----------------------------------------------------------------

CASES = {   # model type (None: the tiny Qwen3.5), tied, expected export format
    "llama": ("llama", False, "ptq1_0-gguf"),
    "qwen3-tied": ("qwen3", True, "ptq1_0-gguf"),
    "qwen2": ("qwen2", False, "latent"),
    "qwen3_5": (None, False, "ptq1_0-gguf"),
}
ARCH = {"llama": "llama", "qwen3": "qwen3", None: "qwen35"}


def _run_ctx(out) -> StageContext:
    run = RunDir(out)
    from openbonsai.config import DEFAULTS, check_keys, deep_merge
    return StageContext(check_keys(deep_merge(DEFAULTS, run.load_config())), run)


@pytest.mark.parametrize("case", list(CASES))
def test_openbonsai_run_end_to_end(case, tmp_path, tiny_dir, monkeypatch):
    model_type, tied, fmt = CASES[case]
    monkeypatch.chdir(ROOT)
    model = tiny_dir if model_type is None else save_tiny_dense(tmp_path / "model", model_type, tied=tied)
    out = tmp_path / "run"
    t0 = time.time()
    assert cli.main(["run", "--model", str(model), "--config", "examples/tiny-dense.yaml", "--out", str(out)]) == 0
    elapsed = time.time() - t0
    assert elapsed < 60, f"{case}: tiny end-to-end run took {elapsed:.1f}s"
    for f in ("traces/traces.jsonl", "calib/tokens.npy", "quantize/state.safetensors", "endpoints/state.safetensors",
              "distill/scales.safetensors", "eval/distill.compare.json", "eval/distill.teacher.json"):
        assert (out / f).exists(), f
    assert set(load_state(out / "endpoints" / "state.safetensors").tensors) == (
        set() if tied else {"lm_head", "embed_tokens"})
    compare = json.loads((out / "eval" / "distill.compare.json").read_text())
    assert compare["tokens"] == 126 and 0 <= compare["top1_agreement"] <= 1 and np.isfinite(compare["kl_full"])
    done = json.loads((out / "export" / "DONE.json").read_text())["result"]
    assert done["format"] == fmt and done["verified"] is True and done["source"] == "distill"
    if fmt == "ptq1_0-gguf":
        gguf = Path(done["output"])
        with GGUFReader(gguf) as r:
            types = {t.name: t.gtype for t in r.tensors}
            assert r.value("general.architecture") == ARCH[model_type]
            assert ("output.weight" in types) is not tied
            assert types["token_embd.weight"] == (GGML_BF16 if tied else GGML_PTQ1_0)
            assert ("token_embd.weight" in r.value("prism.hadamard.inverse_weight_names", [])) is not tied
        if model_type is not None:   # every weight read back from the file computes what distill computed
            assert cli.main(["eval", "--run", str(out), "--from", str(gguf), "--suite", "compare"]) == 0
            back = json.loads((out / "eval" / f"{gguf.stem}.compare.json").read_text())
            assert (back["nll"], back["kl_full"]) == (compare["nll"], compare["kl_full"])
        return
    folder = Path(done["folder"])
    manifest = json.loads((folder / "openbonsai.json").read_text())
    assert manifest["format"] == "openbonsai-latent" and "'qwen2'" in manifest["why_not_gguf"]
    assert set(manifest["files"]) >= {"model.safetensors", "ternary.safetensors", "config.json", "tokenizer.json"}
    from transformers import AutoModelForCausalLM
    latent = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.bfloat16).eval()
    ctx = _run_ctx(out)
    ref, _ = ctx.load_model()
    apply_dense(ctx.adapter(), ref, ctx.state_through("distill").records())
    ids = ctx.calib_tokens()[:2]
    with torch.no_grad():
        assert torch.equal(latent(input_ids=ids).logits, ref(input_ids=ids, use_cache=False).logits)
    # The exact codes and distilled scales: ternary.safetensors installs what distillation evaluated.
    assert cli.main(["eval", "--run", str(out), "--from", str(folder / "ternary.safetensors"), "--install", "ternary",
                     "--suite", "compare"]) == 0
    back = json.loads((out / "eval" / "ternary.compare.json").read_text())
    assert (back["nll"], back["kl_full"]) == (compare["nll"], compare["kl_full"])
    assert cli.main(["inspect", str(folder)]) == 0


def test_latent_export_of_a_multimodal_checkpoint_keeps_its_vision_tower(tiny_mm_dir, tmp_path, monkeypatch):
    """export.format latent works for any adapter; the folder loads whole (vision tower unquantized)."""
    from safetensors import safe_open
    monkeypatch.chdir(ROOT)
    out = tmp_path / "run"
    assert cli.main(["run", "--model", str(tiny_mm_dir), "--config", "examples/tiny-dense.yaml", "--out", str(out),
                     "--set", "export.format=latent", "--set", "distill.train.small=true", "--until", "export"]) == 0
    done = json.loads((out / "export" / "DONE.json").read_text())["result"]
    assert done["format"] == "latent" and done["why_not_gguf"] is None and done["verified"]
    folder = Path(done["folder"])
    with safe_open(str(folder / "model.safetensors"), "pt") as f:
        keys = list(f.keys())
    assert any(".visual." in k for k in keys) and any("language_model.layers.0.linear_attn.A_log" in k for k in keys)
    manifest = json.loads((folder / "openbonsai.json").read_text())
    assert manifest["model_type"] == "qwen3_5" and manifest["ternary"]["tensors"] == 27
    # The distilled high-precision tensors travel with the state: eval --from ternary.safetensors scores the
    # distilled model, not the original checkpoint's norms and A_log.
    from openbonsai.formats.state import read_small, read_small_masters
    assert manifest["small"]["file"] == "small.safetensors" and manifest["small"]["tensors"] > 0
    assert "small.safetensors" in manifest["files"]
    assert any(c["check"].startswith("small.safetensors") and c["ok"] for c in manifest["verify"])
    distill = out / "distill" / "scales.safetensors"
    masters, dtypes = read_small_masters(folder / "small.safetensors")
    want_masters, want_dtypes = read_small_masters(distill)
    assert dtypes == want_dtypes and set(masters) == set(want_masters)
    assert all(torch.equal(masters[k], want_masters[k]) for k in masters)
    assert all(torch.equal(v, read_small(distill)[k]) for k, v in read_small(folder / "small.safetensors").items())
    assert cli.main(["eval", "--run", str(out), "--from", "distill", "--suite", "compare"]) == 0
    want = json.loads((out / "eval" / "distill.compare.json").read_text())
    assert cli.main(["eval", "--run", str(out), "--from", str(folder / "ternary.safetensors"), "--install", "ternary",
                     "--suite", "compare"]) == 0
    got = json.loads((out / "eval" / "ternary.compare.json").read_text())
    assert (got["nll"], got["kl_full"]) == (want["nll"], want["kl_full"])
    # Without them it would score another model.
    (folder / "openbonsai.json").write_text(json.dumps({**manifest, "small": None}), encoding="utf-8")
    assert cli.main(["eval", "--run", str(out), "--from", str(folder / "ternary.safetensors"), "--install", "ternary",
                     "--suite", "compare"]) == 0
    without = json.loads((out / "eval" / "ternary.compare.json").read_text())
    assert without["nll"] != want["nll"]


def test_latent_export_never_deletes_a_folder_it_did_not_write(tiny_dir, tmp_path, monkeypatch):
    """--out names the checkpoint folder itself; an existing folder is replaced only when it is empty or a
    previous latent checkpoint (``export --out D:/models`` must not delete D:/models)."""
    from openbonsai.stages.export import check_latent_out, latent_out_problem
    monkeypatch.chdir(ROOT)
    out = tmp_path / "run"
    assert cli.main(["run", "--model", str(tiny_dir), "--config", "examples/tiny-dense.yaml", "--out", str(out),
                     "--set", "export.format=latent", "--until", "distill"]) == 0
    models = tmp_path / "models"
    (models / "other-model").mkdir(parents=True)
    (models / "other-model" / "weights.bin").write_bytes(b"precious")
    (models / "notes.txt").write_text("keep me")
    gguf = tmp_path / "old.gguf"
    gguf.write_bytes(b"GGUF old")
    for bad in (models, gguf):
        assert latent_out_problem(bad) is not None
        assert cli.main(["export", "--run", str(out), "--out", str(bad)]) == 2
    assert (models / "other-model" / "weights.bin").read_bytes() == b"precious"
    assert (models / "notes.txt").read_text() == "keep me" and gguf.read_bytes() == b"GGUF old"
    assert not list(tmp_path.glob(".*.tmp")) and not list(tmp_path.glob("*.tmp"))
    # A new folder inside it, then the same folder again (a previous latent checkpoint is replaced).
    target = models / "tiny-latent"
    assert cli.main(["export", "--run", str(out), "--out", str(target)]) == 0
    first = json.loads((target / "openbonsai.json").read_text())
    assert first["format"] == "openbonsai-latent"
    check_latent_out(target)
    assert cli.main(["export", "--run", str(out), "--out", str(target), "--force"]) == 0
    assert json.loads((target / "openbonsai.json").read_text())["files"] == first["files"]
    assert sorted(p.name for p in models.iterdir()) == ["notes.txt", "other-model", "tiny-latent"]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert latent_out_problem(empty) is None
    # The preflight of a run refuses it before any stage runs.
    assert cli.main(["run", "--model", str(tiny_dir), "--config", "examples/tiny-dense.yaml",
                     "--out", str(tmp_path / "run2"), "--set", "export.format=latent",
                     "--set", f"export.out={models.as_posix()}"]) == 2
    assert not (tmp_path / "run2" / "traces").exists()


def test_preflight_checks_the_tokenizer_and_the_thinking_modes(tmp_path, monkeypatch):
    """An explicit ptq1_0-gguf with a SentencePiece-style tokenizer, or thinking on for a family without a
    thinking switch, fails before traces, not after quantize and distill."""
    monkeypatch.chdir(ROOT)
    model_dir = save_tiny_dense(tmp_path / "llama", "llama")
    tj = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tj["model"]["byte_fallback"] = True
    (model_dir / "tokenizer.json").write_text(json.dumps(tj), encoding="utf-8")
    out = tmp_path / "run"
    assert cli.main(["run", "--model", str(model_dir), "--config", "examples/tiny-dense.yaml", "--out", str(out),
                     "--set", "export.format=ptq1_0-gguf"]) == 2
    assert not (out / "traces").exists()
    good = save_tiny_dense(tmp_path / "llama-ok", "llama")
    for sets in (["eval.suites=[compare,gsm8k]", "eval.reasoning.thinking=[off,on]"], ["traces.thinking=true"]):
        args = ["run", "--model", str(good), "--config", "examples/tiny-dense.yaml", "--out", str(tmp_path / "r2")]
        assert cli.main(args + [a for s in sets for a in ("--set", s)]) == 2
        assert not (tmp_path / "r2" / "traces").exists()


# -- Prism's runtime (optional) ---------------------------------------------------

PRISM_BIN = os.environ.get("OPENBONSAI_PRISM_BIN")


@pytest.mark.skipif(not PRISM_BIN, reason="set OPENBONSAI_PRISM_BIN to a Prism llama.cpp build's bin directory")
@pytest.mark.parametrize("model_type,tied", [("llama", False), ("qwen3", True)])
def test_prism_runtime_scores_the_exported_gguf_like_the_torch_model(model_type, tied, tmp_path, monkeypatch):
    """Prism's llama-perplexity log-probs over the whole vocabulary against the FP32 fake-quant torch model
    of the same state. Measured: max |diff| 0.06 (Llama) / 0.015 (Qwen3), from Prism's 8-bit activations;
    an export without the llama q/k permutation (or with it on Qwen3) gives about 1.0 and a mean of 0.15."""
    spec = importlib.util.spec_from_file_location("compare_logits", ROOT / "tools" / "prism" / "compare_logits.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    monkeypatch.chdir(ROOT)
    model = save_tiny_dense(tmp_path / "model", model_type, tied=tied)
    out = tmp_path / "run"
    assert cli.main(["run", "--model", str(model), "--config", "examples/tiny-dense.yaml", "--out", str(out),
                     "--until", "export"]) == 0
    gguf = Path(json.loads((out / "export" / "DONE.json").read_text())["result"]["output"])
    lines = (ROOT / "examples" / "tiny-data" / "heldout.jsonl").read_text(encoding="utf-8").splitlines()
    text = tmp_path / "text.txt"
    text.write_text((" ".join(json.loads(x).get("text", "") for x in lines if x.strip()) * 3)[:3000], encoding="utf-8")
    n_ctx, tokens, lp, valid, first = tool.prism_logprobs(PRISM_BIN, gguf, text, out=tmp_path / "prism.logits")
    ctx = _run_ctx(out)
    adapter = ctx.adapter()
    ref, tok = adapter.load(ctx.model_path(), dtype=torch.float32, device="cpu")
    apply_dense(adapter, ref, ctx.state_through("distill").records())
    diffs = []
    for c in range(len(tokens)):
        ids = torch.from_numpy(tokens[c].astype(np.int64))
        if adapter.add_bos:
            ids[0] = tok.bos_token_id
        with torch.no_grad():
            t = ref(input_ids=ids[None], use_cache=False).logits[0].double().log_softmax(-1).numpy()[first:n_ctx - 1]
        diffs.append(np.abs(t - lp[c])[valid[c]])
    d = np.concatenate(diffs)
    assert d.max() < 0.2 and d.mean() < 0.03, (d.max(), d.mean())

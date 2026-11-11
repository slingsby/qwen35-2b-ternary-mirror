"""Torch suites agree with the frozen-manifest contract; assets are pinned; eval stage sources."""
import json

import pytest
import torch

from openbonsai.eval import suites as S
from openbonsai.eval.backends import TransformersBackend
from openbonsai.eval.manifest import evaluate_manifest, freeze_manifest, freeze_text, sliding_windows

IDENTITY = {"kind": "tiny-byte-fixture", "sha256": "0" * 64}


def frozen_manifest(tok, backend):
    def text(key, s):
        ids = tok(s, add_special_tokens=False)["input_ids"]
        return freeze_text(key, raw_text=s, rendered_text=s, input_ids=ids, token_pieces=backend.token_pieces(ids))
    doc = text("doc", "The river turned west below the old mill, where the valley widened into meadows.")
    q = "Question: which bakes bread? Answer:"
    opts = [text(f"opt-{k}", q + c) for k, c in (("A", " A"), ("B", " B"), ("C", " C"))]
    prefix = len(tok(q, add_special_tokens=False)["input_ids"])
    windows = sliding_windows(doc["input_ids"], context_length=32, stride=16)
    return freeze_manifest(
        source_provenance={"fixture": "tiny"}, tokenizer_identity=IDENTITY,
        texts=[doc, *opts],
        nll=[{"id": f"w{k}", "text_id": "doc", **w} for k, w in enumerate(windows)],
        choices=[{"id": "q", "answer_id": "B", "options": [
            {"id": k, "text_id": o["id"], "input_ids": o["input_ids"], "prefix_length": prefix,
             "score_positions": list(range(prefix, len(o["input_ids"])))} for k, o in zip("ABC", opts)]}])


def test_dev_scorer_matches_the_manifest_contract(tiny_dir):
    from openbonsai.adapters import load_adapter
    adapter = load_adapter(tiny_dir)
    model, tok = adapter.load(tiny_dir, device="cpu", dtype=torch.float32, attn="eager")
    backend = TransformersBackend(model, tok, context_length=64, source_identity=IDENTITY,
                                  tokenizer_identity=IDENTITY)
    manifest = frozen_manifest(tok, backend)
    fast = S.dev(model, tok, manifest, generations=False, check=True)
    report = evaluate_manifest(backend, manifest)
    assert fast["nll"] == pytest.approx(report["aggregate"]["mean_nll"], abs=1e-9)
    assert fast["scored_tokens"] == report["aggregate"]["scored_tokens"]
    assert fast["arc_preds"] == report["choices"][0]["selected_id"]
    assert fast["arc_correct"] == report["aggregate"]["choice_correct"]


def test_compare_against_itself_is_exact_and_suites_refuse_other_tokenizers(tiny_dir):
    from openbonsai.adapters import load_adapter
    adapter = load_adapter(tiny_dir)
    model, tok = adapter.load(tiny_dir, device="cpu", dtype=torch.bfloat16)
    blocks = torch.randint(0, 256, (2, 24), generator=torch.Generator().manual_seed(1))
    ref = S.compare_reference(model, blocks, topk=8)
    r = S.compare(model, blocks, ref)
    assert r["tokens"] == 46 and r["top1_agreement"] == 1.0
    assert abs(r["delta_nll"]) < 1e-12 and abs(r["kl_top8"]) < 1e-9
    with pytest.raises(S.SuiteNotApplicable):
        S.dev(model, tok)
    with pytest.raises(S.SuiteNotApplicable):
        S.extended(model, tok)


def test_extended_assets_are_pinned():
    legacy, chat = S.load_extended_inputs()
    assert (len(legacy["nll"]), len(legacy["questions"]), len(chat)) == (32, 100, 16)
    assert legacy["model"] == {"id": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}
    assert len(S.LEGACY_INPUTS.read_bytes()) == 689368 and len(S.CHAT_TRACES.read_bytes()) == 77139
    assert S.DEV_MANIFEST.parent == S.LEGACY_INPUTS.parent == S.ASSETS


def test_eval_stage_sources_and_legacy_states(calibrated, tmp_path):
    from openbonsai.cli import execute
    ctx = calibrated
    for stage in ("quantize", "endpoints"):
        execute(ctx, stage)
    execute(ctx, "eval", source="quantize", suites=["compare", "dev"])
    out = ctx.run.stage_dir("eval")
    q = json.loads((out / "quantize.compare.json").read_text())
    assert q["source"] == "quantize" and q["kl_top16"] > 0 and not (out / "quantize.dev.json").exists()
    # The same decoder state written in the research .pt layout evaluates identically.
    state = ctx.quantize_state()
    modules = {k: {"codes": t.codes, "scales": t.scales, "signs": s, "block": b, "seed": t.codes.shape[1]}
               for k, t, s, b in state.records()}
    legacy = tmp_path / "ternary_state.pt"
    torch.save({"args": {}, "modules": modules}, legacy)
    execute(ctx, "eval", source=str(legacy), suites=["compare"])
    r = json.loads((out / "ternary_state.compare.json").read_text())
    assert r["nll"] == q["nll"] and r["kl_top16"] == q["kl_top16"]
    execute(ctx, "eval", source="bf16", suites=["compare"])
    b = json.loads((out / "bf16.compare.json").read_text())
    assert b["kl_top16"] < 1e-9 and b["top1_agreement"] == 1.0
    execute(ctx, "eval", source="endpoints", suites=["compare"])
    e = json.loads((out / "endpoints.compare.json").read_text())
    assert e["kl_top16"] > q["kl_top16"] * 0.5

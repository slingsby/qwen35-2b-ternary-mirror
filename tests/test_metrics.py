"""Expected acceptance rate and KL against the BF16 reference: full vocabulary and top-k bounds."""
import copy
import json

import numpy as np
import pytest
import torch

from openbonsai.eval import metrics as M
from openbonsai.eval import suites as S


def random_logits(n, v, seed, scale=3.0):
    return torch.randn(n, v, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) * scale


def test_full_vocab_metrics_match_their_definitions():
    p_logits, q_logits = random_logits(40, 97, 0), random_logits(40, 97, 1)
    lq = q_logits.log_softmax(-1)
    ear, kl = M.full_vocab(p_logits, lq)
    p, q = p_logits.softmax(-1), q_logits.softmax(-1)
    assert torch.allclose(ear, torch.minimum(p, q).sum(-1), rtol=0, atol=1e-15)
    assert torch.allclose(ear, 1 - 0.5 * (p - q).abs().sum(-1), rtol=0, atol=1e-14)   # 1 - total variation
    assert torch.allclose(kl, (p * (p.log() - q.log())).sum(-1), rtol=0, atol=1e-13)
    assert ((ear >= 0) & (ear <= 1)).all() and (kl > 0).all()
    same_ear, same_kl = M.full_vocab(p_logits, p_logits.log_softmax(-1))
    assert torch.allclose(same_ear, torch.ones_like(same_ear), rtol=0, atol=1e-14) and (same_kl == 0).all()
    # A token the teacher gives probability 0 (logit -inf) adds nothing, not NaN.
    masked = p_logits.clone()
    masked[:, 5] = float("-inf")
    assert torch.isfinite(M.full_vocab(masked, lq)[1]).all()


@pytest.mark.parametrize("k", [1, 8, 96, 97])
def test_topk_bounds_bracket_the_full_vocabulary_metrics(k):
    p_logits, q_logits = random_logits(64, 97, 2), random_logits(64, 97, 3, scale=2.0)
    lp, lq = p_logits.log_softmax(-1), q_logits.log_softmax(-1)
    tlogp, tidx = lp.topk(k, -1)
    ear, kl = M.full_vocab(p_logits, lq)
    lower, upper = M.topk_ear(tidx, tlogp, lq)
    kl_k = M.topk_kl(tidx, tlogp, lq)
    eps = 1e-14
    assert (lower <= ear + eps).all() and (ear <= upper + eps).all() and (upper <= 1 + eps).all()
    assert (kl_k >= -eps).all() and (kl_k <= kl + eps).all()
    if k == 97:   # the whole vocabulary: both bounds are exact
        assert torch.allclose(lower, ear, rtol=0, atol=1e-14) and torch.allclose(upper, ear, rtol=0, atol=1e-14)


def perturbed(model, rel=0.05, seed=0):
    """A copy of ``model`` with every matrix moved by ``rel`` of its RMS (a stand-in for a compressed model)."""
    out = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in out.parameters():
            if p.dim() == 2:
                p.add_(torch.randn(p.shape, generator=g, dtype=p.dtype) * rel * p.float().pow(2).mean().sqrt())
    return out.eval()


@pytest.fixture(scope="module")
def tiny_pair(tiny_dir):
    from openbonsai.adapters import load_adapter
    model, _ = load_adapter(tiny_dir).load(tiny_dir, device="cpu", dtype=torch.float32, attn="eager")
    model.eval()
    blocks = torch.randint(0, 256, (3, 40), generator=torch.Generator().manual_seed(4))
    logits = {}
    ref = S.compare_reference(model, blocks, topk=8, keep_logits=logits.__setitem__)
    return model, perturbed(model), blocks, ref, logits


def test_compare_identical_models_is_exact(tiny_pair):
    model, _, blocks, ref, logits = tiny_pair
    r = S.compare(model, blocks, ref, ref_logits=logits.__getitem__)
    assert r["tokens"] == 3 * 39 and r["full_vocab"] is True and r["topk"] == 8 and r["vocab"] == 320
    assert abs(r["ear_full"] - 1) < 1e-12 and r["kl_full"] == 0.0
    assert r["top1_agreement"] == 1.0 and r["delta_nll"] == 0.0 and abs(r["kl_top8"]) < 1e-12
    # Against itself the top-8 lower bound is the teacher's top-8 mass; the upper bound adds the whole tail.
    assert 0 < r["ear_top8_lb"] < r["ear_full"] and abs(r["ear_top8_ub"] - 1) < 1e-12


def test_compare_full_vocabulary_metrics_on_a_perturbed_model(tiny_pair):
    model, student, blocks, ref, logits = tiny_pair
    r = S.compare(student, blocks, ref, ref_logits=logits.__getitem__)
    assert 0 <= r["ear_top8_lb"] <= r["ear_full"] <= r["ear_top8_ub"] <= 1
    assert r["ear_full"] < 1 and r["kl_full"] > 0 and 0 <= r["kl_top8"] <= r["kl_full"]
    # Independent arithmetic over whole blocks (no chunks, no top-k).
    with torch.no_grad():
        ears, kls = [], []
        for row in blocks:
            p = model(input_ids=row[None]).logits[0, :-1].double().softmax(-1)
            q = student(input_ids=row[None]).logits[0, :-1].double().softmax(-1)
            ears.append(torch.minimum(p, q).sum(-1))
            kls.append((p * (p.log() - q.log())).sum(-1))
    assert r["ear_full"] == pytest.approx(float(torch.cat(ears).mean()), rel=0, abs=1e-12)
    assert r["kl_full"] == pytest.approx(float(torch.cat(kls).mean()), rel=1e-10, abs=0)
    # Chunking positions only bounds memory.
    for chunk in (1, 7, 1000):
        c = S.compare(student, blocks, ref, ref_logits=logits.__getitem__, chunk=chunk)
        for key in ("ear_full", "kl_full", "ear_top8_lb", "ear_top8_ub", "kl_top8", "nll", "top1_agreement"):
            assert c[key] == pytest.approx(r[key], rel=1e-12, abs=1e-15), (chunk, key)
    # Without the reference logits only the top-k metrics are reported, unchanged.
    t = S.compare(student, blocks, ref)
    assert t["full_vocab"] is False and "ear_full" not in t and "kl_full" not in t
    assert all(t[key] == r[key] for key in ("ear_top8_lb", "ear_top8_ub", "kl_top8", "nll", "top1_agreement"))
    with pytest.raises(ValueError, match="shape"):
        S.compare(student, blocks, ref, ref_logits=lambda b: logits[b][:-1])


def test_teacher_suite_uses_only_stored_topk(tiny_pair, tmp_path):
    from openbonsai.stages.distill import read_teacher_rows, teacher_topk
    model, student, blocks, ref, _ = tiny_pair
    # Same top-k metrics as compare when given compare's own FP64 reference (the last position is unscored).
    pad = lambda a: torch.cat([a, a[:, -1:]], 1)   # noqa: E731
    t = S.teacher(student, blocks, pad(ref["idx"]), pad(ref["logp"]))
    c = S.compare(student, blocks, ref)
    assert t["tokens"] == c["tokens"] and t["full_vocab"] is False and "ear_full" not in t
    for key in ("nll", "top1_agreement", "ear_top8_lb", "ear_top8_ub", "kl_top8"):
        assert t[key] == pytest.approx(c[key], rel=1e-12, abs=1e-15), key
    # Distillation's fp16 shards, read back for a subset of rows.
    teacher_topk(model, blocks, tmp_path, topk=8, shard=2, device="cpu", log=lambda *_: None)
    np.save(tmp_path / "input_ids.npy", blocks.numpy().astype(np.int32))
    ids, idx, logp = read_teacher_rows(tmp_path, 3)
    assert ids.shape == (3, 40) and idx.shape == logp.shape == (3, 40, 8) and logp.dtype == np.float16
    assert read_teacher_rows(tmp_path, 1)[1].shape == (1, 40, 8)
    same = S.teacher(model, ids, idx, logp)
    assert same["top1_agreement"] == 1.0 and abs(same["kl_top8"]) < 1e-3   # fp16 rounding of the stored p
    other = S.teacher(student, ids, idx, logp)
    assert 0 <= other["ear_top8_lb"] <= other["ear_top8_ub"] <= 1 + 1e-6 and other["kl_top8"] > same["kl_top8"]
    assert other["ear_top8_lb"] == pytest.approx(c["ear_top8_lb"], abs=2e-3)


def test_eval_stage_full_and_topk_only(calibrated):
    from openbonsai.cli import execute
    ctx = calibrated
    for stage in ("quantize", "endpoints", "distill"):
        execute(ctx, stage)
    out = ctx.run.stage_dir("eval")
    execute(ctx, "eval", source="bf16", suites=["compare", "teacher"])
    b = json.loads((out / "bf16.compare.json").read_text())
    assert abs(b["ear_full"] - 1) < 1e-12 and b["kl_full"] == 0.0 and b["full_vocab"] is True
    (logits_dir,) = out.glob("compare-reference-*.logits")
    assert sorted(f.name for f in logits_dir.iterdir()) == ["block_0000.pt", "block_0001.pt"]
    bt = json.loads((out / "bf16.teacher.json").read_text())
    assert bt["rows"] == 4 and bt["top1_agreement"] == 1.0 and abs(bt["kl_top16"]) < 1e-3
    execute(ctx, "eval", source="quantize", suites=["compare", "teacher"])
    q = json.loads((out / "quantize.compare.json").read_text())
    assert 0 <= q["ear_top16_lb"] <= q["ear_full"] <= q["ear_top16_ub"] <= 1 and q["ear_full"] < 1
    assert 0 < q["kl_top16"] <= q["kl_full"]
    qt = json.loads((out / "quantize.teacher.json").read_text())
    assert 0 <= qt["ear_top16_lb"] <= qt["ear_top16_ub"] <= 1 + 1e-6 and qt["kl_top16"] > 0
    # eval.compare.full: false reports only the top-k metrics, from the same cached reference.
    ctx.cfg["eval"]["compare"]["full"] = False
    ctx.cfg["eval"]["teacher"]["rows"] = 2
    execute(ctx, "eval", source="quantize", suites=["compare", "teacher"])
    k = json.loads((out / "quantize.compare.json").read_text())
    assert "ear_full" not in k and k["full_vocab"] is False
    assert all(k[key] == q[key] for key in ("nll", "top1_agreement", "kl_top16", "ear_top16_lb", "ear_top16_ub"))
    assert json.loads((out / "quantize.teacher.json").read_text())["rows"] == 2


def test_teacher_suite_without_shards_is_skipped(make_ctx, tmp_path):
    from openbonsai.cli import execute
    ctx = make_ctx(tmp_path / "bare")
    execute(ctx, "eval", source="bf16", suites=["teacher"])
    assert not (ctx.run.stage_dir("eval") / "bf16.teacher.json").exists()

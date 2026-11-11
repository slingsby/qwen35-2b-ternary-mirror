"""Evaluation suites scored in-process with torch.

- ``dev``: the frozen development manifest (assets/bonsai-dev/frozen-inputs.json):
  4 x 513-token WikiText-2 test NLL windows (2048 scored targets), 20 ARC-Easy
  letter choices, 4 greedy generations. FP32 logits, FP64 log-softmax.
  Reference points: BF16 Qwen3.5-9B 2.42, official Ternary Bonsai 3.192 (16/20).
- ``extended``: 16,384 WikiText-2 test tokens, 100 ARC-Easy letters and NLL on
  16 self-generated chat traces disjoint from calibration
  (assets/bonsai-dev/legacy-eval-inputs.json, chat-traces16.jsonl).
- ``compare``: the model against its own BF16 original on held-out blocks:
  NLL of both, top-1 agreement, the full-vocabulary expected acceptance rate
  ``ear_full`` and ``kl_full`` (when the reference's logits are kept), and
  the top-k ``ear_top{k}_lb``/``_ub`` and ``kl_top{k}`` (eval.metrics). Works
  for any model and tokenizer (use it for your fine-tune).
- ``teacher``: the model against stored teacher top-k outputs (distillation's
  teacher shards): top-k metrics only, no BF16 model needed. The shards are
  distillation's training data, so this is in-sample for a distilled model.
- ``gsm8k``, ``humaneval``: greedy chat generation on frozen TEST subsets
  (100 GSM8K, 50 HumanEval), thinking off and/or on; exact match on the final
  number and pass@1 (eval.reasoning, eval.scoring).

``dev`` and ``extended`` are tokenized for Qwen/Qwen3.5-9B and refuse other
tokenizers. Their arithmetic is OpenBonsai research/experiments/quick_eval.py's
``evaluate`` and ``extended_eval``; every input file is SHA256-pinned.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from . import metrics as M
from .manifest import load_manifest
from .metrics import topk_kl

ASSETS = Path(__file__).resolve().parent / "assets" / "bonsai-dev"
DEV_MANIFEST = ASSETS / "frozen-inputs.json"
DEV_SHA256 = "9936a5382e9cc9921ac21ccae0ceb84fab8a9a7681d46fd0f3efbbc05e6b70be"
LEGACY_INPUTS = ASSETS / "legacy-eval-inputs.json"
LEGACY_SHA256 = "2952cf072dff2e9b00b09a59a5f2345a02b3e509725854af07ea8478f8a7a64c"
CHAT_TRACES = ASSETS / "chat-traces16.jsonl"
CHAT_SHA256 = "443aaba5e5b41c641c27cdcd45f34fd5a2e770fa466321c9bbab84436ff40aaf"
SUITES = ("dev", "extended", "compare", "teacher", "gsm8k", "humaneval")   # the last two: eval.reasoning


class SuiteNotApplicable(ValueError):
    """The suite's frozen token IDs do not belong to this model's tokenizer."""


def _pinned_bytes(path: Path, sha: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError(f"{path.name}: SHA256 differs from the pinned asset")
    return data


def load_dev_manifest() -> dict:
    return load_manifest(DEV_MANIFEST, expected_file_sha256=DEV_SHA256)


def load_extended_inputs() -> tuple[dict, list[dict]]:
    legacy = json.loads(_pinned_bytes(LEGACY_INPUTS, LEGACY_SHA256).decode("utf-8"))
    chat = [json.loads(line) for line in _pinned_bytes(CHAT_TRACES, CHAT_SHA256).decode("utf-8").splitlines()
            if line.strip()]
    return legacy, chat


def vocab_size(model) -> int:
    return model.get_output_embeddings().out_features if hasattr(model.get_output_embeddings(), "out_features") \
        else model.config.get_text_config().vocab_size


def check_tokenizer(tokenizer, manifest: dict, model=None) -> None:
    """Refuse a model/tokenizer that cannot reproduce the manifest's frozen IDs."""
    max_id = max(max(t["input_ids"]) for t in manifest["texts"])
    if model is not None and max_id >= vocab_size(model):
        raise SuiteNotApplicable(f"manifest token id {max_id} exceeds the model vocabulary; "
                                 f"it is tokenized for {manifest['tokenizer_identity'].get('id')}")
    for text in manifest["texts"][:4]:
        rendered = bytes.fromhex(text["rendered_utf8_hex"]).decode("utf-8")
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if list(ids) != text["input_ids"]:
            raise SuiteNotApplicable("this tokenizer does not reproduce the frozen token IDs; "
                                     f"the suite is tokenized for {manifest['tokenizer_identity'].get('id')}")


@torch.no_grad()
def dev(model, tokenizer, manifest: dict | None = None, *, generations: bool = True, check: bool = True) -> dict:
    manifest = load_dev_manifest() if manifest is None else manifest
    if check:
        check_tokenizer(tokenizer, manifest, model)
    dev_ = next(model.parameters()).device
    nll_sum, nll_n = 0.0, 0
    for rec in manifest["nll"]:
        ids = torch.tensor([rec["input_ids"]], device=dev_)
        pos = torch.tensor(rec["score_positions"], device=dev_)
        logits = model(input_ids=ids, use_cache=False).logits[0, pos - 1].float().double()
        tgt = ids[0, pos]
        lp = logits.log_softmax(-1).gather(-1, tgt[:, None])[:, 0]
        nll_sum += float(-lp.sum())
        nll_n += tgt.numel()
    correct, preds = 0, []
    for q in manifest["choices"]:
        scores = {}
        for opt in q["options"]:
            ids = torch.tensor([opt["input_ids"]], device=dev_)
            logits = model(input_ids=ids, use_cache=False).logits[0].float().double()
            s = 0.0
            for p in opt["score_positions"]:
                s += float(logits[p - 1].log_softmax(-1)[opt["input_ids"][p]])
            scores[opt["id"]] = s
        pred = max(scores, key=scores.get)
        preds.append(pred)
        correct += pred == q.get("answer_id")
    gens = []
    if generations:
        from ..stages.traces import generate_encoded     # plain greedy: no generation_config.json options
        for g in manifest["generations"]:
            (gen, _), = generate_encoded(model, [list(g["input_ids"])], eos_ids=list(g["eos_ids"]),
                                         max_new_tokens=int(g["max_new_tokens"]), device=dev_, pad_id=g["eos_ids"][0])
            gens.append(tokenizer.decode(gen, skip_special_tokens=False))
    nll = nll_sum / nll_n
    return {"suite": "dev", "nll": nll, "ppl": math.exp(nll), "scored_tokens": nll_n,
            "arc_correct": correct, "arc_total": len(manifest["choices"]), "arc_preds": "".join(preds),
            "manifest_frozen_sha256": manifest["frozen_sha256"], "generations": gens}


@torch.no_grad()
def extended(model, tokenizer=None, *, check: bool = True) -> dict:
    legacy, chat = load_extended_inputs()
    if check:
        if max(max(r["input_ids"]) for r in legacy["nll"]) >= vocab_size(model):
            raise SuiteNotApplicable("extended suite is tokenized for Qwen/Qwen3.5-9B")
        if tokenizer is not None:   # same tokenizer as the dev manifest
            check_tokenizer(tokenizer, load_dev_manifest(), model)
    dev_ = next(model.parameters()).device

    def nll_of(ids, mask):
        ids_t = torch.tensor([ids], device=dev_)
        logits = model(input_ids=ids_t, use_cache=False).logits[0, :-1].float().double()
        lp = logits.log_softmax(-1).gather(-1, ids_t[0, 1:, None])[:, 0]
        m = torch.tensor(mask[1:], device=dev_, dtype=torch.bool) & torch.tensor(mask[:-1], device=dev_, dtype=torch.bool)
        return float(-lp[m].sum()), int(m.sum())

    s, n = 0.0, 0
    for rec in legacy["nll"]:
        a, b = nll_of(rec["input_ids"], rec["attention_mask"])
        s, n = s + a, n + b
    wiki = s / n
    correct = 0
    for q in legacy["questions"]:
        logits = model(input_ids=torch.tensor([q["input_ids"]], device=dev_), use_cache=False
                       ).logits[0, -1].float().double().log_softmax(-1)
        scores = {c["label"]: float(logits[c["answer_ids"][0]]) for c in q["choices"]}
        correct += max(scores, key=scores.get) == q["answer"]
    s, n = 0.0, 0
    for r in chat:
        a, b = nll_of(r["input_ids"], r["attention_mask"])
        s, n = s + a, n + b
    return {"suite": "extended", "wiki16k_nll": wiki, "arc100": correct, "chat_nll": s / n}


@torch.no_grad()
def compare_reference(model, blocks: torch.Tensor, *, topk: int = 64, keep_logits=None, chunk: int = M.CHUNK) -> dict:
    """BF16 reference statistics per scored position (targets 1..T-1 of each block).

    ``keep_logits(b, logits)``, when given, receives block ``b``'s raw logits
    [T-1, V] (the model's dtype, on the CPU) for ``compare``'s full-vocabulary
    metrics. Positions are processed ``chunk`` at a time.
    """
    dev_ = next(model.parameters()).device
    nll, top1, idx, logp = [], [], [], []
    for b, row in enumerate(blocks):
        ids = row[None].to(dev_)
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1]
        if keep_logits is not None:
            keep_logits(b, logits.detach().cpu().clone())
        parts = ([], [], [], [])
        for c0 in range(0, logits.shape[0], chunk):
            lp = logits[c0:c0 + chunk].double().log_softmax(-1)
            parts[0].append(-lp.gather(-1, ids[0, 1 + c0:1 + c0 + lp.shape[0], None])[:, 0].cpu())
            parts[1].append(lp.argmax(-1).cpu())
            v, j = lp.topk(topk, -1)
            parts[2].append(j.cpu())
            parts[3].append(v.cpu())
        for out, part in zip((nll, top1, idx, logp), parts):
            out.append(torch.cat(part))
    return {"nll": torch.stack(nll), "top1": torch.stack(top1), "idx": torch.stack(idx), "logp": torch.stack(logp)}


def _summary(acc: dict, k: int) -> dict:
    """Means of the per-position metrics, labelled by where they come from (see eval.metrics)."""
    mean = {key: float(torch.cat(v).mean()) for key, v in acc.items() if v}
    out = {}
    if "ear_full" in mean:
        out.update(ear_full=mean["ear_full"], kl_full=mean["kl_full"])
    out.update({f"ear_top{k}_lb": mean["ear_lb"], f"ear_top{k}_ub": mean["ear_ub"], f"kl_top{k}": mean["kl_topk"]})
    return out


def _topk_metrics(acc: dict, lq, tidx, tlogp) -> None:
    acc["kl_topk"].append(topk_kl(tidx, tlogp, lq).cpu())
    lower, upper = M.topk_ear(tidx, tlogp, lq)
    acc["ear_lb"].append(lower.cpu())
    acc["ear_ub"].append(upper.cpu())


@torch.no_grad()
def compare(model, blocks: torch.Tensor, reference: dict, *, ref_logits=None, chunk: int = M.CHUNK) -> dict:
    """The model against its BF16 original on held-out blocks.

    ``reference``: ``compare_reference``'s statistics; they give NLL, top-1
    agreement and the top-k metrics ``kl_top{k}``, ``ear_top{k}_lb`` and
    ``ear_top{k}_ub``. ``ref_logits(b)``: the reference's raw logits of block
    ``b`` ([T-1, V], as ``keep_logits`` received them); with it the result also
    has the full-vocabulary ``ear_full`` and ``kl_full``. Positions are processed
    ``chunk`` at a time, so memory stays at a few [chunk, V] FP64 slabs.
    """
    dev_ = next(model.parameters()).device
    k = int(reference["idx"].shape[-1])
    acc = {key: [] for key in ("nll", "agree", "kl_topk", "ear_lb", "ear_ub", "ear_full", "kl_full")}
    vocab = None
    for b, row in enumerate(blocks):
        ids = row[None].to(dev_)
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1]
        vocab = int(logits.shape[-1])
        ref = None if ref_logits is None else ref_logits(b)
        if ref is not None and tuple(ref.shape) != tuple(logits.shape):
            raise ValueError(f"reference logits of block {b} have shape {tuple(ref.shape)}; "
                             f"the model gives {tuple(logits.shape)}")
        for c0 in range(0, logits.shape[0], chunk):
            c1 = min(c0 + chunk, logits.shape[0])
            lq = logits[c0:c1].double().log_softmax(-1)
            acc["nll"].append(-lq.gather(-1, ids[0, 1 + c0:1 + c1, None])[:, 0].cpu())
            acc["agree"].append((lq.argmax(-1).cpu() == reference["top1"][b, c0:c1]).double())
            _topk_metrics(acc, lq, reference["idx"][b, c0:c1].to(dev_), reference["logp"][b, c0:c1].to(dev_))
            if ref is not None:
                ear, kl = M.full_vocab(ref[c0:c1].to(dev_), lq)
                acc["ear_full"].append(ear.cpu())
                acc["kl_full"].append(kl.cpu())
    nll = torch.cat(acc["nll"])
    ref_nll = reference["nll"].reshape(-1)
    return {"suite": "compare", "tokens": int(nll.numel()), "nll": float(nll.mean()), "ref_nll": float(ref_nll.mean()),
            "delta_nll": float(nll.mean() - ref_nll.mean()), "top1_agreement": float(torch.cat(acc["agree"]).mean()),
            **_summary(acc, k), "topk": k, "vocab": vocab, "full_vocab": ref_logits is not None}


@torch.no_grad()
def teacher(model, ids, tidx, tlogp, *, chunk: int = M.CHUNK) -> dict:
    """The model against stored teacher top-k outputs (distillation's teacher shards).

    Needs neither the BF16 model nor its full logits, so it reports only the
    top-k metrics (``kl_top{k}``, ``ear_top{k}_lb``/``_ub``), top-1 agreement
    with the teacher (the model's argmax is a token with the stored maximum
    log-probability; torch.topk sorts, and fp16 rounding makes ties) and the
    model's NLL, over positions 0..T-2 of each row (those distillation trains
    on). ``ids``: [N, T]; ``tidx``: [N, T, k]; ``tlogp``: [N, T, k].
    """
    dev_ = next(model.parameters()).device
    ids, tidx, tlogp = (torch.as_tensor(a) for a in (ids, tidx, tlogp))
    vocab = vocab_size(model)
    if int(ids.max()) >= vocab or int(tidx.max()) >= vocab:
        raise SuiteNotApplicable("the teacher shards hold token ids beyond this model's vocabulary")
    k = int(tidx.shape[-1])
    acc = {key: [] for key in ("nll", "agree", "kl_topk", "ear_lb", "ear_ub", "mass")}
    for r in range(ids.shape[0]):
        x = ids[r:r + 1].long().to(dev_)
        logits = model(input_ids=x, use_cache=False).logits[0, :-1]
        for c0 in range(0, logits.shape[0], chunk):
            c1 = min(c0 + chunk, logits.shape[0])
            lq = logits[c0:c1].double().log_softmax(-1)
            ti, tl = tidx[r, c0:c1].long().to(dev_), tlogp[r, c0:c1].to(dev_)
            acc["nll"].append(-lq.gather(-1, x[0, 1 + c0:1 + c1, None])[:, 0].cpu())
            top = (ti == lq.argmax(-1)[:, None]) & (tl == tl[:, :1])   # ties in the stored top-1 all count
            acc["agree"].append(top.any(-1).double().cpu())
            acc["mass"].append(tl.double().exp().sum(-1).cpu())
            _topk_metrics(acc, lq, ti, tl)
    nll = torch.cat(acc["nll"])
    return {"suite": "teacher", "rows": int(ids.shape[0]), "tokens": int(nll.numel()), "nll": float(nll.mean()),
            "top1_agreement": float(torch.cat(acc["agree"]).mean()), **_summary(acc, k),
            f"teacher_top{k}_mass": float(torch.cat(acc["mass"]).mean()), "topk": k, "full_vocab": False}

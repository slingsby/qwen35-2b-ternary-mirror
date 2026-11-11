"""Score a run's model (BF16, or after quantize/endpoints/refine/distill) on the suites.

Sources (``--from``):

- ``bf16``: the original model;
- ``quantize``, ``endpoints``, ``refine``: dequantized weights (C * s) R^T written
  into the model, as the decoder-only and endpoint results were scored
  (``refine``: refine's decoder state plus the endpoints when they are on);
- ``distill``: TernaryLinear/TernaryEmbedding modules with the distilled fp16
  scales (the runtime arithmetic, as e2e_kd.py scored it), plus the distilled
  high-precision tensors when ``distill.train.small``;
- a path: a state ``.safetensors`` or a legacy research ``.pt`` (decoder
  ``ternary_state.pt`` and/or ``endpoint_state.pt``; comma-separated). Legacy
  per-module signs are applied exactly as ``apply_state`` did. A latent
  checkpoint's ``ternary.safetensors`` also applies the distilled
  high-precision tensors its folder lists (``small.safetensors``), so it scores
  the distilled model;
- a Prism PTQ1_0 ``.gguf`` (ours or the official Ternary Bonsai): the whole
  model is built from the file (``adapter.from_gguf``: ternary modules plus its
  F32 tensors); only the config and tokenizer come from ``model.path``. The
  ``compare`` reference still comes from the BF16 model.

``compare`` caches the BF16 reference in ``eval/compare-reference-<key>.pt``
(NLL, top-1, top-k) and, with ``eval.compare.full`` (the default), the
reference's raw logits per block in ``compare-reference-<key>.logits/`` for
the full-vocabulary ``ear_full`` and ``kl_full`` (n x (seq_len - 1) x vocab in
the model's dtype: about 2 GB for Qwen3.5-9B at the defaults). Later
evaluations of other sources reuse both without loading the BF16 model.
``teacher`` scores against the first ``eval.teacher.rows`` rows of this run's
distillation teacher shards (top-k metrics only; no BF16 model).

``gsm8k`` and ``humaneval`` (``eval.reasoning``) generate greedily with the
model's chat template, once per ``thinking`` mode, after the other suites. A
thinking mode the chat template has no switch for, and (for a stage source)
stored calibration or teacher tokens that contain a scored test problem, fail
before the model is loaded. With
``eval.reasoning.dense`` (the default) a ternary model (distill, a GGUF,
``--install ternary``) generates with dense dequantized weights (``densify``;
the same weights ``--install dense`` writes): much faster, with the same
function up to BF16 rounding order. Generations are cached per batch in
``eval/generations/`` under a hash of the weights' identity (stage output
digests or file SHA256s), so an interrupted run resumes and rescoring reuses
them.

A stage source must be enabled, complete and current (same config, inputs and
upstream outputs as when it ran); ``auto`` picks the last such stage. So a
rerun quantize never meets distilled scales trained on the old codes.

Results go to ``eval/<source>.<suite>.json``, and for the reasoning suites to
``eval/<source>.<suite>.think-<off|on>.json`` (counts, hit-cap counts and
per-problem records).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..config import ConfigError, canonical_hash
from ..eval import reasoning as R
from ..eval import suites as S
from ..formats.atomic import atomic_write_json
from ..formats.state import load_legacy_pt, load_state
from . import calibrate
from .common import apply_dense, apply_small, densify, free, has_ternary, install_ternary, legacy_records, vram_cap

STAGE_SOURCES = ("bf16", "quantize", "endpoints", "refine", "distill")


def is_gguf(source) -> bool:
    return isinstance(source, str) and source.lower().endswith(".gguf") and "," not in source


def resolve_source(ctx, source: str | None) -> str:
    if source in (None, "auto"):
        for stage in ("distill", "refine", "endpoints", "quantize"):
            if ctx.stale_reason(stage) is None:
                return stage
        return "bf16"
    if source in STAGE_SOURCES[1:]:
        ctx.require_current(source)
    return source


def install(ctx, model, source: str, mode: str = "auto") -> str:
    """Put the source's weights into ``model``; returns a label for file names."""
    adapter = ctx.adapter()
    if source == "bf16":
        return "bf16"
    if source in STAGE_SOURCES:
        state = ctx.state_through(source)
        ternary = mode == "ternary" or (mode == "auto" and source == "distill")
        if ternary:
            install_ternary(adapter, model, state.records(), ctx.device)
        else:
            apply_dense(adapter, model, state.records())
        if source == "distill":
            apply_small(adapter, model, ctx.distill_small())
        return source
    from ..formats.state import read_small
    from .export import latent_small_file
    for part in source.split(","):
        path = Path(part)
        if path.suffix == ".safetensors":
            records = load_state(path).records()
        else:
            records = legacy_records(load_legacy_pt(path))
        if mode == "ternary":
            install_ternary(adapter, model, records, ctx.device)
        else:
            apply_dense(adapter, model, records)
        small = latent_small_file(path)
        if small is not None:     # a latent checkpoint's state: its distilled high-precision tensors too
            n = apply_small(adapter, model, read_small(small))
            ctx.log("eval")(f"{path.name}: applied the latent checkpoint's {n} distilled high-precision tensors "
                            f"({small.name})")
    return "+".join(Path(p).stem for p in source.split(","))


def source_files(source: str) -> list[str]:
    """The files a path source reads: its states and a latent checkpoint's distilled tensors."""
    from .export import latent_small_file
    files = []
    for part in source.split(","):
        files.append(part)
        small = latent_small_file(Path(part)) if part.endswith(".safetensors") else None
        if small is not None:
            files.append(str(small))
    return files


def compare_blocks(ctx, tok) -> torch.Tensor:
    c = ctx.cfg["eval"]["compare"]
    adapter = ctx.adapter()
    sources = c.get("sources") or [{"kind": "wikitext2", "n": int(c["n"]),
                                    "file": "wikitext-2-raw-v1/test-00000-of-00001.parquet"}]
    blocks, _ = calibrate.build(tok, sources, seq_len=int(c["seq_len"]), seed=int(c["seed"]),
                                chat_kwargs=adapter.chat_kwargs(thinking=False), bos=adapter.bos_id(tok),
                                traces_path=ctx.run.stage_dir("traces", create=False) / "traces.jsonl",
                                local_files_only=bool(ctx.cfg["model"].get("local_files_only")), where="eval.compare",
                                chat_guard=False)   # held-out scoring data, not training data
    return blocks


def _reference(ctx, blocks, model) -> tuple[dict, object]:
    """BF16 reference statistics of the unmodified model, cached per model and block set.

    ``model``: the model, or a callable returning it (called only when the cache misses).
    Returns (statistics, ``ref_logits(b)`` loader of the cached raw logits, or None
    when ``eval.compare.full`` is off)."""
    from ..config import canonical_hash
    from ..formats.atomic import durable_save
    out = ctx.run.stage_dir("eval")
    m = ctx.cfg["model"]
    c = ctx.cfg["eval"]["compare"]
    key = canonical_hash({"blocks": blocks.tolist(), "model": [m.get(k) for k in ("path", "revision", "dtype", "attn")],
                          "model_sha256": ctx.model_digest(), "topk": int(c["topk"])})[:16]
    path = out / f"compare-reference-{key}.pt"
    full = bool(c.get("full", True))
    logits_dir = out / f"compare-reference-{key}.logits"
    files = [logits_dir / f"block_{b:04d}.pt" for b in range(len(blocks))]
    loader = (lambda b: torch.load(files[b], weights_only=True)) if full else None
    if path.exists() and (not full or all(f.exists() for f in files)):
        return torch.load(path, weights_only=True), loader
    if callable(model) and not isinstance(model, torch.nn.Module):
        model = model()
    model.eval()
    keep = None
    if full:
        vocab = S.vocab_size(model)
        ctx.log("eval")(f"compare: keeping the reference logits for ear_full/kl_full in {logits_dir.name}: "
                        f"{len(blocks)} x {blocks.shape[1] - 1} x {vocab} in {ctx.dtype} "
                        f"(~{len(blocks) * (blocks.shape[1] - 1) * vocab * ctx.dtype.itemsize / 2**30:.2f} GiB)")

        def keep(b, logits):
            durable_save(logits, files[b])
    ref = S.compare_reference(model, blocks, topk=int(c["topk"]), keep_logits=keep)
    durable_save(ref, path)
    return ref, loader


def teacher_rows(ctx):
    """(ids, idx, logp) of the first ``eval.teacher.rows`` rows of the run's distillation teacher shards."""
    from .distill import read_teacher_rows
    folder = ctx.run.stage_dir("distill", create=False) / "teacher"
    try:
        return read_teacher_rows(folder, int(ctx.cfg["eval"]["teacher"]["rows"]))
    except FileNotFoundError as error:
        raise S.SuiteNotApplicable(f"no distillation teacher shards in this run ({error})") from None


def load_gguf(ctx, path, *, small_fp32: bool = False):
    """The native model built from a PTQ1_0 GGUF on the run's device and dtype."""
    m = ctx.cfg["model"]
    model_dir = ctx.model_dir(weights=False)   # config and generation files; the weights come from the GGUF
    return ctx.adapter().from_gguf(path, model_dir=model_dir, dtype=ctx.dtype, device=ctx.device,
                                   attn=m.get("attn") or "sdpa", small_fp32=small_fp32, log=ctx.log("eval"))


def weights_identity(ctx, source: str, weights: str, gguf_small_fp32: bool = False) -> str:
    """Hash of what the generating weights are made of: the model, and the stage's recorded config
    hash and output digest, or the source files' SHA256 (the key of the generation cache)."""
    m = ctx.cfg["model"]
    ident = {"model": {k: m.get(k) for k in ("path", "revision", "adapter", "dtype", "attn")},
             "device": ctx.device.type, "weights": weights, "source": source}
    if source in STAGE_SOURCES[1:]:
        ident["stage"] = ctx.run.done_key(source)        # its hash covers the model and upstream outputs
    elif is_gguf(source):
        ident["files"] = ctx.file_digests([source])      # the whole model comes from the file
        ident["small_fp32"] = bool(gguf_small_fp32)
    else:
        ident["model_sha256"] = ctx.model_digest()
        if source != "bf16":
            ident["files"] = ctx.file_digests(source_files(source))
    return canonical_hash(ident)


def check_test_problems_absent(ctx, source: str, suites) -> None:
    """Refuse to score a stage whose stored calibration (and, for distill, distillation-teacher)
    tokens contain a problem about to be scored. The build-time guard checked them against the test
    problems known then (``heldout``); this covers data built before it, or without the full splits
    (the problems above the frozen subsets). A clean result is remembered per file and problem set."""
    from ..eval import heldout
    from ..formats.atomic import atomic_write_json
    if source not in STAGE_SOURCES[1:]:
        return
    r = ctx.cfg["eval"]["reasoning"]
    stored = [(ctx.run.stage_dir("calibrate", create=False) / "tokens.npy", int(ctx.cfg["calibrate"]["seed"]),
               "calib/tokens.npy")]
    if source == "distill":
        stored.append((ctx.run.stage_dir("distill", create=False) / "teacher" / "input_ids.npy",
                       int(ctx.cfg["distill"]["teacher"]["seed"]), "distill/teacher/input_ids.npy"))
    stored = [s for s in stored if s[0].is_file()]
    if not stored:
        return
    probes = []
    modes = R.thinking_modes(r.get("thinking", "off"))
    for suite in suites:
        sc = r.get(suite) or {}
        # the larger of the modes' subsets (each is the first n of the same selection order)
        size = max(R.FROZEN[suite]["n"] if n is None else n for n in (suite_size(r, suite, m) for m in modes))
        _, problems = R.load_problems(suite, size, path=sc.get("path"),
                                      local_files_only=bool(ctx.cfg["model"].get("local_files_only")))
        probes += heldout.problem_needles(suite, problems)
    memo_path = ctx.run.stage_dir("eval") / "heldout-checked.json"
    try:
        memo = json.loads(memo_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        memo = {}
    tok = None
    for path, seed, name in stored:
        key = canonical_hash({"file": ctx.digests().file(path), "seed": seed, "probes": sorted(probes)})
        if memo.get(name) == key:
            continue
        if tok is None:
            tok = ctx.load_tokenizer()
        import numpy as np
        heldout.check_stored_blocks(tok, torch.from_numpy(np.load(path).astype(np.int64)), seed,
                                    f"{name} (scored by {', '.join(suites)})", probes,
                                    bos=ctx.adapter().bos_id(tok))
        memo[name] = key
        atomic_write_json(memo_path, memo)


def check_thinking_modes(cfg, adapter, tokenizer) -> list[str]:
    """``eval.reasoning.thinking`` as modes; a mode the model's chat template has no switch for, or whose
    switch the tokenizer's template ignores (``traces.checked_chat_kwargs``), is a ConfigError (raised
    before any model is loaded or any result is written)."""
    from .traces import checked_chat_kwargs
    try:
        modes = R.thinking_modes(cfg["eval"]["reasoning"].get("thinking", "off"))
    except ValueError as error:
        raise ConfigError(str(error)) from None
    for mode in modes:
        checked_chat_kwargs(adapter, tokenizer, mode == "on", f"eval.reasoning.thinking {mode}")
    return modes


def suite_size(r: dict, suite: str, mode: str) -> int | None:
    """``eval.reasoning.<suite>.n``, or ``n_thinking`` with thinking on when it is set (None: the frozen subset)."""
    sc = r.get(suite) or {}
    if mode == "on" and sc.get("n_thinking") is not None:
        return int(sc["n_thinking"])
    return None if sc.get("n") is None else int(sc["n"])


def suite_batch(r: dict, mode: str) -> int:
    """``eval.reasoning.batch``, or ``batch_thinking`` with thinking on when it is set."""
    if mode == "on" and r.get("batch_thinking"):
        return int(r["batch_thinking"])
    return int(r.get("batch") or 16)


def run_reasoning(ctx, model, tok, source: str, label: str, suites, *, identity: str, weights: str) -> dict:
    """The gsm8k/humaneval suites in every ``eval.reasoning.thinking`` mode; one result file each."""
    r = ctx.cfg["eval"]["reasoning"]
    log = ctx.log("eval")
    out = ctx.run.stage_dir("eval")
    modes = R.thinking_modes(r.get("thinking", "off"))
    results = {}
    for suite in suites:
        sc = r.get(suite) or {}
        results[suite] = {}
        for mode in modes:
            cap = int(r["max_new_tokens_thinking"] if mode == "on" else r["max_new_tokens"])
            cache = out / "generations" / f"{label}.{suite}.think-{mode}" if r.get("cache", True) else None
            memory_mb = sc.get("memory_mb", 2048)
            try:
                res = R.run_suite(
                    suite, model, tok, ctx.adapter(), thinking=mode, n=suite_size(r, suite, mode),
                    max_new_tokens=cap, batch=suite_batch(r, mode), device=ctx.device, log=log, cache_dir=cache,
                    cache_key=identity, timeout=float(sc.get("timeout") or 10.0),
                    workers=int(sc.get("workers") or 4), path=sc.get("path"),
                    memory_limit=int(memory_mb) * 2**20 if memory_mb else None,
                    local_files_only=bool(ctx.cfg["model"].get("local_files_only")),
                    batch_check=int(r.get("batch_check") or 0))
            except S.SuiteNotApplicable as error:
                log(f"{suite}: skipped ({error})")
                results[suite] = {"suite": suite, "skipped": str(error)}
                break
            res = {"source": source, "weights": weights, **res}
            atomic_write_json(out / f"{label}.{suite}.think-{mode}.json", res)
            rate = res["tokens_per_second"]
            unfinished = f", unfinished thinking {res['unfinished_thinking']}" if mode == "on" else ""
            log(f"{suite} think={mode}: {res['score']}, hit cap {res['hit_cap']}/{res['total']} at {cap} tokens"
                f"{unfinished}{f', {rate:.1f} tok/s' if rate else ''} "
                f"({res['cached_batches']}/{res['batches']} batches from the cache)")
            results[suite][mode] = {k: v for k, v in res.items() if k != "problems"}
    return results


def run(ctx, source: str | None = None, suites=None, install_mode: str = "auto",
        gguf_small_fp32: bool = False) -> dict:
    e = ctx.cfg["eval"]
    log = ctx.log("eval")
    source = resolve_source(ctx, source or e.get("from"))
    suites = list(suites or e.get("suites") or ["dev"])
    unknown = set(suites) - set(S.SUITES)
    if unknown:
        raise ValueError(f"unknown suites {sorted(unknown)}; known: {S.SUITES}")
    generating = [s for s in suites if s in R.SUITES]
    suites = [s for s in suites if s not in R.SUITES]       # scored first, on the installed weights
    dense_gen = bool(e["reasoning"].get("dense", True))
    if generating:
        check_thinking_modes(ctx.cfg, ctx.adapter(), ctx.load_tokenizer())     # bad values fail before loading
        check_test_problems_absent(ctx, source, generating)
        if not suites and dense_gen and install_mode == "auto" and not is_gguf(source):
            install_mode = "dense"    # generation only: write the dequantized weights directly
    out = ctx.run.stage_dir("eval")
    results = {}
    reference = blocks = ref_logits = None
    if is_gguf(source):
        if install_mode == "dense":
            raise ConfigError("a GGUF source is installed as ternary modules; --install dense does not apply")
        tok = ctx.load_tokenizer()
        if "compare" in suites:
            blocks = compare_blocks(ctx, tok)
            reference, ref_logits = _reference(ctx, blocks, lambda: ctx.load_model()[0])
            free(ctx.device)
        model = load_gguf(ctx, source, small_fp32=gguf_small_fp32)
        label = Path(source).stem + ("-fp32small" if gguf_small_fp32 else "")
    else:
        model, tok = ctx.load_model()
        if "compare" in suites:
            blocks = compare_blocks(ctx, tok)
            model.eval()
            reference, ref_logits = _reference(ctx, blocks, model)   # before any ternary weights go in
        label = install(ctx, model, source, install_mode)
    model.eval()
    for suite in suites:
        try:
            if suite == "dev":
                r = S.dev(model, tok, generations=bool(e.get("generations", True)))
            elif suite == "extended":
                r = S.extended(model, tok)
            elif suite == "teacher":
                r = S.teacher(model, *teacher_rows(ctx))
            else:
                r = S.compare(model, blocks, reference, ref_logits=ref_logits)
        except S.SuiteNotApplicable as error:
            log(f"{suite}: skipped ({error})")
            results[suite] = {"suite": suite, "skipped": str(error)}
            continue
        r = {"source": source, **r}
        atomic_write_json(out / f"{label}.{suite}.json", r)
        log(json.dumps({k: v for k, v in r.items() if k != "generations"}))
        results[suite] = r
    if generating:
        weights = "ternary" if has_ternary(model) else "dense"
        if weights == "ternary" and dense_gen:
            replaced = densify(model, ctx.dtype)
            free(ctx.device)
            weights = "dense"
            log(f"reasoning: generating with dense dequantized weights ({replaced} ternary modules replaced)")
        with vram_cap(ctx.device, log):    # a KV cache grown a token per step must not spill to system RAM
            results.update(run_reasoning(ctx, model, tok, source, label, generating, weights=weights,
                                         identity=weights_identity(ctx, source, weights, gguf_small_fp32)))
    return results

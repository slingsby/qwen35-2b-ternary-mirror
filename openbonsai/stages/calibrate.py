"""Fixed-length calibration token blocks from configurable sources.

Port of OpenBonsai research/experiments/calib_data.py with the same block arithmetic:
WikiText-2 *train* windows at a seed-dependent phase, chat conversations
rendered with the model's own chat template (rows below ``min_row`` are left
for trace prompts), the model's self-generated traces, and the user's
own JSONL. Blocks are concatenated in source order and permuted with the seed.

Files come from a local ``path`` or from the Hugging Face hub at a pinned
revision (no machine-specific paths). For adapters that add BOS, every block
starts with BOS and carries seq_len - 1 data tokens.

The reasoning suites' TEST problems (GSM8K, HumanEval) never enter the blocks:
a source naming a test split is refused, and so are blocks whose text contains
a frozen test problem (openbonsai.eval.heldout). Math or code calibration data
must come from training splits. Blocks that contain a prompt or an answer of the
extended suite's chat NLL are refused too (``heldout.chat_needles``).

Output: ``calib/tokens.npy`` (int32 [N, seq_len]) and ``calib/sources.json``.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np
import torch

from ..digest import file_sha256
from ..formats.atomic import atomic_write_bytes, atomic_write_json

HUB = {
    "wikitext2": {"repo": "Salesforce/wikitext", "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
                  "file": "wikitext-2-raw-v1/train-00000-of-00001.parquet"},
    "ultrachat": {"repo": "HuggingFaceH4/ultrachat_200k", "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
                  "file": "data/train_sft-00000-of-00003-a3ecf92756993583.parquet"},
}
KINDS = ("wikitext2", "ultrachat", "traces", "jsonl")


def resolve_file(src: dict, *, local_files_only: bool = False, default_path=None) -> Path:
    """A local ``path``, or ``repo``/``file``/``revision`` downloaded from the hub (dataset repo)."""
    if src.get("path"):
        path = Path(src["path"])
        if not path.exists():
            raise FileNotFoundError(f"{src['kind']} source not found: {path}")
        return path
    if default_path is not None:
        return Path(default_path)
    hub = {**HUB.get(src["kind"], {}), **{k: src[k] for k in ("repo", "revision", "file") if src.get(k)}}
    if not hub.get("repo") or not hub.get("file"):
        raise ValueError(f"{src['kind']} source needs a path or repo/file")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(hub["repo"], hub["file"], revision=hub.get("revision"), repo_type="dataset",
                                local_files_only=local_files_only))


def pack(stream, seq_len, n_blocks, bos=None):
    """Concatenate token lists and cut consecutive blocks (BOS-prefixed when ``bos``)."""
    take = seq_len - (bos is not None)
    blocks, buf = [], []
    for ids in stream:
        buf.extend(ids)
        while len(buf) >= take and len(blocks) < n_blocks:
            blocks.append(([bos] if bos is not None else []) + buf[:take])
            buf = buf[take:]
        if len(blocks) >= n_blocks:
            break
    return blocks


def chat_text(tok, messages, chat_kwargs) -> str:
    if getattr(tok, "chat_template", None):
        from ..adapters.base import template_vars
        return tok.apply_chat_template(messages, tokenize=False, **template_vars(chat_kwargs))
    return "\n\n".join(str(m.get("content", "")) for m in messages)   # base model: plain text


def wiki_blocks(tok, path, seq_len, n, seed, bos=None):
    import pyarrow.parquet as pq
    text = "".join(pq.read_table(path).column("text").to_pylist())
    ids = tok(text, add_special_tokens=False)["input_ids"]
    take = seq_len - (bos is not None)
    rng = random.Random(seed)
    off = rng.randrange(take)  # non-overlapping windows at a seed-dependent phase
    starts = sorted(rng.sample(range(off, len(ids) - take, take), n))
    return [([bos] if bos is not None else []) + ids[s:s + take] for s in starts]


def ultrachat_blocks(tok, path, seq_len, n, seed, *, min_row=6000, chat_kwargs=None, bos=None):
    import pyarrow.parquet as pq
    # Rows < min_row are the pool for trace prompts.
    rows = pq.read_table(path, columns=["messages"]).column("messages").to_pylist()[min_row:]
    random.Random(seed).shuffle(rows)
    kwargs = chat_kwargs or {}

    def gen():
        for msgs in rows:
            yield tok(chat_text(tok, msgs, kwargs), add_special_tokens=False)["input_ids"]

    return pack(gen(), seq_len, n, bos)


def trace_blocks(path, seq_len, n, bos=None):
    def gen():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                mask = r.get("attention_mask") or [1] * len(r["input_ids"])
                yield [t for t, m in zip(r["input_ids"], mask) if m]

    return pack(gen(), seq_len, n, bos)


def jsonl_blocks(tok, path, seq_len, n, seed, *, chat_kwargs=None, bos=None, shuffle=False):
    """User data: one JSON object per line with ``messages`` (chat), ``text`` or ``input_ids``."""
    rows = []
    for k, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if not isinstance(r, dict) or not ({"messages", "text", "input_ids"} & set(r)):
            raise ValueError(f"{path}:{k}: expected an object with messages, text or input_ids")
        rows.append(r)
    if shuffle:
        random.Random(seed).shuffle(rows)
    kwargs = chat_kwargs or {}

    def gen():
        for r in rows:
            if "input_ids" in r:
                yield [int(t) for t in r["input_ids"]]
            elif "messages" in r:
                yield tok(chat_text(tok, r["messages"], kwargs), add_special_tokens=False)["input_ids"]
            else:
                yield tok(r["text"], add_special_tokens=False)["input_ids"]

    return pack(gen(), seq_len, n, bos)


def build(tok, sources, *, seq_len=2048, seed=0, chat_kwargs=None, bos=None, traces_path=None,
          local_files_only=False, where="calibration", chat_guard=True):
    """Blocks from every source in order, then one seeded permutation.

    Refuses evaluation test data (``heldout``; ``where`` names the data in the error) and, with
    ``chat_guard``, the extended suite's chat-NLL prompts and answers (training data only: calibration
    and the distillation teacher). Returns (LongTensor [N, seq_len], per-source report).
    """
    from ..eval import heldout
    probes = heldout.needles() + (heldout.chat_needles(tok) if chat_guard else ())
    blocks, report = [], []
    for k, src in enumerate(sources):
        kind, n = src.get("kind"), int(src.get("n") or 0)
        if kind not in KINDS:
            raise ValueError(f"unknown calibration source kind {kind!r}; known: {KINDS}")
        if n <= 0:
            continue
        s_seed = int(src.get("seed", seed))
        heldout.check_source(src, f"{where} source {k} ({kind})")
        path = resolve_file(src, local_files_only=local_files_only,
                            default_path=traces_path if kind == "traces" and not src.get("path") else None)
        heldout.check_source(src, f"{where} source {k} ({kind})", path=path if src.get("path") else None)
        if kind == "wikitext2":
            got = wiki_blocks(tok, path, seq_len, n, s_seed, bos)
        elif kind == "ultrachat":
            got = ultrachat_blocks(tok, path, seq_len, n, s_seed, min_row=int(src.get("min_row", 6000)),
                                   chat_kwargs=chat_kwargs, bos=bos)
        elif kind == "traces":
            if not Path(path).exists():
                raise FileNotFoundError(f"traces file {path} missing; run the traces stage or set a path")
            got = trace_blocks(path, seq_len, n, bos)
        else:
            got = jsonl_blocks(tok, path, seq_len, n, s_seed, chat_kwargs=chat_kwargs, bos=bos,
                               shuffle=bool(src.get("shuffle", False)))
        if len(got) != n:
            advice = ""
            if kind == "traces":
                from .traces import SUPPLY_ADVICE
                advice = f" ({path}); {SUPPLY_ADVICE}"
            raise ValueError(f"{kind}: only {len(got)} blocks of {seq_len} tokens available, {n} requested{advice}")
        heldout.check_blocks(tok, got, f"{where} source {k} ({kind})", bos=bos, probes=probes)
        blocks += got
        report.append({"kind": kind, "blocks": n, "path": str(path), "sha256": file_sha256(path),
                       **{k: src[k] for k in ("repo", "revision", "file", "min_row", "seed") if k in src}})
    if not blocks:
        raise ValueError("no calibration blocks requested")
    out = torch.tensor(blocks, dtype=torch.long)
    perm = torch.randperm(len(out), generator=torch.Generator().manual_seed(seed))
    return out[perm], report


def save_tokens(path, tokens: torch.Tensor) -> None:
    buf = io.BytesIO()
    np.save(buf, tokens.numpy().astype(np.int32))
    atomic_write_bytes(path, buf.getvalue())


def run(ctx) -> dict:
    c = ctx.cfg["calibrate"]
    log = ctx.log("calibrate")
    adapter = ctx.adapter()
    tok = ctx.load_tokenizer()
    out = ctx.run.stage_dir("calibrate")
    tokens, report = build(tok, c["sources"], seq_len=int(c["seq_len"]), seed=int(c["seed"]),
                           chat_kwargs=adapter.chat_kwargs(thinking=False), bos=adapter.bos_id(tok),
                           traces_path=ctx.run.stage_dir("traces", create=False) / "traces.jsonl",
                           local_files_only=bool(ctx.cfg["model"].get("local_files_only")))
    save_tokens(out / "tokens.npy", tokens)
    summary = {"shape": list(tokens.shape), "tokens": int(tokens.numel()), "seed": int(c["seed"]),
               "sources": report}
    atomic_write_json(out / "sources.json", summary)
    log(f"calibration {tuple(tokens.shape)} = {tokens.numel()} tokens from "
        + ", ".join(f"{r['kind']}x{r['blocks']}" for r in report))
    return summary

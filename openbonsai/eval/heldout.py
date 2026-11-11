"""Keep evaluation data out of calibration data and trace prompts.

The ``gsm8k`` and ``humaneval`` suites (openbonsai.eval.reasoning) score TEST
problems. If calibration blocks (calibrate, the distillation teacher's data)
or trace prompts contained them, the compressed model would be fitted on what
it is scored on. Three checks run on every build:

- ``check_source``: a hub source naming a reasoning dataset's test split, or a
  local file whose bytes are a pinned test parquet, is refused;
- ``check_texts``: trace prompts that contain a test problem are refused;
- ``check_blocks``: a source's token blocks are decoded as one text (packed
  sources are consecutive cuts of one stream, so a problem split across two
  blocks is still found) and refused if it contains one.

The test problems (``needles``) are every problem of both pinned test splits
(1,319 GSM8K, 164 HumanEval) when the split is in the Hugging Face cache,
else the frozen subsets the suites score by default (an ``n`` above them scores
problems that only the full splits guard). So eval checks again when it scores
a stage: ``check_stored_blocks`` decodes the stored calibration and teacher
tokens in their build order and refuses them if they contain a problem about
to be scored (data built before a guard existed, or without the full splits).

A problem is contained when its whitespace-normalized text is a substring: a
GSM8K question, or a HumanEval prompt from ``def <entry_point>(`` on.

Trace prompts drawn from training splits (``traces.sources``) are also
screened for near duplicates (``near_duplicate``): a row that shares a run of
13 words with a test problem, such as a GSM8K train question written from the
same template as a test question, is left out of the sampling pool and listed
in the traces summary; a row that contains a whole test problem is refused.

The ``extended`` suite's chat NLL scores 16 Qwen3.5-9B answers to UltraChat
first-turn prompts (assets/bonsai-dev/chat-traces16.jsonl, rows below 5,000).
``chat_needles`` decodes their prompts and answers with the run's tokenizer,
and ``build`` (calibration and the distillation teacher) and the trace prompts
refuse data that contains one. A tokenizer that cannot reproduce the suite's
token ids gets no chat needles: the extended suite refuses such a model, so
its chat NLL cannot be fitted on.

Math or code calibration must come from TRAINING splits, e.g. ``openai/gsm8k``
``main/train-00000-of-00001.parquet`` (HumanEval has no training split; use
another dataset's training split, such as MBPP's). The traces stage's
``gsm8k_train``, ``math_train``, ``mbpp_train`` and ``opencodeinstruct``
kinds read pinned training files and refuse a ``file`` that is not one.
"""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import re


class ContaminationError(ValueError):
    """Calibration or trace data contains evaluation test problems."""


ADVICE = ("these are evaluation TEST problems (openbonsai.eval.reasoning); math or code calibration data "
          "must come from training splits (e.g. openai/gsm8k main/train-00000-of-00001.parquet)")
CHAT_PREFIX = "chat-nll/"
CHAT_ADVICE = ("these are the extended suite's chat-NLL prompts or answers (UltraChat rows below 5,000, "
               "openbonsai/eval/assets/bonsai-dev/chat-traces16.jsonl); take trace prompts from rows 5,000 on "
               "(--set 'traces.prompts.rows=[5000,6000]') and UltraChat calibration from min_row 6000")


def normalize(text: str) -> str:
    return " ".join(str(text).split())


def problem_needles(name: str, problems) -> list[tuple[str, str]]:
    """(normalized text, problem id) of reasoning problems (manifest problem records)."""
    if name == "gsm8k":
        return [(normalize(p["question"]), p["id"]) for p in problems]
    out = []
    for p in problems:
        k = p["prompt"].find(f"def {p['entry_point']}(")
        out.append((normalize(p["prompt"][max(k, 0):]), p["id"]))
    return out


def _full_split(name: str) -> list[dict] | None:
    """Every problem of the pinned test split, if its parquet is available offline (else None)."""
    from .reasoning import _problem, read_rows
    try:
        rows = read_rows(name, local_files_only=True)
    except Exception:          # not in the cache (or no hub client): the frozen subset guards
        return None
    return [_problem(name, r, rec) for r, rec in enumerate(rows)]


def _test_problems(name: str) -> list[dict]:
    from .reasoning import load_frozen
    return _full_split(name) or load_frozen(name)["problems"]


@lru_cache(maxsize=None)
def needles() -> tuple[tuple[str, str], ...]:
    """(normalized text, problem id) of every test problem: the full pinned splits when available
    offline, else the frozen subsets."""
    out = []
    for name in ("gsm8k", "humaneval"):
        out += problem_needles(name, _test_problems(name))
    return tuple(out)


# Near duplicates: a text sharing a run of NGRAM words with a test problem (the GPT-3 paper's
# 13-gram rule). Words are lowercase \w+ runs, so spacing, punctuation and case do not matter.
NGRAM = 13
_WORD = re.compile(r"\w+")


def word_ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    words = _WORD.findall(str(text).lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


@lru_cache(maxsize=None)
def ngram_index() -> dict[tuple[str, ...], str]:
    """Every NGRAM-word run of every test problem (GSM8K question; HumanEval prompt, signature and
    docstring) -> its problem id; the same problems as ``needles``."""
    index: dict[tuple[str, ...], str] = {}
    for name in ("gsm8k", "humaneval"):
        for p in _test_problems(name):
            for gram in word_ngrams(p["question"] if name == "gsm8k" else p["prompt"]):
                index.setdefault(gram, p["id"])
    return index


def near_duplicate(text: str, where: str) -> str | None:
    """The id of a test problem ``text`` shares an NGRAM-word run with (None when it shares none).
    A text that contains a whole test problem is refused (ContaminationError), as ``check_texts`` does;
    one that only shares a run (a template twin with other numbers, a copied problem description) is
    the caller's to drop. Trace prompts from training splits use it: GSM8K train holds 4 such rows,
    OpenCodeInstruct's first shard 3, MBPP and MATH train none."""
    index = ngram_index()
    hit = next((index[g] for g in word_ngrams(text) if g in index), None)
    if hit is not None:
        check_texts([text], where)
    return hit


def contained(texts, probes=None) -> list[tuple[int, str]]:
    """(text index, problem id) for every text that contains a test problem (``probes``: (needle, id)
    pairs; default ``needles()``)."""
    probes = needles() if probes is None else probes
    hits = []
    for i, text in enumerate(texts):
        flat = normalize(text)
        hits += [(i, pid) for needle, pid in probes if needle in flat]
    return hits


def _trace_parts(text: str) -> tuple[str, str]:
    """(user prompt, answer) of a chat trace rendered by Qwen3.5's template (the empty think block dropped)."""
    prompt = text.split("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0]
    answer = text.split("<|im_start|>assistant\n", 1)[-1]
    if "</think>" in answer:
        answer = answer.split("</think>", 1)[1]
    answer = answer.split("<|im_end|>", 1)[0].rstrip("�")   # a cut multi-byte character at the cap
    return prompt, answer


def chat_needles(tokenizer) -> tuple[tuple[str, str], ...]:
    """(normalized text, id) of the prompt and the answer of each of the extended suite's 16 chat-NLL
    traces, decoded with ``tokenizer``; none when ``tokenizer`` cannot reproduce the suite's token ids
    (the extended suite refuses such a model, so its chat NLL cannot be contaminated)."""
    if tokenizer is None:
        return ()
    from .suites import check_tokenizer, load_dev_manifest, load_extended_inputs
    try:
        check_tokenizer(tokenizer, load_dev_manifest())
    except Exception:          # SuiteNotApplicable, or a tokenizer that cannot even encode the texts
        return ()
    out = []
    for rec in load_extended_inputs()[1]:
        ids = [int(t) for t, m in zip(rec["input_ids"], rec["attention_mask"]) if m]
        prompt, answer = _trace_parts(tokenizer.decode(ids, skip_special_tokens=False))
        row = rec["source_row_index"]
        out += [(normalize(text), f"{CHAT_PREFIX}row-{row}/{part}")
                for text, part in ((prompt, "prompt"), (answer, "answer")) if normalize(text)]
    return tuple(out)


def _advice(ids) -> str:
    chat = any(i.startswith(CHAT_PREFIX) for i in ids)
    test = any(not i.startswith(CHAT_PREFIX) for i in ids)
    return "; ".join(a for a, on in ((ADVICE, test), (CHAT_ADVICE, chat)) if on)


def check_texts(texts, where: str, probes=None) -> None:
    hits = contained(texts, probes)
    if hits:
        ids = sorted({pid for _, pid in hits})
        raise ContaminationError(f"{where}: {len({i for i, _ in hits})} item(s) contain {len(ids)} evaluation "
                                 f"item(s) ({', '.join(ids[:5])}{', ...' if len(ids) > 5 else ''}); {_advice(ids)}")


def check_blocks(tokenizer, blocks, where: str, bos: int | None = None, probes=None) -> None:
    """Decode one source's token blocks in order as one text (block-leading ``bos`` dropped) and
    refuse it if it contains a test problem. ``blocks``: lists of ids, or a 2-D array or tensor."""
    if isinstance(blocks, list):
        ids = [int(t) for b in blocks for t in (b[1:] if bos is not None and len(b) and int(b[0]) == bos else b)]
    else:
        import numpy as np
        arr = np.asarray(blocks, dtype=np.int64)
        keep = np.ones(arr.shape, dtype=bool)
        if bos is not None and arr.size:
            keep[:, 0] = arr[:, 0] != bos
        ids = arr[keep].tolist()
    check_texts([tokenizer.decode(ids, skip_special_tokens=False)], where, probes)


def build_order(tokens, seed: int):
    """Stored blocks back in the order ``calibrate.build`` made them (it saves ``tokens[perm]`` with
    ``perm = randperm(n, seed)``), so a problem cut across two consecutive blocks is whole again."""
    import torch
    perm = torch.randperm(len(tokens), generator=torch.Generator().manual_seed(int(seed)))
    tokens = torch.as_tensor(tokens)
    out = torch.empty_like(tokens)
    out[perm] = tokens
    return out


def check_stored_blocks(tokenizer, tokens, seed: int, where: str, probes, bos: int | None = None) -> None:
    """Refuse stored blocks (``calibrate.build`` output saved as tokens[perm]) that contain a problem."""
    import torch
    check_blocks(tokenizer, build_order(torch.as_tensor(tokens), seed), where, bos=bos, probes=probes)


def check_source(src: dict, where: str, path=None) -> None:
    """Refuse a data source that is a reasoning dataset's test split (by hub name or by file bytes)."""
    from ..digest import file_sha256
    from .reasoning import DATASETS
    repo, file = src.get("repo"), str(src.get("file") or "")
    for d in DATASETS.values():
        if repo == d["repo"] and (not file or d["split"] in Path(file).name.lower() or file == d["file"]):
            raise ContaminationError(f"{where}: {repo} {file or '(no file)'} is not a training split; {ADVICE}")
    if path is not None and Path(path).is_file():
        size = os.path.getsize(path)
        for d in DATASETS.values():
            if size == d["bytes"] and file_sha256(path) == d["sha256"]:
                raise ContaminationError(f"{where}: {path} is {d['repo']} {d['file']}; {ADVICE}")

"""GSM8K and HumanEval reasoning suites on frozen TEST subsets.

Data
  GSM8K (``main`` test split, 1,319 problems) and HumanEval (164) come from the
  Hugging Face hub at pinned revisions (``DATASETS``; the local HF cache is
  used, and HF_HUB_OFFLINE works) or from a local parquet ``path``. The suites
  score fixed subsets frozen in ``assets/reasoning/``: 100 GSM8K and 50
  HumanEval problems picked by the seeded selector ``sha256-rank/1``. Each
  manifest records the dataset (repo, config, split, revision, file and its
  SHA256), the seed and selector rule, the row ids, every problem (GSM8K
  question and final number; HumanEval prompt, test, entry point and
  canonical solution) and the SHA256 of its own content; the files' SHA256
  are pinned in ``FROZEN``. ``n`` below the frozen size takes the first ``n``
  problems in selection order, which is what the selector picks for ``n``; a
  larger ``n`` reads the dataset (the extra problems are not frozen).

  These are TEST problems. calibrate and traces refuse sources and prompts
  that contain them (openbonsai.eval.heldout; every test problem of both
  splits when they are in the cache, else the frozen subsets), and eval
  refuses to score a stage whose stored calibration or distillation-teacher
  tokens contain a problem it is about to score; math or code calibration data
  must come from training splits.

Generation
  Greedy (plain argmax: the checkpoint's ``generation_config.json`` is not
  applied, see ``stages.traces.generate_encoded``), with the model's own chat
  template (``adapter.chat_kwargs``; templates that print today's date see the
  fixed ``adapters.base.CHAT_DATE``), in left-padded batches with the KV cache,
  stopping at the adapter's EOS ids. Thinking off: at most 512 new tokens;
  thinking on: 4096. Rows that reach the cap are counted (``hit_cap``). With
  thinking on, the answer is the text after the last ``adapter.think_end``
  (``</think>``); a generation that never closes its thinking has no answer
  (``unfinished_thinking``). Generations are cached per batch, so an
  interrupted run resumes and rescoring needs no GPU. In BF16 a row's tokens can
  depend on the batch it shares (left padding changes the prefill arithmetic),
  so scores reproduce at the same ``batch``; ``batch_check`` measures it.

Scoring: openbonsai.eval.scoring (GSM8K final-number exact match; HumanEval
pass@1 with every program run in a separate ``python -I`` process with a
timeout, which is not a security sandbox).

Rebuild or check the frozen manifests (needs the datasets: hub cache or paths):

    python -m openbonsai.eval.reasoning check [--gsm8k PATH] [--humaneval PATH]
    python -m openbonsai.eval.reasoning freeze [--out DIR]
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from ..formats.atomic import atomic_write_bytes, atomic_write_json
from . import scoring
from .suites import SuiteNotApplicable

SUITES = ("gsm8k", "humaneval")
ASSETS = Path(__file__).resolve().parent / "assets" / "reasoning"
FORMAT = "openbonsai-reasoning/1"
SEED = 2026
SELECTOR = "sha256-rank/1"
SELECTOR_RULE = ("rank every row index r (0-based, in the split's parquet file order) by the hex SHA256 of "
                 "'{repo}:{split}:{seed}:{r}' (UTF-8), ascending; the first n rows in that order are the subset, "
                 "kept in that order")
DATASETS = {
    "gsm8k": {"repo": "openai/gsm8k", "config": "main", "split": "test",
              "revision": "740312add88f781978c0658806c59bc2815b9866",
              "file": "main/test-00000-of-00001.parquet", "rows": 1319, "bytes": 419088,
              "sha256": "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"},
    "humaneval": {"repo": "openai/openai_humaneval", "config": "openai_humaneval", "split": "test",
                  "revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                  "file": "openai_humaneval/test-00000-of-00001.parquet", "rows": 164, "bytes": 83920,
                  "sha256": "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6"},
}
FROZEN = {
    "gsm8k": {"file": "gsm8k-test-100.json", "n": 100,
              "sha256": "f1783b93eb7a7d14be338cb36545441b6572ce65ffeecc7103616caf3dd59d0a"},
    "humaneval": {"file": "humaneval-test-50.json", "n": 50,
                  "sha256": "392f9271c8c43989ea096efd6817579064c821516f0e18a4d7ee5fe30887e9ad"},
}
THINKING_CAPS = {"off": 512, "on": 4096}
GSM8K_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
HUMANEVAL_INSTRUCTION = ("Complete the following Python function. Reply with the complete function, "
                         "including its signature, in a single ```python code block.")


# -- selection and manifests -----------------------------------------------------

def select_rows(name: str, n: int, seed: int = SEED, n_rows: int | None = None) -> list[int]:
    """The ``sha256-rank/1`` selector (``SELECTOR_RULE``): row indices in selection order."""
    d = DATASETS[name]
    n_rows = int(d["rows"] if n_rows is None else n_rows)
    if not 0 < n <= n_rows:
        raise ValueError(f"{name}: cannot select {n} of {n_rows} rows")
    key = f"{d['repo']}:{d['split']}:{int(seed)}:"
    return sorted(range(n_rows), key=lambda r: hashlib.sha256(f"{key}{r}".encode("utf-8")).hexdigest())[:n]


def dataset_file(name: str, path=None, *, local_files_only: bool = False) -> Path:
    """The split's parquet: a local ``path``, else the pinned hub file (cache first; HF_HUB_OFFLINE works)."""
    if path:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"{name} parquet not found: {path}")
        return path
    from huggingface_hub import hf_hub_download
    d = DATASETS[name]
    return Path(hf_hub_download(d["repo"], d["file"], revision=d["revision"], repo_type="dataset",
                                local_files_only=local_files_only))


def read_rows(name: str, path=None, *, local_files_only: bool = False, check: bool = True) -> list[dict]:
    """Every row of the split; ``check`` refuses a file whose SHA256 differs from the pinned one."""
    import pyarrow.parquet as pq
    from ..digest import file_sha256
    file = dataset_file(name, path, local_files_only=local_files_only)
    if check and file_sha256(file) != DATASETS[name]["sha256"]:
        raise ValueError(f"{file}: SHA256 differs from the pinned {DATASETS[name]['repo']} "
                         f"{DATASETS[name]['file']} @ {DATASETS[name]['revision'][:12]}")
    return pq.read_table(file).to_pylist()


def _problem(name: str, row: int, rec: dict) -> dict:
    if name == "gsm8k":
        return {"id": f"gsm8k/test/{row}", "row": row, "question": rec["question"],
                "answer": scoring.gsm8k_reference(rec["answer"])}
    return {"id": rec["task_id"], "row": row, "prompt": rec["prompt"], "test": rec["test"],
            "entry_point": rec["entry_point"], "canonical_solution": rec["canonical_solution"]}


def content_sha256(manifest: dict) -> str:
    body = {k: v for k, v in manifest.items() if k != "content_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def build_manifest(name: str, n: int, *, seed: int = SEED, rows: list[dict] | None = None, path=None,
                   local_files_only: bool = False) -> dict:
    """The manifest of ``n`` problems chosen by the selector (``rows``: the split, else read)."""
    rows = read_rows(name, path, local_files_only=local_files_only) if rows is None else rows
    d = DATASETS[name]
    if len(rows) != d["rows"]:
        raise ValueError(f"{name}: {len(rows)} rows, the pinned split has {d['rows']}")
    picked = select_rows(name, n, seed, len(rows))
    manifest = {"format": FORMAT, "suite": name,
                "dataset": {k: d[k] for k in ("repo", "config", "split", "revision", "file", "rows", "sha256")},
                "selector": {"algorithm": SELECTOR, "rule": SELECTOR_RULE, "seed": int(seed), "n": int(n)},
                "row_ids": picked, "problems": [_problem(name, r, rows[r]) for r in picked]}
    manifest["content_sha256"] = content_sha256(manifest)
    return manifest


def manifest_bytes(manifest: dict) -> bytes:
    return (json.dumps(manifest, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def verify_manifest(manifest: dict) -> dict:
    if manifest.get("format") != FORMAT or manifest.get("suite") not in SUITES:
        raise ValueError("not a reasoning manifest")
    if manifest.get("content_sha256") != content_sha256(manifest):
        raise ValueError(f"{manifest['suite']} manifest: content SHA256 does not match its content")
    if [p["row"] for p in manifest["problems"]] != manifest["row_ids"]:
        raise ValueError(f"{manifest['suite']} manifest: problems and row_ids disagree")
    return manifest


def load_manifest_file(path, sha256: str | None = None) -> dict:
    data = Path(path).read_bytes()
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{Path(path).name}: SHA256 differs from the pinned asset")
    return verify_manifest(json.loads(data.decode("utf-8")))


def load_frozen(name: str) -> dict:
    f = FROZEN[name]
    return load_manifest_file(ASSETS / f["file"], f["sha256"])


def load_problems(name: str, n: int | None = None, *, manifest=None, path=None,
                  local_files_only: bool = False) -> tuple[dict, list[dict]]:
    """(manifest summary, problems): the first ``n`` of the frozen subset (or of ``manifest``, a file);
    an ``n`` above it selects from the dataset (``path`` or the hub) with the same rule."""
    if name not in SUITES:
        raise ValueError(f"unknown reasoning suite {name!r}; known: {SUITES}")
    m = load_manifest_file(manifest) if manifest else load_frozen(name)
    if m["suite"] != name:
        raise ValueError(f"{manifest}: a {m['suite']} manifest, not {name}")
    frozen = True
    n = len(m["problems"]) if n is None else int(n)
    if n < 1:
        raise ValueError(f"{name}: n must be positive")
    if n > len(m["problems"]):
        if manifest:
            raise ValueError(f"{manifest} holds {len(m['problems'])} problems, {n} requested")
        m = build_manifest(name, n, seed=m["selector"]["seed"], path=path, local_files_only=local_files_only)
        frozen = False
    info = {"file": None if not frozen else (str(manifest) if manifest else FROZEN[name]["file"]),
            "content_sha256": m["content_sha256"], "frozen": frozen, "n": n,
            "dataset": m["dataset"], "selector": m["selector"]}
    return info, m["problems"][:n]


# -- prompts and generation ------------------------------------------------------

def messages(name: str, problem: dict) -> list[dict]:
    """The single user turn for a problem (the instruction strings above are part of the protocol)."""
    if name == "gsm8k":
        return [{"role": "user", "content": f"{problem['question']}\n{GSM8K_INSTRUCTION}"}]
    code = problem["prompt"].strip("\n")
    return [{"role": "user", "content": f"{HUMANEVAL_INSTRUCTION}\n\n```python\n{code}\n```"}]


def thinking_modes(value) -> list[str]:
    """``eval.reasoning.thinking`` as a list of "off"/"on" (YAML reads bare off/on as booleans)."""
    out = []
    for v in value if isinstance(value, (list, tuple)) else [value]:
        s = str(v).strip().lower()
        if v is True or s in ("on", "true", "yes", "1"):
            mode = "on"
        elif v is False or s in ("off", "false", "no", "0"):
            mode = "off"
        else:
            raise ValueError(f"eval.reasoning.thinking: {v!r} is not off or on")
        if mode not in out:
            out.append(mode)
    if not out:
        raise ValueError("eval.reasoning.thinking is empty")
    return out


def split_thinking(text: str, think_end: str | None, thinking: bool) -> tuple[str, bool]:
    """(answer, closed): the text after the last ``think_end``; with thinking on and no
    ``think_end`` the model never finished thinking, so there is no answer."""
    if think_end and think_end in text:
        return text.rsplit(think_end, 1)[1], True
    if thinking and think_end:
        return "", False
    return text, True


def _clock(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


def _kernel_roots(obj, depth: int = 3) -> set[str]:
    """Top-level packages of the CUDA kernel functions (FLA, causal-conv1d) a function's closure
    binds (transformers wraps the chosen implementation, or its torch fallback)."""
    from ..adapters.base import CUDA_KERNEL_PACKAGES
    found = set()
    for cell in getattr(obj, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        root = (getattr(value, "__module__", None) or "").split(".")[0]
        if root in CUDA_KERNEL_PACKAGES:
            found.add(root)
        elif depth > 1 and callable(value):
            found |= _kernel_roots(value, depth - 1)
    return found


def runtime_info(model) -> dict:
    """What decides the generated tokens besides the weights and the prompts: the torch and
    transformers versions, the CUDA device, and the kernels the model's module bound (FLA and
    causal-conv1d on CUDA when installed; the torch paths otherwise)."""
    import sys
    import torch
    import transformers
    module = sys.modules.get(type(model).__module__)
    kernels = set()
    for obj in (vars(module).values() if module else ()):
        if callable(obj):
            kernels |= _kernel_roots(obj)
    info = {"torch": torch.__version__, "transformers": transformers.__version__, "kernels": sorted(kernels)}
    device = next(model.parameters()).device
    if device.type == "cuda":
        info.update({"cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)})
    return info


def _first_difference(a: list[int], b: list[int]) -> int | None:
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return k if k is not None or len(a) == len(b) else min(len(a), len(b))


def generate(model, tokenizer, prompts: list[list[dict]], *, chat_kwargs: dict, eos_ids: list[int],
             max_new_tokens: int, batch: int, device, log=None, cache_dir=None, cache_key=None,
             label: str = "generate", batch_check: int = 0) -> tuple[list[dict], dict]:
    """Greedy left-padded batches; returns (one record per prompt, statistics).

    A record: ``text`` (decoded with special tokens, the final EOS dropped), ``generated_tokens``
    (EOS included), ``prompt_tokens``, ``hit_cap``. With ``cache_dir`` every finished batch is
    stored under a hash of ``cache_key`` (the weights' identity), the decoding, the runtime
    (``runtime_info``), the cap, EOS and the batch's prompt ids, and reused. Logs tokens/s and the
    ETA after each generated batch.

    Left padding changes the prefill arithmetic, so in BF16 a row's greedy tokens can depend on the
    batch it is in. ``batch_check`` regenerates that many rows of the first freshly generated
    batch (the most padded: shortest prompts first) one at a time and records whether their
    tokens match (``stats["batch_check"]``)."""
    from ..config import canonical_hash
    from ..stages.traces import DECODING, encode_chat, generate_encoded, pad_token
    if cache_dir is not None and cache_key is None:
        raise ValueError("a generation cache needs cache_key, the identity of the weights")
    encoded = [encode_chat(tokenizer, p, chat_kwargs) for p in prompts]
    pad = pad_token(tokenizer, eos_ids)
    eos = set(int(e) for e in eos_ids)
    batch = max(1, int(batch))
    n_batches = (len(encoded) + batch - 1) // batch
    records: list[dict] = []
    runtime = runtime_info(model)
    stats = {"batches": n_batches, "cached_batches": 0, "generated_tokens": 0, "decode_steps": 0,
             "seconds": 0.0, "fresh_tokens": 0, "runtime": runtime, "batch_check": None}
    fresh_rows = 0
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    for k in range(n_batches):
        rows = encoded[k * batch:(k + 1) * batch]
        key = canonical_hash({"weights": cache_key, "decoding": DECODING, "runtime": runtime,
                              "max_new_tokens": int(max_new_tokens), "eos": sorted(eos), "pad": int(pad),
                              "rows": rows})[:24]
        path = None if cache_dir is None else cache_dir / f"{key}.json"
        if path is not None and path.exists():
            got = json.loads(path.read_text(encoding="utf-8"))
            stats["cached_batches"] += 1
        else:
            t0 = time.monotonic()
            out = generate_encoded(model, rows, eos_ids=list(eos_ids), max_new_tokens=int(max_new_tokens),
                                   device=device, pad_id=pad)
            seconds = time.monotonic() - t0
            recs = []
            for ids, (gen, hit) in zip(rows, out):
                body = gen[:-1] if gen and gen[-1] in eos else gen
                recs.append({"prompt_tokens": len(ids), "generated_tokens": len(gen), "hit_cap": bool(hit),
                             "text": tokenizer.decode(body, skip_special_tokens=False)})
            got = {"seconds": seconds, "steps": max(len(g) for g, _ in out), "records": recs}
            if path is not None:
                atomic_write_json(path, got)
            if batch_check and stats["batch_check"] is None and len(rows) > 1:
                checked = []    # the most padded rows (shortest prompts) first
                for r in sorted(range(len(rows)), key=lambda i: (len(rows[i]), i))[:int(batch_check)]:
                    single, _ = generate_encoded(model, [rows[r]], eos_ids=list(eos_ids),
                                                 max_new_tokens=int(max_new_tokens), device=device, pad_id=pad)[0]
                    checked.append({"row": k * batch + r, "first_difference": _first_difference(out[r][0], single),
                                    "batched_tokens": len(out[r][0]), "single_tokens": len(single)})
                same = sum(c["first_difference"] is None for c in checked)
                stats["batch_check"] = {"batch": k, "rows": len(checked), "identical": same, "rows_checked": checked}
                if log is not None:
                    log(f"{label}: batch check: {same}/{len(checked)} rows of batch {k + 1} generate the same tokens "
                        f"alone as in the batch of {len(rows)}"
                        + "".join(f"; row {c['row']} differs from token {c['first_difference']}"
                                  for c in checked if c["first_difference"] is not None))
            tokens = sum(r["generated_tokens"] for r in recs)
            stats["seconds"] += seconds
            stats["fresh_tokens"] += tokens
            fresh_rows += len(rows)
            if log is not None:
                remaining = len(encoded) - (k + 1) * batch
                eta = stats["seconds"] / fresh_rows * max(0, remaining)
                log(f"{label}: batch {k + 1}/{n_batches}: {tokens} new tokens in {seconds:.1f} s = "
                    f"{tokens / max(seconds, 1e-9):.1f} tok/s ({got['steps']} steps, "
                    f"{got['steps'] / max(seconds, 1e-9):.2f} steps/s), hit cap {sum(r['hit_cap'] for r in recs)}"
                    f"/{len(recs)}; ETA {_clock(eta)}")
        records += got["records"]
        stats["generated_tokens"] += sum(r["generated_tokens"] for r in got["records"])
        stats["decode_steps"] += int(got["steps"])
    stats["tokens_per_second"] = stats["fresh_tokens"] / stats["seconds"] if stats["seconds"] else None
    return records, stats


# -- suites ------------------------------------------------------------------------

def _score_gsm8k(problems, answers) -> list[dict]:
    out = []
    for p, answer in zip(problems, answers):
        pred, method = scoring.extract_answer(answer)
        out.append({"id": p["id"], "row": p["row"], "answer": p["answer"], "pred": pred, "method": method,
                    "correct": pred is not None and pred == p["answer"]})
    return out


def _score_humaneval(problems, answers, *, timeout: float, workers: int,
                     memory_limit: int | None = scoring.MEMORY_LIMIT) -> list[dict]:
    programs, meta = [], []
    for p, answer in zip(problems, answers):
        code, how = scoring.extract_code(answer, p["entry_point"])
        prepared = scoring.prepare_program(p["prompt"], code, p["test"], p["entry_point"])
        programs.append(prepared["program"])
        meta.append((how, prepared["mode"], prepared["dropped"]))
    results = scoring.run_programs(programs, timeout=timeout, workers=workers, memory_limit=memory_limit)
    out = []
    for p, (how, mode, dropped), r in zip(problems, meta, results):
        rec = {"id": p["id"], "row": p["row"], "status": r["status"], "passed": r["status"] == "passed",
               "code_from": how, "program": mode, "dropped_statements": dropped, "returncode": r["returncode"],
               "seconds": r["seconds"]}
        if r["status"] != "passed":
            rec["stderr"] = r["stderr"][-600:]
        out.append(rec)
    return out


def run_suite(name: str, model, tokenizer, adapter, *, thinking: str = "off", n: int | None = None,
              max_new_tokens: int | None = None, batch: int = 16, device=None, log=None, cache_dir=None,
              cache_key=None, timeout: float = 10.0, workers: int = 4, memory_limit: int | None = scoring.MEMORY_LIMIT,
              manifest=None, path=None, local_files_only: bool = False, batch_check: int = 0) -> dict:
    """Generate and score one suite in one thinking mode; the result has the counts and per-problem records."""
    from ..adapters.base import CHAT_DATE
    from ..stages.traces import DECODING
    if thinking not in THINKING_CAPS:
        raise ValueError(f"thinking must be off or on, not {thinking!r}")
    if not getattr(tokenizer, "chat_template", None):
        raise SuiteNotApplicable("the tokenizer has no chat template")
    on = thinking == "on"
    try:
        chat_kwargs = adapter.chat_kwargs(thinking=on)
    except ValueError as error:
        raise SuiteNotApplicable(f"thinking {thinking}: {error}") from None
    info, problems = load_problems(name, n, manifest=manifest, path=path, local_files_only=local_files_only)
    cap = int(max_new_tokens or THINKING_CAPS[thinking])
    if device is None:
        device = next(model.parameters()).device
    gens, stats = generate(model, tokenizer, [messages(name, p) for p in problems],
                           chat_kwargs=chat_kwargs, eos_ids=adapter.eos_ids(tokenizer, model),
                           max_new_tokens=cap, batch=batch, device=device, log=log,
                           cache_dir=cache_dir, cache_key=cache_key, label=f"{name} think={thinking}",
                           batch_check=batch_check)
    think_end = getattr(adapter, "think_end", None)
    split = [split_thinking(g["text"], think_end, on) for g in gens]
    answers = [a for a, _ in split]
    if name == "gsm8k":
        scored = _score_gsm8k(problems, answers)
    else:
        scored = _score_humaneval(problems, answers, timeout=timeout, workers=workers, memory_limit=memory_limit)
    records = []
    for rec, g, (_, closed) in zip(scored, gens, split):
        records.append({**rec, "prompt_tokens": g["prompt_tokens"], "generated_tokens": g["generated_tokens"],
                        "hit_cap": g["hit_cap"], **({"finished_thinking": closed} if on else {}),
                        "completion": g["text"]})
    total = len(records)
    hit = sum(r["hit_cap"] for r in records)
    common = {"thinking": thinking, "total": total, "hit_cap": hit,
              "unfinished_thinking": sum(not c for _, c in split), "max_new_tokens": cap, "batch": int(batch),
              "generated_tokens": stats["generated_tokens"], "mean_generated_tokens": stats["generated_tokens"] / total,
              "generation_seconds": round(stats["seconds"], 3), "tokens_per_second": stats["tokens_per_second"],
              "decode_steps": stats["decode_steps"], "cached_batches": stats["cached_batches"],
              "batches": stats["batches"], "manifest": info, "chat_kwargs": chat_kwargs,
              "chat_date": "%04d-%02d-%02d" % CHAT_DATE, "decoding": DECODING, "runtime": stats["runtime"],
              **({"batch_check": stats["batch_check"]} if batch_check else {})}
    if name == "gsm8k":
        correct = sum(r["correct"] for r in records)
        methods = {}
        for r in records:
            methods[r["method"]] = methods.get(r["method"], 0) + 1
        summary = {"suite": name, "score": f"{correct}/{total}", "correct": correct, "accuracy": correct / total,
                   **common, "methods": methods, "instruction": GSM8K_INSTRUCTION}
    else:
        counts = {s: sum(r["status"] == s for r in records) for s in ("passed", "failed", "timeout")}
        summary = {"suite": name, "score": f"{counts['passed']}/{total}", "pass@1": counts["passed"] / total,
                   **counts, **common, "timeout_seconds": float(timeout),
                   "memory_limit_bytes": memory_limit, "dropped_statements": sum(r["dropped_statements"] for r in records),
                   "instruction": HUMANEVAL_INSTRUCTION}
    summary["problems"] = records
    return summary


# -- freezing the assets -----------------------------------------------------------

def freeze(out_dir=ASSETS, *, paths=None, local_files_only: bool = False) -> dict:
    """Write the frozen manifests (``FROZEN`` sizes, ``SEED``) to ``out_dir``; returns {name: file SHA256}."""
    out = {}
    for name, f in FROZEN.items():
        data = manifest_bytes(build_manifest(name, f["n"], path=(paths or {}).get(name),
                                             local_files_only=local_files_only))
        atomic_write_bytes(Path(out_dir) / f["file"], data)
        out[name] = hashlib.sha256(data).hexdigest()
    return out


def check_frozen(*, paths=None, local_files_only: bool = False) -> dict:
    """Rebuild every frozen manifest from the dataset and compare it byte for byte with the asset."""
    out = {}
    for name, f in FROZEN.items():
        data = manifest_bytes(build_manifest(name, f["n"], path=(paths or {}).get(name),
                                             local_files_only=local_files_only))
        out[name] = {"identical": data == (ASSETS / f["file"]).read_bytes(),
                     "sha256": hashlib.sha256(data).hexdigest(), "pinned": f["sha256"]}
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m openbonsai.eval.reasoning",
                                 description="Freeze or check the GSM8K/HumanEval test-subset manifests.")
    ap.add_argument("action", choices=("check", "freeze"))
    ap.add_argument("--gsm8k", help="local GSM8K main test parquet (default: the pinned hub file)")
    ap.add_argument("--humaneval", help="local HumanEval test parquet (default: the pinned hub file)")
    ap.add_argument("--out", default=str(ASSETS), help="freeze: output directory")
    ap.add_argument("--local-files-only", action="store_true", help="never download")
    args = ap.parse_args(argv)
    paths = {"gsm8k": args.gsm8k, "humaneval": args.humaneval}
    if args.action == "freeze":
        result = freeze(args.out, paths=paths, local_files_only=args.local_files_only)
    else:
        result = check_frozen(paths=paths, local_files_only=args.local_files_only)
    print(json.dumps(result, indent=1))
    return 0 if args.action == "freeze" or all(r["identical"] for r in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

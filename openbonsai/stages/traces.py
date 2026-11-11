"""Self-generated traces: the original model answers prompts with its own chat template.

The traces become calibration and distillation data, so the quantized model
is fitted on the distribution it actually produces: its thinking and answers
to math and code problems as much as its chat (calibration on generic text
alone tends to keep perplexity but lose reasoning).

Two config forms
  ``traces.prompts`` (one source; the original form, unchanged): kind
  ``jsonl`` (``path``; the first ``n`` lines) or ``ultrachat`` (first user
  turns of rows ``rows``, default [5000, 6000), seeded sample of ``n``), all
  answered with ``traces.thinking``, ``traces.max_new_tokens`` and
  ``traces.batch``, in prompt order.

  ``traces.sources`` (a list; replaces ``prompts``): every entry is a prompt
  source with its own ``n`` (required), ``seed``, ``thinking`` (on/off),
  ``max_new_tokens`` and ``batch`` (unset: the ``traces.*`` values), and an
  optional ``name`` (default: its kind) and ``instruction`` (text appended to
  the prompt on a new line; ``null`` or ``""`` for none). Kinds:

  ==================  ===========================================================
  ultrachat           first user turn of UltraChat rows ``rows`` (default [5000, 6000))
  gsm8k_train         openai/gsm8k main TRAIN ``question``; instruction default:
                      the GSM8K suite's boxed-answer line (Qwen's math prompt)
  math_train          EleutherAI/hendrycks_math TRAIN ``problem``, ``subjects``
                      (default all 7), optional ``levels`` ([1, 2, 3] or "Level 3");
                      same default instruction
  mbpp_train          google-research-datasets/mbpp full TRAIN: ``text`` plus its
                      first test as a format hint (``MBPP_TEMPLATE``)
  opencodeinstruct    nvidia/OpenCodeInstruct first shard ``input``; inputs longer
                      than ``max_chars`` (default 2000) are skipped
  parquet, jsonl      generic: a ``column`` of a local ``path`` or of a hub
                      ``repo``/``revision``/``file`` (parquet) or of a local JSONL;
                      jsonl without ``column`` reads prompt, question or messages
  ==================  ===========================================================

  The hub kinds read files pinned by revision (``HUB``) through the Hugging
  Face cache (HF_HUB_OFFLINE works; ``model.local_files_only`` never
  downloads); ``path`` reads a local copy instead (math_train: a folder with
  the hub's <subject>/train-... layout, or one parquet). ``max_chars`` skips
  longer prompts for every kind, ``shuffle: false`` takes rows in file order.
  A source draws ``n`` distinct prompts (duplicates by normalized text are
  dropped) in a seeded order, so a larger ``n`` extends a smaller one.

Test data stays out (openbonsai.eval.heldout): a source naming a GSM8K or
HumanEval test split, or a local copy of one, is refused, and so is any
prompt that contains a test problem or one of the extended suite's chat-NLL
prompts. The built-in kinds refuse a ``file`` that is not a training split.
With ``sources``, rows that share a 13-word run with a test problem (near
duplicates such as GSM8K train questions written from a test question's
template) are skipped while sampling and listed in ``summary.json``. A generated
answer that repeats a test problem or a chat-NLL item (its prompt did not) is
dropped and listed (``answers_dropped``): calibrate and the distillation
teacher would refuse the blocks that hold it, hours after the traces.

Generation is greedy (``generate_encoded``), left-padded, with the model's own
chat template and thinking switch; the stored ids are exactly the rendered
prompt plus what the model generated (its thinking block too, with thinking
on), through the first EOS or the cap. With ``sources``, each source's prompts
are sorted by length (longest first, so a batch that does not fit in memory
fails first) and run in batches of its ``batch``. ``traces.batch_tokens`` (both
forms; null: no limit) also caps a batch's tokens in flight, rows x (its longest
prompt + the cap): what the left-padded KV cache holds at the last step, so a
few long prompts (UltraChat has some of 2,000 tokens) get smaller batches
instead of a batch that does not fit. On Windows the driver can page GPU
memory to system memory instead of raising out-of-memory (the CUDA sysmem
fallback), and generation then crawls instead of failing and splitting; so
from the first generated batch the stage caps its CUDA allocator at the VRAM
free when the model loads (``common.cap_vram``, lifted when it ends), which
also makes the allocator return the blocks a growing KV cache leaves behind
before it would pass the cap. Each batch is cached in ``traces/cache/`` under a hash of its prompt ids, cap,
decoding, the model's identity and the runtime (torch and transformers
versions, device), so an
interrupted run resumes after the last finished batch, and a config edit
regenerates only batches whose prompts or cap changed (``--force`` clears the
cache). A batch that runs out of GPU memory is split in two and retried.

Output: ``traces/traces.jsonl``, one record per trace: ``input_ids`` (prompt +
generated ids), ``attention_mask``, ``source`` (name), ``kind``, ``row`` (row in
its file; ``id`` too when the data has one), ``thinking``, ``max_new_tokens``,
``hit_cap``, ``prompt_token_count``, ``generated_token_count``,
``finished_thinking`` (thinking on: the generation closed its thinking),
``prompt`` (user text) and ``completion`` (the generated ids decoded, special
tokens kept). With ``sources`` the records are interleaved in an order seeded
by ``traces.seed``, so any prefix of the file (what calibrate and the
distillation teacher read) holds every source in proportion; with ``prompts``
they stay in prompt order. ``traces/summary.json`` has counts per source.

Calibrate and the teacher concatenate whole records (template tokens, the
thinking block and the closing EOS included) and cut blocks of ``seq_len``
from that stream, as for UltraChat: a record that crosses a block boundary
continues at the start of the next block without its prompt (with thinking
traces of ~1,500 tokens and 2,048-token blocks, about half of the records,
a third of the tokens).

Trace-backed blocks are never capped (``trace_demands``): calibrate and the
distillation teacher read ``n`` blocks from the start of traces.jsonl and fail
if it is shorter, since a silent cap would change the data mix, calibrate's
multiple of quantize.batch and the distillation schedule. It is checked as
early as it can be: before generating (when even traces that all ran to
their caps could not supply them), and in ``openbonsai run`` before the first
stage when the traces are current, else right after the traces stage, for
every later stage the run includes (a run stops before quantize, not at
distill). The traces stage itself only logs the supply and a warning, since
its output is complete either way.
"""
from __future__ import annotations

from contextlib import ExitStack
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from ..formats.atomic import atomic_write_bytes, atomic_write_json
from .common import vram_cap

# Training splits at pinned revisions (dataset repos). math_train's file is per subject.
HUB = {
    "gsm8k_train": {"repo": "openai/gsm8k", "revision": "740312add88f781978c0658806c59bc2815b9866",
                    "file": "main/train-00000-of-00001.parquet"},
    "math_train": {"repo": "EleutherAI/hendrycks_math", "revision": "21a5633873b6a120296cce3e2df9d5550074f4a3",
                   "file": "{subject}/train-00000-of-00001.parquet"},
    "mbpp_train": {"repo": "google-research-datasets/mbpp", "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
                   "file": "full/train-00000-of-00001.parquet"},
    "opencodeinstruct": {"repo": "nvidia/OpenCodeInstruct", "revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d",
                         "file": "data/train-00000-of-00050.parquet"},
}
MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                 "prealgebra", "precalculus")
# The GSM8K suite's instruction (eval.reasoning.GSM8K_INSTRUCTION), Qwen's recommended math prompt.
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
# MBPP's own prompt (Austin et al. 2021) with one of its tests as the format hint.
MBPP_TEMPLATE = "You are an expert Python programmer, and here is your task: {text} Your code should pass this test:\n\n{test}"
DEFAULT_INSTRUCTION = {"gsm8k_train": MATH_INSTRUCTION, "math_train": MATH_INSTRUCTION}
DEFAULT_MAX_CHARS = {"opencodeinstruct": 2000}
DEFAULT_SEED = {"ultrachat": 5090}
COLUMNS = {"gsm8k_train": "question", "math_train": "problem", "opencodeinstruct": "input"}
TRAIN_KINDS = ("gsm8k_train", "math_train", "mbpp_train", "opencodeinstruct")


# -- prompts -----------------------------------------------------------------------

def _record_messages(record, location: str) -> list[dict]:
    """A JSONL prompt record (``prompt``, ``question`` or ``messages``) as chat messages."""
    if not isinstance(record, dict):
        raise ValueError(f"{location}: expected a JSON object")
    if "messages" in record:
        msgs = record["messages"]
        if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) for m in msgs):
            raise ValueError(f"{location}: messages must be a nonempty list of objects")
        return msgs
    if "prompt" in record and "question" in record and record["prompt"] != record["question"]:
        raise ValueError(f"{location}: conflicting prompt and question fields")
    prompt = record.get("prompt", record.get("question"))
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{location}: prompt or question must be a nonempty string")
    return [{"role": "user", "content": prompt}]


def _jsonl_records(path):
    """(line number, object) of every nonempty line."""
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            yield line_number, json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error


def _first_prompts(path, max_samples: int | None = None) -> list[tuple[int, list[dict]]]:
    """(line number, chat messages) of the first ``max_samples`` JSONL prompt records."""
    prompts = []
    for line_number, record in _jsonl_records(path):
        if max_samples is not None and len(prompts) >= max_samples:
            break
        prompts.append((line_number, _record_messages(record, f"{path}:{line_number}")))
    if not prompts:
        raise ValueError(f"{path}: no prompts found")
    return prompts


def read_prompts(path, max_samples: int | None = None) -> list[list[dict]]:
    """JSONL prompts as chat message lists (the first ``max_samples``)."""
    return [msgs for _, msgs in _first_prompts(path, max_samples)]


def _first_user(msgs) -> str | None:
    if isinstance(msgs, str):
        return msgs
    if isinstance(msgs, (list, tuple)):
        return next((m.get("content") for m in msgs if isinstance(m, dict) and m.get("role") == "user"), None)
    return None


def _normalize(text) -> str:
    return " ".join(str(text).split())


def _user_text(msgs: list[dict]) -> str:
    return "\n".join(str(m.get("content", "")) for m in msgs)


def ultrachat_prompts(path, *, rows=(5000, 6000), n=128, seed=5090) -> list[list[dict]]:
    """First user message of UltraChat rows [rows[0], rows[1]), deduplicated, seeded sample."""
    spec = {"kind": "ultrachat", "rows": list(rows), "n": n, "seed": seed, "shuffle": True, "instruction": None,
            "max_chars": None, "where": "ultrachat"}
    return [msgs for _, msgs, _ in _sample(spec, _candidates(spec, _ultrachat_rows(path, rows)), screen=False)[0]]


def _ultrachat_rows(path, rows):
    import pyarrow.parquet as pq
    a, b = int(rows[0]), int(rows[1])
    table = pq.read_table(path, columns=["messages"]).column("messages").to_pylist()[a:b]
    return [(a + k, _first_user(msgs), None) for k, msgs in enumerate(table)]


# Every candidate is (row, prompt text before the instruction, dataset id or None).

def _parquet_rows(path, column: str, *, ids: str | None = None, prefix: str = ""):
    import pyarrow.parquet as pq
    names = pq.ParquetFile(path).schema_arrow.names
    if column not in names:
        raise ValueError(f"{path}: no column {column!r}; columns: {names}")
    cols = [column] + ([ids] if ids and ids in names else [])
    table = pq.read_table(path, columns=cols)
    values = table.column(column).to_pylist()
    keys = table.column(ids).to_pylist() if len(cols) > 1 else [None] * len(values)
    return [(f"{prefix}{r}" if prefix else r, _first_user(v), k) for r, (v, k) in enumerate(zip(values, keys))]


def _math_rows(src, paths: dict[str, Path]):
    import pyarrow.parquet as pq
    levels = _levels(src.get("levels"), src["where"])
    out = []
    for subject, path in paths.items():
        table = pq.read_table(path, columns=["problem", "level"]).to_pylist()
        prefix = f"{subject}/" if subject else ""
        for r, rec in enumerate(table):
            if levels is None or str(rec.get("level")) in levels:
                out.append((f"{prefix}{r}", rec["problem"], None))
    return out


def _mbpp_rows(path):
    import pyarrow.parquet as pq
    out = []
    for r, rec in enumerate(pq.read_table(path, columns=["task_id", "text", "test_list", "test_setup_code"]).to_pylist()):
        tests = rec.get("test_list") or []
        if not rec.get("text") or not tests:
            continue
        setup = (rec.get("test_setup_code") or "").strip()
        test = (setup + "\n" if setup else "") + str(tests[0]).strip()
        out.append((r, MBPP_TEMPLATE.format(text=str(rec["text"]).strip(), test=test), rec.get("task_id")))
    return out


def _jsonl_rows(path, column: str | None):
    out = []
    for line_number, record in _jsonl_records(path):
        location = f"{path}:{line_number}"
        if column is None:
            msgs = _record_messages(record, location)
            out.append((line_number, msgs, None))
        else:
            if not isinstance(record, dict) or column not in record:
                raise ValueError(f"{location}: no field {column!r}")
            out.append((line_number, _first_user(record[column]), None))
    return out


def _levels(value, where: str) -> set[str] | None:
    from ..config import ConfigError
    if value is None:
        return None
    out = set()
    for v in value if isinstance(value, (list, tuple)) else [value]:
        s = str(v).strip()
        s = s if s.lower().startswith("level") else f"Level {s}"
        if s not in {f"Level {k}" for k in range(1, 6)}:
            raise ConfigError(f"{where}.levels: {v!r} is not a MATH level (1 to 5)")
        out.add(s)
    return out


def _hub_file(src: dict, file: str, *, local_files_only: bool) -> tuple[Path, dict]:
    """The hub file of a source (``repo`` overrides the kind's pinned repo and revision)."""
    from huggingface_hub import hf_hub_download
    hub = HUB.get(src["kind"], {})
    repo = src.get("repo") or hub.get("repo")
    revision = src.get("revision") if src.get("repo") else (src.get("revision") or hub.get("revision"))
    path = Path(hf_hub_download(repo, file, revision=revision, repo_type="dataset",
                                local_files_only=local_files_only))
    return path, {"repo": repo, "revision": revision, "file": file}


def _source_files(src: dict, *, local_files_only: bool) -> tuple[dict, list[dict]]:
    """{label: local file} of a source and what each one is (repo, revision, file or path, SHA-256)."""
    from ..digest import file_sha256
    from ..eval import heldout
    from .calibrate import resolve_file
    kind, where = src["kind"], src["where"]
    if kind == "jsonl":
        files = {"": Path(src["path"])}
        info = [{"path": str(src["path"])}]
    elif kind == "ultrachat":
        files = {"": resolve_file({"kind": "ultrachat", **{k: src.get(k) for k in ("path", "repo", "revision", "file")}},
                                  local_files_only=local_files_only)}
        info = [{"path": str(files[""])} if src.get("path") else _ultrachat_hub(src)]
    elif kind == "math_train" and src.get("path") and Path(src["path"]).is_dir():
        files = {s: Path(src["path"]) / s / "train-00000-of-00001.parquet" for s in _subjects(src, src["path"])}
        info = [{"path": str(p)} for p in files.values()]
    elif src.get("path"):
        files = {"": Path(src["path"])}
        info = [{"path": str(src["path"])}]
    elif kind == "math_train":
        files, info = {}, []
        for s in _subjects(src):
            files[s], meta = _hub_file(src, HUB["math_train"]["file"].format(subject=s),
                                       local_files_only=local_files_only)
            info.append(meta)
    else:
        file = src.get("file") or HUB.get(kind, {}).get("file")
        files, info = {}, []
        files[""], meta = _hub_file(src, file, local_files_only=local_files_only)
        info.append(meta)
    for (label, path), meta in zip(files.items(), info):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{where}: {path} not found")
        heldout.check_source({}, where, path=path)       # a local copy of a test split
        meta["sha256"] = file_sha256(path)
    return files, info


def _ultrachat_hub(src: dict) -> dict:
    """The hub file an ultrachat source reads (``calibrate.resolve_file``'s rule)."""
    from .calibrate import HUB as CALIB_HUB
    return {**CALIB_HUB["ultrachat"], **{k: src[k] for k in ("repo", "revision", "file") if src.get(k)}}


def _subjects(src: dict, folder=None) -> list[str]:
    """math_train's subjects: ``subjects``, else all seven (those a local ``folder`` holds)."""
    from ..config import ConfigError
    subjects = src.get("subjects")
    if not subjects:
        subjects = [s for s in MATH_SUBJECTS if folder is None
                    or (Path(folder) / s / "train-00000-of-00001.parquet").is_file()]
        if not subjects:
            raise ConfigError(f"{src['where']}: {folder} holds no <subject>/train-00000-of-00001.parquet")
    subjects = [subjects] if isinstance(subjects, str) else list(subjects)
    unknown = sorted(set(subjects) - set(MATH_SUBJECTS))
    if unknown:
        raise ConfigError(f"{src['where']}.subjects: unknown {unknown}; known: {list(MATH_SUBJECTS)}")
    return subjects


def _rows(src: dict, files: dict) -> list:
    kind = src["kind"]
    path = next(iter(files.values()))
    if kind == "ultrachat":
        return _ultrachat_rows(path, src.get("rows") or (5000, 6000))
    if kind == "math_train":
        return _math_rows(src, files)
    if kind == "mbpp_train":
        return _mbpp_rows(path)
    if kind == "jsonl":
        return _jsonl_rows(path, src.get("column"))
    column = src.get("column") or COLUMNS[kind]
    return _parquet_rows(path, column, ids="id" if kind == "opencodeinstruct" else None)


def _candidates(src: dict, rows) -> list[tuple[Any, list[dict], Any]]:
    """(row, messages, id) of every usable row in file order: nonempty, at most ``max_chars``, the
    instruction appended, the first of rows with the same normalized text."""
    instruction = src.get("instruction")
    max_chars = src.get("max_chars")
    seen, out = set(), []
    for row, value, ident in rows:
        if isinstance(value, list):        # a JSONL messages record, used as written
            msgs = value
        else:
            if not isinstance(value, str) or not value.strip():
                continue
            text = f"{value}\n{instruction}" if instruction else value
            msgs = [{"role": "user", "content": text}]
        text = _user_text(msgs)
        if max_chars is not None and len(text) > int(max_chars):
            continue
        key = _normalize(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((row, msgs, ident))
    return out


def _sample(src: dict, candidates, *, screen: bool):
    """``n`` candidates in a seeded order (file order with ``shuffle: false``), skipping near duplicates
    of test problems when ``screen``; returns (picked, screened [{row, problem}])."""
    from ..eval import heldout
    order = list(candidates)
    if src.get("shuffle", True):
        random.Random(int(src["seed"])).shuffle(order)
    n = int(src["n"]) if src.get("n") is not None else len(order)
    picked, screened = [], []
    for cand in order:
        if len(picked) >= n:
            break
        if screen:
            problem = heldout.near_duplicate(_user_text(cand[1]), src["where"])
            if problem is not None:
                screened.append({"row": cand[0], "problem": problem})
                continue
        picked.append(cand)
    if len(picked) < n:
        what = "distinct UltraChat prompts in rows " + str(tuple(src.get("rows") or (5000, 6000))) \
            if src["kind"] == "ultrachat" else "usable prompts"
        raise ValueError(f"{src['where']}: only {len(picked)} {what}, {n} requested"
                         + (f" ({len(screened)} near duplicates of test problems skipped)" if screened else ""))
    return picked, screened


def _thinking(value, where: str) -> bool:
    from ..config import ConfigError
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("on", "true", "yes", "1"):
        return True
    if s in ("off", "false", "no", "0", "none"):
        return False
    raise ConfigError(f"{where}: thinking must be on or off, not {value!r}")


def _positive(value, where: str) -> int:
    from ..config import ConfigError
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = 0
    if out < 1 or (isinstance(value, float) and value != out):
        raise ConfigError(f"{where} must be a positive integer, not {value!r}")
    return out


def trace_sources(cfg: dict) -> list[dict]:
    """The prompt sources of the traces stage with every default filled in (``traces.prompts`` as one
    legacy source); each has ``where`` (its config path), ``name``, ``index`` and ``legacy``."""
    from ..config import DEFAULTS, ConfigError
    t = cfg["traces"]
    common = {"thinking": t.get("thinking", False), "max_new_tokens": t.get("max_new_tokens"),
              "batch": t.get("batch") or 8}
    budget = None if t.get("batch_tokens") is None else _positive(t["batch_tokens"], "traces.batch_tokens")
    if t.get("sources") is None:
        spec = dict(t.get("prompts") or {})
        kind = spec.get("kind", "jsonl")
        if kind not in ("jsonl", "ultrachat"):
            raise ConfigError(f"unknown traces.prompts.kind {kind!r}; known: jsonl, ultrachat "
                              "(traces.sources has more kinds)")
        if kind == "jsonl" and not spec.get("path"):
            raise ConfigError("traces.prompts.path is required for kind jsonl")
        src = {**spec, "kind": kind, "where": "traces.prompts", "name": kind, "index": 0, "legacy": True,
               "shuffle": kind == "ultrachat", "instruction": None, "max_chars": None,
               "seed": int(spec.get("seed", 5090)),
               "n": spec.get("n") if kind == "jsonl" else int(spec.get("n") or 128),
               "thinking": _thinking(common["thinking"], "traces.thinking"),
               "max_new_tokens": _positive(common["max_new_tokens"], "traces.max_new_tokens"),
               "batch": _positive(common["batch"], "traces.batch"), "batch_tokens": budget}
        return [src]
    if (t.get("prompts") or DEFAULTS["traces"]["prompts"]) != DEFAULTS["traces"]["prompts"]:
        raise ConfigError("set traces.sources or traces.prompts, not both: traces.sources replaces prompts "
                          "(--set traces.prompts=null)")
    out, names = [], set()
    for i, s in enumerate(t["sources"]):
        where = f"traces.sources.{i}"
        kind = s["kind"]
        src = {k: v for k, v in s.items() if v is not None or k in ("instruction", "max_chars")}
        src.update(where=where, index=i, legacy=False)
        for key in ("thinking", "max_new_tokens", "batch"):
            if src.get(key) is None:
                src[key] = common[key]
        src["thinking"] = _thinking(src["thinking"], f"{where}.thinking")
        src["n"] = _positive(s.get("n"), f"{where}.n")
        src["max_new_tokens"] = _positive(src["max_new_tokens"], f"{where}.max_new_tokens")
        src["batch"] = _positive(src["batch"], f"{where}.batch")
        src["batch_tokens"] = budget
        src["seed"] = int(s["seed"]) if s.get("seed") is not None else DEFAULT_SEED.get(kind, 0)
        src["shuffle"] = s.get("shuffle", True) is not False
        if "instruction" not in s:
            src["instruction"] = DEFAULT_INSTRUCTION.get(kind)
        src["instruction"] = src["instruction"] or None
        if "max_chars" not in s:
            src["max_chars"] = DEFAULT_MAX_CHARS.get(kind)
        if src["max_chars"] is not None:
            src["max_chars"] = _positive(src["max_chars"], f"{where}.max_chars")
        name = str(s.get("name") or (kind if kind not in names else f"{kind}.{i}"))
        if name in names:
            raise ConfigError(f"{where}.name {name!r} is used by another source")
        names.add(name)
        src["name"] = name
        if kind == "jsonl" and not s.get("path"):
            raise ConfigError(f"{where}: kind jsonl needs a path")
        if kind == "parquet" and not (s.get("path") or (s.get("repo") and s.get("file"))):
            raise ConfigError(f"{where}: kind parquet needs a path, or a repo and a file")
        if kind == "parquet" and not s.get("column"):
            raise ConfigError(f"{where}: kind parquet needs a column")
        if kind in TRAIN_KINDS and s.get("file"):
            from ..eval import heldout
            heldout.check_source({"repo": s.get("repo") or HUB[kind]["repo"], "file": s["file"]}, where)
            if "train" not in Path(str(s["file"])).name.lower():
                raise ConfigError(f"{where}: {kind} reads a training split; {s['file']} is not one")
        if kind == "math_train":
            _subjects(src)
            _levels(src.get("levels"), where)
            if s.get("file"):
                raise ConfigError(f"{where}: math_train reads <subject>/train-00000-of-00001.parquet per "
                                  "subject; give a path for another file")
        if kind == "ultrachat":
            rows = src.get("rows") or (5000, 6000)
            if not (isinstance(rows, (list, tuple)) and len(rows) == 2 and 0 <= int(rows[0]) < int(rows[1])):
                raise ConfigError(f"{where}.rows must be [first, end) row numbers, not {rows!r}")
        out.append(src)
    return out


def load_sources(ctx, sources: list[dict] | None = None) -> list[tuple[dict, list, dict]]:
    """(source, picked [(row, messages, id)], data info) for every trace source.

    Refuses test splits by name or file bytes, and prompts that contain a test problem or one of the
    extended suite's chat-NLL prompts or answers (eval.heldout); ``sources`` entries also skip near
    duplicates of test problems while sampling."""
    from ..eval import heldout
    sources = trace_sources(ctx.cfg) if sources is None else sources
    local = bool(ctx.cfg["model"].get("local_files_only"))
    probes = heldout.needles() + heldout.chat_needles(ctx.load_tokenizer())
    out = []
    for src in sources:
        where = src["where"]
        hub = {} if src.get("path") else (_ultrachat_hub(src) if src["kind"] == "ultrachat" else
                                         {"repo": src.get("repo") or HUB.get(src["kind"], {}).get("repo"),
                                          "file": src.get("file") or HUB.get(src["kind"], {}).get("file")})
        heldout.check_source({k: v for k, v in hub.items() if k in ("repo", "file")}, where)
        files, info = _source_files(src, local_files_only=local)
        if src["legacy"] and src["kind"] == "jsonl":
            picked = [(line, msgs, None) for line, msgs in _first_prompts(src["path"], src.get("n"))]
            screened = []
        else:
            picked, screened = _sample(src, _candidates(src, _rows(src, files)), screen=not src["legacy"])
        heldout.check_texts([_user_text(msgs) for _, msgs, _ in picked], where, probes)
        out.append((src, picked, {"files": info, "screened": screened, "parts": [k for k in files if k]}))
    return out


def load_prompts(ctx) -> list[list[dict]]:
    """The trace prompts of every source as chat messages, after the test-data checks (``load_sources``)."""
    return [msgs for _, picked, _ in load_sources(ctx) for _, msgs, _ in picked]


# -- generation ----------------------------------------------------------------------

def encode_chat(tokenizer, msgs: list[dict], chat_kwargs: dict) -> list[int]:
    """Token ids of ``msgs`` rendered by the model's own chat template with the generation prompt
    (and the fixed date of ``adapters.base.template_vars``)."""
    from ..adapters.base import template_vars
    ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False,
                                        **template_vars(chat_kwargs))
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    ids = [int(t) for t in (ids[0] if ids and isinstance(ids[0], list) else ids)]
    if not ids:
        raise ValueError("chat template produced no tokens")
    return ids


# What generate_encoded does, for result files and cache keys (bump when the decoding changes).
DECODING = {"strategy": "greedy", "num_beams": 1, "kv_cache": True,
            "model_generation_config": "ignored: transformers' library defaults (no repetition penalty, "
                                       "n-gram blocking, suppressed or bad tokens, minimum length)"}


def greedy_config(*, max_new_tokens: int, eos_ids, pad_id: int):
    """The complete GenerationConfig of a greedy generation."""
    from transformers import GenerationConfig
    return GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=int(max_new_tokens),
                            eos_token_id=[int(e) for e in eos_ids], pad_token_id=int(pad_id), use_cache=True)


@torch.no_grad()
def generate_encoded(model, encoded: list[list[int]], *, eos_ids: list[int], max_new_tokens: int, device,
                     pad_id: int) -> list[tuple[list[int], bool]]:
    """Greedy, left-padded batch generation with the KV cache: plain argmax decoding.

    ``generate`` fills every option a call leaves unset from ``model.generation_config``, i.e. a
    checkpoint's ``generation_config.json`` (``repetition_penalty``, ``no_repeat_ngram_size``,
    ``bad_words_ids``, ``suppress_tokens``, ``min_length`` ...), which a model built from a GGUF
    does not have. So the call runs with the library defaults in its place: the same decoding for
    every model, however it was built.

    Returns per prompt (generated ids through the first EOS, hit ``max_new_tokens``)."""
    from transformers import GenerationConfig
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    width = max(len(ids) for ids in encoded)
    input_ids = torch.tensor([[pad_id] * (width - len(ids)) + ids for ids in encoded], device=device)
    mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in encoded], device=device)
    config = greedy_config(max_new_tokens=max_new_tokens, eos_ids=eos_ids, pad_id=pad_id)
    saved = model.generation_config
    model.generation_config = GenerationConfig()
    try:
        out = model.generate(input_ids=input_ids, attention_mask=mask, generation_config=config)
    finally:
        model.generation_config = saved
    eos = set(eos_ids)
    results = []
    for row in range(len(encoded)):
        gen = out[row, width:].tolist()
        stop = next((k for k, t in enumerate(gen) if t in eos), None)
        gen = gen[:stop + 1] if stop is not None else gen
        if not gen:
            raise ValueError("model generated no tokens")
        results.append((gen, stop is None))
    return results


# The conversation checked_chat_kwargs renders in both thinking modes.
THINKING_PROBE = [{"role": "user", "content": "What is 2 + 2?"}]


def checked_chat_kwargs(adapter, tokenizer, thinking: bool, where: str) -> dict:
    """``adapter.chat_kwargs``; a thinking mode the chat template has no switch for is a ConfigError, and so
    is thinking on when the tokenizer's chat template renders it as the same prompt as thinking off (a
    template that ignores the switch the model family has, e.g. one that always closes an empty think
    block). Thinking off passes, as for a template without the switch; a tokenizer without a chat template
    is left to the stages (traces refuses it, eval skips its reasoning suites)."""
    from ..config import ConfigError
    try:
        kwargs = adapter.chat_kwargs(thinking=bool(thinking))
    except ValueError as error:
        raise ConfigError(f"{where}: {error}") from None
    if thinking and getattr(tokenizer, "chat_template", None):
        off = adapter.chat_kwargs(thinking=False)
        if encode_chat(tokenizer, THINKING_PROBE, kwargs) == encode_chat(tokenizer, THINKING_PROBE, off):
            switch = ", ".join(sorted(kwargs)) or "a thinking switch"
            raise ConfigError(f"{where}: the model's chat template ignores {switch}: thinking on renders the same "
                              "prompt as thinking off; use thinking off")
    return kwargs


def pad_token(tokenizer, eos_ids) -> int:
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else int(eos_ids[0])


def generate_batch(model, tokenizer, prompts: list[list[dict]], *, chat_kwargs: dict, eos_ids: list[int],
                   max_new_tokens: int, device, pad_id: int | None = None) -> list[dict[str, Any]]:
    """Greedy, left-padded batch generation; returns one record per prompt."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    encoded = [encode_chat(tokenizer, msgs, chat_kwargs) for msgs in prompts]
    if pad_id is None:
        pad_id = pad_token(tokenizer, eos_ids)
    generated = generate_encoded(model, encoded, eos_ids=eos_ids, max_new_tokens=max_new_tokens, device=device,
                                 pad_id=pad_id)
    return [_record(ids, gen, hit, msgs) for msgs, ids, (gen, hit) in zip(prompts, encoded, generated)]


def _record(ids: list[int], gen: list[int], hit: bool, msgs: list[dict], **meta) -> dict:
    return {**meta, "hit_cap": bool(hit), "prompt_token_count": len(ids), "generated_token_count": len(gen),
            "prompt": next((m.get("content") for m in msgs if m.get("role") == "user"), None),
            "input_ids": ids + gen, "attention_mask": [1] * (len(ids) + len(gen))}


def plan_batches(lengths: list[int], batch: int, *, sort: bool, cap: int = 0,
                 budget: int | None = None) -> list[list[int]]:
    """Row indices per batch: longest prompts first (ties in row order) when ``sort``, else in order.

    A batch holds at most ``batch`` rows and, with a ``budget``, at most ``budget`` tokens in flight:
    rows x (its longest prompt + ``cap``), what the left-padded KV cache holds at the last step (a row
    over the budget on its own is a batch of one)."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)) if sort else list(range(len(lengths)))
    out: list[list[int]] = []
    for i in order:
        cur = out[-1] if out else None
        if cur is not None and len(cur) < batch and (
                budget is None or (len(cur) + 1) * (max(max(lengths[k] for k in cur), lengths[i]) + cap) <= budget):
            cur.append(i)
        else:
            out.append([i])
    return out


_OOM = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


def runtime_identity(ctx) -> dict:
    """What, besides the prompts, cap and decoding, decides the generated tokens (cache keys)."""
    import transformers
    from ..config import stage_inputs
    device = ctx.device
    return {"model": stage_inputs(ctx.cfg, "traces")["model"], "model_sha256": ctx.model_digest(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "device": device.type + (f":{torch.cuda.get_device_name(device)}" if device.type == "cuda" else "")}


class BatchCache:
    """Generated batches in ``cache/<key>.json``; ``<key>.split`` marks a batch that ran out of memory
    and was generated as two halves."""

    def __init__(self, root: Path, identity: dict):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity

    def key(self, rows: list[list[int]], cap: int) -> str:
        from ..config import canonical_hash
        return canonical_hash({"identity": self.identity, "decoding": DECODING, "max_new_tokens": int(cap),
                               "rows": rows})[:24]

    def get(self, rows, cap, generate, log=None) -> tuple[list[tuple[list[int], bool]], dict]:
        """Generated (ids, hit cap) per row, from the cache or ``generate(rows, cap)``; and
        {"cached": rows from the cache, "seconds", "tokens": generated now}."""
        key = self.key(rows, cap)
        path = self.root / f"{key}.json"
        if path.exists():
            got = json.loads(path.read_text(encoding="utf-8"))
            if got.get("prompt_tokens") == [len(r) for r in rows]:
                return [(g, bool(h)) for g, h in zip(got["generated"], got["hit_cap"])], \
                    {"cached": len(rows), "seconds": 0.0, "tokens": 0}
        marker = self.root / f"{key}.split"
        out = None
        if not (len(rows) > 1 and marker.exists()):
            t0 = time.monotonic()
            try:
                out = generate(rows, cap)
            except _OOM:
                if len(rows) == 1:
                    raise
            if out is not None:
                seconds = time.monotonic() - t0
                atomic_write_json(path, {"prompt_tokens": [len(r) for r in rows], "max_new_tokens": int(cap),
                                         "seconds": round(seconds, 3), "steps": max(len(g) for g, _ in out),
                                         "generated": [g for g, _ in out], "hit_cap": [bool(h) for _, h in out]})
                return out, {"cached": 0, "seconds": seconds, "tokens": sum(len(g) for g, _ in out)}
            from .common import free
            free(None)
            atomic_write_json(marker, {"rows": len(rows), "max_new_tokens": int(cap)})
            if log is not None:
                log(f"out of memory at batch {len(rows)}; generating it as two halves")
        half = (len(rows) + 1) // 2
        a, sa = self.get(rows[:half], cap, generate, log)
        b, sb = self.get(rows[half:], cap, generate, log)
        return a + b, {k: sa[k] + sb[k] for k in sa}


def _clock(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


# -- trace supply (calibrate and the distillation teacher read blocks from traces.jsonl) ---------------

def trace_demands(cfg: dict, stages=("calibrate", "distill")) -> list[dict]:
    """Every block count that reads traces.jsonl: {stage, where, n, seq_len}."""
    from ..config import stage_enabled
    out = []
    if "calibrate" in stages and stage_enabled(cfg, "calibrate"):
        seq = int(cfg["calibrate"]["seq_len"])
        for i, s in enumerate(cfg["calibrate"].get("sources") or ()):
            if s.get("kind") == "traces" and not s.get("path") and int(s.get("n") or 0) > 0:
                out.append({"stage": "calibrate", "where": f"calibrate.sources.{i}.n", "n": int(s["n"]),
                            "seq_len": seq})
    if "distill" in stages and stage_enabled(cfg, "distill"):
        from .distill import teacher_sources
        t = cfg["distill"]["teacher"]
        explicit = bool(t.get("sources"))
        for i, s in enumerate(teacher_sources(cfg)):
            if s.get("kind") == "traces" and not s.get("path") and int(s.get("n") or 0) > 0:
                where = f"distill.teacher.sources.{i}.n" if explicit else \
                    "distill.teacher.n_seq (the default mix's trace share: n_seq/32, at most 24)"
                out.append({"stage": "distill", "where": where, "n": int(s["n"]), "seq_len": int(t["seq_len"])})
    return out


def supply(cfg: dict, tokens: int, bos, stages=("calibrate", "distill")) -> list[dict]:
    """Each demand with the blocks ``tokens`` trace tokens make at its seq_len (``available``)."""
    return [{**d, "available": int(tokens) // (d["seq_len"] - (bos is not None))} for d in trace_demands(cfg, stages)]


def supply_problems(cfg: dict, tokens: int, bos, stages=("calibrate", "distill")) -> list[str]:
    return [f"{d['where']} = {d['n']} blocks of {d['seq_len']} tokens, but {tokens} trace tokens make "
            f"{d['available']}" for d in supply(cfg, tokens, bos, stages) if d["n"] > d["available"]]


SUPPLY_ADVICE = ("trace-backed blocks are never capped: lower those counts to at most the blocks available "
                 "(keep calibrate's total a multiple of quantize.batch, and the teacher's mix by moving the "
                 "difference to another source), or make more traces (traces sources' n or max_new_tokens)")


def check_trace_supply(cfg: dict, tokens: int, bos, *, stages=("calibrate", "distill"),
                       what: str = "traces/traces.jsonl") -> None:
    from ..config import ConfigError
    problems = supply_problems(cfg, tokens, bos, stages)
    if problems:
        raise ConfigError(f"{what} is too short: " + "; ".join(problems) + f"; {SUPPLY_ADVICE}")


def trace_tokens(traces_dir: Path) -> int | None:
    """Tokens in traces.jsonl (from summary.json when it describes it), None when it is missing."""
    path = Path(traces_dir) / "traces.jsonl"
    if not path.is_file():
        return None
    try:
        summary = json.loads((Path(traces_dir) / "summary.json").read_text(encoding="utf-8"))
        if isinstance(summary.get("tokens"), int) and summary.get("bytes") == path.stat().st_size:
            return summary["tokens"]
    except (OSError, ValueError):
        pass
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            total += sum(1 for m in (r.get("attention_mask") or [1] * len(r["input_ids"])) if m)
    return total


# -- the stage ---------------------------------------------------------------------------

def run(ctx) -> dict:
    t = ctx.cfg["traces"]
    log = ctx.log("traces")
    out = ctx.run.stage_dir("traces")
    sources = trace_sources(ctx.cfg)
    adapter = ctx.adapter()
    tok = ctx.load_tokenizer()
    kwargs = {s["index"]: checked_chat_kwargs(adapter, tok, s["thinking"], f"{s['where']}.thinking"
                                              if not s["legacy"] else "traces.thinking")
              for s in sources}                                   # before loading the model
    bos = adapter.bos_id(tok)
    loaded = load_sources(ctx, sources)
    encoded = {src["index"]: [encode_chat(tok, msgs, kwargs[src["index"]]) for _, msgs, _ in picked]
               for src, picked, _ in loaded}
    upper = sum(len(ids) + src["max_new_tokens"] for src, _, _ in loaded for ids in encoded[src["index"]])
    check_trace_supply(ctx.cfg, upper, bos, what=f"the traces (at most {upper} tokens, every one at its cap)")
    plans = {src["index"]: plan_batches([len(i) for i in encoded[src["index"]]], src["batch"], sort=not src["legacy"],
                                        cap=src["max_new_tokens"], budget=src["batch_tokens"])
             for src, _, _ in loaded}
    for src, picked, data in loaded:
        sizes = [len(b) for b in plans[src["index"]]]
        log(f"{src['name']}: {len(picked)} prompts, thinking {'on' if src['thinking'] else 'off'}, cap "
            f"{src['max_new_tokens']}, batch {src['batch']}"
            + (f" and at most {src['batch_tokens']} tokens in flight" if src["batch_tokens"] else "")
            + f": {len(sizes)} batches of {min(sizes)} to {max(sizes)} prompts"
            + (f"; {len(data['screened'])} near duplicates of test problems skipped" if data["screened"] else ""))
    cache = BatchCache(out / "cache", runtime_identity(ctx))
    state: dict = {}
    vram = ExitStack()

    def generate(rows, cap):
        if "model" not in state:
            vram.enter_context(vram_cap(ctx.device, log))
            state["model"], state["tok"] = ctx.load_model()
            torch.manual_seed(0)
            state["eos"] = adapter.eos_ids(state["tok"], state["model"])
        return generate_encoded(state["model"], rows, eos_ids=state["eos"], max_new_tokens=cap, device=ctx.device,
                                pad_id=pad_token(state["tok"], state["eos"]))

    remaining = sum(len(encoded[s["index"]]) * s["max_new_tokens"] for s in sources)
    stats = {"batches": sum(len(p) for p in plans.values()), "cached_batches": 0, "seconds": 0.0, "fresh_tokens": 0}
    done_units = 0
    think_end = getattr(adapter, "think_end", None)
    from ..eval import heldout
    probes = heldout.needles() + heldout.chat_needles(tok)
    records_by_source, per_source = [], []
    with vram:            # capped from the first generated batch to the last
        for src, picked, data in loaded:
            i, cap = src["index"], src["max_new_tokens"]
            rows = encoded[i]
            gens: list = [None] * len(rows)
            for b, members in enumerate(plans[i]):
                batch_rows = [rows[k] for k in members]
                got, info = cache.get(batch_rows, cap, generate, log)
                for k, g in zip(members, got):
                    gens[k] = g
                remaining -= len(members) * cap
                if info["cached"] == len(members):
                    stats["cached_batches"] += 1
                    continue
                stats["seconds"] += info["seconds"]
                stats["fresh_tokens"] += info["tokens"]
                done_units += len(members) * cap
                eta = stats["seconds"] / max(done_units, 1) * remaining
                log(f"{src['name']}: batch {b + 1}/{len(plans[i])}: {info['tokens']} new tokens in "
                    f"{info['seconds']:.1f} s = {info['tokens'] / max(info['seconds'], 1e-9):.1f} tok/s, hit cap "
                    f"{sum(h for _, h in got)}/{len(got)}; ETA {_clock(eta)}")
            recs = []
            for (row, msgs, ident), ids, (gen, hit) in zip(picked, rows, gens):
                meta = {"source": src["name"], "kind": src["kind"], "row": row,
                        **({"id": ident} if ident is not None else {}), "thinking": src["thinking"], "max_new_tokens": cap}
                rec = _record(ids, gen, hit, msgs, **meta)
                completion = tok.decode(gen, skip_special_tokens=False)
                if src["thinking"] and think_end:
                    rec["finished_thinking"] = think_end in completion
                rec["completion"] = completion
                recs.append(rec)
            # The prompts were checked; an answer that repeats an evaluation item is dropped here, since
            # calibrate and the distillation teacher refuse blocks that contain one.
            hits = heldout.contained([r["completion"] for r in recs], probes)
            dropped = [{"row": recs[k]["row"], "items": sorted({pid for j, pid in hits if j == k})}
                       for k in sorted({j for j, _ in hits})]
            if dropped:
                log(f"WARNING: {src['name']}: {len(dropped)} answers repeat an evaluation item and are left out: "
                    + ", ".join(f"row {d['row']} ({', '.join(d['items'])})" for d in dropped[:5]))
                recs = [r for k, r in enumerate(recs) if k not in {j for j, _ in hits}]
            records_by_source.append(recs)
            gen_tokens = sum(r["generated_token_count"] for r in recs)
            per_source.append({
                "name": src["name"], "kind": src["kind"], "where": src["where"], "n": len(recs), "seed": src["seed"],
                "thinking": src["thinking"], "max_new_tokens": cap, "batch": src["batch"],
                "batches": len(plans[i]), "batch_tokens": src["batch_tokens"],
                "instruction": src.get("instruction"), "max_chars": src.get("max_chars"),
                **({"levels": src.get("levels"), "subjects": data["parts"]} if src["kind"] == "math_train" else {}),
                **({"rows": list(src.get("rows") or (5000, 6000))} if src["kind"] == "ultrachat" else {}),
                "records": len(recs), "prompt_tokens": sum(r["prompt_token_count"] for r in recs),
                "generated_tokens": gen_tokens, "mean_generated_tokens": round(gen_tokens / max(len(recs), 1), 1),
                "hit_cap": sum(r["hit_cap"] for r in recs),
                **({"finished_thinking": sum(r.get("finished_thinking", False) for r in recs)}
                   if src["thinking"] and think_end else {}),
                "data": data["files"], "near_duplicates_skipped": data["screened"],
                "answers_dropped": dropped})
    if any(s["legacy"] for s in sources):
        rows_out = records_by_source[0]
    else:
        rows_out = [r for recs in records_by_source for r in recs]
        random.Random(int(t.get("seed") or 0)).shuffle(rows_out)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_out).encode("utf-8")
    atomic_write_bytes(out / "traces.jsonl", data)
    tokens = sum(len(r["input_ids"]) for r in rows_out)
    summary = {"records": len(rows_out), "tokens": tokens, "bytes": len(data),
               "generated_tokens": sum(r["generated_token_count"] for r in rows_out),
               "hit_cap": sum(r["hit_cap"] for r in rows_out),
               "form": "prompts" if sources[0]["legacy"] else "sources",
               **({"thinking": sources[0]["thinking"], "max_new_tokens": sources[0]["max_new_tokens"]}
                  if sources[0]["legacy"] else {"seed": int(t.get("seed") or 0)}),
               "batches": stats["batches"], "cached_batches": stats["cached_batches"],
               "generation_seconds": round(stats["seconds"], 3),
               "tokens_per_second": stats["fresh_tokens"] / stats["seconds"] if stats["seconds"] else None,
               "decoding": DECODING, "sources": per_source, "supply": supply(ctx.cfg, tokens, bos)}
    atomic_write_json(out / "summary.json", summary)
    log(f"{summary['records']} traces, {tokens} tokens ({summary['generated_tokens']} generated, hit cap "
        f"{summary['hit_cap']}); " + ", ".join(f"{s['name']} {s['generated_tokens']}" for s in per_source))
    for d in summary["supply"]:
        log(f"{d['where']}: {d['n']} of the {d['available']} blocks of {d['seq_len']} tokens the traces make")
    for problem in supply_problems(ctx.cfg, tokens, bos):
        log(f"WARNING: {problem}; the stage that reads them will refuse this ({SUPPLY_ADVICE})")
    return summary

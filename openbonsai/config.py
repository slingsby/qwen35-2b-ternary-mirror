"""Run configuration: built-in defaults, a YAML file, then ``--set a.b=c`` overrides.

The resolved config is saved as ``run.yaml`` in the run directory. Each stage
hashes the config sections it depends on, the contents of the files they name
(``path:`` entries, a local model directory) and its upstream stages' config
hashes and output digests; a stage whose output exists with the same hash is
skipped (openbonsai.stages.common.StageContext.stage_hash).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

STAGES = ("traces", "calibrate", "quantize", "endpoints", "refine", "distill", "export", "eval")

DEFAULTS: dict = {
    "model": {
        "path": None,              # HF id or local directory (your fine-tune works the same way)
        "revision": None,          # pin a commit for hub ids
        "adapter": "auto",
        "dtype": "bfloat16",
        "device": "auto",          # auto = cuda if available, else cpu
        "attn": "sdpa",
        "keep_vision": False,
        "local_files_only": False,
    },
    "rotation": {
        "group": 128,
        "max_block": 4096,
        "signs": {"mode": "seeded", "seed": "width", "path": None},  # path: explicit signs.safetensors
        "impl": "dense",           # dense = the experiments' arithmetic; fwht = no dense R
    },
    "traces": {
        "enabled": True,
        # one prompt source (kind jsonl or ultrachat), answered with max_new_tokens, thinking and batch below
        "prompts": {"kind": "jsonl", "path": None, "n": None},
        # several prompt sources instead of prompts: a list of mappings, each with its own kind, n, seed,
        # thinking, max_new_tokens and batch (defaults: the values below); stages/traces.py lists the kinds
        "sources": None,
        "max_new_tokens": 512,
        "thinking": False,
        "batch": 8,
        # tokens in flight per batch, rows x (longest prompt + cap); null: no limit (stages/traces.py)
        "batch_tokens": None,
        "seed": 0,                 # sources only: seeds the interleaving of the sources in traces.jsonl
    },
    "calibrate": {
        "seq_len": 2048,
        "seed": 0,
        "sources": [
            {"kind": "wikitext2", "n": 128},
            {"kind": "ultrachat", "n": 96, "min_row": 6000},
            {"kind": "traces", "n": 32},
        ],
    },
    "quantize": {"damp": 0.01, "batch": 4},
    # head / embedding: ternary, bf16 (kept high precision) or auto (ternary unless the model ties the
    # two: a tied table stays high precision by default). tied: ternary explicitly opts into joint GPTQ.
    "endpoints": {"enabled": True, "head": "auto", "embedding": "auto", "tied": "bf16", "damp": 0.01,
                  "chunk": 16384, "batch": 4},
    # Optional block-wise training of the codes and scales, layer by layer (docs/refine.md; stages/refine.py).
    "refine": {
        "enabled": False,
        "layers": None,            # [start, stop) of decoder layers; null = all
        "window": 1,               # layers trained together (stride = window)
        "blocks": 128,             # training blocks: seeded subset of calibrate's, stratified by source
        "val_blocks": 8,           # held out: per-layer metrics and rollback
        "seed": 0,
        "batch": 1,                # blocks per optimizer step (the layer kwargs are captured at this batch size)
        "epochs": 4,
        "target": "original",      # original (original prefix, GPTAQ-style) | quantized (student prefix) | mix
        "mix": 0.5,                # weight of the quantized-target term under target: mix
        "assign": "nearest",       # nearest (threshold s/2) | quantile (`zeros` zeros per group)
        "zeros": 42,               # zeros per group under assign: quantile (Bonsai: 42 of 128)
        "zero_anneal": 0.25,       # fraction of the steps to move from GPTQ's zero count to `zeros`
        "init": "center",          # center (latent = input codes) | original (+ clipped residual of the weights)
        "lr_codes": 5.0e-3,        # latent weights, in units of the group's input scale
        "lr_scales": 5.0e-4,       # log multipliers of the scales
        "warmup": 0.05,
        "freeze_codes_after": 0.8,  # fraction of the steps; 0 = scales only, 1 = never freeze
        "eps": 1.0e-10,            # AdamW eps
        "checkpoint": "auto",      # gradient checkpointing per layer: auto = on for window >= 2 or batch > 2
        "cache": "auto",           # activation streams: auto | cuda | cpu | disk (not part of the hash)
        "rollback": True,          # a window whose held-out error did not go down keeps its input codes
    },
    "distill": {
        "enabled": True,
        "teacher": {"n_seq": 2048, "seq_len": 2048, "seed": 1, "shard": 64, "topk": 64, "sources": None},
        # scales: the group scales (log multipliers; always on). small: also every high-precision
        # decoder tensor (norms, conv1d, in_proj_a/b, A_log, dt_bias, final norm) through FP32
        # masters, as e2e_kd.py --train-small; the export writes them. (The list form [scales] is accepted.)
        "train": {"scales": True, "small": False},
        "endpoints": True,         # train endpoint scales too when the endpoints are ternary
        "lr": 1.0e-3,              # every measured 9B run used e2e_kd.py --lr 1e-3 (the script default was 2e-3)
        "lr_small": 1.0e-4,        # vector-like small tensors (norms, A_log, dt_bias)
        "lr_small_mat": 2.0e-5,    # matrix-like small tensors (conv1d, in_proj_a, in_proj_b)
        "epochs": 1,
        "accum": 8,
        "warmup": 10,
        "chunk": 512,
        "log_every": 5,
        "ckpt_every": 16,
        "max_steps": None,         # train only this many optimizer steps; the cosine schedule ends there
    },
    "export": {
        "enabled": True,
        # ptq1_0-gguf: Prism's PTQ1_0 GGUF (runs in Prism's llama.cpp fork); latent: a Hugging Face
        # checkpoint of the dequantized weights plus the exact codes and scales (ternary.safetensors);
        # auto: the GGUF when the adapter can write one for this model, else latent (the reason is logged)
        "format": "auto",
        "out": None,               # default: <run>/export/<name>.gguf (latent: <run>/export/<name>/)
        "name": None,              # general.name; default OpenBonsai-<model name>-PTQ1_0 (latent: -ternary)
        "size_label": None,        # general.size_label; default from the model name (e.g. 9B)
        "verify": True,            # re-read the file: structure, exact codes/scales/F32, function check
        "chunk_rows": 16384,
    },
    "eval": {
        "from": "auto",            # auto = the last stage whose output is current
        "suites": ["dev", "extended", "compare"],  # dev/extended need the Qwen3.5 tokenizer; compare works for any
        "generations": True,
        # full: keep the BF16 logits (n x (seq_len-1) x vocab on disk) for ear_full/kl_full;
        # off: top-k metrics only (ear_top{k}_lb/_ub, kl_top{k}). See docs/evaluation.md.
        "compare": {"seq_len": 512, "n": 8, "seed": 7, "topk": 64, "full": True, "sources": None},
        "teacher": {"rows": 8},    # suite "teacher": rows of the distillation teacher shards scored
        # suites gsm8k, humaneval: greedy chat generation on frozen TEST subsets (eval/reasoning.py)
        "reasoning": {
            "thinking": ["off"],           # off, on or [off, on] (the chat template's thinking switch)
            "max_new_tokens": 512,         # cap with thinking off
            "max_new_tokens_thinking": 4096,  # cap with thinking on; rows reaching a cap are counted
            "batch": 16,                   # prompts per left-padded generate call (KV cache)
            "batch_thinking": None,        # batch with thinking on (None: batch)
            "batch_check": 0,              # regenerate this many (most padded) rows of the first new batch alone; compare
            "dense": True,                 # ternary sources generate with dense dequantized weights (faster)
            "cache": True,                 # keep generations per batch in eval/generations/ (resume, rescore)
            # first n of the frozen 100; n_thinking: first n with thinking on (None: n); path: local test parquet
            "gsm8k": {"n": 100, "n_thinking": None, "path": None},
            # timeout: seconds per program; memory_mb: memory of each program and what it starts
            "humaneval": {"n": 50, "n_thinking": None, "path": None, "timeout": 10.0, "workers": 4,
                          "memory_mb": 2048},
        },
    },
}


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


# Keys a data source entry (calibrate.sources, distill.teacher.sources,
# eval.compare.sources) or a trace prompt spec may carry; which ones apply
# depends on its ``kind``.
SOURCE_KEYS = frozenset({"kind", "n", "path", "seed", "min_row", "repo", "revision", "file", "shuffle"})
PROMPT_KEYS = frozenset({"kind", "path", "n", "rows", "seed", "repo", "revision", "file"})
# traces.sources entries (stages/traces.py documents each key and kind).
TRACE_KINDS = ("ultrachat", "gsm8k_train", "math_train", "mbpp_train", "opencodeinstruct", "parquet", "jsonl")
TRACE_SOURCE_KEYS = frozenset({"kind", "name", "n", "seed", "shuffle", "thinking", "max_new_tokens", "batch",
                               "path", "repo", "revision", "file", "column", "instruction", "max_chars", "rows",
                               "subjects", "levels"})
_SOURCE_LISTS = {("calibrate", "sources"), ("distill", "teacher", "sources"), ("eval", "compare", "sources")}
_PROMPTS = ("traces", "prompts")
_TRACE_SOURCES = ("traces", "sources")


def _check_trace_sources(value, path: tuple) -> None:
    where = ".".join(path)
    if value is None:
        return
    if not isinstance(value, list) or not value or not all(isinstance(s, dict) for s in value):
        raise ConfigError(f"{where} must be a nonempty list of mappings (or null for traces.prompts)")
    for i, src in enumerate(value):
        unknown = sorted(set(src) - TRACE_SOURCE_KEYS)
        if unknown:
            raise ConfigError(f"unknown keys in {where}.{i}: {unknown}; known: {sorted(TRACE_SOURCE_KEYS)}")
        if src.get("kind") not in TRACE_KINDS:
            raise ConfigError(f"{where}.{i}: kind must be one of {list(TRACE_KINDS)}, not {src.get('kind')!r}")


def _check_sources(value, path: tuple) -> None:
    where = ".".join(path)
    if value is None and path != ("calibrate", "sources"):
        return
    if not isinstance(value, list) or not all(isinstance(s, dict) for s in value):
        raise ConfigError(f"{where} must be a list of mappings")
    for i, src in enumerate(value):
        unknown = sorted(set(src) - SOURCE_KEYS)
        if unknown:
            raise ConfigError(f"unknown keys in {where}[{i}]: {unknown}; known: {sorted(SOURCE_KEYS)}")
        if "kind" not in src:
            raise ConfigError(f"{where}[{i}] needs a kind")


def _check_mapping(value, schema: dict, path: tuple) -> None:
    where = ".".join(path)
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping")
    allowed = PROMPT_KEYS if path == _PROMPTS else frozenset(schema)
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"unknown keys in {where!r}: {unknown}; known: {sorted(allowed)}")
    if path == _PROMPTS:
        return
    for key, item in value.items():
        sub = path + (key,)
        if sub in _SOURCE_LISTS:
            _check_sources(item, sub)
        elif sub == _TRACE_SOURCES:
            _check_trace_sources(item, sub)
        elif sub == _PROMPTS:
            if item is not None:
                _check_mapping(item, {}, sub)
        elif isinstance(schema[key], dict):
            if item is not None:
                _check_mapping(item, schema[key], sub)
        elif isinstance(item, dict):
            raise ConfigError(f"{'.'.join(sub)} must be a value, not a mapping")


_TRAIN_KEYS = ("scales", "small")


def _normalize(cfg: dict) -> None:
    """Accept older spellings: ``distill.train: [scales, small]`` -> {scales: true, small: true}."""
    d = cfg.get("distill")
    if isinstance(d, dict) and isinstance(d.get("train"), (list, tuple)):
        unknown = sorted(set(d["train"]) - set(_TRAIN_KEYS))
        if unknown:
            raise ConfigError(f"unknown distill.train entries {unknown}; known: {list(_TRAIN_KEYS)}")
        d["train"] = {k: k in d["train"] for k in _TRAIN_KEYS}


def check_keys(cfg: dict) -> dict:
    """Reject unknown sections and keys at every level (typos), and values whose
    shape (mapping, list of sources, scalar) does not match the defaults."""
    _normalize(cfg)
    for section, value in cfg.items():
        if section not in DEFAULTS:
            raise ConfigError(f"unknown config section {section!r}; known: {sorted(DEFAULTS)}")
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ConfigError(f"config section {section!r} must be a mapping")
        _check_mapping(value, DEFAULTS[section], (section,))
    return cfg


def _index(items: list, part: str, where: str) -> int:
    if not part.isdigit() or int(part) >= len(items):
        raise ConfigError(f"--set {where}: expected a list index below {len(items)}")
    return int(part)


def apply_set(cfg: dict, assignment: str) -> dict:
    """``a.b.c=value`` (YAML value). A numeric part indexes a list: ``calibrate.sources.2.path=x``."""
    if "=" not in assignment:
        raise ConfigError(f"--set expects key=value, got {assignment!r}")
    key, raw = assignment.split("=", 1)
    parts = [p for p in key.strip().split(".") if p]
    if len(parts) < 2:
        raise ConfigError(f"--set needs section.key=value, got {assignment!r}")
    value = yaml.safe_load(raw) if raw.strip() else None
    node = cfg
    for depth, part in enumerate(parts[:-1]):
        where = ".".join(parts[:depth + 1])
        if isinstance(node, list):
            node = node[_index(node, part, where)]
            continue
        if part not in node:
            node[part] = {}            # a new key: check_keys below rejects it unless the schema has it
        elif not isinstance(node[part], (dict, list)):
            raise ConfigError(f"--set {key}: {where} is {node[part]!r}, not a mapping or list")
        node = node[part]
    if isinstance(node, list):
        node[_index(node, parts[-1], key)] = value
    else:
        node[parts[-1]] = value
    return check_keys(cfg)


def load_config(path=None, sets=(), model: str | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a mapping")
        cfg = check_keys(deep_merge(cfg, data))
    for assignment in sets:
        apply_set(cfg, assignment)
    if model is not None:
        cfg["model"]["path"] = model
    return cfg


def dump(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
                          .encode("utf-8")).hexdigest()


_MODEL_IDENTITY = ("path", "revision", "adapter", "dtype", "attn")
# Settings that cannot change a stage's output (logging and checkpoint cadence, write chunking).
_UNHASHED = {"distill": ("log_every", "ckpt_every"), "export": ("chunk_rows",)}


def refine_hashed(refine: dict) -> dict:
    """The refine settings that can change its result: not where the streams live (``cache``), and
    ``mix`` / ``zeros`` / ``zero_anneal`` only under the target and assign rule that use them."""
    out = {k: v for k, v in refine.items() if k != "cache"}
    if out.get("target") != "mix":
        out.pop("mix", None)
    if out.get("assign") != "quantile":
        out.pop("zeros", None)
        out.pop("zero_anneal", None)
    return out


def stage_enabled(cfg: dict, stage: str) -> bool:
    section = cfg.get(stage) or {}
    return section.get("enabled", True) is not False


def stage_inputs(cfg: dict, stage: str) -> dict:
    """The config sections a stage's output depends on (device and logging excluded)."""
    model = {k: cfg["model"].get(k) for k in _MODEL_IDENTITY}
    sections = {
        "traces": ("traces",),
        "calibrate": ("calibrate",),
        "quantize": ("rotation", "quantize"),
        "endpoints": ("rotation", "endpoints"),
        "refine": ("rotation", "refine"),
        "distill": ("rotation", "distill"),
        "export": ("export",),
        "eval": ("eval",),
    }[stage]
    out = {"stage": stage, "model": model}
    for s in sections:
        value = copy.deepcopy(cfg[s])
        for key in _UNHASHED.get(s, ()):
            value.pop(key, None)
        if s == "distill" and not (value.get("train") or {}).get("small"):
            value.pop("lr_small", None)
            value.pop("lr_small_mat", None)
        if s == "traces":
            # Only the form in use: a single-source config hashes as it did before sources existed.
            for key in (("sources", "seed") if value.get("sources") is None else ("prompts",)):
                value.pop(key, None)
            if value.get("batch_tokens") is None:
                value.pop("batch_tokens", None)
        if s == "refine":
            value = refine_hashed(value)
        out[s] = value
    return out


def _source_paths(sources) -> list[str]:
    return [str(s["path"]) for s in sources or () if s.get("path")]


def input_paths(cfg: dict, stage: str) -> list[str]:
    """Local files (or folders) a stage reads through ``path:`` entries (their contents are part of its hash)."""
    paths = []
    if stage == "traces":
        if cfg["traces"].get("sources") is not None:
            paths += _source_paths(cfg["traces"]["sources"])
        else:
            prompts = cfg["traces"].get("prompts") or {}
            paths += [str(prompts["path"])] if prompts.get("path") else []
    elif stage == "calibrate":
        paths += _source_paths(cfg["calibrate"]["sources"])
    if stage in ("quantize", "endpoints", "refine", "distill"):
        signs = cfg["rotation"].get("signs") or {}
        paths += [str(signs["path"])] if signs.get("path") else []
    if stage == "distill":
        paths += _source_paths(cfg["distill"]["teacher"].get("sources"))
    return paths


def uses_traces(sources) -> bool:
    """True when a source list reads the traces stage output (a traces source without its own path)."""
    return any(s.get("kind") == "traces" and not s.get("path") for s in sources or ())


def distill_uses_endpoints(cfg: dict) -> bool:
    return bool(cfg["distill"].get("endpoints", True)) and stage_enabled(cfg, "endpoints")


def upstream(cfg: dict, stage: str) -> tuple[str, ...]:
    """Stages whose outputs this stage reads under this config.

    Refine (when enabled) reads quantize's state, the calibration blocks and, when endpoints is on, its
    ternary embedding; distill and the export then read refine's decoder state instead of quantize's.
    With refine disabled every other stage's list is what it was before refine existed."""
    teacher_sources = cfg["distill"]["teacher"].get("sources")
    distill_traces = teacher_sources is None or uses_traces(teacher_sources)
    refine = (("refine",) if stage_enabled(cfg, "refine") else ())
    if stage == "export":
        return (next((s for s in ("distill", "refine", "endpoints") if stage_enabled(cfg, s)), "quantize"),)
    return {
        "traces": (),
        "calibrate": ("traces",) if uses_traces(cfg["calibrate"]["sources"]) else (),
        "quantize": ("calibrate",),
        "endpoints": ("quantize", "calibrate"),
        "refine": ("quantize", "calibrate") + (("endpoints",) if stage_enabled(cfg, "endpoints") else ()),
        "distill": ("quantize",) + refine + (("endpoints",) if distill_uses_endpoints(cfg) else ())
                   + (("traces",) if distill_traces else ()),
        "eval": (),
    }[stage]

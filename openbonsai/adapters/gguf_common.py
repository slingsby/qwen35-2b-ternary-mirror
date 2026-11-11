"""Helpers adapters share when they describe a GGUF file (no architecture knowledge).

- ``gpt2_vocab``: the tokenizer KVs Prism's converter writes for a Hugging
  Face byte-level BPE tokenizer (``TextModel._set_vocab_gpt2`` plus
  ``gguf.SpecialVocab``), derived from ``tokenizer.json``,
  ``tokenizer_config.json``, ``config.json`` and ``chat_template.jinja``.
  Ported unchanged from OpenBonsai research/experiments/export_gguf.py, which matched the
  official Ternary-Bonsai-2-9B tokens, types, merges and ids.
- ``chat_template_kvs``: the chat template KVs (string or named-list form).
- ``model_basename`` / ``size_label`` / ``rounded_count``: names for
  ``general.name`` and ``general.size_label``.
- ``reference_kv_equal``: compare a KV with a header dump entry (arrays by
  SHA-256 of their JSON, long strings by prefix and length).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct

from ..formats.gguf import GGUFError

TOKEN_NORMAL, TOKEN_CONTROL, TOKEN_USER_DEFINED, TOKEN_UNUSED = 1, 3, 4, 5
SPECIAL_TYPES = ("bos", "eos", "unk", "sep", "pad", "cls", "mask")
SPECIAL_KV = {"bos": "bos_token_id", "eos": "eos_token_id", "unk": "unknown_token_id", "sep": "seperator_token_id",
              "pad": "padding_token_id", "cls": "cls_token_id", "mask": "mask_token_id"}

# The converter's get_vocab_base_pre probe text (its SHA-256 guards the transcription).
CHKTXT = ('\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n\U0001f680 (normal) \U0001f636‍\U0001f32b️ '
          '(multiple emojis concatenated) ✅ \U0001f999\U0001f999 3 33 333 3333 33333 333333 3333333 33333333 '
          '3.3 3..3 3...3 កាន់តែពិសេសអាច'
          '\U0001f601 ?我想在apple工作1314151天～ ------======= '
          'нещо на Български '
          '\'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? '
          '\'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL')
CHKTXT_SHA256 = "a36179c1e2b1cfcfbfad1dc779addad2ffc778201e4cf0d538bddeafa2c526da"


def does_token_look_special(t: str) -> bool:
    return (t in ("<pad>", "<mask>", "<2mass>", "[@BOS@]") or (t.startswith("<|") and t.endswith("|>"))
            or (t.startswith("<｜") and t.endswith("｜>")) or (t.startswith("<unused") and t.endswith(">")))


def gpt2_vocab(model_dir, vocab_size: int) -> dict:
    """Tokens, token types, merges, special ids (converter order), add_* flags, chat template.

    Returns {"tokens", "token_type", "merges", "ids" (ordered {type: id}), "add" (ordered {type: bool}),
    "chat_template", "chkhsh" (pre-tokenizer probe hash or None), "notes"}."""
    root = Path(model_dir)
    tj = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
    tcfg_path = root / "tokenizer_config.json"
    tcfg = json.loads(tcfg_path.read_text(encoding="utf-8")) if tcfg_path.exists() else {}
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    notes = []
    if tj["model"]["type"] != "BPE":
        raise ValueError(f"{root}: tokenizer model {tj['model']['type']!r} is not byte-level BPE (gpt2)")
    # AutoTokenizer.vocab = base vocab + added tokens (tokenizer.json added_tokens, then tokenizer_config's
    # added_tokens_decoder).
    reverse = {i: t for t, i in tj["model"]["vocab"].items()}
    added = {}
    for a in tj.get("added_tokens", []):
        added[int(a["id"])] = {"content": a["content"], "special": bool(a.get("special", False)),
                               "normalized": bool(a.get("normalized", False))}
    for k, a in (tcfg.get("added_tokens_decoder") or {}).items():
        new = {"content": a["content"], "special": bool(a.get("special", False)),
               "normalized": bool(a.get("normalized", False))}
        if int(k) in added and added[int(k)] != new:
            notes.append(f"added token {k}: tokenizer.json {added[int(k)]} vs tokenizer_config {new}; "
                         "using tokenizer_config")
        added[int(k)] = new
    for i, a in added.items():
        reverse[i] = a["content"]
    added_vocab = {a["content"]: i for i, a in added.items()}
    if reverse and max(reverse) >= vocab_size:
        raise ValueError(f"{root}: tokenizer id {max(reverse)} exceeds the model vocabulary ({vocab_size})")
    try:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(root / "tokenizer.json"))
    except Exception as e:  # pragma: no cover
        tk = None
        notes.append(f"tokenizers unavailable ({e}); non-normalized added tokens not round-tripped")
    tokens, types = [], []
    for i in range(vocab_size):
        if i not in reverse:
            tokens.append(f"[PAD{i}]")
            types.append(TOKEN_UNUSED)
            continue
        tok = reverse[i]
        if tok in added_vocab:
            if not added[i]["normalized"] and tk is not None:
                rt = tk.decode(tk.encode(tok, add_special_tokens=False).ids, skip_special_tokens=False)
                if rt != tok:
                    notes.append(f"added token {i} {tok!r} round-trips to {rt!r}")
                    tok = rt
            if added[i]["special"] or does_token_look_special(tok):
                types.append(TOKEN_CONTROL)
            else:
                tok = tok.replace("▁", " ")
                types.append(TOKEN_USER_DEFINED)
        else:
            types.append(TOKEN_NORMAL)
        tokens.append(tok)
    chkhsh = None
    if tk is not None:
        if hashlib.sha256(CHKTXT.encode()).hexdigest() != CHKTXT_SHA256:  # pragma: no cover
            raise AssertionError("pre-tokenizer probe text transcription")
        chkhsh = hashlib.sha256(str(tk.encode(CHKTXT).ids).encode()).hexdigest()
    merges = tj["model"].get("merges") or []
    if merges and not isinstance(merges[0], str):
        merges = [" ".join("".join(chr(ord(c) + 256) if c == " " else c for c in part) for part in pair)
                  for pair in merges]
    # special token ids: tokenizer_config.json first, then config.json (incl. text_config)
    ids, add_flags = {}, {}

    def set_special(typ, tid):
        if isinstance(tid, int) and 0 <= tid < vocab_size and typ not in ids:
            ids[typ] = tid

    for typ in SPECIAL_TYPES:
        a = tcfg.get(f"add_{typ}_token")
        if isinstance(a, bool):
            add_flags[typ] = a
        e = tcfg.get(f"{typ}_token")
        content = e if isinstance(e, str) else (e.get("content") if isinstance(e, dict) else None)
        if content is None:
            continue
        set_special(typ, next((at["id"] for at in tj.get("added_tokens", []) if at["content"] == content), None))
    for typ in SPECIAL_TYPES:
        tid = cfg.get(f"{typ}_token_id")
        if tid is None and isinstance(cfg.get("text_config"), dict):
            tid = cfg["text_config"].get(f"{typ}_token_id")
        set_special(typ, tid)
    template = tcfg.get("chat_template")
    jinja = root / "chat_template.jinja"
    if jinja.is_file():
        jt = jinja.read_text(encoding="utf-8")
        if template is None:
            template = jt
        elif template != jt:
            notes.append("tokenizer_config chat_template differs from chat_template.jinja; using tokenizer_config's")
    return {"tokens": tokens, "token_type": types, "merges": merges, "ids": ids, "add": add_flags,
            "chat_template": template, "chkhsh": chkhsh, "notes": notes, "vocab": reverse}


def chat_template_kvs(template) -> list[tuple[str, str, object]]:
    """The chat template KVs gguf-py's ``add_chat_template`` writes: a string as
    ``tokenizer.chat_template``; tokenizer_config's list form [{"name", "template"}] as
    ``tokenizer.chat_template.<name>`` per named template (name reduced to [A-Za-z0-9_]), the
    ``tokenizer.chat_templates`` name list, then the ``default`` one as ``tokenizer.chat_template``."""
    if template is None:
        return []
    if isinstance(template, str):
        return [("tokenizer.chat_template", "str", template)]
    if not isinstance(template, (list, tuple)) or not all(isinstance(c, dict) for c in template):
        raise GGUFError(f"chat_template of type {type(template).__name__}: expected a string or a list of "
                         "{name, template} entries")
    kvs, names, default = [], [], None
    for choice in template:
        name = "".join(c if c.isascii() and c.isalnum() else "_" for c in str(choice.get("name", "")))
        body = choice.get("template")
        if not name or body is None:
            continue
        if not isinstance(body, str):
            raise GGUFError(f"chat_template {name!r}: template is not a string")
        if name == "default":
            default = body
        elif name not in names:
            names.append(name)
            kvs.append((f"tokenizer.chat_template.{name}", "str", body))
    if names:
        kvs.append(("tokenizer.chat_templates", "arr[str]", names))
    if default is not None:
        kvs.append(("tokenizer.chat_template", "str", default))
    return kvs


# -- names --------------------------------------------------------------------

def model_basename(path) -> str:
    """'Qwen/Qwen3.5-9B' -> 'Qwen3.5-9B'; an HF cache snapshot dir -> its repo name; else the dir name."""
    text = str(path).replace("\\", "/").rstrip("/")
    for part in reversed(text.split("/")):
        if part.startswith("models--"):
            return part.split("--")[-1]
    return text.split("/")[-1] or "model"


_SIZE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)([KMBT])(?![A-Za-z0-9])", re.I)


def rounded_count(n: int, min_digits: int = 2) -> str:
    """gguf-py's model_weight_count_rounded_notation."""
    if n > 1e12:
        scaled, suffix = n * 1e-12, "T"
    elif n > 1e9:
        scaled, suffix = n * 1e-9, "B"
    elif n > 1e6:
        scaled, suffix = n * 1e-6, "M"
    else:
        scaled, suffix = n * 1e-3, "K"
    fix = max(min_digits - len(str(round(scaled)).lstrip("0")), 0)
    return f"{scaled:.{fix}f}{suffix}"


def size_label(name: str, n_params: int) -> str:
    """The size in the model name (as gguf-py's metadata heuristics read it), else the rounded count."""
    m = _SIZE.search(name)
    return f"{m.group(1)}{m.group(2).upper()}" if m else rounded_count(n_params)


# -- comparing against a header dump ------------------------------------------------

_TRUNC = re.compile(r"^(.*)\.\.\.<(\d+) chars>$", re.S)


def sha_items(items) -> str:
    """Array hash of the header dumps: sha256(json.dumps(items, ensure_ascii=False))."""
    return hashlib.sha256(json.dumps(list(items), ensure_ascii=False).encode()).hexdigest()


def reference_kv_equal(entry: dict, typ: str, val) -> tuple[bool, str]:
    """(equal, detail) of a KV against one header-dump entry."""
    if typ.startswith("arr["):
        ok = entry.get("len") == len(val) and entry.get("sha256") == sha_items(val)
        return ok, f"len {len(val)} vs {entry.get('len')}, sha256 {'equal' if ok else 'different'}"
    hv = entry["value"]
    if typ == "str":
        m = _TRUNC.match(hv) if isinstance(hv, str) else None
        if m:   # the dump keeps only a prefix of long strings
            ok = val.startswith(m.group(1)) and len(val) == int(m.group(2))
            return ok, f"prefix and length {len(val)} vs {m.group(2)}"
        return val == hv, f"{val!r} vs {hv!r}"
    if typ == "f32":
        return struct.pack("<f", val) == struct.pack("<f", hv), f"{val!r} vs {hv!r}"
    return val == hv, f"{val!r} vs {hv!r}"

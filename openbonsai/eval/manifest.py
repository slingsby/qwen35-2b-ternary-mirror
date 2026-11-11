"""Frozen evaluation manifests and the FP64 scoring contract.

A manifest freezes explicit token IDs (with their raw bytes) for NLL windows,
multiple-choice continuations and greedy generations, and seals them with a
canonical SHA256. Scoring normalizes FP32 logits with an FP64 logsumexp over
the full vocabulary; target j uses the prediction at j - 1. Nothing here
renders chat templates, guesses continuation boundaries or truncates inputs.

Any backend with ``metadata``, ``tokenize``, ``token_pieces``,
``score_tokens`` and ``generate`` can be scored against a manifest
(eval/backends.py has Transformers and Prism backends). The fast in-process
torch scorer for the development suite is eval/suites.py.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import math
from pathlib import Path

import torch

EVALUATION_PROTOCOL = "openbonsai.frozen-evaluation.v1"
REDUCTION = "fp32-logits-fp64-logsumexp-v1"
FROZEN_MANIFEST_FORMAT = "openbonsai.frozen-inputs.v1"
_ROLES = {"calibration", "search_validation", "evaluation"}


def _canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("Expected finite JSON-compatible data") from error


def canonical_sha256(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _identity(value, name):
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty mapping")
    result = copy.deepcopy(dict(value))
    _canonical(result)
    return result


def _keys(value, required, optional=(), *, name):
    if not isinstance(value, dict) or not set(required).issubset(value) or set(value) - set(required) - set(optional):
        raise ValueError(f"Invalid {name} fields; padding/masks and implicit defaults are unsupported")


def validate_token_ids(input_ids, *, vocab_size=None, allow_empty=False):
    """Validate explicit one-dimensional IDs, never a mask/padded batch."""
    if not isinstance(input_ids, (list, tuple)) or (not input_ids and not allow_empty):
        raise ValueError("input_ids must be a nonempty one-dimensional sequence")
    if vocab_size is not None:
        _positive_int(vocab_size, "vocab_size")
    if any(type(token) is not int or token < 0 or (vocab_size is not None and token >= vocab_size) for token in input_ids):
        raise ValueError("input_ids contain invalid or out-of-vocabulary token IDs")
    return list(input_ids)


def validate_score_positions(input_ids, score_positions, *, context_length=None, vocab_size=None):
    ids = validate_token_ids(input_ids, vocab_size=vocab_size)
    if context_length is not None:
        _positive_int(context_length, "context_length", 2)
        if len(ids) > context_length:
            raise ValueError("Scoring input exceeds context_length; truncation is forbidden")
    if (not isinstance(score_positions, (list, tuple)) or not score_positions
            or any(type(j) is not int or not 1 <= j < len(ids) for j in score_positions)
            or any(a >= b for a, b in zip(score_positions, score_positions[1:]))):
        raise ValueError("score_positions must be nonempty, strictly increasing target indices j >= 1")
    return ids, list(score_positions)


@torch.no_grad()
def reduce_target_logprobs(logits, target_ids):
    """Reference full-vocabulary FP64 normalization of FP32 prediction rows.

    Callers select logits at j-1 and targets at j. They must bound the number
    of supplied rows; this helper never approximates the vocabulary normalizer.
    """
    if (not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32
            or logits.ndim != 2 or min(logits.shape) < 1 or logits.is_meta):
        raise ValueError("logits must be a materialized FP32 [rows, vocabulary] tensor")
    if isinstance(target_ids, torch.Tensor):
        if target_ids.ndim != 1 or target_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("target_ids must be one-dimensional integer IDs")
        target_ids = target_ids.detach().cpu().tolist()
    ids = validate_token_ids(target_ids, vocab_size=logits.shape[1])
    if len(ids) != logits.shape[0] or not bool(torch.isfinite(logits).all()):
        raise ValueError("Finite full logits and one target per prediction row are required")
    values = logits.to(torch.float64)
    # Subtract the shared offset before both terms. Even FP64 loses log(V)
    # when subtracting two ~1e38 quantities despite finite FP32 source logits.
    values = values - values.amax(dim=1, keepdim=True)
    targets = torch.tensor(ids, dtype=torch.long, device=logits.device)
    probabilities = values.gather(1, targets[:, None]).squeeze(1) - torch.logsumexp(values, dim=1)
    result = probabilities.cpu().tolist()
    if any(not math.isfinite(p) or p > 0 for p in result):
        raise FloatingPointError("Invalid normalized target log probability")
    return result


def sliding_windows(input_ids, *, context_length=2048, stride=512):
    """Freeze overlapping contexts, scoring each document target exactly once.

    The first window scores positions 1..context-1. Subsequent windows advance
    by stride and retain the preceding context. stride < context guarantees a
    visible predecessor for every newly scored target, including the tail.
    Returned token_start/token_end are half-open global document coordinates.
    """
    ids = validate_token_ids(input_ids)
    _positive_int(context_length, "context_length", 2)
    _positive_int(stride, "stride")
    if stride >= context_length or len(ids) < 2:
        raise ValueError("A scored document needs >=2 tokens and 1 <= stride < context_length")
    windows, next_target, end = [], 1, min(len(ids), context_length)
    while next_target < len(ids):
        start = max(0, end - context_length)
        positions = list(range(next_target - start, end - start))
        validate_score_positions(ids[start:end], positions, context_length=context_length)
        windows.append({"token_start": start, "token_end": end,
                        "input_ids": ids[start:end], "score_positions": positions})
        next_target = end
        end = min(len(ids), end + stride)
    return windows


def _bytes(value, name):
    if isinstance(value, str):
        value = value.encode("utf-8")
    if not isinstance(value, bytes):
        raise ValueError(f"{name} must be str or UTF8 bytes")
    try:
        value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be valid UTF8") from error
    return value


def _unhex(value, name, *, utf8=False):
    if not isinstance(value, str) or len(value) % 2 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must contain canonical lowercase hex bytes")
    result = bytes.fromhex(value)
    if utf8:
        _bytes(result, name)
    return result


def freeze_text(text_id, *, raw_text, rendered_text, input_ids, token_pieces, parse_special=False):
    """Freeze caller-rendered UTF8 and caller-tokenized IDs without guessing.

    token_pieces are raw per-token bytes; individual pieces may split a UTF8
    character. No BOS, template, normalization or special token is added here.
    """
    _identifier(text_id, "text_id")
    ids = validate_token_ids(input_ids)
    if type(parse_special) is not bool:
        raise ValueError("parse_special must be bool")
    if (not isinstance(token_pieces, (list, tuple)) or len(token_pieces) != len(ids)
            or any(not isinstance(piece, bytes) for piece in token_pieces)):
        raise ValueError("token_pieces must supply raw bytes for every frozen token")
    raw, rendered = _bytes(raw_text, "raw_text"), _bytes(rendered_text, "rendered_text")
    return {"id": text_id, "raw_utf8_hex": raw.hex(), "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "rendered_utf8_hex": rendered.hex(), "rendered_sha256": hashlib.sha256(rendered).hexdigest(),
            "input_ids": ids, "token_pieces_hex": [piece.hex() for piece in token_pieces],
            "parse_special": parse_special}


def freeze_manifest(*, source_provenance, tokenizer_identity, texts, nll=(), choices=(), generations=(),
                    role="evaluation", metadata=None):
    """Seal already explicit records; this does not build benchmark prompts.

    The caller supplies dataset and reference-tokenizer identity (free-form
    JSON, sealed by the checksum); this pure function contacts nothing.
    """
    manifest = {"format": FROZEN_MANIFEST_FORMAT, "protocol": EVALUATION_PROTOCOL, "reduction": REDUCTION,
                "role": role, "source_provenance": copy.deepcopy(source_provenance),
                "tokenizer_identity": copy.deepcopy(tokenizer_identity), "texts": copy.deepcopy(list(texts)),
                "nll": copy.deepcopy(list(nll)), "choices": copy.deepcopy(list(choices)),
                "generations": copy.deepcopy(list(generations)), "metadata": copy.deepcopy(metadata or {})}
    manifest["frozen_sha256"] = canonical_sha256(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest, *, context_length=None, vocab_size=None):
    """Fail closed on changed bytes/IDs, invalid boundaries or repeated targets."""
    _keys(manifest, {"format", "protocol", "reduction", "role", "source_provenance", "tokenizer_identity",
                     "texts", "nll", "choices", "generations", "metadata", "frozen_sha256"}, name="manifest")
    if (manifest["format"] != FROZEN_MANIFEST_FORMAT or manifest["protocol"] != EVALUATION_PROTOCOL
            or manifest["reduction"] != REDUCTION or manifest["role"] not in _ROLES):
        raise ValueError("Unsupported frozen evaluation format, numerical protocol or role")
    body = {k: v for k, v in manifest.items() if k != "frozen_sha256"}
    if canonical_sha256(body) != manifest["frozen_sha256"]:
        raise ValueError("Frozen input checksum mismatch")
    _identity(manifest["source_provenance"], "source_provenance")
    _identity(manifest["tokenizer_identity"], "tokenizer_identity")
    if not isinstance(manifest["metadata"], dict):
        raise ValueError("metadata must be a JSON dictionary")
    if not isinstance(manifest["texts"], list) or not manifest["texts"]:
        raise ValueError("Frozen texts are required")
    texts = {}
    for text in manifest["texts"]:
        _keys(text, {"id", "raw_utf8_hex", "raw_sha256", "rendered_utf8_hex", "rendered_sha256",
                     "input_ids", "token_pieces_hex", "parse_special"}, name="text")
        text_id = _identifier(text["id"], "text id")
        if text_id in texts or type(text["parse_special"]) is not bool:
            raise ValueError("Duplicate text ID or invalid parse_special")
        for prefix in ("raw", "rendered"):
            value = _unhex(text[f"{prefix}_utf8_hex"], prefix, utf8=True)
            if hashlib.sha256(value).hexdigest() != text[f"{prefix}_sha256"]:
                raise ValueError("Frozen raw/rendered byte checksum mismatch")
        ids = validate_token_ids(text["input_ids"], vocab_size=vocab_size)
        if not isinstance(text["token_pieces_hex"], list) or len(text["token_pieces_hex"]) != len(ids):
            raise ValueError("Frozen token-piece count mismatch")
        for piece in text["token_pieces_hex"]:
            _unhex(piece, "token piece")
        texts[text_id] = text
    record_ids, coverage = set(), {}
    def record(record, required, optional=(), name="record"):
        _keys(record, {"id", *required}, optional, name=name)
        key = _identifier(record["id"], "record id")
        if key in record_ids:
            raise ValueError("Record IDs must be unique across tasks")
        record_ids.add(key)
    def text_for(value):
        if value not in texts:
            raise ValueError("Record refers to missing frozen text")
        return texts[value]
    for field in ("nll", "choices", "generations"):
        if not isinstance(manifest[field], list):
            raise ValueError(f"{field} must be a list")
    for item in manifest["nll"]:
        record(item, {"text_id", "token_start", "token_end", "input_ids", "score_positions"}, name="NLL record")
        text = text_for(item["text_id"])
        start, end = item["token_start"], item["token_end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text["input_ids"]):
            raise ValueError("Invalid document token interval")
        ids, positions = validate_score_positions(item["input_ids"], item["score_positions"],
                                                  context_length=context_length, vocab_size=vocab_size)
        if ids != text["input_ids"][start:end]:
            raise ValueError("Window tokens differ from the frozen document")
        counts = coverage.setdefault(item["text_id"], [0] * len(text["input_ids"]))
        for position in positions:
            counts[start + position] += 1
    if any(counts[0] != 0 or any(count != 1 for count in counts[1:]) for counts in coverage.values()):
        raise ValueError("NLL windows must score every document token after the first exactly once")
    for item in manifest["choices"]:
        record(item, {"options"}, {"answer_id"}, name="choice record")
        if not isinstance(item["options"], list) or len(item["options"]) < 2:
            raise ValueError("A choice record needs at least two options")
        option_ids, common_prefix = set(), None
        for option in item["options"]:
            _keys(option, {"id", "text_id", "input_ids", "prefix_length", "score_positions"}, name="choice option")
            key = _identifier(option["id"], "option id")
            if key in option_ids:
                raise ValueError("Duplicate option ID")
            option_ids.add(key)
            text = text_for(option["text_id"])
            ids, positions = validate_score_positions(option["input_ids"], option["score_positions"],
                                                      context_length=context_length, vocab_size=vocab_size)
            boundary = _positive_int(option["prefix_length"], "prefix_length")
            if ids != text["input_ids"] or positions != list(range(boundary, len(ids))):
                raise ValueError("Choice must score its exact frozen continuation boundary and all continuation tokens")
            prefix = ids[:boundary]
            if common_prefix is not None and prefix != common_prefix:
                raise ValueError("Choice options must share identical frozen conditioning token IDs")
            common_prefix = prefix
        if "answer_id" in item and item["answer_id"] not in option_ids:
            raise ValueError("answer_id is not a declared option")
    for item in manifest["generations"]:
        record(item, {"text_id", "input_ids", "max_new_tokens", "eos_ids"}, name="generation record")
        text = text_for(item["text_id"])
        ids = validate_token_ids(item["input_ids"], vocab_size=vocab_size)
        cap = _positive_int(item["max_new_tokens"], "max_new_tokens")
        eos = validate_token_ids(item["eos_ids"], vocab_size=vocab_size)
        if ids != text["input_ids"] or len(eos) != len(set(eos)):
            raise ValueError("Generation tokens differ from frozen prefix or EOS IDs are duplicated")
        if context_length is not None and len(ids) + cap > context_length:
            raise ValueError("Generation prefix plus cap exceeds context_length")
    if not record_ids:
        raise ValueError("At least one scored or generated record is required")
    return copy.deepcopy(manifest)


def validate_score_result(result, input_ids, score_positions):
    _, positions = validate_score_positions(input_ids, score_positions)
    if not isinstance(result, Mapping):
        raise ValueError("Backend score result must be an object")
    returned = result.get("score_positions")
    values = result.get("target_logprobs")
    count, nll = result.get("count"), result.get("nll_sum")
    if (not isinstance(returned, list) or any(type(j) is not int for j in returned)
            or returned != positions or type(count) is not int or count != len(positions)
            or not isinstance(values, (list, tuple)) or len(values) != count
            or any(isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p) or p > 0 for p in values)
            or isinstance(nll, bool) or not isinstance(nll, (float, int)) or not math.isfinite(nll) or nll < 0):
        raise ValueError("Backend returned invalid/nonfinite target log probabilities, count or positions")
    expected = -math.fsum(values)
    if not math.isclose(nll, expected, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("Backend NLL sum does not match its per-target log probabilities")
    return {"score_positions": positions, "target_logprobs": [float(v) for v in values],
            "nll_sum": float(nll), "count": count}


def validate_generation_result(result, *, max_new_tokens, eos_ids, vocab_size=None):
    _positive_int(max_new_tokens, "max_new_tokens")
    eos = set(validate_token_ids(eos_ids, vocab_size=vocab_size))
    if not isinstance(result, Mapping):
        raise ValueError("Backend generation result must be an object")
    ids = validate_token_ids(result.get("generated_ids"), vocab_size=vocab_size)
    reason = result.get("stop_reason")
    hits = [i for i, token in enumerate(ids) if token in eos]
    if (len(ids) > max_new_tokens or reason not in {"eos", "length"}
            or (reason == "eos" and hits != [len(ids) - 1])
            or (reason == "length" and (hits or len(ids) != max_new_tokens))):
        raise ValueError("Generation violated explicit EOS, exact cap or stop-reason semantics")
    return {"generated_ids": ids, "stop_reason": reason}


def _backend_metadata(backend):
    value = copy.deepcopy(backend.metadata())
    if (not isinstance(value, dict) or value.get("protocol") != EVALUATION_PROTOCOL
            or value.get("reduction") != REDUCTION):
        raise ValueError("Backend does not declare the frozen-evaluation protocol and FP64 reduction")
    _positive_int(value.get("context_length"), "backend context_length", 2)
    _positive_int(value.get("vocab_size"), "backend vocab_size")
    _canonical(value)
    return value


def verify_tokenization(backend, manifest):
    """Require exact frozen IDs and raw token pieces for every declared text."""
    for text in manifest["texts"]:
        rendered = bytes.fromhex(text["rendered_utf8_hex"]).decode("utf-8")
        ids = backend.tokenize(rendered, parse_special=text["parse_special"])
        if validate_token_ids(ids) != text["input_ids"]:
            raise ValueError(f"Tokenization mismatch for frozen text {text['id']}")
        pieces = backend.token_pieces(list(text["input_ids"]))
        if (not isinstance(pieces, (list, tuple)) or any(not isinstance(piece, bytes) for piece in pieces)
                or [piece.hex() for piece in pieces] != text["token_pieces_hex"]):
            raise ValueError(f"Raw token-piece mismatch for frozen text {text['id']}")
    return {"texts": len(manifest["texts"]), "exact_ids": True, "exact_raw_pieces": True,
            "reference_identity": copy.deepcopy(manifest["tokenizer_identity"])}


def evaluate_manifest(backend, manifest):
    """Score frozen NLL/continuations and raw greedy generations, retaining items.

    The backend must reproduce every frozen token ID and raw piece first, so a
    different tokenizer container (GGUF vs HF files) is fine and a different
    tokenizer is refused before any scoring.
    """
    metadata = _backend_metadata(backend)
    frozen = validate_manifest(manifest, context_length=metadata["context_length"], vocab_size=metadata["vocab_size"])
    verified = verify_tokenization(backend, frozen)
    text_hashes = {text["id"]: text["rendered_sha256"] for text in frozen["texts"]}
    report = {"protocol": EVALUATION_PROTOCOL, "reduction": REDUCTION, "frozen_sha256": frozen["frozen_sha256"],
              "role": frozen["role"], "source_provenance": frozen["source_provenance"], "backend": metadata,
              "tokenization": verified, "nll": [], "choices": [], "generations": [],
              "legacy_fp32_cross_entropy_compatible": False,
              "semantics": {"score_positions": "target j uses full-vocabulary prediction at j-1",
                            "choice_reduction": "summed continuation log probability; first declared option wins ties",
                            "generation": "raw greedy argmax; lowest token ID wins ties; EOS included",
                            "state": "every window, option and generation is an independent backend call"}}
    def score(item):
        result = backend.score_tokens(list(item["input_ids"]), list(item["score_positions"]))
        return validate_score_result(result, item["input_ids"], item["score_positions"])
    for item in frozen["nll"]:
        report["nll"].append({**copy.deepcopy(item), **score(item), "rendered_sha256": text_hashes[item["text_id"]]})
    for item in frozen["choices"]:
        options = []
        for option in item["options"]:
            result = score(option)
            options.append({**copy.deepcopy(option), **result, "logprob_sum": math.fsum(result["target_logprobs"]),
                            "rendered_sha256": text_hashes[option["text_id"]]})
        winner = max(range(len(options)), key=lambda index: options[index]["logprob_sum"])
        selected = options[winner]["id"]
        report["choices"].append({"id": item["id"], "options": options, "selected_id": selected,
                                   "answer_id": item.get("answer_id"),
                                   "correct": selected == item["answer_id"] if "answer_id" in item else None})
    for item in frozen["generations"]:
        result = validate_generation_result(
            backend.generate(list(item["input_ids"]), item["max_new_tokens"], list(item["eos_ids"])),
            max_new_tokens=item["max_new_tokens"], eos_ids=item["eos_ids"], vocab_size=metadata["vocab_size"])
        pieces = backend.token_pieces(result["generated_ids"])
        if (not isinstance(pieces, (list, tuple)) or len(pieces) != len(result["generated_ids"])
                or any(not isinstance(piece, bytes) for piece in pieces)):
            raise ValueError("Backend must return a raw byte piece for every generated token")
        raw = b"".join(pieces)
        try:
            raw.decode("utf-8", errors="strict")
            complete = True
        except UnicodeDecodeError:
            complete = False
        report["generations"].append({**copy.deepcopy(item), **result, "token_pieces_hex": [p.hex() for p in pieces],
                                        "generated_bytes_hex": raw.hex(), "utf8_complete": complete,
                                        "display_text": raw.decode("utf-8", errors="replace"),
                                        "rendered_sha256": text_hashes[item["text_id"]]})
    count = sum(item["count"] for item in report["nll"])
    nll = math.fsum(item["nll_sum"] for item in report["nll"])
    mean = nll / count if count else None
    labelled = [item["correct"] for item in report["choices"] if item["correct"] is not None]
    report["aggregate"] = {"nll_sum": nll, "scored_tokens": count, "mean_nll": mean,
                           "perplexity": math.exp(mean) if mean is not None and mean < 709 else None,
                           "choice_correct": sum(labelled), "choice_labelled": len(labelled),
                           "choice_accuracy": sum(labelled) / len(labelled) if labelled else None,
                           "generations": len(report["generations"])}
    _canonical(report)
    return report


def assert_disjoint_manifests(*manifests):
    """Reject exact frozen text/token overlap across calibration/search/eval sets.

    This checks exact overlap, including frozen windows, not semantic benchmark
    contamination. Declared split provenance remains the caller's responsibility.
    """
    seen = {}
    for index, value in enumerate(manifests):
        manifest = validate_manifest(value)
        fingerprints = set()
        referenced = {item["text_id"] for field in ("nll", "generations") for item in manifest[field]}
        referenced.update(option["text_id"] for item in manifest["choices"] for option in item["options"])
        for text in manifest["texts"]:
            # Shared tokenizer-only probes do not constitute fitting/scoring
            # overlap. Original nonempty text still overlaps across templates.
            if text["id"] not in referenced:
                continue
            fingerprints.update(("bytes:" + text["rendered_sha256"], "ids:" + canonical_sha256(text["input_ids"])))
            if text["raw_utf8_hex"]:
                fingerprints.add("raw:" + text["raw_sha256"])
        for field in ("nll", "generations"):
            fingerprints.update("ids:" + canonical_sha256(item["input_ids"]) for item in manifest[field])
        for fingerprint in fingerprints:
            if fingerprint in seen:
                raise ValueError(f"Exact frozen-input overlap between manifests {seen[fingerprint]} and {index}")
            seen[fingerprint] = index
    return {"manifests": len(manifests), "exact_overlap": False,
            "semantic_contamination_checked": False}


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path, *, expected_file_sha256=None, context_length=None, vocab_size=None):
    """Read, optionally pin the exact file bytes, and validate a frozen manifest."""
    data = Path(path).read_bytes()
    if expected_file_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_file_sha256:
        raise ValueError(f"{path}: file SHA256 differs from the pinned value")
    return validate_manifest(json.loads(data.decode("utf-8")), context_length=context_length,
                             vocab_size=vocab_size)

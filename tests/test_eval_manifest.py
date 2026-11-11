"""Independent numerical/window oracles and frozen-manifest protocol checks."""
import copy
import hashlib
import math

import pytest
import torch

from openbonsai.eval.manifest import (
    EVALUATION_PROTOCOL, REDUCTION, assert_disjoint_manifests, canonical_sha256,
    evaluate_manifest, freeze_manifest, freeze_text, reduce_target_logprobs,
    sliding_windows, validate_generation_result, validate_manifest,
    validate_score_positions, validate_score_result,
)


SOURCE = {"source": "synthetic-document-fixture", "revision": "fixture-v1", "split": "unused-test",
          "content_sha256": hashlib.sha256(b"synthetic-document-fixture-v1").hexdigest()}
TOKENIZER = {"source": "raw-byte-fixture", "revision": "fixture-v1",
             "content_sha256": hashlib.sha256(b"raw-byte-fixture-v1").hexdigest()}


class Backend:
    """Tiny deterministic fake, with explicit independent-call score behavior."""
    def __init__(self):
        self.calls = []
        self.tokenization_calls = []
        self.generation_script = [67, 256]
        self.score_overrides = {}
        self.identity = {"protocol": EVALUATION_PROTOCOL, "reduction": REDUCTION,
                         "context_length": 64, "vocab_size": 260,
                         "cache_mode": "full-prefix-reference", "reset_policy": "per-record",
                         "source_identity": {"source": "fake-model", "sha256": "a" * 64},
                         "tokenizer_identity": TOKENIZER, "runtime": {"name": "pure-fixture"}}

    def metadata(self):
        return copy.deepcopy(self.identity)

    def tokenize(self, text, parse_special=False):
        self.tokenization_calls.append((text, parse_special))
        if parse_special and "<special>" in text:
            prefix, suffix = text.split("<special>", 1)
            return list(prefix.encode()) + [258] + self.tokenize(suffix, parse_special=True)
        return list(text.encode())

    def token_pieces(self, ids):
        special = {256: b"<eos1>", 257: b"<eos2>", 258: b"<special>", 259: b""}
        return [bytes([token]) if token < 256 else special[token] for token in ids]

    @staticmethod
    def logits(prefix):
        result = torch.arange(260, dtype=torch.float32) / 100
        result[(sum(prefix) + len(prefix)) % 260] += 3
        return result

    def score_tokens(self, input_ids, score_positions):
        self.calls.append(("score", list(input_ids), list(score_positions)))
        override = self.score_overrides.get(tuple(input_ids))
        logprobs = (list(override) if override is not None else reduce_target_logprobs(
            torch.stack([self.logits(input_ids[:j]) for j in score_positions]),
            [input_ids[j] for j in score_positions]))
        return {"score_positions": list(score_positions), "target_logprobs": logprobs,
                "nll_sum": -math.fsum(logprobs), "count": len(logprobs)}

    def generate(self, input_ids, max_new_tokens, eos_ids):
        self.calls.append(("generate", list(input_ids), max_new_tokens, list(eos_ids)))
        generated = []
        for index in range(max_new_tokens):
            token = self.generation_script[index % len(self.generation_script)]
            generated.append(token)
            if token in eos_ids:
                return {"generated_ids": generated, "stop_reason": "eos"}
        return {"generated_ids": generated, "stop_reason": "length"}


def text(backend, key, rendered, *, raw=None, parse_special=False):
    ids = backend.tokenize(rendered, parse_special=parse_special)
    return freeze_text(key, raw_text=rendered if raw is None else raw, rendered_text=rendered,
                       input_ids=ids, token_pieces=backend.token_pieces(ids), parse_special=parse_special)


def manifest(backend=None, *, role="evaluation"):
    backend = Backend() if backend is None else backend
    first, second = text(backend, "document-a", "abcdefg"), text(backend, "document-b", "hi")
    first_option, second_option = text(backend, "option-a", "Q: AB"), text(backend, "option-b", "Q: C")
    prompt = text(backend, "prompt", "Hi")
    nll = [{"id": f"a-{i}", "text_id": first["id"], **window}
           for i, window in enumerate(sliding_windows(first["input_ids"], context_length=5, stride=2))]
    nll += [{"id": "b-0", "text_id": second["id"], **sliding_windows(second["input_ids"], context_length=5, stride=2)[0]}]
    options = [{"id": label, "text_id": item["id"], "input_ids": item["input_ids"],
                "prefix_length": 3, "score_positions": list(range(3, len(item["input_ids"])))}
               for label, item in (("A", first_option), ("B", second_option))]
    return freeze_manifest(source_provenance=SOURCE, tokenizer_identity=TOKENIZER,
        texts=[first, second, first_option, second_option, prompt], nll=nll,
        choices=[{"id": "question", "options": options, "answer_id": "B"}],
        generations=[{"id": "generation", "text_id": "prompt", "input_ids": prompt["input_ids"],
                      "max_new_tokens": 3, "eos_ids": [256, 257]}], role=role)


def reseal(value):
    value["frozen_sha256"] = canonical_sha256({k: v for k, v in value.items() if k != "frozen_sha256"})
    return value


def independent_logprob(logits, target):
    values = [float(v) for v in logits]
    maximum = max(values)
    centered = [value - maximum for value in values]
    return centered[target] - math.log(math.fsum(math.exp(value) for value in centered))


def test_fp64_reference_matches_independent_full_vocabulary_formula_and_target_shift():
    logits = torch.tensor([[0., 1., -2.], [2., -1., .5], [-2., -2., -2.]], dtype=torch.float32)
    targets = [2, 0, 1]
    actual = reduce_target_logprobs(logits, targets)
    expected = [independent_logprob(row, target) for row, target in zip(logits, targets)]
    assert actual == pytest.approx(expected, abs=1e-14)
    backend = Backend()
    ids, positions = [3, 7, 19, 2], [1, 3]
    result = backend.score_tokens(ids, positions)
    expected = [independent_logprob(backend.logits(ids[:j]), ids[j]) for j in positions]
    assert result["target_logprobs"] == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("magnitude", [0., 1e12, 1e38, -1e38])
def test_uniform_huge_logits_preserve_log_vocabulary_normalizer(magnitude):
    logits = torch.full((2, 13), magnitude, dtype=torch.float32)
    assert reduce_target_logprobs(logits, [0, 12]) == pytest.approx([-math.log(13)] * 2, abs=1e-14)


def test_nonuniform_reduction_offset_invariant_and_not_legacy_fp32_ce():
    logits = torch.tensor([[1., -2., 7., 0.], [-4., -3., 1., 0.]], dtype=torch.float32)
    expected = reduce_target_logprobs(logits, [0, 3])
    for offset in (-65536., 65536.):
        assert reduce_target_logprobs(logits + offset, [0, 3]) == pytest.approx(expected, abs=1e-14)
    legacy = -torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3]), reduction="none")
    assert any(float(a) != b for a, b in zip(legacy, expected))


@pytest.mark.parametrize("logits,targets", [
    (torch.ones(2, 3, dtype=torch.float64), [0, 1]),
    (torch.ones(3), [0]), (torch.ones(0, 3), []),
    (torch.tensor([[float("nan"), 0.]]), [0]),
    (torch.tensor([[float("inf"), 0.]]), [0]),
    (torch.ones(2, 3), [0]), (torch.ones(1, 3), [3]),
    (torch.ones(1, 3), [True]), (torch.ones(1, 3), torch.tensor([1.])),
])
def test_invalid_reduction_inputs_fail_closed(logits, targets):
    with pytest.raises(ValueError):
        reduce_target_logprobs(logits, targets)


@pytest.mark.parametrize("length", [2, 4, 5, 6, 9, 10, 11, 17, 39])
@pytest.mark.parametrize("context,stride", [(2, 1), (5, 1), (5, 2), (5, 4), (8, 3)])
def test_windows_match_independent_target_partition_and_keep_overlap_visible(length, context, stride):
    ids = list(range(length))
    windows = sliding_windows(ids, context_length=context, stride=stride)
    actual = []
    for index, window in enumerate(windows):
        start, end = window["token_start"], window["token_end"]
        assert window["input_ids"] == ids[start:end]
        assert 2 <= len(window["input_ids"]) <= context
        assert all(position >= 1 for position in window["score_positions"])
        actual.extend(start + j for j in window["score_positions"])
        if index:
            previous = windows[index - 1]
            assert start < previous["token_end"]  # Context was not deleted by a loss mask.
            assert start + window["score_positions"][0] == previous["token_end"]
    # Independent membership oracle: every eligible global target belongs to
    # exactly one interval ending at context + k*stride (clipped at the tail).
    expected = []
    for target in range(1, length):
        owner = 0 if target < context else (target - context) // stride + 1
        expected.append(owner)
    observed_owners = [index for index, w in enumerate(windows) for _ in w["score_positions"]]
    assert actual == list(range(1, length))
    assert observed_owners == expected
    assert windows[-1]["token_end"] == length


@pytest.mark.parametrize("kwargs", [{"context_length": 1}, {"stride": 0}, {"context_length": 3, "stride": 3},
                                    {"context_length": True}, {"stride": 1.5}])
def test_invalid_windows_rejected(kwargs):
    with pytest.raises(ValueError):
        sliding_windows([1, 2, 3], **kwargs)


@pytest.mark.parametrize("ids", [[], [1]])
def test_no_eligible_document_targets_rejected(ids):
    with pytest.raises(ValueError):
        sliding_windows(ids)


def test_length_weighted_nll_and_summed_multitoken_choices_preserve_individual_items():
    backend = Backend()
    frozen = manifest(backend)
    for item, logprobs in zip(frozen["nll"], [[-1.] * 4, [-10.] * 2, [-.5]]):
        backend.score_overrides[tuple(item["input_ids"])] = logprobs
    options = frozen["choices"][0]["options"]
    backend.score_overrides[tuple(options[0]["input_ids"])] = [-.5, -.5]
    backend.score_overrides[tuple(options[1]["input_ids"])] = [-.75]
    report = evaluate_manifest(backend, frozen)
    assert report["aggregate"]["nll_sum"] == 24.5
    assert report["aggregate"]["scored_tokens"] == 7
    assert report["aggregate"]["mean_nll"] == 3.5
    assert report["choices"][0]["selected_id"] == "B"  # Sum, not per-option token average.
    assert report["aggregate"]["choice_accuracy"] == 1.
    assert report["choices"][0]["options"][0]["target_logprobs"] == [-.5, -.5]
    assert report["generations"][0]["generated_ids"] == [67, 256]
    assert report["generations"][0]["stop_reason"] == "eos"
    assert report["frozen_sha256"] == frozen["frozen_sha256"]
    assert report["source_provenance"] == SOURCE
    assert report["backend"]["source_identity"] == backend.identity["source_identity"]
    assert not report["legacy_fp32_cross_entropy_compatible"]
    assert [call[0] for call in backend.calls] == ["score"] * 5 + ["generate"]
    assert [call[1:] for call in backend.calls[:3]] == [(item["input_ids"], item["score_positions"]) for item in frozen["nll"]]


def test_choice_ties_use_frozen_order_without_retokenizing_boundary():
    backend = Backend()
    frozen = manifest(backend)
    options = frozen["choices"][0]["options"]
    backend.score_overrides[tuple(options[0]["input_ids"])] = [-.5, -.5]
    backend.score_overrides[tuple(options[1]["input_ids"])] = [-1.]
    backend.tokenization_calls.clear()
    report = evaluate_manifest(backend, frozen)
    assert report["choices"][0]["selected_id"] == "A"
    rendered = [bytes.fromhex(t["rendered_utf8_hex"]).decode() for t in frozen["texts"]]
    assert [call[0] for call in backend.tokenization_calls] == rendered
    assert "AB" not in rendered and "C" not in rendered  # No separately guessed answer tokenization.


@pytest.mark.parametrize("value,parse_special", [
    ("Türkçe: ı İ ş ğ ç ö ü", False), ("e\u0301 versus é", False),
    ("🪴🙂", False), ("def f(x):\n\treturn x + 1\n", False),
    (" A\n B\t", False), ("<special>", False), ("<special>", True),
])
def test_raw_rendered_and_token_piece_identities_cover_text_edge_cases(value, parse_special):
    backend = Backend()
    frozen_text = text(backend, "probe", value, raw="raw source", parse_special=parse_special)
    frozen = freeze_manifest(source_provenance=SOURCE, tokenizer_identity=TOKENIZER, texts=[frozen_text],
        generations=[{"id": "g", "text_id": "probe", "input_ids": frozen_text["input_ids"],
                      "max_new_tokens": 2, "eos_ids": [256, 257]}])
    report = evaluate_manifest(backend, frozen)
    assert report["tokenization"]["exact_ids"] and report["tokenization"]["exact_raw_pieces"]
    assert frozen_text["raw_sha256"] == hashlib.sha256(b"raw source").hexdigest()
    assert bytes.fromhex(frozen_text["rendered_utf8_hex"]).decode() == value
    assert b"".join(bytes.fromhex(piece) for piece in frozen_text["token_pieces_hex"]) == value.encode()


def test_incomplete_utf8_at_exact_cap_is_preserved_without_extra_tokens():
    backend = Backend()
    backend.generation_script = [0xf0, 0x9f]
    frozen = manifest(backend)
    frozen["generations"][0]["max_new_tokens"] = 2
    reseal(frozen)
    result = evaluate_manifest(backend, frozen)["generations"][0]
    assert result["generated_ids"] == [0xf0, 0x9f]
    assert result["generated_bytes_hex"] == "f09f" and not result["utf8_complete"]
    assert result["stop_reason"] == "length"


@pytest.mark.parametrize("change", ["ids", "raw", "rendered", "answer", "provenance"])
def test_manifest_hash_binds_ids_bytes_answers_and_provenance(change):
    frozen = manifest()
    if change == "ids": frozen["texts"][0]["input_ids"][0] += 1
    elif change == "raw": frozen["texts"][0]["raw_utf8_hex"] += "20"
    elif change == "rendered": frozen["texts"][0]["rendered_utf8_hex"] += "20"
    elif change == "answer": frozen["choices"][0]["answer_id"] = "A"
    elif change == "provenance": frozen["source_provenance"]["revision"] = "different"
    with pytest.raises(ValueError, match="checksum"):
        validate_manifest(frozen)


@pytest.mark.parametrize("change", ["skip", "duplicate", "position_zero", "boundary", "prefix", "mask", "empty_scores", "duplicate_id", "unknown_text", "bad_pieces"])
def test_resealed_but_invalid_frozen_contract_rejected(change):
    frozen = manifest()
    if change == "skip": frozen["nll"][0]["score_positions"].pop()
    elif change == "duplicate": frozen["nll"][1]["score_positions"] = [1, 2, 3, 4]
    elif change == "position_zero": frozen["nll"][0]["score_positions"] = [0, 1, 2, 3, 4]
    elif change == "boundary": frozen["choices"][0]["options"][0]["prefix_length"] = 4
    elif change == "prefix": frozen["choices"][0]["options"][1]["prefix_length"] = 2
    elif change == "mask": frozen["nll"][0]["attention_mask"] = [0] * 5
    elif change == "empty_scores": frozen["nll"][0]["score_positions"] = []
    elif change == "duplicate_id": frozen["generations"][0]["id"] = frozen["nll"][0]["id"]
    elif change == "unknown_text": frozen["generations"][0]["text_id"] = "absent"
    elif change == "bad_pieces": frozen["texts"][0]["token_pieces_hex"][0] = "FF"
    with pytest.raises(ValueError):
        validate_manifest(reseal(frozen))


@pytest.mark.parametrize("change", ["tokens", "pieces", "old_reduction", "context", "generation_context", "vocab"])
def test_backend_mismatch_rejected_before_scoring(monkeypatch, change):
    backend = Backend()
    frozen = manifest(backend)
    if change == "tokens": monkeypatch.setattr(backend, "tokenize", lambda *a, **k: [1])
    elif change == "pieces": monkeypatch.setattr(backend, "token_pieces", lambda ids: [b"wrong"] * len(ids))
    elif change == "old_reduction": backend.identity["reduction"] = "fp32-ce-then-fp64-sum"
    elif change == "context": backend.identity["context_length"] = 4
    elif change == "generation_context": frozen["generations"][0]["max_new_tokens"] = 63; reseal(frozen)
    elif change == "vocab": backend.identity["vocab_size"] = 7
    with pytest.raises(ValueError):
        evaluate_manifest(backend, frozen)
    assert not backend.calls


def test_backend_different_tokenizer_container_hash_allowed_only_after_exact_tokens():
    backend = Backend()
    backend.identity["tokenizer_identity"] = {"source": "GGUF-tokenizer", "sha256": "b" * 64}
    result = evaluate_manifest(backend, manifest(backend))
    assert result["tokenization"]["reference_identity"] == TOKENIZER
    assert result["backend"]["tokenizer_identity"]["sha256"] == "b" * 64


@pytest.mark.parametrize("change", ["nan", "inf", "positive", "count", "sum", "positions", "bool_position", "missing"])
def test_invalid_backend_probability_result_fails_closed(change):
    result = {"score_positions": [1, 2], "target_logprobs": [-.5, -1.], "nll_sum": 1.5, "count": 2}
    if change == "nan": result["target_logprobs"][0] = float("nan")
    elif change == "inf": result["nll_sum"] = float("inf")
    elif change == "positive": result["target_logprobs"][0] = .1
    elif change == "count": result["count"] = True
    elif change == "sum": result["nll_sum"] = .75
    elif change == "positions": result["score_positions"] = [2, 1]
    elif change == "bool_position": result["score_positions"] = [True, 2]
    elif change == "missing": result.pop("target_logprobs")
    with pytest.raises(ValueError):
        validate_score_result(result, [3, 4, 5], [1, 2])


@pytest.mark.parametrize("ids,reason,cap", [
    ([], "eos", 3), ([1, 9, 2], "eos", 3), ([9, 9], "eos", 3),
    ([1, 2], "eos", 3), ([1], "length", 3), ([1, 9], "length", 2),
    ([1, 2, 3, 4], "length", 3), ([1], "stop-string", 3),
])
def test_invalid_generation_eos_length_or_stop_reason_rejected(ids, reason, cap):
    with pytest.raises(ValueError):
        validate_generation_result({"generated_ids": ids, "stop_reason": reason},
                                   max_new_tokens=cap, eos_ids=[9, 10], vocab_size=20)


@pytest.mark.parametrize("eos", [9, 10])
def test_both_eos_ids_retained_and_length_cap_exact(eos):
    assert validate_generation_result({"generated_ids": [1, eos], "stop_reason": "eos"},
        max_new_tokens=2, eos_ids=[9, 10])["generated_ids"] == [1, eos]
    assert validate_generation_result({"generated_ids": [1, 2], "stop_reason": "length"},
        max_new_tokens=2, eos_ids=[9, 10])["stop_reason"] == "length"


def test_exact_split_overlap_rejected_and_disjoint_sets_preserve_roles():
    calibration = manifest(role="calibration")
    evaluation = manifest(role="evaluation")
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_manifests(calibration, evaluation)
    backend = Backend()
    independent = text(backend, "new", "completely different")
    search = freeze_manifest(source_provenance={**SOURCE, "split": "search"}, tokenizer_identity=TOKENIZER,
        texts=[independent], role="search_validation", generations=[{"id": "other", "text_id": "new",
            "input_ids": independent["input_ids"], "max_new_tokens": 2, "eos_ids": [256]}])
    result = assert_disjoint_manifests(calibration, search)
    assert result["manifests"] == 2 and not result["semantic_contamination_checked"]


def test_raw_text_overlap_survives_different_templates_but_unscored_probes_do_not_count():
    backend = Backend()
    probe = text(backend, "shared-probe", "Turkish ı🙂")
    def frozen(raw, rendered, role):
        value = text(backend, "actual-input", rendered, raw=raw)
        return freeze_manifest(source_provenance=SOURCE, tokenizer_identity=TOKENIZER, texts=[value, probe],
            role=role, generations=[{"id": "g", "text_id": "actual-input", "input_ids": value["input_ids"],
                                     "max_new_tokens": 2, "eos_ids": [256]}])
    calibration = frozen("same question", "User: same question", "calibration")
    evaluation = frozen("same question", "[USER]same question[ASSISTANT]", "evaluation")
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_manifests(calibration, evaluation)
    disjoint = frozen("unrelated question", "[USER]unrelated question", "evaluation")
    assert not assert_disjoint_manifests(calibration, disjoint)["exact_overlap"]


def test_dev_asset_manifest_is_pinned_and_valid():
    from openbonsai.eval.manifest import load_manifest
    from openbonsai.eval.suites import DEV_MANIFEST, DEV_SHA256
    frozen = load_manifest(DEV_MANIFEST, expected_file_sha256=DEV_SHA256)
    assert frozen["frozen_sha256"] == "8021d9cc8f54943a3b816af74cbc41d588173d385f51b73a681e979393ef5fdf"
    assert (len(frozen["texts"]), len(frozen["nll"]), len(frozen["choices"]), len(frozen["generations"])) == (88, 4, 20, 4)
    assert sum(len(r["score_positions"]) for r in frozen["nll"]) == 2048


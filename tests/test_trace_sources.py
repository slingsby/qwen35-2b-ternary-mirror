"""Multi-source traces (traces.sources): prompt kinds, mixed thinking, metadata, the per-batch cache
(resume, out-of-memory splits), the test-data guard, the trace supply checks and the reasoning example."""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from openbonsai import cli
from openbonsai.config import ConfigError, apply_set, load_config, stage_inputs
from openbonsai.eval import heldout as H
from openbonsai.eval import reasoning as R
from openbonsai.stages import traces as T

ROOT = Path(__file__).resolve().parents[1]
REASONING = ROOT / "examples" / "qwen3_5-9b-reasoning.yaml"


def _parquet(path, columns: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return path


@pytest.fixture
def data(tmp_path):
    """Small local stand-ins for every prompt kind (same columns as the hub files)."""
    d = tmp_path / "data"
    gsm = _parquet(d / "gsm8k.parquet", {
        "question": [f"Ann has {k} apples and buys {k + 3} more. How many apples does she have now?" for k in range(6)],
        "answer": ["#### 0"] * 6})
    for subject, levels in (("algebra", ["Level 1", "Level 2", "Level 5"]), ("geometry", ["Level 2", "Level 4", "Level 1"])):
        _parquet(d / "math" / subject / "train-00000-of-00001.parquet", {
            "problem": [f"{subject} problem {k}: solve x + {k} = {2 * k}." for k in range(3)], "level": levels,
            "type": [subject] * 3, "solution": ["x"] * 3})
    mbpp = _parquet(d / "mbpp.parquet", {
        "task_id": [601, 602, 603], "text": ["Write a function to add two numbers.", "Write a function to reverse a list.",
                                            "Write a function to square a number."],
        "code": ["", "", ""], "test_list": [["assert add(1, 2) == 3", "assert add(0, 0) == 0"], ["assert rev([1, 2]) == [2, 1]"],
                                            ["assert sq(3) == 9"]],
        "test_setup_code": ["", "import math", ""], "challenge_test_list": [[], [], []]})
    oci = _parquet(d / "oci.parquet", {"id": ["a1", "b2", "c3", "d4"],
                                       "input": ["Implement fizzbuzz for n.", "Parse a CSV line into fields. " * 20,
                                                 "Return the n-th Fibonacci number.", "Sort a list of words by length."]})
    chat = _parquet(d / "ultrachat.parquet", {"messages": [
        [{"role": "user", "content": f"Tell me about river number {k}."}, {"role": "assistant", "content": "It flows."}]
        for k in range(8)]})
    generic = _parquet(d / "generic.parquet", {"instruction": ["Name a prime.", "Name a colour.", "Name a tree."]})
    jsonl = d / "q.jsonl"
    jsonl.write_text("".join(json.dumps({"q": q}) + "\n" for q in ("First question?", "Second question?")),
                     encoding="utf-8")
    return {"gsm": gsm, "math": d / "math", "mbpp": mbpp, "oci": oci, "chat": chat, "generic": generic, "jsonl": jsonl}


def mixed_sources(data) -> list[dict]:
    p = {k: Path(v).as_posix() for k, v in data.items()}
    return [
        {"kind": "gsm8k_train", "path": p["gsm"], "n": 3, "seed": 1, "thinking": True, "max_new_tokens": 10, "batch": 2},
        {"kind": "math_train", "path": p["math"], "n": 3, "levels": [1, 2], "thinking": "on", "max_new_tokens": 8,
         "batch": 2},
        {"kind": "mbpp_train", "path": p["mbpp"], "n": 2, "max_new_tokens": 6, "batch": 4},
        {"kind": "opencodeinstruct", "path": p["oci"], "n": 3, "max_chars": 200, "thinking": "off",
         "max_new_tokens": 6, "batch": 1},
        {"kind": "ultrachat", "path": p["chat"], "rows": [2, 8], "n": 2, "max_new_tokens": 5},
        {"kind": "parquet", "path": p["generic"], "column": "instruction", "n": 2, "instruction": "Answer briefly.",
         "name": "generic"},
        {"kind": "jsonl", "path": p["jsonl"], "column": "q", "n": 1, "shuffle": False},
    ]


def sources_sets(sources, *more) -> list[str]:
    return ["traces.prompts=null", "traces.sources=" + json.dumps(sources), *more]


def _rows(ctx):
    path = ctx.run.stage_dir("traces") / "traces.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _capture(ctx):
    msgs = []
    log = ctx.log
    ctx.log = lambda stage: (lambda m: (msgs.append(m), log(stage)(m)))
    return msgs


# -- one run with every kind ---------------------------------------------------------------

def test_multi_source_traces_mix_thinking_and_record_what_each_trace_is(make_ctx, tmp_path, data):
    from openbonsai.cli import execute
    ctx = make_ctx(tmp_path / "run", *sources_sets(mixed_sources(data)))
    assert execute(ctx, "traces") == "ran"
    adapter, tok = ctx.adapter(), ctx.load_tokenizer()
    eos = set(adapter.eos_ids(tok, ctx.load_model()[0]))
    rows = _rows(ctx)
    expected = {"gsm8k_train": (3, True, 10), "math_train": (3, True, 8), "mbpp_train": (2, False, 6),
                "opencodeinstruct": (3, False, 6), "ultrachat": (2, False, 5), "generic": (2, False, 24),
                "jsonl": (1, False, 24)}
    assert len(rows) == sum(n for n, _, _ in expected.values())
    for r in rows:
        n, thinking, cap = expected[r["source"]]
        assert (r["thinking"], r["max_new_tokens"]) == (thinking, cap)
        # The stored ids are the prompt in the model's own template (with this source's thinking switch)
        # followed by exactly what the model generated.
        prompt_ids = T.encode_chat(tok, [{"role": "user", "content": r["prompt"]}], adapter.chat_kwargs(thinking=thinking))
        assert r["input_ids"][:r["prompt_token_count"]] == prompt_ids
        gen = r["input_ids"][r["prompt_token_count"]:]
        assert len(gen) == r["generated_token_count"] <= cap and r["attention_mask"] == [1] * len(r["input_ids"])
        assert r["hit_cap"] == (gen[-1] not in eos) and tok.decode(gen, skip_special_tokens=False) == r["completion"]
        assert ("finished_thinking" in r) == thinking
        assert tok.decode(prompt_ids).endswith("<think>\n\n</think>\n\n") != thinking   # the tiny template's switch
    by = {}
    for r in rows:
        by.setdefault(r["source"], []).append(r)
    assert {k: len(v) for k, v in by.items()} == {k: v[0] for k, v in expected.items()}
    assert all(r["prompt"].endswith("\n" + R.GSM8K_INSTRUCTION) for r in by["gsm8k_train"] + by["math_train"])
    assert T.MATH_INSTRUCTION == R.GSM8K_INSTRUCTION
    math_rows = [r["row"] for r in by["math_train"]]           # levels 1 and 2 only
    assert len(set(math_rows)) == 3 and set(math_rows) <= {"algebra/0", "algebra/1", "geometry/0", "geometry/2"}
    mbpp = {r["id"]: r["prompt"] for r in by["mbpp_train"]}
    assert set(mbpp) <= {601, 602, 603}
    if 602 in mbpp:   # the setup code goes before the one test
        assert mbpp[602].endswith("Your code should pass this test:\n\nimport math\nassert rev([1, 2]) == [2, 1]")
    for task, text, test in ((601, "Write a function to add two numbers.", "assert add(1, 2) == 3"),
                             (603, "Write a function to square a number.", "assert sq(3) == 9")):
        if task in mbpp:
            assert mbpp[task] == T.MBPP_TEMPLATE.format(text=text, test=test)
    assert sorted(r["id"] for r in by["opencodeinstruct"]) == ["a1", "c3", "d4"]          # b2 is over max_chars
    assert all(r["prompt"].startswith("Tell me about river number") and 2 <= r["row"] < 8 for r in by["ultrachat"])
    assert all(r["prompt"].endswith("\nAnswer briefly.") for r in by["generic"])
    assert [(r["row"], r["prompt"]) for r in by["jsonl"]] == [(1, "First question?")]     # shuffle: false
    # The sources are interleaved (seeded), so any prefix of the file holds every source.
    order = [r["source"] for r in rows]
    assert order != sorted(order, key=list(expected).index)
    summary = json.loads((ctx.run.stage_dir("traces") / "summary.json").read_text(encoding="utf-8"))
    assert summary["form"] == "sources" and summary["records"] == len(rows) and summary["seed"] == 0
    per = {s["name"]: s for s in summary["sources"]}
    assert per["math_train"]["subjects"] == ["algebra", "geometry"] and per["gsm8k_train"]["thinking"] is True
    assert per["opencodeinstruct"]["max_chars"] == 200 and per["generic"]["instruction"] == "Answer briefly."
    assert all(len(f["sha256"]) == 64 for s in summary["sources"] for f in s["data"])
    assert summary["tokens"] == sum(len(r["input_ids"]) for r in rows)
    assert {d["where"] for d in summary["supply"]} == {"calibrate.sources.2.n", "distill.teacher.sources.1.n"}
    # Another interleaving seed reorders the same traces without generating anything.
    first = sorted(json.dumps(r, sort_keys=True) for r in rows)
    ctx2 = make_ctx(tmp_path / "run", *sources_sets(mixed_sources(data), "traces.seed=7"))
    msgs = _capture(ctx2)
    assert execute(ctx2, "traces") == "ran"
    assert not any(": batch " in m for m in msgs)
    rows2 = _rows(ctx2)
    assert sorted(json.dumps(r, sort_keys=True) for r in rows2) == first and rows2 != rows


def test_sources_draw_distinct_prompts_in_a_seeded_order(data):
    src = T.trace_sources(load_config(None, sources_sets([{"kind": "gsm8k_train", "path": str(data["gsm"]), "n": 4,
                                                           "seed": 3}])))[0]
    rows = T._rows(src, {"": data["gsm"]})
    a, _ = T._sample(src, T._candidates(src, rows), screen=True)
    b, _ = T._sample({**src, "n": 6}, T._candidates(src, rows), screen=True)
    assert [x[0] for x in b[:4]] == [x[0] for x in a]          # a larger n extends a smaller one
    c, _ = T._sample({**src, "seed": 4}, T._candidates(src, rows), screen=True)
    assert [x[0] for x in c] != [x[0] for x in a]
    dup = rows + [(99, rows[0][1].replace(" ", "  "), None)]   # same text up to whitespace
    assert len(T._candidates(src, dup)) == len(rows)
    with pytest.raises(ValueError, match="only 6 usable prompts, 7 requested"):
        T._sample({**src, "n": 7}, T._candidates(src, rows), screen=True)


def test_the_original_ultrachat_form_picks_the_prompts_it_always_picked(tmp_path):
    """traces.prompts kind ultrachat: first user turns of rows [a, b), deduplicated, shuffled with the seed."""
    import random
    msgs = [[{"role": "user", "content": f"Question {k % 5}?"}, {"role": "assistant", "content": "A."}]
            for k in range(30)] + [[{"role": "assistant", "content": "no user turn"}]]
    path = _parquet(tmp_path / "chat.parquet", {"messages": msgs})
    seen, pool = set(), []                       # the implementation before traces.sources existed
    for m in pq.read_table(path, columns=["messages"]).column("messages").to_pylist()[3:31]:
        first = next((x["content"] for x in m if x.get("role") == "user"), None)
        key = " ".join(str(first).split()) if first else None
        if key and key not in seen:
            seen.add(key)
            pool.append([{"role": "user", "content": first}])
    random.Random(11).shuffle(pool)
    assert T.ultrachat_prompts(path, rows=(3, 31), n=4, seed=11) == pool[:4]
    with pytest.raises(ValueError, match=r"only 5 distinct UltraChat prompts in rows \(3, 31\)"):
        T.ultrachat_prompts(path, rows=(3, 31), n=6, seed=11)


# -- batching, resume and the cache ----------------------------------------------------------

def test_batches_hold_the_longest_prompts_first():
    assert T.plan_batches([3, 9, 5, 9, 1], 2, sort=True) == [[1, 3], [2, 0], [4]]
    assert T.plan_batches([3, 9, 5, 9, 1], 2, sort=False) == [[0, 1], [2, 3], [4]]
    # traces.batch_tokens: rows x (longest prompt + cap) stays within the budget; a row over it alone is
    # a batch of one; without a budget nothing changes.
    lengths = [30, 90, 50, 10, 10, 10, 10, 200]
    assert T.plan_batches(lengths, 4, sort=True, cap=10, budget=240) == [[7], [1, 2], [0, 3, 4, 5], [6]]
    assert T.plan_batches(lengths, 4, sort=False, cap=10, budget=240) == [[0, 1], [2, 3, 4, 5], [6], [7]]
    for sort in (True, False):
        assert T.plan_batches(lengths, 4, sort=sort, cap=10) == T.plan_batches(lengths, 4, sort=sort)
        assert T.plan_batches(lengths, 4, sort=sort, cap=10, budget=10**6) == T.plan_batches(lengths, 4, sort=sort)
    for batch in T.plan_batches(lengths, 4, sort=True, cap=10, budget=240)[1:]:
        assert len(batch) * (max(lengths[i] for i in batch) + 10) <= 240


def test_a_token_budget_shrinks_the_batches_of_long_prompts(make_ctx, tmp_path, data):
    """traces.batch_tokens splits the batches that would hold the most tokens; the traces are the same rows."""
    sources = [{"kind": "ultrachat", "path": Path(data["chat"]).as_posix(), "rows": [0, 8], "n": 6,
                "max_new_tokens": 4, "batch": 6}]
    whole = make_ctx(tmp_path / "whole", *sources_sets(sources))
    T.run(whole)
    widest = max(r["prompt_token_count"] for r in _rows(whole))
    budget = 3 * (widest + 4)
    split = make_ctx(tmp_path / "split", *sources_sets(sources, f"traces.batch_tokens={budget}"))
    summary = T.run(split)
    assert summary["sources"][0]["batches"] == 2 and summary["sources"][0]["batch_tokens"] == budget
    assert [r["row"] for r in _rows(split)] == [r["row"] for r in _rows(whole)]
    assert stage_inputs(split.cfg, "traces")["traces"]["batch_tokens"] == budget
    assert "batch_tokens" not in stage_inputs(whole.cfg, "traces")["traces"]      # unset: the hash is unchanged
    with pytest.raises(ConfigError, match="traces.batch_tokens must be a positive integer"):
        T.trace_sources(load_config(None, sources_sets(sources, "traces.batch_tokens=0")))


def test_an_interrupted_multi_source_run_resumes_after_its_last_batch(make_ctx, tmp_path, data, monkeypatch):
    from openbonsai.cli import execute
    sources = mixed_sources(data)[:4]
    reference = make_ctx(tmp_path / "ref", *sources_sets(sources))
    execute(reference, "traces")
    real = T.generate_encoded
    calls = []

    def crashing(model, rows, **kw):
        if len(calls) == 3:
            raise KeyboardInterrupt("power cut")
        calls.append([len(r) for r in rows])
        return real(model, rows, **kw)

    monkeypatch.setattr(T, "generate_encoded", crashing)
    ctx = make_ctx(tmp_path / "run", *sources_sets(sources))
    with pytest.raises(KeyboardInterrupt):
        execute(ctx, "traces")
    cache = ctx.run.stage_dir("traces") / "cache"
    assert len(list(cache.glob("*.json"))) == 3 and not (ctx.run.stage_dir("traces") / "traces.jsonl").exists()
    for f in cache.glob("*.json"):           # each batch's prompts, longest first
        lengths = json.loads(f.read_text())["prompt_tokens"]
        assert lengths == sorted(lengths, reverse=True)
    monkeypatch.setattr(T, "generate_encoded", lambda model, rows, **kw: (calls.append(None), real(model, rows, **kw))[1])
    assert execute(ctx, "traces") == "ran"
    n_batches = json.loads((ctx.run.stage_dir("traces") / "summary.json").read_text())["batches"]
    assert len(calls) == n_batches and n_batches == 2 + 2 + 1 + 3    # 3 before the cut, the rest after
    assert (ctx.run.stage_dir("traces") / "traces.jsonl").read_bytes() == \
        (reference.run.stage_dir("traces") / "traces.jsonl").read_bytes()
    # A config edit regenerates only the batches it changes (here one source's cap).
    calls.clear()
    edited = make_ctx(tmp_path / "run", *sources_sets(sources, "traces.sources.2.max_new_tokens=5"))
    assert execute(edited, "traces") == "ran" and len(calls) == 1
    # --force clears the cache.
    calls.clear()
    assert execute(edited, "traces", force=True) == "ran" and len(calls) == n_batches


def test_a_batch_out_of_gpu_memory_is_generated_in_halves(make_ctx, tmp_path, data, monkeypatch):
    from openbonsai.cli import execute
    sources = [dict(s, batch=3) for s in mixed_sources(data)[:2]]
    single = make_ctx(tmp_path / "single", *sources_sets([dict(s, batch=1) for s in sources]))
    execute(single, "traces")
    real = T.generate_encoded
    sizes = []

    def small_gpu(model, rows, **kw):
        sizes.append(len(rows))
        if len(rows) > 1:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(model, rows, **kw)

    monkeypatch.setattr(T, "generate_encoded", small_gpu)
    ctx = make_ctx(tmp_path / "run", *sources_sets(sources))
    msgs = _capture(ctx)
    assert execute(ctx, "traces") == "ran"
    assert any("out of memory at batch 3" in m for m in msgs) and 3 in sizes
    assert len(list((ctx.run.stage_dir("traces") / "cache").glob("*.split"))) >= 2
    # Halves of a sorted batch are the batch-1 run's batches: the same traces.
    assert (ctx.run.stage_dir("traces") / "traces.jsonl").read_bytes() == \
        (single.run.stage_dir("traces") / "traces.jsonl").read_bytes()
    sizes.clear()
    ctx.run.clear_done("traces")
    assert execute(ctx, "traces") == "ran" and sizes == []   # the markers lead straight to the cached halves


def test_generation_runs_under_a_vram_cap_that_is_lifted_after(make_ctx, tmp_path, data, monkeypatch):
    from contextlib import contextmanager
    from openbonsai.cli import execute
    events = []

    @contextmanager
    def cap(device, log):
        events.append("cap")
        try:
            yield 30.0
        finally:
            events.append("lift")

    real = T.generate_encoded
    monkeypatch.setattr(T, "vram_cap", cap)
    monkeypatch.setattr(T, "generate_encoded", lambda model, rows, **kw: (events.append("batch"), real(model, rows, **kw))[1])
    ctx = make_ctx(tmp_path / "run", *sources_sets(mixed_sources(data)[:2]))
    assert execute(ctx, "traces") == "ran"
    assert events[0] == "cap" and events[-1] == "lift" and set(events[1:-1]) == {"batch"}
    events.clear()
    ctx.run.clear_done("traces")
    assert execute(ctx, "traces") == "ran" and events == []      # every batch cached: no model, no cap

    def crash(model, rows, **kw):
        raise KeyboardInterrupt("power cut")

    monkeypatch.setattr(T, "generate_encoded", crash)
    with pytest.raises(KeyboardInterrupt):
        execute(ctx, "traces", force=True)
    assert events == ["cap", "lift"]


# -- config ----------------------------------------------------------------------------------

def test_trace_sources_config_is_checked_and_filled_in(data):
    gsm = str(data["gsm"])

    def resolve(*sources, sets=()):
        return T.trace_sources(load_config(None, sources_sets(list(sources), *sets)))

    bad = {
        "unknown keys": {"kind": "gsm8k_train", "n": 2, "temperature": 0.7},
        "kind must be one of": {"kind": "gsm8k", "n": 2},
        "traces.sources.0.n must be a positive integer": {"kind": "gsm8k_train"},
        "must be a positive integer, not 0": {"kind": "gsm8k_train", "n": 0},
        "thinking must be on or off": {"kind": "gsm8k_train", "n": 2, "thinking": "maybe"},
        "needs a column": {"kind": "parquet", "path": gsm, "n": 2},
        "needs a path, or a repo and a file": {"kind": "parquet", "column": "question", "n": 2},
        "kind jsonl needs a path": {"kind": "jsonl", "n": 2},
        "not a MATH level": {"kind": "math_train", "n": 2, "levels": [6]},
        "subjects: unknown": {"kind": "math_train", "n": 2, "subjects": ["calculus"]},
        "math_train reads <subject>": {"kind": "math_train", "n": 2, "file": "algebra/train-00000-of-00001.parquet"},
        "reads a training split": {"kind": "mbpp_train", "n": 2, "file": "full/test-00000-of-00001.parquet"},
        "rows must be": {"kind": "ultrachat", "n": 2, "rows": [6000, 5000]},
    }
    for match, src in bad.items():
        with pytest.raises(ConfigError, match=match):
            resolve(src)
    with pytest.raises(H.ContaminationError, match="not a training split"):
        resolve({"kind": "gsm8k_train", "n": 2, "file": "main/test-00000-of-00001.parquet"})
    with pytest.raises(ConfigError, match="used by another source"):
        resolve({"kind": "gsm8k_train", "n": 2, "name": "x"}, {"kind": "mbpp_train", "n": 2, "name": "x"})
    with pytest.raises(ConfigError, match="not both"):
        T.trace_sources(load_config(None, ["traces.sources=[{kind: gsm8k_train, n: 2}]",
                                           "traces.prompts.path=examples/prompts.jsonl"]))
    a, b, c, d = resolve({"kind": "gsm8k_train", "n": 2}, {"kind": "gsm8k_train", "n": 3, "instruction": None},
                         {"kind": "opencodeinstruct", "n": 1, "batch": 4},
                         {"kind": "opencodeinstruct", "n": 1, "max_chars": None, "thinking": "on"},
                         sets=["traces.thinking=true", "traces.max_new_tokens=99", "traces.batch=5"])
    assert (a["name"], b["name"], c["name"], d["name"]) == ("gsm8k_train", "gsm8k_train.1", "opencodeinstruct",
                                                            "opencodeinstruct.3")
    assert (a["thinking"], a["max_new_tokens"], a["batch"], a["seed"]) == (True, 99, 5, 0)
    assert a["instruction"] == T.MATH_INSTRUCTION and b["instruction"] is None
    assert (c["max_chars"], c["batch"], d["max_chars"], d["thinking"]) == (2000, 4, None, True)
    assert resolve({"kind": "ultrachat", "n": 2})[0]["seed"] == 5090
    # The original form is one legacy source, and its stage hash is what it was before sources existed.
    legacy = load_config()
    assert stage_inputs(legacy, "traces")["traces"] == {"enabled": True, "prompts": {"kind": "jsonl", "path": None, "n": None},
                                                         "max_new_tokens": 512, "thinking": False, "batch": 8}
    multi = load_config(None, sources_sets([{"kind": "gsm8k_train", "n": 2}]))
    assert "prompts" not in stage_inputs(multi, "traces")["traces"]
    assert stage_inputs(multi, "traces") != stage_inputs(load_config(None, sources_sets([{"kind": "gsm8k_train",
                                                                                          "n": 2}], "traces.seed=1")),
                                                         "traces")
    assert apply_set(multi, "traces.sources.0.n=5")["traces"]["sources"][0]["n"] == 5
    [one] = T.trace_sources(load_config(ROOT / "examples" / "tiny-qwen3_5.yaml"))
    assert (one["legacy"], one["kind"], one["where"], one["max_new_tokens"], one["batch"]) == \
        (True, "jsonl", "traces.prompts", 24, 2)






def test_eval_thinking_subsets_default_to_the_full_ones():
    from openbonsai.stages.evaluate import suite_batch, suite_size
    r = load_config()["eval"]["reasoning"]
    assert [suite_size(r, "gsm8k", m) for m in ("off", "on")] == [100, 100] and suite_batch(r, "on") == 16
    r = load_config(None, ["eval.reasoning.humaneval.n_thinking=10", "eval.reasoning.batch_thinking=4"])["eval"]["reasoning"]
    assert (suite_size(r, "humaneval", "on"), suite_size(r, "humaneval", "off"), suite_batch(r, "on")) == (10, 50, 4)


# -- the test-data guard -------------------------------------------------------------------------

def _load(make_ctx, tmp_path, sources):
    ctx = make_ctx(tmp_path / "run", *sources_sets(sources))
    return T.load_sources(ctx)


def test_trace_prompts_refuse_test_problems_and_skip_near_duplicates(make_ctx, tmp_path, monkeypatch):
    gsm = R.load_frozen("gsm8k")["problems"]
    he = R.load_frozen("humaneval")["problems"]
    clean = [f"Bob has {k} pears and eats {k} of them. How many pears are left?" for k in range(4)]
    # A template twin: the first 13 words of a test question, then other numbers and another question.
    words = gsm[0]["question"].split()
    twin = " ".join(words[:14]) + " Then 7 more arrive. How many are there in total?"
    assert H.near_duplicate(twin, "x") == gsm[0]["id"] and H.near_duplicate(clean[0], "x") is None
    path = _parquet(tmp_path / "train.parquet", {"question": clean + [twin]})
    [(src, picked, info)] = _load(make_ctx, tmp_path, [{"kind": "gsm8k_train", "path": path.as_posix(), "n": 4,
                                                         "shuffle": False}])
    assert [row for row, _, _ in picked] == [0, 1, 2, 3] and info["screened"] == []
    with pytest.raises(ValueError, match=r"only 4 usable prompts, 5 requested \(1 near duplicates of test problems"):
        _load(make_ctx, tmp_path, [{"kind": "gsm8k_train", "path": path.as_posix(), "n": 5}])
    first = _parquet(tmp_path / "first.parquet", {"question": [twin] + clean})      # every kind is screened
    [(_, picked, info)] = _load(make_ctx, tmp_path, [{"kind": "parquet", "path": first.as_posix(), "column": "question",
                                                      "n": 4, "shuffle": False}])
    assert [row for row, _, _ in picked] == [1, 2, 3, 4] and info["screened"] == [{"row": 0, "problem": gsm[0]["id"]}]
    # A row that holds a whole test problem is refused, not skipped.
    leaked = _parquet(tmp_path / "leaked.parquet", {"question": clean + ["Warm-up: " + gsm[1]["question"]]})
    with pytest.raises(H.ContaminationError, match=gsm[1]["id"]):
        _load(make_ctx, tmp_path, [{"kind": "gsm8k_train", "path": leaked.as_posix(), "n": 5}])
    code = _parquet(tmp_path / "code.parquet", {"id": ["x"], "input": ["Finish this:\n" + he[0]["prompt"]]})
    with pytest.raises(H.ContaminationError, match=he[0]["id"].replace("/", ".")):
        _load(make_ctx, tmp_path, [{"kind": "opencodeinstruct", "path": code.as_posix(), "n": 1, "max_chars": None}])
    # Test splits by name, before any download.
    with pytest.raises(H.ContaminationError, match="not a training split"):
        _load(make_ctx, tmp_path, [{"kind": "parquet", "repo": "openai/openai_humaneval", "column": "prompt", "n": 1,
                                    "file": "openai_humaneval/test-00000-of-00001.parquet"}])
    # The extended suite's chat-NLL prompts, for every kind.
    fake = ((H.normalize(clean[2]), "chat-nll/row-859/prompt"),)
    monkeypatch.setattr(H, "chat_needles", lambda tok: fake)
    with pytest.raises(H.ContaminationError, match="chat-nll/row-859/prompt"):
        _load(make_ctx, tmp_path, [{"kind": "parquet", "path": path.as_posix(), "column": "question", "n": 4,
                                    "shuffle": False}])


def test_a_screened_near_duplicate_is_listed_in_the_summary(make_ctx, tmp_path):
    from openbonsai.cli import execute
    gsm = R.load_frozen("gsm8k")["problems"]
    twin = " ".join(gsm[0]["question"].split()[:14]) + " Then 7 more arrive. How many are there in total?"
    path = _parquet(tmp_path / "train.parquet", {"question": [twin, "What is two plus two?", "What is three plus one?"]})
    ctx = make_ctx(tmp_path / "run", *sources_sets([{"kind": "gsm8k_train", "path": path.as_posix(), "n": 2,
                                                     "shuffle": False, "max_new_tokens": 30}]))
    assert execute(ctx, "traces") == "ran"
    summary = json.loads((ctx.run.stage_dir("traces") / "summary.json").read_text(encoding="utf-8"))
    assert summary["sources"][0]["near_duplicates_skipped"] == [{"row": 0, "problem": gsm[0]["id"]}]
    assert sorted(r["row"] for r in _rows(ctx)) == [1, 2]


def test_an_answer_that_repeats_a_test_problem_is_left_out(make_ctx, tmp_path, monkeypatch):
    """The prompts are clean, but the model's answer quotes a test question: the trace is dropped and listed,
    so calibrate and the teacher (which refuse such blocks) never see it."""
    gsm = R.load_frozen("gsm8k")["problems"]
    path = _parquet(tmp_path / "train.parquet", {"question": ["What is two plus two?", "What is three plus one?",
                                                              "What is one plus one?"]})
    ctx = make_ctx(tmp_path / "run", *sources_sets([{"kind": "gsm8k_train", "path": path.as_posix(), "n": 3,
                                                     "shuffle": False, "max_new_tokens": 400, "batch": 1}]))
    tok = ctx.load_tokenizer()
    leak = tok(" Like: " + gsm[3]["question"], add_special_tokens=False)["input_ids"]
    clean = tok(" Four.", add_special_tokens=False)["input_ids"]
    monkeypatch.setattr(T, "generate_encoded", lambda model, rows, **kw: [
        (leak if "three plus one" in tok.decode(r) else clean, True) for r in rows])
    T.run(ctx)
    summary = json.loads((ctx.run.stage_dir("traces") / "summary.json").read_text(encoding="utf-8"))
    assert summary["sources"][0]["answers_dropped"] == [{"row": 1, "items": [gsm[3]["id"]]}]
    assert sorted(r["row"] for r in _rows(ctx)) == [0, 2] and summary["records"] == 2


# -- trace supply -------------------------------------------------------------------------------

def test_too_few_trace_blocks_fail_before_generating_or_right_after_the_traces(tiny_dir, tmp_path, monkeypatch,
                                                                              capsys):
    from openbonsai.rundir import RunDir
    from openbonsai.stages.common import StageContext
    monkeypatch.chdir(ROOT)
    base = ["run", "--model", str(tiny_dir), "--config", "examples/tiny-qwen3_5.yaml"]
    # Even traces that all ran to their caps could not make 402 blocks: refused before any generation.
    out = tmp_path / "a"
    assert cli.main(base + ["--out", str(out), "--set", "calibrate.sources.2.n=402"]) == 2
    err = capsys.readouterr().err
    assert "every one at its cap" in err and "calibrate.sources.2.n = 402 blocks" in err
    assert not (out / "traces" / "cache").exists() or not list((out / "traces" / "cache").glob("*.json"))
    # Traces that stop early make fewer blocks than the cap allows: the run stops after the traces stage,
    # naming the count, before calibrate (and hours of quantize) start.
    from openbonsai.adapters import load_adapter
    adapter = load_adapter(tiny_dir)
    tok = adapter.load_tokenizer(tiny_dir)
    prompts = T.read_prompts(ROOT / "examples" / "prompts.jsonl")
    prompt_tokens = sum(len(T.encode_chat(tok, p, adapter.chat_kwargs(thinking=False))) for p in prompts)
    seq = prompt_tokens + 2 * 24           # one block if every trace reaches its 24-token cap
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    monkeypatch.setattr(T, "generate_encoded", lambda model, rows, **kw: [([eos], False) for _ in rows])
    out = tmp_path / "b"
    sets = ["--set", f"distill.teacher.seq_len={seq}", "--set", "distill.teacher.sources.1.n=1"]
    assert cli.main(base + ["--out", str(out)] + sets) == 2
    err = capsys.readouterr().err
    assert "traces/traces.jsonl is too short: distill.teacher.sources.1.n = 1 blocks" in err
    assert (out / "traces" / "DONE.json").exists() and not (out / "calib").exists()
    # A rerun refuses before the first stage; the traces are current and complete.
    ctx = StageContext(load_config(ROOT / "examples" / "tiny-qwen3_5.yaml",
                                   [f"distill.teacher.seq_len={seq}", "distill.teacher.sources.1.n=1"],
                                   model=str(tiny_dir)), RunDir(out))
    with pytest.raises(ConfigError, match="too short"):
        cli.preflight(ctx)
    cli.preflight(ctx, ["traces", "calibrate"])            # a run that stops before distill does not need them


# -- real data (skipped unless the pinned training files are in the Hugging Face cache) -------------

def _cached(kind: str, file: str | None = None) -> bool:
    from huggingface_hub import hf_hub_download
    hub = T.HUB[kind]
    try:
        hf_hub_download(hub["repo"], file or hub["file"], revision=hub["revision"], repo_type="dataset",
                        local_files_only=True)
        return True
    except Exception:
        return False


def test_the_reasoning_example_prompts_resolve_from_the_pinned_training_files():
    import types
    from tiny_models import byte_tokenizer
    missing = [k for k in ("gsm8k_train", "mbpp_train", "opencodeinstruct") if not _cached(k)]
    missing += [] if all(_cached("math_train", T.HUB["math_train"]["file"].format(subject=s))
                         for s in T.MATH_SUBJECTS) else ["math_train"]
    if missing:
        pytest.skip(f"not in the Hugging Face cache: {missing}")
    cfg = load_config(REASONING)
    cfg["model"]["local_files_only"] = True
    srcs = [s for s in T.trace_sources(cfg) if s["kind"] != "ultrachat"]
    ctx = types.SimpleNamespace(cfg=cfg, load_tokenizer=byte_tokenizer)
    loaded = T.load_sources(ctx, srcs)
    for src, picked, info in loaded:
        assert len(picked) == src["n"] and len({r for r, _, _ in picked}) == src["n"]
        assert all(len(f["sha256"]) == 64 and f["revision"] == T.HUB[src["kind"]]["revision"] for f in info["files"])
        texts = [T._user_text(m) for _, m, _ in picked]
        if src["kind"] in ("gsm8k_train", "math_train"):
            assert all(t.endswith("\n" + T.MATH_INSTRUCTION) for t in texts)
        if src["kind"] == "mbpp_train":
            assert all(t.startswith("You are an expert Python programmer, and here is your task: ")
                       and "Your code should pass this test:\n\n" in t for t in texts)
        if src["kind"] == "opencodeinstruct":
            assert max(map(len, texts)) <= 2000
    assert loaded == T.load_sources(ctx, srcs)            # deterministic
    # The near-duplicate screen over all of GSM8K train: the 4 template twins of test questions.
    if H._full_split("gsm8k") is not None and H._full_split("humaneval") is not None:
        src = srcs[0]
        rows = T._rows(src, T._source_files(src, local_files_only=True)[0])
        assert [row for row, text, _ in rows if H.near_duplicate(text, "gsm8k_train") is not None] == [20, 406, 1314, 5162]

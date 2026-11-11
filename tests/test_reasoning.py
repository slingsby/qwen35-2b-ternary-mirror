"""GSM8K/HumanEval reasoning suites: answer extraction, the HumanEval executor, frozen manifests,
the test-data guard, dense generation weights and a tiny-model end-to-end run."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pytest
import torch

from openbonsai.eval import heldout as H
from openbonsai.eval import reasoning as R
from openbonsai.eval import scoring as SC

ROOT = Path(__file__).resolve().parents[1]


# -- GSM8K answers ------------------------------------------------------------------

@pytest.mark.parametrize("text, value, method", [
    (r"so the total is \boxed{18}.", "18", "boxed"),
    (r"$\boxed{\$1,080}$", "1080", "boxed"),
    (r"\boxed{1{,}234.50}", "1234.5", "boxed"),
    (r"\boxed{\frac{3}{4}}", "0.75", "boxed"),
    (r"\boxed{\dfrac{12}{4}}", "3", "boxed"),
    (r"\boxed{18 \text{ eggs}}", "18", "boxed"),
    (r"\boxed{5 + 3 = 8}", "8", "boxed"),
    (r"I will put it in \boxed{} ... \boxed{42} and later \boxed{}", "42", "boxed"),
    (r"\boxed{-3}", "-3", "boxed"),
    (r"First \boxed{7}, corrected: \boxed{9}", "9", "boxed"),
    ("She makes 9 * 2 = $18 every day.\n#### 18", "18", "####"),
    ("#### 1,000.00", "1000", "####"),
    ("#### +5", "5", "####"),
    ("The answer is 18.", "18", "answer is"),
    ("The final answer is: $1,234", "1234", "answer is"),
    ("Final Answer: -7 degrees", "-7", "answer is"),
    ("So the answer is −5.", "-5", "answer is"),
    ("The answer is 12 apples, not 13.", "12", "answer is"),
    # The marker's own clause: the number after its last "=", else its first number.
    ("Answer: 16 - 3 - 4 = 9 ... 18", "9", "answer is"),
    ("So the answer is: 3 + 4 = 7. Check: 7 - 4 = 3", "7", "answer is"),
    ("The answer is 18 (6 boxes of 3).", "18", "answer is"),
    ("**Answer:**\n\n18", "18", "answer is"),
    ("The answer is 1,080.", "1080", "answer is"),
    ("The answer is $3.50 per day.", "3.5", "answer is"),
    ("The answer is 5 + 3 = 8, so she keeps 8 of 10.", "8", "answer is"),
    ("The answer is x = ?\n42", "42", "answer is"),
    ("Step 1: 5 eggs. Step 2: 10 eggs. So 20 eggs in total.", "20", "last number"),
    ("x = 5-3 = 2", "2", "last number"),
    ("between 1,2,3 and 4", "4", "last number"),
    ("It costs 18.00 dollars", "18", "last number"),
    ("no digits at all", None, "none"),
    ("", None, "none"),
])
def test_gsm8k_answer_extraction(text, value, method):
    assert SC.extract_answer(text) == (value, method)


def test_marker_precedence_boxed_over_hash_over_answer_is_over_last_number():
    assert SC.extract_answer(r"The answer is 3. #### 4 \boxed{5} then 6") == ("5", "boxed")
    assert SC.extract_answer("The answer is 3. #### 4 then 6") == ("4", "####")
    assert SC.extract_answer("The answer is 3, then 6") == ("3", "answer is")


@pytest.mark.parametrize("raw, norm", [("18", "18"), ("18.0", "18"), ("18.", "18"), ("1,080", "1080"),
                                       ("$1,080.00", "1080"), ("+5", "5"), ("-0", "0"), ("-0.50", "-0.5"),
                                       ("−12", "-12"), ("007", "7"), ("abc", None), ("", None),
                                       ("inf", None), ("nan", None), ("9" * 64, "9" * 64), ("9" * 65, None),
                                       ("1" * 5000, None)])
def test_normalize_number(raw, norm):
    assert SC.normalize_number(raw) == norm


def test_a_looping_digit_run_is_scored_wrong_not_a_crash():
    """str(int(x)) refuses more than 4300 digits; such a run is a looping generation, never an answer."""
    loop = "The total is " + "1234567890" * 600
    assert SC.extract_answer(loop) == (None, "last number")
    assert SC.extract_answer("The answer is " + "7" * 5000) == (None, "answer is")
    assert SC.extract_answer(r"\boxed{12} then \boxed{" + "3" * 5000 + "}") == ("12", "boxed")
    scored = R._score_gsm8k([{"id": "x", "row": 0, "answer": "12"}], [loop])
    assert scored[0]["correct"] is False and scored[0]["pred"] is None


def test_gsm8k_reference_parsing():
    assert SC.gsm8k_reference("He sells 9 * 2 = <<9*2=18>>18.\n#### 1,018") == "1018"
    with pytest.raises(ValueError):
        SC.gsm8k_reference("no marker")


# -- HumanEval code extraction and execution ------------------------------------------

PROMPT = 'from typing import List\n\n\ndef add(xs: List[int]) -> int:\n    """Sum the list."""\n'
TEST = "def check(candidate):\n    assert candidate([1, 2, 3]) == 6\n    assert candidate([]) == 0\n"


def program_for(completion):
    code, how = SC.extract_code(completion, "add")
    program, mode = SC.build_program(PROMPT, code, TEST, "add")
    return program, how, mode


@pytest.mark.parametrize("completion, how, mode", [
    ("Here it is:\n```python\ndef add(xs):\n    return sum(xs)\n```\nDone.", "fenced", "function"),
    ("```python\ndef add(xs):\n    return sum(xs)\n\n\nif __name__ == '__main__':\n    print(input())\n```",
     "fenced", "function"),
    ("```py\ndef add(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total", "fenced", "function"),
    ("```\n    return sum(xs)\n```", "fenced", "continuation"),
    ("```python\nreturn sum(xs)\n```", "fenced", "body"),
    ("    return sum(xs)\n", "raw", "continuation"),
    ("Sure, here you go:\nfrom typing import List\ndef add(xs: List[int]) -> int:\n    return sum(xs)\n", "raw",
     "function"),
    ("```python\ndef add(xs):\n    return 0\n```\nBetter:\n```python\ndef add(xs):\n    return sum(xs)\n```",
     "fenced", "function"),
    ("```python\nimport math\n\ndef helper(xs):\n    return math.fsum(xs)\n\ndef add(xs):\n    return int(helper(xs))\n```",
     "fenced", "function"),
])
def test_code_extraction_and_passing_programs(completion, how, mode):
    program, got_how, got_mode = program_for(completion)
    assert (got_how, got_mode) == (how, mode)
    assert program.startswith(PROMPT) and program.rstrip().endswith("check(add)")
    assert "__main__" not in program
    assert SC.run_program(program, timeout=10)["status"] == "passed"


def test_executor_pass_fail_timeout_and_early_exit():
    assert SC.run_program("def f(x):\n    return x + 1\n\nassert f(1) == 2\n")["status"] == "passed"
    failed = SC.run_program("def f(x):\n    return x\n\nassert f(1) == 2, 'wrong'\n")
    assert failed["status"] == "failed" and failed["returncode"] != 0 and "AssertionError" in failed["stderr"]
    assert SC.run_program("def f(:\n")["status"] == "failed"                       # syntax error
    hang = SC.run_program("while True:\n    pass\n", timeout=1.0)
    assert hang["status"] == "timeout" and hang["seconds"] < 8
    # An exit before the test finishes is not a pass, even with exit code 0.
    assert SC.run_program("import sys\nsys.exit(0)\nassert False\n")["status"] == "failed"
    assert SC.run_program("import os\nos._exit(0)\n")["status"] == "failed"
    assert SC.run_program("x = input()\n")["status"] == "failed"                  # stdin is closed


def test_executor_runs_in_a_separate_isolated_process(tmp_path):
    probe = (f"import os, sys\n"
             f"assert os.getpid() != {os.getpid()}\n"
             f"assert os.path.abspath(os.getcwd()) != {str(Path.cwd().resolve())!r}\n"
             f"assert sorted(os.listdir('.')) == ['program.py']\n"
             f"assert sys.flags.isolated == 1\n"
             f"assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'\n"
             f"open('scratch.txt', 'w').write('x')\n")
    assert SC.run_program(probe)["status"] == "passed"
    assert not (Path.cwd() / "scratch.txt").exists()
    results = SC.run_programs(["assert True\n", "assert False\n", "assert True\n"], workers=3)
    assert [r["status"] for r in results] == ["passed", "failed", "passed"]


@pytest.mark.parametrize("completion, mode, dropped", [
    # A fence indented under a markdown list item is dedented, not taken as a continuation.
    ("Steps:\n1. Write it:\n   ```python\n   def add(xs):\n       return sum(xs)\n   ```\n", "function", 0),
    ("1. Body:\n   ```python\n   total = 0\n   for x in xs:\n       total += x\n   return total\n   ```",
     "continuation", 0),
    # A future import in the model's code moves to the top of the program.
    ("```python\nfrom __future__ import annotations\n\ndef add(xs: list[int]) -> int:\n    return sum(xs)\n```",
     "function", 0),
    # The model's own (wrong) asserts, prints, example calls and input() never run.
    ("```python\ndef add(xs):\n    return sum(xs)\n\nassert add([1]) == 2, 'my own wrong test'\nprint(add([1, 2]))\n```",
     "function", 2),
    ("```python\ndef add(xs):\n    n = len(xs)\n    return sum(xs)\n\nn = int(input())\nfor _ in range(n):\n    pass\n```",
     "function", 2),
    ("    return sum(xs)\n\nprint(add([5]))\ninput()\n", "continuation", 2),
    # Module-level values the definitions read stay (transitively), with their imports.
    ("```python\nimport math\nBASE = 0\nOFFSET = BASE * 2\nunused = input()\n\ndef add(xs):\n"
     "    return int(math.fsum(xs)) + OFFSET\n\nresult = add([1])\n```", "function", 2),
    # Non-ASCII output works on every OS (python -X utf8).
    ("```python\ndef add(xs):\n    print('→ summing Σ')\n    return sum(xs)\n```", "function", 0),
])
def test_humaneval_programs_are_cleaned_like_evalplus(completion, mode, dropped):
    code, _ = SC.extract_code(completion, "add")
    p = SC.prepare_program(PROMPT, code, TEST, "add")
    assert (p["mode"], p["dropped"]) == (mode, dropped)
    result = SC.run_program(p["program"])
    assert result["status"] == "passed", (p["program"], result["stderr"])
    if "__future__" in completion:
        assert p["program"].startswith("from __future__ import annotations\n")
    if dropped or "__future__" in completion:      # uncleaned, these fail (the review's reproductions)
        raw = SC.prepare_program(PROMPT, code, TEST, "add", clean=False)["program"]
        assert SC.run_program(raw)["status"] != "passed"


def test_unparseable_code_is_left_as_written():
    code = "def add(xs):\n    return sum(xs\n\nprint(1)\n"
    p = SC.prepare_program(PROMPT, code, TEST, "add")
    assert p["dropped"] == 0 and "print(1)" in p["program"]
    assert SC.run_program(p["program"])["status"] == "failed"
    assert SC.clean_program("x = 1\ny = f(x)\n", 1) == ("", 2)          # no definition reads them
    assert SC.clean_program("def f():\n    return x\nx = 1\n", 1) == ("def f():\n    return x\nx = 1\n", 0)
    # Lines before first_line (the problem's prompt) are never touched.
    assert SC.clean_program("print(1)\ndef f(): pass\nprint(2)\n", 2) == ("print(1)\ndef f(): pass\n", 1)


def test_executor_kills_every_process_the_program_starts(tmp_path):
    psutil = pytest.importorskip("psutil")
    pid_file = tmp_path / "pids.json"
    spawn = ("import json, os, subprocess, sys, time\n"
             "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
             f"open({str(pid_file)!r}, 'w').write(json.dumps([os.getpid(), p.pid]))\n")
    for tail, status in (("", "passed"), ("time.sleep(60)\n", "timeout")):
        pid_file.unlink(missing_ok=True)
        t0 = time.monotonic()
        result = SC.run_program(spawn + tail, timeout=3)
        assert result["status"] == status and time.monotonic() - t0 < 15
        pids = json.loads(pid_file.read_text())
        deadline = time.monotonic() + 3
        while any(_alive(psutil, p) for p in pids) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not any(_alive(psutil, p) for p in pids), f"left running after a {status}: {pids}"


def _alive(psutil, pid) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_executor_limits_output_and_memory():
    t0 = time.monotonic()
    flood = SC.run_program("import sys\nwhile True:\n    sys.stdout.write('x' * 65536)\n", timeout=20,
                           output_limit=2**20)
    assert flood["status"] == "failed" and "bytes of output" in flood["stderr"] and time.monotonic() - t0 < 10
    loud = SC.run_program("print('y' * 100000)\n", output_limit=2**20)      # under the limit: fine
    assert loud["status"] == "passed"
    if os.name == "nt" or sys.platform.startswith("linux"):
        hog = SC.run_program("x = bytearray(3 * 2**30)\n", memory_limit=256 * 2**20)
        assert hog["status"] == "failed" and "MemoryError" in hog["stderr"]
    assert SC.run_program("x = bytearray(16 * 2**20)\n", memory_limit=256 * 2**20)["status"] == "passed"


def test_humaneval_canonical_solutions_pass_and_wrong_ones_fail():
    _, problems = R.load_problems("humaneval", 4)
    for p in problems:
        program, mode = SC.build_program(p["prompt"], p["canonical_solution"], p["test"], p["entry_point"])
        assert mode == "continuation"
        whole = p["prompt"] + p["canonical_solution"]
        fenced = f"Sure.\n```python\n{whole}```\n"
        code, how = SC.extract_code(fenced, p["entry_point"])
        program2, mode2 = SC.build_program(p["prompt"], code, p["test"], p["entry_point"])
        assert (how, mode2) == ("fenced", "function")
        wrong, _ = SC.build_program(p["prompt"], "    return None\n", p["test"], p["entry_point"])
        assert [r["status"] for r in SC.run_programs([program, program2, wrong], workers=3)] == \
            ["passed", "passed", "failed"], p["id"]


def test_every_canonical_solution_passes_as_a_body_and_as_a_function():
    """All frozen problems (every problem when the split is cached): canonical solutions continue the
    prompt (some open with a nested helper def, which must not be dedented), and prompt + solution is a
    full function (whose repeated prompt may lose an unused constant, HumanEval/64's FIX)."""
    problems = R.load_frozen("humaneval")["problems"]
    try:
        problems = [R._problem("humaneval", i, r) for i, r in enumerate(R.read_rows("humaneval", local_files_only=True))]
    except Exception:
        pass
    programs = []
    for p in problems:
        body = SC.prepare_program(p["prompt"], p["canonical_solution"], p["test"], p["entry_point"])
        whole = SC.prepare_program(p["prompt"], p["prompt"] + p["canonical_solution"], p["test"], p["entry_point"])
        assert (body["mode"], whole["mode"], body["dropped"]) == ("continuation", "function", 0), p["id"]
        programs += [body["program"], whole["program"]]
    results = SC.run_programs(programs, workers=4)
    failed = [(problems[k // 2]["id"], r["stderr"][-200:]) for k, r in enumerate(results) if r["status"] != "passed"]
    assert failed == []


# -- frozen manifests -----------------------------------------------------------------

@pytest.mark.parametrize("name, n", [("gsm8k", 100), ("humaneval", 50)])
def test_frozen_manifests_are_pinned_and_follow_the_selector(name, n):
    f = R.FROZEN[name]
    data = (R.ASSETS / f["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == f["sha256"] and f["n"] == n
    m = R.load_frozen(name)
    assert R.manifest_bytes(m) == data                       # canonical serialization round-trips
    assert m["content_sha256"] == R.content_sha256(m)
    assert m["row_ids"] == R.select_rows(name, n) == [p["row"] for p in m["problems"]]
    assert len(set(m["row_ids"])) == n
    assert m["selector"] == {"algorithm": "sha256-rank/1", "rule": R.SELECTOR_RULE, "seed": R.SEED, "n": n}
    d = R.DATASETS[name]
    assert m["dataset"] == {k: d[k] for k in ("repo", "config", "split", "revision", "file", "rows", "sha256")}
    # A smaller n is the selector's own choice for that n: a prefix of the frozen order.
    assert R.select_rows(name, 10) == m["row_ids"][:10]
    info, first = R.load_problems(name, 7)
    assert [p["row"] for p in first] == m["row_ids"][:7] and info["frozen"] and info["n"] == 7
    if name == "gsm8k":
        assert all(SC.normalize_number(p["answer"]) == p["answer"] for p in m["problems"])
        assert all(p["id"] == f"gsm8k/test/{p['row']}" for p in m["problems"])
    else:
        assert all(p["id"] == f"HumanEval/{p['row']}" for p in m["problems"])
        assert all(f"def {p['entry_point']}(" in p["prompt"] and "def check(" in p["test"] for p in m["problems"])


def test_manifest_building_is_deterministic_and_tamper_evident():
    rows = [{"question": f"Q{i}: how many?", "answer": f"steps\n#### {i * 3}"} for i in range(1319)]
    a = R.manifest_bytes(R.build_manifest("gsm8k", 12, rows=rows))
    b = R.manifest_bytes(R.build_manifest("gsm8k", 12, rows=list(rows)))
    assert a == b
    m = json.loads(a)
    assert m["row_ids"] == R.select_rows("gsm8k", 12) and m["problems"][0]["answer"] == str(m["row_ids"][0] * 3)
    other_seed = R.build_manifest("gsm8k", 12, rows=rows, seed=R.SEED + 1)
    assert other_seed["row_ids"] != m["row_ids"]
    m["problems"][0]["answer"] = "999999"
    with pytest.raises(ValueError, match="content SHA256"):
        R.verify_manifest(m)
    with pytest.raises(ValueError, match="rows"):
        R.build_manifest("gsm8k", 12, rows=rows[:-1])


def test_frozen_manifests_rebuild_byte_for_byte_from_the_cached_datasets():
    """Needs both pinned parquet files in the local HF cache (HF_HOME); skipped otherwise."""
    try:
        paths = {name: R.dataset_file(name, local_files_only=True) for name in R.SUITES}
    except Exception as error:   # not in this machine's HF cache
        pytest.skip(f"dataset not cached: {type(error).__name__}")
    result = R.check_frozen(paths=paths)
    assert all(r["identical"] and r["sha256"] == r["pinned"] for r in result.values()), result


def test_problem_counts_and_thinking_modes():
    with pytest.raises(ValueError):
        R.load_problems("gsm8k", 0)
    with pytest.raises(FileNotFoundError):   # beyond the frozen 50: needs the dataset
        R.load_problems("humaneval", 51, path=ROOT / "missing.parquet")
    assert R.thinking_modes([False, True]) == ["off", "on"]
    assert R.thinking_modes("on") == ["on"] and R.thinking_modes(["off", "off"]) == ["off"]
    with pytest.raises(ValueError):
        R.thinking_modes(["maybe"])
    from openbonsai.config import load_config
    cfg = load_config(None, ["eval.reasoning.thinking=[off,on]", "eval.reasoning.gsm8k.n=20"])
    assert R.thinking_modes(cfg["eval"]["reasoning"]["thinking"]) == ["off", "on"]
    assert R.split_thinking("plan</think>\n\nanswer 4", "</think>", True) == ("\n\nanswer 4", True)
    assert R.split_thinking("still thinking 4", "</think>", True) == ("", False)
    assert R.split_thinking("answer 4", "</think>", False) == ("answer 4", True)
    assert R.split_thinking("answer 4", None, True) == ("answer 4", True)


# -- test problems never reach calibration or trace prompts -----------------------------

def test_guard_finds_test_problems_in_any_whitespace():
    gsm = R.load_frozen("gsm8k")["problems"][3]
    he = R.load_frozen("humaneval")["problems"][5]
    chat = "<|im_start|>user\n" + "  \n".join(gsm["question"].split(" ")) + "<|im_end|>"
    assert H.contained([chat, "clean text"]) == [(0, gsm["id"])]
    assert H.contained(["notes:\n" + he["prompt"] + "    pass\n"]) == [(0, he["id"])]
    with pytest.raises(H.ContaminationError, match="training splits"):
        H.check_texts(["fine", chat], "unit")
    H.check_texts(["The river turned west below the old mill."], "unit")


def test_guard_refuses_test_split_sources(tmp_path, monkeypatch):
    for src in ({"kind": "jsonl", "repo": "openai/gsm8k", "file": "main/test-00000-of-00001.parquet"},
                {"kind": "jsonl", "repo": "openai/openai_humaneval", "file": "openai_humaneval/test-00000-of-00001.parquet"},
                {"kind": "jsonl", "repo": "openai/gsm8k"}):
        with pytest.raises(H.ContaminationError):
            H.check_source(src, "unit")
    H.check_source({"kind": "jsonl", "repo": "openai/gsm8k", "file": "main/train-00000-of-00001.parquet"}, "unit")
    # A local copy of a test parquet is recognized by its bytes.
    copy = tmp_path / "renamed.parquet"
    copy.write_bytes(b"pretend parquet bytes")
    monkeypatch.setitem(R.DATASETS, "gsm8k", {**R.DATASETS["gsm8k"], "bytes": copy.stat().st_size,
                                               "sha256": hashlib.sha256(copy.read_bytes()).hexdigest()})
    with pytest.raises(H.ContaminationError):
        H.check_source({"kind": "jsonl", "path": str(copy)}, "unit", path=copy)


def test_calibration_and_trace_prompts_refuse_test_problems(tiny_dir, make_ctx, tmp_path):
    from openbonsai.adapters import load_adapter
    from openbonsai.stages import calibrate, traces
    tok = load_adapter(tiny_dir).load_tokenizer(tiny_dir)
    he = R.load_frozen("humaneval")["problems"][0]
    gsm = R.load_frozen("gsm8k")["problems"][0]
    data = tmp_path / "data.jsonl"
    rows = [{"text": "Plain calibration text about rivers and mills. " * 4},
            {"messages": [{"role": "user", "content": "Finish this:\n" + he["prompt"]},
                          {"role": "assistant", "content": he["canonical_solution"] + " filler" * 20}]}]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    clean = tmp_path / "clean.jsonl"
    clean.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    calibrate.build(tok, [{"kind": "jsonl", "path": str(clean), "n": 2}], seq_len=64)
    # The problem spans many 64-token blocks; the source's blocks are checked as one stream.
    n_blocks = len(tok(calibrate.chat_text(tok, rows[1]["messages"], {}), add_special_tokens=False)["input_ids"]) // 64 + 3
    with pytest.raises(H.ContaminationError, match=he["id"].replace("/", ".")):
        calibrate.build(tok, [{"kind": "jsonl", "path": str(data), "n": n_blocks}], seq_len=64)
    with pytest.raises(H.ContaminationError, match="calibration source"):
        calibrate.build(tok, [{"kind": "wikitext2", "n": 1, "repo": "openai/gsm8k",
                               "file": "main/test-00000-of-00001.parquet"}], seq_len=64)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt": "Hi there."}) + "\n" + json.dumps({"prompt": gsm["question"]}) + "\n",
                       encoding="utf-8")
    ctx = make_ctx(tmp_path / "run", f"traces.prompts.path={prompts.as_posix()}")
    with pytest.raises(H.ContaminationError, match="traces.prompts"):
        traces.load_prompts(ctx)


def test_shipped_calibration_and_prompt_data_hold_no_test_problems():
    from openbonsai.stages import calibrate
    assert not {d["repo"] for d in R.DATASETS.values()} & {h["repo"] for h in calibrate.HUB.values()}
    texts = []
    for f in [ROOT / "examples" / "prompts.jsonl", *sorted((ROOT / "examples" / "tiny-data").glob("*.jsonl"))]:
        texts += f.read_text(encoding="utf-8").splitlines()
    assert len(texts) > 5 and H.contained(texts) == []


# -- dense generation weights and the end-to-end run -------------------------------------

def test_densify_writes_the_weights_apply_dense_writes(calibrated):
    from openbonsai.cli import execute
    from openbonsai.core.ternary import TernaryLinear
    from openbonsai.stages.common import apply_dense, densify, has_ternary, install_ternary
    ctx = calibrated
    execute(ctx, "quantize")
    execute(ctx, "endpoints")
    state = ctx.state_through("endpoints")
    adapter = ctx.adapter()
    dense, _ = ctx.load_model()
    apply_dense(adapter, dense, state.records())
    tern, _ = ctx.load_model()
    install_ternary(adapter, tern, state.records(), ctx.device)
    assert has_ternary(tern) and any(isinstance(m, TernaryLinear) for m in tern.modules())
    assert densify(tern, ctx.dtype) == len(state.tensors) and not has_ternary(tern)
    a, b = dense.state_dict(), tern.state_dict()
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    ids = torch.tensor([[5, 9, 33, 70, 101]])
    assert torch.equal(dense(input_ids=ids).logits, tern(input_ids=ids).logits)


def test_reasoning_suites_end_to_end_on_the_tiny_model(calibrated, monkeypatch):
    from contextlib import contextmanager
    from openbonsai.cli import execute
    from openbonsai.stages import evaluate as EV
    from openbonsai.stages import traces as T
    ctx = calibrated
    events = []

    @contextmanager
    def cap(device, log):
        events.append("cap")
        try:
            yield None
        finally:
            events.append("lift")

    real = T.generate_encoded
    monkeypatch.setattr(EV, "vram_cap", cap)
    monkeypatch.setattr(T, "generate_encoded", lambda model, rows, **kw: (events.append("batch"), real(model, rows, **kw))[1])
    for key, value in (("thinking", [False, True]), ("max_new_tokens", 6), ("max_new_tokens_thinking", 5),
                       ("batch", 2), ("batch_check", 1)):
        ctx.cfg["eval"]["reasoning"][key] = value
    ctx.cfg["eval"]["reasoning"]["gsm8k"]["n"] = 3
    ctx.cfg["eval"]["reasoning"]["humaneval"]["n"] = 2
    execute(ctx, "quantize")
    msgs = []
    log = ctx.log
    ctx.log = lambda stage: (lambda m: (msgs.append(m), log(stage)(m)))
    execute(ctx, "eval", source="quantize", suites=["humaneval", "gsm8k"], install_mode="ternary")
    assert events[0] == "cap" and events[-1] == "lift" and set(events[1:-1]) == {"batch"}   # generation is capped
    assert any("ternary modules replaced" in m for m in msgs)
    assert any("tok/s" in m and "ETA" in m for m in msgs)
    assert any("batch check:" in m for m in msgs)
    out = ctx.run.stage_dir("eval")
    first = {}
    for suite, n in (("gsm8k", 3), ("humaneval", 2)):
        for mode, cap in (("off", 6), ("on", 5)):
            r = json.loads((out / f"quantize.{suite}.think-{mode}.json").read_text(encoding="utf-8"))
            first[suite, mode] = r
            assert r["source"] == "quantize" and r["weights"] == "dense" and r["thinking"] == mode
            assert r["total"] == n == len(r["problems"]) and r["max_new_tokens"] == cap
            assert r["batches"] == (n + 1) // 2 and r["cached_batches"] == 0
            assert 0 <= r["hit_cap"] <= n and r["generated_tokens"] <= n * cap
            assert r["manifest"]["frozen"] and r["manifest"]["n"] == n
            assert r["chat_kwargs"] == {"enable_thinking": mode == "on"} and r["chat_date"] == "2024-07-26"
            assert r["decoding"]["strategy"] == "greedy" and r["runtime"]["kernels"] == []
            assert r["runtime"]["torch"] == torch.__version__ and "transformers" in r["runtime"]
            check = r["batch_check"]
            assert check["batch"] == 0 and check["rows"] == 1 and check["rows_checked"][0]["row"] in (0, 1)
            assert check["identical"] == (check["rows_checked"][0]["first_difference"] is None)
            for p in r["problems"]:
                assert 1 <= p["generated_tokens"] <= cap and (not p["hit_cap"] or p["generated_tokens"] == cap)
                assert ("finished_thinking" in p) == (mode == "on")
            if suite == "gsm8k":
                correct = sum(p["correct"] for p in r["problems"])
                assert r["score"] == f"{correct}/{n}" and r["correct"] == correct
                assert sum(r["methods"].values()) == n
            else:
                assert r["passed"] + r["failed"] + r["timeout"] == n and r["score"] == f"{r['passed']}/{n}"
                assert all(p["status"] in ("passed", "failed", "timeout") for p in r["problems"])
            if mode == "on":
                assert r["unfinished_thinking"] == sum(not p["finished_thinking"] for p in r["problems"])
    # A rerun takes every generation from the cache and scores identically.
    execute(ctx, "eval", source="quantize", suites=["gsm8k", "humaneval"], install_mode="ternary")
    for (suite, mode), r in first.items():
        again = json.loads((out / f"quantize.{suite}.think-{mode}.json").read_text(encoding="utf-8"))
        assert again["cached_batches"] == again["batches"] and again["tokens_per_second"] is None
        strip = ("seconds", "returncode")
        assert [{k: v for k, v in p.items() if k not in strip} for p in again["problems"]] == \
            [{k: v for k, v in p.items() if k not in strip} for p in r["problems"]]


def test_runtime_info_names_the_kernels_a_module_bound():
    """The generation cache key records whether transformers bound FLA / causal-conv1d or the torch paths."""
    import types
    fake = types.ModuleType("openbonsai_fake_modeling")

    def kernel(x):
        return x
    kernel.__module__ = "fla.ops.gated_delta_rule"

    def torch_path(x):
        return x

    def bind(impl):
        def wrapped(*args):
            return impl(*args)
        return wrapped
    fake.chunk_gated_delta_rule = bind(kernel)
    fake.causal_conv1d_fn = bind(torch_path)
    sys.modules[fake.__name__] = fake
    try:
        assert R.runtime_info(torch.nn.Linear(2, 2))["kernels"] == []
        Model = type("Model", (torch.nn.Linear,), {"__module__": fake.__name__})
        assert R.runtime_info(Model(2, 2))["kernels"] == ["fla"]
    finally:
        del sys.modules[fake.__name__]


def test_guard_covers_every_test_problem_when_the_splits_are_cached(monkeypatch):
    H.needles.cache_clear()
    try:
        full = H.needles()
        frozen = 100 + 50
        if H._full_split("gsm8k") is None:
            pytest.skip("the GSM8K/HumanEval test splits are not in the Hugging Face cache")
        assert len(full) == 1319 + 164
        # A problem outside the frozen subsets is found too.
        outside = next(p for p in H._full_split("gsm8k") if p["row"] not in R.load_frozen("gsm8k")["row_ids"])
        assert H.contained(["notes " + outside["question"]]) == [(0, outside["id"])]
        H.needles.cache_clear()
        monkeypatch.setattr(H, "_full_split", lambda name: None)
        assert len(H.needles()) == frozen
    finally:
        H.needles.cache_clear()


def test_stored_blocks_are_checked_in_their_build_order(tiny_dir):
    """calibrate saves its blocks permuted; a problem cut across two consecutive blocks is only whole
    again in build order."""
    from openbonsai.adapters import load_adapter
    tok = load_adapter(tiny_dir).load_tokenizer(tiny_dir)
    gsm = R.load_frozen("gsm8k")["problems"][7]
    ids = tok("Some notes. " * 30 + gsm["question"] + " More notes." * 60, add_special_tokens=False)["input_ids"]
    seq = 48
    blocks = torch.tensor([ids[k:k + seq] for k in range(0, len(ids) - seq + 1, seq)])
    seed = 3
    perm = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(seed))
    stored = blocks[perm]
    assert torch.equal(H.build_order(stored, seed), blocks)
    probes = H.problem_needles("gsm8k", [gsm])
    with pytest.raises(H.ContaminationError, match="gsm8k/test"):
        H.check_stored_blocks(tok, stored, seed, "calib/tokens.npy", probes)
    clean = torch.tensor([ids[k:k + seq] for k in range(0, 30 * 3, seq)])
    H.check_stored_blocks(tok, clean, seed, "calib/tokens.npy", probes)


def test_eval_refuses_a_stage_whose_stored_data_holds_a_scored_problem(calibrated):
    import numpy as np
    from openbonsai.stages.calibrate import save_tokens
    from openbonsai.stages.evaluate import check_test_problems_absent
    ctx = calibrated
    ctx.cfg["eval"]["reasoning"]["gsm8k"]["n"] = 3
    check_test_problems_absent(ctx, "quantize", ["gsm8k"])          # clean: remembered
    memo = json.loads((ctx.run.stage_dir("eval") / "heldout-checked.json").read_text(encoding="utf-8"))
    assert set(memo) == {"calib/tokens.npy"}
    check_test_problems_absent(ctx, "bf16", ["gsm8k"])              # the BF16 model is not fitted on it
    # Calibration data made before the guard knew this problem (e.g. a problem beyond the frozen subset).
    path = ctx.run.stage_dir("calibrate") / "tokens.npy"
    tokens = torch.from_numpy(np.load(path).astype(np.int64))
    tok = ctx.load_tokenizer()
    problem = R.load_problems("gsm8k", 3)[1][2]
    ids = tok(problem["question"], add_special_tokens=False)["input_ids"]
    order = H.build_order(tokens, int(ctx.cfg["calibrate"]["seed"]))
    width = tokens.shape[1]
    flat = order.reshape(-1).clone()
    flat[5:5 + len(ids)] = torch.tensor(ids)
    seed = int(ctx.cfg["calibrate"]["seed"])
    perm = torch.randperm(len(order), generator=torch.Generator().manual_seed(seed))
    save_tokens(path, flat.reshape(-1, width)[perm])
    with pytest.raises(H.ContaminationError, match=problem["id"]):
        check_test_problems_absent(ctx, "quantize", ["gsm8k"])
    ctx.cfg["eval"]["reasoning"]["gsm8k"]["n"] = 2                 # not scored: no refusal
    check_test_problems_absent(ctx, "quantize", ["gsm8k"])

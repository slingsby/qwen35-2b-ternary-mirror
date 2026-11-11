"""CPU checks for the recipe scripts: the start builder end to end on the tiny model, and the generation checks."""
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'recipe' / f'{name}.py')
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def fake_wiki(path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = random.Random(0); words = ['tree', 'river', 'stone', 'light', 'north', 'paper', 'glass', 'field']
    lines = []
    for i in range(400):
        lines.append(f' = Article {i} = \n')
        lines.append(' '.join(rng.choice(words) for _ in range(500)) + f' {i}\n')
    pq.write_table(pa.table({'text': lines}), path)


def test_build_start_writes_a_resumable_ternary_state(tiny_dir, tmp_path):
    from openbonsai.formats.state import load_state
    build_start = load('build_start')
    wiki = tmp_path / 'wiki.parquet'; fake_wiki(wiki)
    args = SimpleNamespace(model_dir=str(tiny_dir), model_revision='tiny', wiki=str(wiki), out=str(tmp_path / 'start'), device='cpu')
    build_start.build(args)
    out = tmp_path / 'start'
    state = load_state(out / 'initial.safetensors')
    assert 'embed_tokens' in state.tensors and len(state.tensors) > 1
    for t in state.tensors.values():
        assert set(np.unique(t.codes.numpy())) <= {-1, 0, 1}
    first = build_start.sha(out / 'initial.safetensors')
    assert json.loads((out / 'start-complete.json').read_text())['initial_sha256'] == first
    data = json.loads((out / 'data.json').read_text())
    assert data['splits']['train']['blocks'] == 256 and data['splits']['test']['blocks'] == 24
    build_start.build(args)  # resume: per-layer files are reused and the result is identical
    assert build_start.sha(out / 'initial.safetensors') == first


def test_generation_checks():
    checks = load('checks')
    assert checks.validate_response({'validator': 'integer', 'expected': 7}, 'so the answer is 7') == (True, 'arithmetic final-value check')
    assert not checks.validate_response({'validator': 'integer', 'expected': 7}, 'Answer: 8')[0]
    py = {'validator': 'python', 'function': 'f'}
    assert checks.validate_response(py, '```python\ndef f(x):\n    return x\n```')[0]
    assert not checks.validate_response(py, 'def f(x):\n    pass')[0]
    tool = {'validator': 'json', 'expected': {'name': 'a', 'arguments': {'n': 1}}}
    assert checks.validate_response(tool, '{"name": "a", "arguments": {"n": 1}}')[0]
    assert checks.validate_response(tool, '{"name": "a", "arguments": {"n": 1}}\n</think>\n{}') == (False, 'response parse failure')
    assert checks.looping_tail([1, 2, 3] * 40) and not checks.looping_tail(list(range(120)))

"""Build the GPTQ ternary start state that recipe/train.py begins from.

1. Calibration data: 256-token blocks from WikiText-2 (raw) *train* and eight Python
   standard-library modules, split by document hash into train/validation/test
   (only train blocks reach the Hessians; 64 seeded train blocks are used).
2. Rotation: Hadamard blocks of at most 512 with seeded random signs.
3. Sequential GPTQ to {-1, 0, +1} with an FP16 scale per 128 weights, layer by layer
   in data-flow order (inputs propagated through the already-quantized layers).
4. The tied embedding/head is quantized with a joint head/embedding Hessian.

Writes <out>/initial.safetensors plus data.json (block sources and hashes) and prepared.json
(fingerprint of data, model config and geometry). Resumable per layer.
The published results used a start built this way on an RTX 4070 SUPER (12 GB), with
SHA-256 32b338a2cea0ed6291398270dee4b9c8025555816731d58849519c4a998644da. Python
stdlib text differs between Python versions, so a rebuild matches it only on the same
Python and library versions.

Example:
  python recipe/build_start.py --model-dir runs/qwen2b/model \
      --wiki runs/data/wikitext-2-raw-v1/train-00000-of-00001.parquet --out runs/start
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import sysconfig
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
PYTHON_FILES = {'train': ['collections/__init__.py', 'functools.py', 'statistics.py', 'textwrap.py',
                          'calendar.py', 'difflib.py', 'csv.py', 'ipaddress.py'],
                'validation': ['fractions.py', 'heapq.py'], 'test': ['bisect.py', 'numbers.py', 'string.py']}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 2**20):
            h.update(block)
    return h.hexdigest()


def dataset(tok, wiki, out, seq_len=256):
    """Disjoint document-level train/validation/test blocks; returns the manifest."""
    import numpy as np
    import pyarrow.parquet as pq
    quotas = {'train': {'wiki': 160, 'python': 96}, 'validation': {'wiki': 16, 'python': 8}, 'test': {'wiki': 16, 'python': 8}}
    splits = ('train', 'validation', 'test')
    documents = {s: [] for s in splits}
    seen = set()
    text = '\n'.join(pq.read_table(wiki, columns=['text'])['text'].to_pylist())
    for body in re.split(r'(?m)^ = [^=\n]+ =\s*$', text):
        h = hashlib.sha256(body.encode()).hexdigest()
        if h in seen or len(body) < 2000:
            continue
        seen.add(h)
        bucket = int(h[:8], 16) % 10
        split = 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'
        documents[split].append(('wiki', h, body))
    lib = Path(sysconfig.get_path('stdlib'))
    for split, names in PYTHON_FILES.items():
        for name in names:
            body = (lib / name).read_text(encoding='utf-8')
            h = hashlib.sha256(body.encode()).hexdigest()
            assert h not in seen
            seen.add(h)
            documents[split].append(('python:' + name, h, body))
            snapshot = out / 'sources' / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_text(body, encoding='utf-8')
    rng = random.Random(42)
    manifest = {'wiki_sha256': sha(wiki), 'python': sys.version, 'seq_len': seq_len, 'splits': {}}
    for split in splits:
        blocks, sources = [], []
        for family in ('wiki', 'python'):
            quota = quotas[split][family]
            candidates = [d for d in documents[split] if d[0].startswith(family)]
            rng.shuffle(candidates)
            streams = [(s, h, tok.encode(b, add_special_tokens=False)) for s, h, b in candidates]
            cursor = 0
            while quota:
                progressed = False
                for source, h, tokens in streams:
                    if len(tokens) >= (cursor + 1) * seq_len:
                        blocks.append(tokens[cursor * seq_len:(cursor + 1) * seq_len])
                        sources.append({'source': source, 'document_sha256': h, 'start': cursor * seq_len})
                        quota -= 1
                        progressed = True
                        if not quota:
                            break
                if not progressed:
                    raise RuntimeError('not enough disjoint data')
                cursor += 1
        np.save(out / (split + '.npy'), np.asarray(blocks, dtype=np.int64))
        manifest['splits'][split] = {'blocks': len(blocks), 'sha256': sha(out / (split + '.npy')), 'documents': sources}
    keys = [{r['document_sha256'] for r in manifest['splits'][s]['documents']} for s in splits]
    assert all(not keys[i] & keys[j] for i in range(3) for j in range(i))
    return manifest


def build(a):
    import numpy as np
    import torch
    from openbonsai.adapters import load_adapter
    from openbonsai.core.geometry import derive_geometry, make_signs
    from openbonsai.core.gptq import quantize_rows
    from openbonsai.formats.atomic import atomic_write_json
    from openbonsai.formats.state import TernaryState, TernaryTensor, save_state, load_state
    from openbonsai.stages.endpoints import head_hessian, joint_hessian
    from openbonsai.stages.quantize import capture_layer0_inputs, capture_layer_kwargs, quantize_layer, layer_forward, apply_layer
    root, dev = Path(a.out), a.device
    root.mkdir(parents=True, exist_ok=True)
    adapter = load_adapter(a.model_dir, local_files_only=True)
    model, tok = adapter.load(a.model_dir, device=dev, dtype=torch.bfloat16, local_files_only=True)
    if not (root / 'data.json').exists():
        atomic_write_json(root / 'data.json', dataset(tok, a.wiki, root))
    manifest = json.loads((root / 'data.json').read_text())
    for s, info in manifest['splits'].items():
        assert sha(root / (s + '.npy')) == info['sha256'], f'{s}.npy changed'
    geometry = derive_geometry(adapter.quantized_widths(model), adapter.hidden_size, max_block=512)
    signs = make_signs(geometry)
    basis = {'data_sha256': sha(root / 'data.json'), 'model_config_sha256': sha(Path(a.model_dir) / 'config.json'),
             'geometry': geometry.to_json(), 'model_revision': a.model_revision}
    fingerprint = hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()
    if (root / 'prepared.json').exists():
        assert json.loads((root / 'prepared.json').read_text())['fingerprint'] == fingerprint, 'inputs changed; refusing to resume'
    atomic_write_json(root / 'prepared.json', {**basis, 'fingerprint': fingerprint})
    # A seeded subset of train blocks; validation/test never enter Hessians.
    ids = torch.from_numpy(np.load(root / 'train.npy'))
    picks = torch.randperm(len(ids), generator=torch.Generator().manual_seed(42))[:64]
    ids = ids[picks]
    np.save(root / 'calibration-indices.npy', picks.numpy())
    kw = capture_layer_kwargs(model, adapter, ids[:1], dev)
    inps = capture_layer0_inputs(model, adapter, ids, 1, dev)
    model.cpu()
    if dev.startswith('cuda'): torch.cuda.empty_cache()
    tensors = {}
    layers_dir = root / 'layers'
    layers_dir.mkdir(exist_ok=True)
    for i, layer in enumerate(adapter.layers(model)):
        t0 = time.monotonic()
        layer.to(dev)
        path = layers_dir / f'{i:04d}.safetensors'
        if path.exists():
            st = load_state(path)
            assert st.meta['fingerprint'] == fingerprint
            apply_layer(adapter, model, i, st)
        else:
            result = quantize_layer(adapter, model, i, inps, kw[i], signs, geometry.block, group=128, damp=.01, impl='dense')
            st = TernaryState({k: v[0] for k, v in result.items()}, signs, geometry.block, 128,
                              {'fingerprint': fingerprint, 'stats': {k: v[1] for k, v in result.items()}})
            save_state(path, st)
        tensors.update(st.tensors)
        with torch.no_grad():
            inps = [layer_forward(layer, x, kw[i]).detach() for x in inps]
        layer.cpu()
        if dev.startswith('cuda'): torch.cuda.empty_cache()
        print(json.dumps({'quantized_layer': i, 'seconds': time.monotonic() - t0}), flush=True)
    del inps, kw
    model.to(dev)
    ep_path = root / 'endpoints.safetensors'
    if ep_path.exists():
        ep = load_state(ep_path)
        assert ep.meta['fingerprint'] == fingerprint
    else:
        h = head_hessian(adapter.text_model(model), ids, 1, torch.device(dev))
        metric = joint_hessian(h, adapter.embedding_gain(model).to(dev))
        c, s, dense = quantize_rows(adapter.embedding(model).weight.detach().cpu(), metric, signs[geometry.hidden],
                                    geometry.block, chunk=2048, impl='dense')
        ep = TernaryState({'embed_tokens': TernaryTensor(c, s)}, {geometry.hidden: signs[geometry.hidden]},
                          geometry.block, 128, {'fingerprint': fingerprint})
        save_state(ep_path, ep)
        del dense, h, metric
    save_state(root / 'initial.safetensors', TernaryState(tensors, signs, geometry.block, 128, {'fingerprint': fingerprint}).merged(ep))
    atomic_write_json(root / 'start-complete.json', {'fingerprint': fingerprint, 'initial_sha256': sha(root / 'initial.safetensors'),
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name() if dev.startswith('cuda') else dev})
    print(json.dumps({'initial': str(root / 'initial.safetensors'), 'sha256': sha(root / 'initial.safetensors')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-dir', required=True, help='BF16 Hugging Face model directory')
    p.add_argument('--model-revision', default='15852e8c16360a2fea060d615a32b45270f8a8fc', help='recorded in the fingerprint')
    p.add_argument('--wiki', required=True, help='WikiText-2 raw *train* parquet')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    build(p.parse_args())


if __name__ == '__main__':
    main()

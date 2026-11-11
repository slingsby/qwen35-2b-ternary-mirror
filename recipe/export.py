"""Export a train.py run's best checkpoint to a verified PTQ1_0 GGUF.

Writes <run>/trained.gguf and <run>/export.json (file hash, input hashes, changed-code count).
export_gguf verifies the written file against the source state.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--initial', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--name', required=True)
    args = p.parse_args()
    from openbonsai.adapters import load_adapter
    from openbonsai.formats.state import load_state, read_small
    from openbonsai.stages.export import export_gguf
    run = Path(args.run_dir)
    state, original = load_state(run / 'best.safetensors'), load_state(args.initial)
    assert state.tensors.keys() == original.tensors.keys()
    changed = sum(int((state.tensors[k].codes != original.tensors[k].codes).sum()) for k in state.tensors)
    adapter = load_adapter(args.model_dir, local_files_only=True)
    report = export_gguf(adapter, state, args.model_dir, run / 'trained.gguf', name=args.name,
                         small=read_small(run / 'best-small.safetensors', cast=True), chunk_rows=1024)
    (run / 'export.json').write_text(json.dumps({'bytes': (run / 'trained.gguf').stat().st_size, 'sha256': sha(run / 'trained.gguf'),
        'state_sha256': sha(run / 'best.safetensors'), 'small_sha256': sha(run / 'best-small.safetensors'),
        'codes_changed_vs_initial': changed, 'report': report}, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'gguf_bytes': (run / 'trained.gguf').stat().st_size, 'codes_changed': changed}), flush=True)


if __name__ == '__main__':
    main()

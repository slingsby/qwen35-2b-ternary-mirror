"""Evaluate a candidate against an online BF16 teacher on the sealed split (and a legacy test set), once.

Run as `python recipe/evaluate.py ...` (checks.py sits beside it).
Candidates: --candidate bf16 | gptq:<state.safetensors> | gguf:<file.gguf>. Uses the sealed
split's token arrays and prompts (hash-verified against its manifest), appends to
<out>/access-log.jsonl, refuses a second evaluation of the same --label, and writes
<out>/results-<label>.json. Metrics: document NLL/PPL plus full-vocabulary teacher KL and
top-1 agreement (sealed WikiText-2 test, sealed Python), chat KL on the sealed teacher-response
tokens, optional legacy test set, and generation checks on the sealed prompts (format/syntax
validators, not functional correctness).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-dir', required=True, help='BF16 teacher model directory')
    p.add_argument('--candidate', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--sealed', required=True, help='sealed split directory (manifest.json, arrays, specs.json)')
    p.add_argument('--legacy-test', help='optional .npy of token blocks (all positions scored)')
    p.add_argument('--out', required=True)
    p.add_argument('--max-new-tokens', type=int, default=256)
    args = p.parse_args()
    import numpy as np
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    from openbonsai.adapters import load_adapter
    from openbonsai.formats.state import load_state
    from openbonsai.stages.common import install_ternary
    from openbonsai.core.method_compare import full_kd_ce
    from checks import looping_tail, validate_response
    torch.manual_seed(42); torch.use_deterministic_algorithms(True)
    sealed, out = Path(args.sealed), Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((sealed / 'manifest.json').read_text())
    needed = ['doc-wiki.npy', 'doc-python.npy', 'chat-ids.npy', 'chat-attention.npy', 'chat-target-mask.npy', 'specs.json']
    for rel in needed:
        assert sha(sealed / rel) == manifest['files_sha256'][rel], f'sealed file changed: {rel}'
    target = out / f'results-{args.label}.json'
    if target.exists(): raise SystemExit(f'{args.label} already evaluated; refusing a second look')
    kind, _, source = args.candidate.partition(':')
    with (out / 'access-log.jsonl').open('a', encoding='utf-8') as log:
        log.write(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'label': args.label, 'candidate': args.candidate,
                              'candidate_sha256': sha(source) if source else 'teacher', 'sealed_manifest_sha256': sha(sealed / 'manifest.json'),
                              'teacher_config_sha256': sha(Path(args.model_dir) / 'config.json')}) + '\n')
    start = time.monotonic()
    adapter = load_adapter(args.model_dir, local_files_only=True)
    teacher, tok = adapter.load(args.model_dir, device='cuda:0', dtype=torch.bfloat16, local_files_only=True); teacher.eval()
    if kind == 'bf16':
        model = teacher
    elif kind == 'gptq':
        model, _ = adapter.load(args.model_dir, device='cuda:0', dtype=torch.bfloat16, local_files_only=True)
        install_ternary(adapter, model, load_state(source).records(), torch.device('cuda:0'))
    elif kind == 'gguf':
        model = adapter.from_gguf(source, device='cuda:0', dtype=torch.bfloat16)
    else:
        raise SystemExit('candidate must be bf16, gptq:<path> or gguf:<path>')
    model.eval(); lm, head = adapter.text_model(model), adapter.head(model)
    tlm, thead = adapter.text_model(teacher), adapter.head(teacher)
    result = {'label': args.label, 'candidate': args.candidate, 'metrics': {}}

    def score(ids, attention, targets):
        nll = kl = 0.; agree = n = 0
        with torch.no_grad():
            for i, row in enumerate(ids):
                x = torch.from_numpy(row[None]).to('cuda:0'); m = None if attention is None else torch.from_numpy(attention[i:i + 1]).to('cuda:0')
                h = lm(input_ids=x, attention_mask=m, use_cache=False).last_hidden_state[0]
                th = tlm(input_ids=x, attention_mask=m, use_cache=False).last_hidden_state[0]
                L = len(row)
                for lo in range(0, L - 1, 32):
                    hi = min(lo + 32, L - 1); keep = torch.from_numpy(targets[i, lo:hi]).to('cuda:0'); k = int(keep.sum())
                    if not k: continue
                    z = head(h[lo:hi])[keep]; tz = thead(th[lo:hi])[keep]
                    gold = torch.from_numpy(row[lo + 1:hi + 1]).to('cuda:0')[keep]
                    nll += float(torch.nn.functional.cross_entropy(z.float(), gold, reduction='sum'))
                    kl += float(full_kd_ce(z, tz)) * k; agree += int((z.argmax(-1) == tz.argmax(-1)).sum()); n += k
        return {'targets': n, 'nll': nll / n, 'perplexity': math.exp(nll / n), 'full_vocab_kl': kl / n, 'teacher_top1_agreement': agree / n}

    splits = [('sealed-doc-wiki', np.load(sealed / 'doc-wiki.npy'), None, None), ('sealed-doc-python', np.load(sealed / 'doc-python.npy'), None, None),
              ('sealed-chat', np.load(sealed / 'chat-ids.npy'), np.load(sealed / 'chat-attention.npy'), np.load(sealed / 'chat-target-mask.npy'))]
    if args.legacy_test:
        splits.append(('legacy-test', np.load(args.legacy_test), None, None))
    for name, ids, attention, targets in splits:
        if targets is None:
            targets = np.ones(ids.shape, dtype=np.bool_); targets[:, -1] = False
        result['metrics'][name] = score(ids, attention, targets)
        print(json.dumps({'label': args.label, 'split': name, **result['metrics'][name]}), flush=True)
    specs = json.loads((sealed / 'specs.json').read_text())['specs']; eos = adapter.eos_ids(tok, model); rows = []
    for spec in specs:
        x = tok.apply_chat_template([{'role': 'user', 'content': spec['prompt']}], tokenize=True, add_generation_prompt=True,
                                    enable_thinking=False, return_dict=False, return_tensors='pt').to('cuda:0')
        class Loop(StoppingCriteria):
            triggered = False
            def __call__(self, ids, scores, **kwargs):
                self.triggered = looping_tail(ids[0, x.shape[1]:].tolist()); return torch.tensor([self.triggered], device=ids.device)
        guard = Loop()
        with torch.no_grad():
            gen = model.generate(input_ids=x, do_sample=False, max_new_tokens=args.max_new_tokens, use_cache=True, eos_token_id=eos,
                                 pad_token_id=tok.pad_token_id, stopping_criteria=StoppingCriteriaList([guard]))
        resp = gen[0, x.shape[1]:].tolist(); text = tok.decode(resp, skip_special_tokens=True)
        ok, reason = validate_response(spec, text); finished = bool(resp) and resp[-1] in eos
        rows.append({'id': spec['id'], 'domain': spec['domain'], 'text': text, 'tokens': len(resp), 'ended_at_eos': finished,
                     'repetition_failure_stop': guard.triggered, 'declared_check_pass': bool(ok and finished and not guard.triggered), 'check_reason': reason})
    by_domain = {}
    for r in rows:
        d = by_domain.setdefault(r['domain'], {'pass': 0, 'n': 0}); d['n'] += 1; d['pass'] += r['declared_check_pass']
    result['generation'] = rows
    result['generation_summary'] = {'declared_pass': sum(r['declared_check_pass'] for r in rows), 'n': len(rows), 'eos': sum(r['ended_at_eos'] for r in rows),
                                    'loops': sum(r['repetition_failure_stop'] for r in rows), 'by_domain': by_domain}
    result['seconds'] = time.monotonic() - start; result['gpu_peak_allocated'] = torch.cuda.max_memory_allocated()
    target.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'label': args.label, 'generation': result['generation_summary'], 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()

"""Ternary recovery training: two-step stochastic mirror descent on PTQ1_0 codes, online teacher distillation.

- Start from a GPTQ ternary state (g128 FP16 scales, Hadamard-rotated, tied embedding/head); see build_start.py.
- Every table's trainable code tensor is the mirror-descent dual of a prune+quantize potential
  psi(w) = sum max(|w|^(1+eps), beta |w|^q) with eps=.05, q=8, beta=1 (openbonsai.core.mirror.PruneQuantize),
  initialized at grad_psi(codes), so training starts exactly at the GPTQ codes.
- The dual steps along an Adafactor direction (code LR 0.02, warmup 32, cosine to 0 over the run).
- The shared embedding/head codes stay frozen; group scales and small high-precision tensors
  train with AdamW (cosine to 0.1x over a 2000-update horizon).
- Loss: full-vocabulary KL(teacher||student) on scored targets; the BF16 teacher runs online.
- 2,044 scored targets per update; validation every N updates (mean of document and chat KL);
  best checkpoint kept; patience early stopping; exact resume (duals, optimizers, RNG).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=1) + '\n', encoding='utf-8')


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return type(value)(cpu_tree(v) for v in value)
    return value


def load_split(directory):
    import numpy as np
    d = Path(directory)
    return np.load(d / 'ids.npy'), np.load(d / 'mask.npy')


def train(args):
    import numpy as np
    import torch
    from torch.utils.checkpoint import checkpoint
    from transformers import Adafactor
    from openbonsai.adapters import load_adapter
    from openbonsai.core.data_recovery import target_windows, masked_kd_mean
    from openbonsai.core.method_compare import TrainEmbedding, TrainLinear
    from openbonsai.core.mirror import PruneQuantize, mirror_table, patience_exhausted
    from openbonsai.formats.state import load_state, save_state, save_scales, TernaryState, TernaryTensor
    from openbonsai.formats.atomic import durable_save
    from openbonsai.stages.common import install_ternary
    from openbonsai.stages.distill import make_small_trainable, trained_small

    torch.set_num_threads(8); torch.manual_seed(args.seed); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction); torch.cuda.reset_peak_memory_stats()
    started = time.monotonic(); run = Path(args.run_dir); run.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    inputs = {'initial': args.initial, 'train_ids': Path(args.train) / 'ids.npy', 'train_mask': Path(args.train) / 'mask.npy',
              'val_doc': args.val_doc, 'val_chat_ids': Path(args.val_chat) / 'ids.npy', 'val_chat_mask': Path(args.val_chat) / 'mask.npy',
              'model_config': model_dir / 'config.json'}
    recipe = {'version': 'push-1', 'seed': args.seed, 'fill_seed': args.seed + 800, 'steps': args.steps, 'targets_per_update': 2044,
              'validate_every': args.validate_every, 'patience': args.patience,
              'potential': 'PruneQuantize eps=.05 q=8 beta=1 kink=mid (patent Eq. 23), two-step SMD, dual = trainable tensor',
              'code_update': 'Adafactor(beta1=None, decay_rate=-.8, no parameter scaling) direction on the dual',
              'code_lr': args.code_lr, 'code_schedule': 'warmup 32, cosine from peak to 0 over the run length',
              'scale_lr': .001, 'vector_lr': .0001, 'matrix_lr': .00002, 'small_schedule': 'warmup 32, cosine to 0.1x over horizon 2000',
              'endpoint_codes': 'frozen', 'loss': 'full-vocab masked KL(teacher||student), T1, online BF16 teacher',
              'smoke': args.smoke,
              'inputs_sha256': {k: sha(v) for k, v in inputs.items()},
              'source_sha256': {p: sha(ROOT / p) for p in ['recipe/train.py', 'openbonsai/core/mirror.py', 'openbonsai/core/method_compare.py',
                                                           'openbonsai/core/data_recovery.py']}}
    path = run / 'recipe.json'
    if path.exists(): assert json.loads(path.read_text()) == recipe, 'recipe or inputs changed; refusing to resume'
    else: write(path, recipe)

    adapter = load_adapter(model_dir, local_files_only=True)
    teacher_model, _ = adapter.load(model_dir, device='cuda:0', dtype=torch.bfloat16, local_files_only=True)
    teacher_model.eval(); [p.requires_grad_(False) for p in teacher_model.parameters()]
    tlm = adapter.text_model(teacher_model); teacher_head = adapter.head(teacher_model).weight
    model, _ = adapter.load(model_dir, device='cuda:0', dtype=torch.bfloat16, local_files_only=True)
    assert adapter.embedding(model).weight is adapter.head(model).weight, 'expected tied embedding/head'
    state = load_state(args.initial)
    install_ternary(adapter, model, state.records(), torch.device('cuda:0'))
    for p in model.parameters(): p.requires_grad_(False)
    lm = adapter.text_model(model); vec, mat = make_small_trainable(lm); tables = {}
    potential = PruneQuantize(eps=.05, q=8., beta=1., kink='mid')
    for name in sorted(state.tensors):
        module = adapter.embedding(model) if name == 'embed_tokens' else getattr(lm.get_submodule(name.rpartition('.')[0]), name.rpartition('.')[2])
        table = mirror_table(module, potential); table.master.requires_grad_(name != 'embed_tokens'); table.log_mult.requires_grad_(True)
        tables[name] = table
        if name == 'embed_tokens':
            head = adapter.head(model); adapter.set_embedding(model, TrainEmbedding(module, table)); adapter.set_head(model, TrainLinear(head, table)); del head
        else:
            parent, _, attr = name.rpartition('.'); setattr(lm.get_submodule(parent), attr, TrainLinear(module, table))
        del module
    signs, block = state.signs, state.block; del state
    assert adapter.embedding(model).table is adapter.head(model).table
    optimizer = torch.optim.AdamW([{'params': [t.log_mult for t in tables.values()], 'lr': .001},
                                   {'params': list(vec.values()), 'lr': .0001}, {'params': list(mat.values()), 'lr': .00002}],
                                  betas=(.9, .95), eps=1e-8, weight_decay=0, foreach=False, fused=False)
    duals = [t.master for t in tables.values() if t.master.requires_grad]
    code_optimizer = Adafactor(duals, lr=args.code_lr, scale_parameter=False, relative_step=False, warmup_init=False,
                               beta1=None, decay_rate=-.8, weight_decay=0)
    owned = [p for g in optimizer.param_groups for p in g['params']] + duals
    assert len(owned) == len({id(p) for p in owned}), 'duplicate optimizer ownership'
    for t in tables.values(): assert torch.equal(t.codes().cpu(), t.initial_codes), 'dual init moved a code'
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    train_ids, train_mask = load_split(args.train); chat_ids, chat_mask = load_split(args.val_chat); doc = np.load(args.val_doc)

    def loss_for(ids, mask, score, normalizer, backward):
        x = torch.from_numpy(ids[None]).to('cuda:0'); m = torch.from_numpy(mask[None]).to('cuda:0')
        with torch.no_grad():
            th = tlm(input_ids=x, attention_mask=m, use_cache=False).last_hidden_state[0]
        h = lm(input_ids=x, attention_mask=m, use_cache=False).last_hidden_state[0]
        head = adapter.head(model); q = head.table.weight()
        hr = ((h * head.signs).reshape(len(h), head.in_features // head.block, head.block) @ head.h).reshape_as(h)
        score = score.to('cuda:0'); loss = 0.
        def chunk(hh, tt, weight, mk):
            return masked_kd_mean(torch.nn.functional.linear(hh, weight), torch.nn.functional.linear(tt, teacher_head), mk)
        for lo in range(0, 511, 32):
            hi = min(lo + 32, 511); mk = score[lo:hi]; n = int(mk.sum())
            if not n: continue
            value = checkpoint(chunk, hr[lo:hi], th[lo:hi], q, mk, use_reentrant=False) if backward else chunk(hr[lo:hi], th[lo:hi], q, mk)
            loss = loss + value * (n / normalizer)
        return loss

    @torch.no_grad()
    def validate():
        model.eval(); values = {}
        for split in ('document', 'chat'):
            total = 0.; count = 0
            rows = len(doc) if split == 'document' else len(chat_ids)
            for i in range(rows):
                ids, mask = (doc[i], np.ones(doc.shape[1], dtype=np.bool_)) if split == 'document' else (chat_ids[i], chat_mask[i])
                score = torch.from_numpy(mask[1:] & mask[:-1])
                total += float(loss_for(ids, mask, score, 1, False)); count += int(score.sum())
            values[split] = total / count
        values['combined_kl'] = .5 * (values['document'] + values['chat'])
        assert all(math.isfinite(v) for v in values.values())
        return values

    def hard_state():
        return TernaryState({k: TernaryTensor(t.codes().cpu(), t.scales().detach().half().cpu()) for k, t in tables.items()}, signs, block, 128)

    def code_changes():
        out = {'changed': 0, 'zero_to_nonzero': 0, 'nonzero_to_zero': 0, 'sign_flip': 0, 'total': 0}
        for t in tables.values():
            new, old = t.codes().cpu(), t.initial_codes; diff = new != old
            out['changed'] += int(diff.sum()); out['total'] += old.numel()
            out['zero_to_nonzero'] += int((diff & (old == 0)).sum()); out['nonzero_to_zero'] += int((diff & (new == 0)).sum())
            out['sign_flip'] += int((diff & (old != 0) & (new != 0)).sum())
        return out

    primary = torch.Generator().manual_seed(args.seed); fill = torch.Generator().manual_seed(args.seed + 800)
    small = {**vec, **mat}; history = []; best = float('inf'); start_step = 0

    def snapshot(step):
        nonlocal best
        history[-1] = {**history[-1], 'codes': code_changes()}
        if history[-1]['combined_kl'] < best:
            best = history[-1]['combined_kl']; save_state(run / 'best.safetensors', hard_state())
            values, dtypes = trained_small(model, adapter); save_scales(run / 'best-small.safetensors', {}, history[-1], values, dtypes)
        write(run / 'history.json', history)
        durable_save({'recipe': recipe, 'step': step, 'scales': {k: t.log_mult.detach().cpu().clone() for k, t in tables.items()},
                      'small': {k: p.detach().cpu().clone() for k, p in small.items()}, 'optimizer': cpu_tree(optimizer.state_dict()),
                      'duals': {k: t.master.detach().cpu().clone() for k, t in tables.items()}, 'code_optimizer': cpu_tree(code_optimizer.state_dict()),
                      'primary_rng': primary.get_state(), 'fill_rng': fill.get_state(), 'rng': torch.get_rng_state(),
                      'cuda_rng': torch.cuda.get_rng_state(), 'best': best, 'history': history}, run / 'latest.pt')

    if (run / 'latest.pt').exists():
        saved = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False); assert saved['recipe'] == recipe
        with torch.no_grad():
            for k, t in tables.items(): t.log_mult.copy_(saved['scales'][k].to('cuda:0')); t.master.copy_(saved['duals'][k].to('cuda:0'))
            for k, p in small.items(): p.copy_(saved['small'][k].to('cuda:0'))
        optimizer.load_state_dict(saved['optimizer']); code_optimizer.load_state_dict(saved['code_optimizer'])
        primary.set_state(saved['primary_rng']); fill.set_state(saved['fill_rng'])
        torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
        start_step, best, history = saved['step'], saved['best'], saved['history']; del saved
    else:
        history.append({'step': 0, **validate()}); snapshot(0)
        print(json.dumps({'validation': history[-1]}), flush=True)

    limit = min(args.steps, args.stop_after or args.steps); stopped_by = None
    for step in range(start_step, limit):
        model.train(); windows = target_windows(train_mask, primary, fill)
        assert sum(int(s.sum()) for _, s in windows) == 2044
        total = 0.
        for i, score in windows:
            value = loss_for(train_ids[i], train_mask[i], score, 2044, True)
            if not torch.isfinite(value): raise RuntimeError('nonfinite loss')
            value.backward(); total += float(value.detach())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        small_mult = min(1., (step + 1) / 32) * (.1 + .9 * (1 + math.cos(math.pi * step / 2000)) / 2)
        code_mult = min(1., (step + 1) / 32) * (1 + math.cos(math.pi * step / args.steps)) / 2
        for group, lr in zip(optimizer.param_groups, [.001, .0001, .00002]): group['lr'] = lr * small_mult
        code_optimizer.param_groups[0]['lr'] = args.code_lr * code_mult
        code_optimizer.step(); code_optimizer.zero_grad(set_to_none=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        print(json.dumps({'step': step + 1, 'train_loss': total, 'gradient_norm': float(norm),
                          'rows': [i for i, _ in windows], 'seconds': time.monotonic() - started,
                          'gpu_peak_allocated': torch.cuda.max_memory_allocated()}), flush=True)
        finish = step + 1 >= limit or time.monotonic() - started >= args.timeout - 120
        if (step + 1) % args.validate_every == 0 or finish or args.smoke and step + 1 <= 2:
            history.append({'step': step + 1, **validate()}); snapshot(step + 1)
            print(json.dumps({'validation': history[-1]}), flush=True)
            if not finish and patience_exhausted(history, args.patience):
                stopped_by = 'patience'; print(json.dumps({'early_stop': 'patience', 'step': step + 1}), flush=True); break
        if finish:
            stopped_by = 'complete' if step + 1 >= limit else 'time cap'; break
    write(run / 'result.json', {'step': history[-1]['step'], 'best_combined_kl': best, 'stopped_by': stopped_by or 'resumed-complete',
                                'codes': code_changes(), 'seconds': time.monotonic() - started,
                                'gpu_peak_allocated': torch.cuda.max_memory_allocated(), 'gpu_peak_reserved': torch.cuda.max_memory_reserved()})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-dir', required=True, help='BF16 Hugging Face model directory (student base and online teacher)')
    p.add_argument('--initial', required=True, help='GPTQ ternary state (.safetensors) to start from')
    p.add_argument('--train', required=True, help='directory with ids.npy and mask.npy (rows x 512)')
    p.add_argument('--val-doc', required=True, help='.npy of 512-token document blocks for validation')
    p.add_argument('--val-chat', required=True, help='directory with chat ids.npy and mask.npy for validation')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--validate-every', type=int, default=100)
    p.add_argument('--patience', type=int, default=3, help='stop after N validations without a new best (0 disables)')
    p.add_argument('--code-lr', type=float, default=.02)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mem-fraction', type=float, default=.92)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--timeout', type=int, default=7200)
    p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    if not 180 <= args.timeout <= 43200: p.error('timeout must be 180..43200 s')
    if args.worker: return train(args)
    import psutil
    run = Path(args.run_dir); run.mkdir(parents=True, exist_ok=True); start = time.monotonic(); peak = 0; reason = None
    with (run / 'train.log').open('a', encoding='utf-8') as log:
        proc = subprocess.Popen([sys.executable, '-u', __file__, *sys.argv[1:], '--worker'], stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            try:
                tree = [psutil.Process(proc.pid)] + psutil.Process(proc.pid).children(recursive=True)
                peak = max(peak, sum(x.memory_info().rss for x in tree if x.is_running()))
                if psutil.virtual_memory().available < 4 * 2**30: reason = 'RAM cap'
                if time.monotonic() - start > args.timeout: reason = 'wall time cap'
                if reason:
                    for x in reversed(tree): x.kill()
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(1)
        rc = proc.wait()
    write(run / f'supervisor-{time.time_ns()}.json', {'returncode': rc, 'reason': reason, 'seconds': time.monotonic() - start, 'peak_process_tree_rss': peak})
    print(json.dumps({'returncode': rc, 'reason': reason, 'seconds': time.monotonic() - start}), flush=True)
    raise SystemExit(rc)


if __name__ == '__main__':
    main()

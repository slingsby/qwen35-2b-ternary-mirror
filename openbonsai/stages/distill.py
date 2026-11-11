"""End-to-end distillation of the ternary group scales (codes frozen).

Port of OpenBonsai research/experiments/e2e_kd.py with the same arithmetic. The codes
from GPTQ stay fixed; every (row, group) scale gets a trainable log
multiplier, trained through the whole model against the original BF16
model's top-k next-token distribution (cross-entropy on the top-k plus a
tail bucket for the remaining mass). This targets cross-layer error that
layer-local reconstruction cannot see.

``train.small`` (e2e_kd.py ``--train-small``) also trains every high-precision
decoder tensor that is not ternary (norms, conv1d, in_proj_a/b, A_log,
dt_bias, the final norm): each gets an FP32 master behind a parametrization
that casts it to its original dtype on every access, so the forward is
unchanged; vector-like tensors use ``lr_small``, matrix-like ones
``lr_small_mat``. The masters are saved with the scales (``small::<name>``,
names relative to the text model) and the export writes the values the
model computed with.

Two phases, both resumable:

- teacher: ``distill/teacher/input_ids.npy`` then ``topk_NNNNNN.npz`` shards
  (int32 ids, fp16 log-probs), each written atomically; finished shards are
  skipped on rerun.
- train: AdamW on the log multipliers with warmup + cosine; a checkpoint
  (parameters, optimizer, scheduler, position) every ``ckpt_every`` steps is
  resumed exactly. The result is ``distill/scales.safetensors`` (fp16); its
  metadata records the quantize/endpoints output digests it was trained on.

``max_steps`` shortens training: the cosine schedule ends at that step. With
it unset the schedule spans ``epochs * n_seq / accum`` steps, as in the script.

With the refine stage enabled, distillation starts from refine's codes and
scales (``StageContext.decoder_state``) and records refine's output digest.
"""
from __future__ import annotations

import io
import json
import math
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from ..config import distill_uses_endpoints
from ..core.losses import kd_chunk_loss
from ..core.ternary import TernaryEmbedding, TernaryLinear
from ..formats.atomic import atomic_write_bytes, durable_save
from ..formats.state import save_scales
from . import calibrate
from .common import free, install_ternary


def teacher_sources(cfg: dict) -> list[dict]:
    """The experiments' mix: a quarter WikiText (<= 1000), 1/32 traces (<= 24), the rest chat."""
    t = cfg["distill"]["teacher"]
    if t.get("sources"):
        return t["sources"]
    n = int(t["n_seq"])
    n_wiki, n_ayot = min(n // 4, 1000), min(n // 32, 24)
    return [{"kind": "wikitext2", "n": n_wiki}, {"kind": "ultrachat", "n": n - n_wiki - n_ayot, "min_row": 6000},
            {"kind": "traces", "n": n_ayot}]


def _npy_bytes(arr) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


@torch.no_grad()
def teacher_topk(model, data, out_dir, *, topk=64, shard=64, device, log=print):
    """Teacher top-k in atomic shards of ``shard`` rows; rerunning resumes after the last shard."""
    n, t = data.shape
    t0 = time.time()
    for s0 in range(0, n, shard):
        path = out_dir / f"topk_{s0:06d}.npz"
        if path.exists():
            continue
        rows = range(s0, min(s0 + shard, n))
        idx = np.empty((len(rows), t, topk), np.int32)
        lp = np.empty((len(rows), t, topk), np.float16)
        for k, i in enumerate(rows):
            logp = model(input_ids=data[i:i + 1].to(device), use_cache=False).logits[0].float().log_softmax(-1)
            v, j = logp.topk(topk, -1)
            idx[k] = j.int().cpu().numpy()
            lp[k] = v.half().cpu().numpy()
        atomic_write_bytes(path, _npz_bytes(idx=idx, logp=lp))
        log(f"teacher {s0 + len(rows)}/{n} {time.time() - t0:.0f}s "
            f"mass={np.exp(lp.astype(np.float32)).sum(-1).mean():.3f}")


def load_teacher(data_dir):
    ids = np.load(data_dir / "input_ids.npy")
    shards = sorted(data_dir.glob("topk_*.npz"))
    starts = [int(f.stem.split("_")[1]) for f in shards]
    idx = [np.load(f)["idx"] for f in shards]
    lp = [np.load(f)["logp"] for f in shards]
    expected = 0
    for s, a in zip(starts, idx):
        if s != expected:
            raise ValueError(f"teacher shards are not contiguous at row {expected}")
        expected += len(a)
    if expected != len(ids):
        raise ValueError(f"teacher incomplete: {expected}/{len(ids)} rows")
    return ids, np.concatenate(idx), np.concatenate(lp)


def read_teacher_rows(data_dir, n: int | None = None):
    """The first ``n`` rows (all when None) of the teacher shards as (ids, idx, logp),
    reading only the shards those rows are in."""
    ids_path = data_dir / "input_ids.npy"
    if not ids_path.exists():
        raise FileNotFoundError(f"{ids_path} is missing")
    ids = np.load(ids_path, mmap_mode="r")
    n = len(ids) if n is None else min(int(n), len(ids))
    idx, lp, row = [], [], 0
    while row < n:
        path = data_dir / f"topk_{row:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing")
        with np.load(path) as z:
            a, b = z["idx"], z["logp"]
        take = min(len(a), n - row)
        idx.append(a[:take])
        lp.append(b[:take])
        row += len(a)
    return np.array(ids[:n]), np.concatenate(idx), np.concatenate(lp)


def _teacher_complete(out, shard: int) -> tuple[int, int] | None:
    """(rows, seq_len) when the inputs and every shard exist."""
    ids_path = out / "input_ids.npy"
    if not ids_path.exists():
        return None
    n, t = np.load(ids_path, mmap_mode="r").shape
    if all((out / f"topk_{s0:06d}.npz").exists() for s0 in range(0, n, shard)):
        return int(n), int(t)
    return None


def run_teacher(ctx) -> dict:
    d = ctx.cfg["distill"]
    t = d["teacher"]
    log = ctx.log("distill")
    out = ctx.run.stage_dir("distill") / "teacher"
    out.mkdir(parents=True, exist_ok=True)
    ids_path = out / "input_ids.npy"
    done = _teacher_complete(out, int(t["shard"]))
    if done is not None:
        log(f"teacher: all {done[0]} rows present; model not loaded")
        return {"rows": done[0], "seq_len": done[1]}
    adapter = ctx.adapter()
    model, tok = ctx.load_model()
    if ids_path.exists():
        data = torch.from_numpy(np.load(ids_path).astype(np.int64))
    else:
        data, report = calibrate.build(tok, teacher_sources(ctx.cfg), seq_len=int(t["seq_len"]), seed=int(t["seed"]),
                                       chat_kwargs=adapter.chat_kwargs(thinking=False), bos=adapter.bos_id(tok),
                                       traces_path=ctx.run.stage_dir("traces", create=False) / "traces.jsonl",
                                       local_files_only=bool(ctx.cfg["model"].get("local_files_only")),
                                       where="distill.teacher")
        atomic_write_bytes(out / "sources.json", json.dumps(report, indent=1).encode("utf-8"))
        atomic_write_bytes(ids_path, _npy_bytes(data.numpy().astype(np.int32)))
    teacher_topk(model, data, out, topk=int(t["topk"]), shard=int(t["shard"]), device=ctx.device, log=log)
    return {"rows": int(data.shape[0]), "seq_len": int(data.shape[1])}


class _CastTo(torch.nn.Module):
    """Parametrization: FP32 master -> the tensor's original dtype on every access."""

    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return x.to(self.dtype)


def make_small_trainable(lm):
    """Give every non-ternary decoder tensor (norms, conv1d, in_proj_a/b, A_log, dt_bias, final
    norm) an FP32 master; returns {name: master} split into (vector-like, matrix-like) dicts.

    e2e_kd.py's make_small_params_trainable, unchanged (same tensors, same order)."""
    import torch.nn.utils.parametrize as parametrize
    vec, mat = {}, {}
    targets = []  # collect first: registering parametrizations adds submodules
    for mname, mod in lm.named_modules():
        if isinstance(mod, (TernaryLinear, TernaryEmbedding)) or "parametrizations" in mname:
            continue
        for pname, p in mod.named_parameters(recurse=False):
            full = f"{mname}.{pname}" if mname else pname
            if full.startswith("layers.") or full == "norm.weight":
                targets.append((mod, pname, full, p))
    for mod, pname, full, p in targets:
        dtype = p.dtype
        mod._parameters[pname] = torch.nn.Parameter(p.detach().float())
        parametrize.register_parametrization(mod, pname, _CastTo(dtype), unsafe=True)
        master = mod.parametrizations[pname].original
        master.requires_grad_(True)
        (mat if master.dim() >= 2 else vec)[full] = master
    return vec, mat


def trained_small(model, adapter) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """(FP32 masters, compute dtype names) of the tensors ``make_small_trainable`` wrapped."""
    import torch.nn.utils.parametrize as parametrize
    masters, dtypes = {}, {}
    for mname, mod in adapter.text_model(model).named_modules():
        if "parametrizations" in mname or not parametrize.is_parametrized(mod):
            continue
        for pname, plist in mod.parametrizations.items():
            casts = [c for c in plist if isinstance(c, _CastTo)]
            if not casts:
                continue
            full = f"{mname}.{pname}" if mname else pname
            masters[full] = plist.original.detach().float().cpu()
            dtypes[full] = str(casts[0].dtype).removeprefix("torch.")
    return masters, dtypes


def train(model, adapter, ids, tidx_all, tlogp_all, out_dir, *, lr=1e-3, epochs=1, accum=8, warmup=10,
          chunk=512, log_every=5, ckpt_every=16, max_steps=None, small=False, lr_small=1e-4, lr_small_mat=2e-5,
          device, log=print, stop_after=None):
    """Train every TernaryLinear/TernaryEmbedding log multiplier (and, with ``small``, the
    high-precision decoder tensors); returns the per-step loss history.

    ``max_steps`` ends training (and the cosine schedule) early; ``stop_after``
    (tests) raises KeyboardInterrupt after that step, as a crash would."""
    torch.manual_seed(0)
    for p in model.parameters():
        p.requires_grad_(False)
    scale_params = list({id(m.log_mult): m.log_mult for m in model.modules()
                         if isinstance(m, (TernaryLinear, TernaryEmbedding))}.values())
    for p in scale_params:
        p.requires_grad_(True)
    lm = adapter.text_model(model)
    small_vec, small_mat = make_small_trainable(lm) if small else ({}, {})
    small_params = list(small_vec.values()) + list(small_mat.values())
    log(f"trainable: {sum(p.numel() for p in scale_params)} scale params, "
        f"{sum(p.numel() for p in small_params)} small high-precision params "
        f"({len(small_vec)} vector, {len(small_mat)} matrix tensors)")
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    head = adapter.head(model)
    n = ids.shape[0]
    groups = [{"params": scale_params, "lr": lr}]
    if small_vec:
        groups.append({"params": list(small_vec.values()), "lr": lr_small})
    if small_mat:
        groups.append({"params": list(small_mat.values()), "lr": lr_small_mat})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-12)
    trainable = scale_params + small_params
    total_steps = max(1, epochs * n // accum)
    if max_steps is not None:
        total_steps = min(total_steps, max(1, int(max_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    order = []
    for ep in range(epochs):
        order += torch.randperm(n, generator=torch.Generator().manual_seed(ep)).tolist()
    ckpt = out_dir / "checkpoint.pt"
    history_path = out_dir / "train.jsonl"
    step, start_it = 0, 0
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        if len(ck["params"]) != len(trainable):
            raise ValueError(f"{ckpt} holds {len(ck['params'])} tensors, this run trains {len(trainable)}")
        for p, v in zip(trainable, ck["params"]):
            p.data.copy_(v)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, start_it = ck["step"], ck["it"]
        log(json.dumps({"resumed_at_step": step}))
    if history_path.exists():   # drop records logged after the checkpoint (they are recomputed)
        kept = [line for line in history_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line)["step"] <= step]
        atomic_write_bytes(history_path, "".join(line + "\n" for line in kept).encode("utf-8"))
    model.train()
    t0, run_loss, run_tok, step_loss, step_tok = time.time(), 0.0, 0, 0.0, 0
    history = []
    for it, i in enumerate(order):
        if it < start_it:
            continue
        if step >= total_steps:
            break
        x = torch.from_numpy(ids[i:i + 1].astype(np.int64)).to(device)
        tidx = torch.from_numpy(tidx_all[i].astype(np.int64)).to(device)
        tlogp = torch.from_numpy(np.asarray(tlogp_all[i])).to(device)
        h = lm(input_ids=x, use_cache=False).last_hidden_state[0]
        loss = 0.0
        tlen = h.shape[0] - 1
        for c0 in range(0, tlen, chunk):
            c1 = min(c0 + chunk, tlen)
            loss = loss + checkpoint(kd_chunk_loss, h[c0:c1], head, tidx[c0:c1], tlogp[c0:c1], use_reentrant=False)
        (loss / (tlen * accum)).backward()
        run_loss += loss.item()
        run_tok += tlen
        step_loss += loss.item()
        step_tok += tlen
        if (it + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            record = {"step": step, "kd_ce": step_loss / step_tok, "lr": sched.get_last_lr()[0]}
            history.append(record)
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            step_loss, step_tok = 0.0, 0
            if step % log_every == 0:
                el = time.time() - t0
                lm_abs = torch.cat([p.detach().flatten() for p in scale_params]).abs()
                log(json.dumps({"step": step, "kd_ce": run_loss / run_tok, "tok_s": run_tok / el, "elapsed": el,
                                "lr": sched.get_last_lr()[0], "mult_absmean": lm_abs.mean().item(),
                                "mult_absmax": lm_abs.max().item()}))
                run_loss, run_tok, t0 = 0.0, 0, time.time()
            if step % ckpt_every == 0:
                durable_save({"params": [p.detach() for p in trainable], "opt": opt.state_dict(),
                              "sched": sched.state_dict(), "step": step, "it": it + 1}, ckpt)
            if stop_after is not None and step >= stop_after:
                raise KeyboardInterrupt(f"stopped after step {step}")
    model.eval()
    return history


def trained_scales(model, adapter) -> dict[str, torch.Tensor]:
    out = {}
    for name, m in adapter.layers(model).named_modules():
        if isinstance(m, TernaryLinear):
            out[f"layers.{name}"] = m.scales().detach().half().cpu()
    for key, m in (("lm_head", adapter.head(model)), ("embed_tokens", adapter.embedding(model))):
        if key == "lm_head" and adapter.tied and isinstance(adapter.embedding(model), TernaryEmbedding):
            continue  # shared scale is serialized once under the canonical embedding key
        if isinstance(m, (TernaryLinear, TernaryEmbedding)):
            out[key] = m.scales().detach().half().cpu()
    return out


def run_train(ctx) -> dict:
    d = ctx.cfg["distill"]
    what = d.get("train") or {}
    if not what.get("scales", True):
        raise NotImplementedError("distill.train.scales: false is not supported (the scales are always trained)")
    small = bool(what.get("small"))
    log = ctx.log("distill")
    out = ctx.run.stage_dir("distill")
    adapter = ctx.adapter()
    model, _ = ctx.load_model()
    state = ctx.decoder_state()     # refine's codes and scales when refine is enabled
    if distill_uses_endpoints(ctx.cfg):
        ep = ctx.endpoints_state()
        if ep.tensors:
            state = state.merged(ep)
    install_ternary(adapter, model, state.records(), ctx.device)
    ids, tidx, tlogp = load_teacher(out / "teacher")
    history = train(model, adapter, ids, tidx, tlogp, out, lr=float(d["lr"]), epochs=int(d["epochs"]),
                    accum=int(d["accum"]), warmup=int(d["warmup"]), chunk=int(d["chunk"]),
                    log_every=int(d["log_every"]), ckpt_every=int(d["ckpt_every"]),
                    max_steps=None if d.get("max_steps") is None else int(d["max_steps"]),
                    small=small, lr_small=float(d["lr_small"]), lr_small_mat=float(d["lr_small_mat"]),
                    device=ctx.device, log=log)
    scales = trained_scales(model, adapter)
    masters, dtypes = trained_small(model, adapter) if small else ({}, {})
    save_scales(out / "scales.safetensors", scales,
                {"stage": "distill", "steps_this_run": len(history), "base": ctx.distill_base()},
                small=masters, small_dtypes=dtypes)
    return {"scales": len(scales), "small": len(masters), "steps_this_run": len(history)}


def _reset_if_changed(marker, value: str, paths) -> None:
    """Drop resumable outputs made under different inputs, then record the inputs."""
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("hash") != value:
        for p in paths:
            if p.is_dir():
                for f in p.glob("*"):
                    f.unlink()
            elif p.exists():
                p.unlink()
    atomic_write_bytes(marker, json.dumps({"hash": value}).encode("utf-8"))


def run(ctx) -> dict:
    from ..config import canonical_hash, stage_inputs, uses_traces
    out = ctx.run.stage_dir("distill")
    teacher_dir = out / "teacher"
    teacher_dir.mkdir(exist_ok=True)
    sources = teacher_sources(ctx.cfg)
    traces_key = ctx.run.done_key("traces") if uses_traces(sources) else None
    teacher_hash = canonical_hash({"model": stage_inputs(ctx.cfg, "distill")["model"],
                                   "model_sha256": ctx.model_digest(), "teacher": ctx.cfg["distill"]["teacher"],
                                   "files": ctx.file_digests([str(s["path"]) for s in sources if s.get("path")]),
                                   "traces": traces_key})
    _reset_if_changed(out / "TEACHER_INPUTS.json", teacher_hash, [teacher_dir])
    _reset_if_changed(out / "TRAIN_INPUTS.json", ctx.stage_hash("distill"),
                      [out / "checkpoint.pt", out / "train.jsonl", out / "scales.safetensors"])
    teacher = run_teacher(ctx)
    free(ctx.device)   # the teacher model is gone before the student loads
    return {"teacher": teacher, "train": run_train(ctx)}

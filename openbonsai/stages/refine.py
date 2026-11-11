"""Refine: block-wise training of the ternary codes and scales, layer by layer (docs/refine.md).

Optional (``refine.enabled``, off by default); it runs after quantize and
endpoints and before distill. Quantize fixes every code by GPTQ and distill
trains only the scales; refine lets the codes change, one decoder layer (or
``window`` layers) at a time, against the original model's layer output.

Each planned linear of the layer in training becomes a
``LatentTernaryLinear`` (core/ternary.py): an FP32 latent weight in units of
its group's GPTQ scale, a log multiplier per scale, codes by the ``assign``
rule, and a straight-through estimator whose forward is the hard ternary
layer bit for bit. The latent starts at the input codes, so a run with no
steps (``epochs: 0``, or both learning rates 0) reproduces the quantize state
exactly. Codes freeze after ``freeze_codes_after`` of the steps; only the
scales train after that.

Activation streams, one tensor [batch, seq, hidden] per element of
``batch`` calibration blocks:

- X_q, the student's input: the hidden states through the already-refined
  layers (and the ternary embedding from endpoints when there is one);
- X_fp, the original model's input (original embedding and layers), kept
  when the target needs it and always for the held-out blocks.

The target (``target``) is the original layer's output on X_fp
(``original``, GPTAQ-style), on X_q (``quantized``, the objective GPTQ
already minimizes), or both (``mix``). The loss is the per-token relative
error ||y_t - T_t||^2 / max(||T_t||^2, 1e-6 * mean ||T||^2) in FP32. Every
window is scored on ``val_blocks`` held-out blocks against both targets
before and after training; with ``rollback`` a window whose held-out error
against its own target did not go down keeps its input codes and scales.

Training blocks are a seeded subset of the calibration blocks, stratified by
calibration source (calib/sources.json), with the held-out blocks drawn
first. The GPU holds the layer in training (and the streams, under ``cache:
auto`` when they fit); the rest of the model waits in CPU memory.

Output: ``refine/state.safetensors``, a TernaryState with quantize's keys,
signs, block and group (layers outside ``refine.layers`` are quantize's
unchanged), and ``refine/report.json``. Each finished layer is written
atomically to ``refine/layers/NNNN.safetensors`` with its statistics; a rerun
resumes after the last finished window, rebuilding the streams by pushing
the blocks through the saved layers (no stream files are kept).
"""
from __future__ import annotations

import copy
import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from ..config import ConfigError, refine_hashed
from ..core.hadamard import rotate
from ..core.ternary import ASSIGNS, INITS, LatentTernaryLinear, TernaryEmbedding, TernaryLinear, shared_hadamard
from ..formats.atomic import atomic_write_json
from ..formats.state import TernaryState, TernaryTensor, load_state, save_state
from .common import cap_vram, cuda_index, free
from .quantize import _Stop, capture_layer_kwargs, layer_forward

TARGETS = ("original", "quantized", "mix")
CACHES = ("auto", "cuda", "cpu", "disk")
CHECKPOINT = ("auto", True, False)
BETAS = (0.9, 0.95)
GPU_STREAM_BUDGET = 8 * 2**30      # cache auto: streams stay on the GPU up to this many bytes
REL_FLOOR = 1e-6                   # relative floor of a token's target energy (no absolute epsilon)


# -- configuration -----------------------------------------------------------------------------------------

def _number(r: dict, key: str, *, lo=None, hi=None, lo_open=False, hi_open=False, integer=False):
    value = r.get(key)
    if integer:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"refine.{key} must be an integer, not {value!r}")
    else:
        try:
            if isinstance(value, bool):
                raise TypeError
            value = float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"refine.{key} must be a number, not {value!r}") from None
        if not math.isfinite(value):
            raise ConfigError(f"refine.{key} must be finite")
    if lo is not None and (value <= lo if lo_open else value < lo):
        raise ConfigError(f"refine.{key} must be {'>' if lo_open else '>='} {lo}, not {value!r}")
    if hi is not None and (value >= hi if hi_open else value > hi):
        raise ConfigError(f"refine.{key} must be {'<' if hi_open else '<='} {hi}, not {value!r}")
    return value


def layer_range(cfg: dict, n_layers: int) -> tuple[int, int]:
    """``refine.layers`` as [start, stop) of decoder layers (null: all)."""
    layers = cfg["refine"].get("layers")
    if layers is None:
        return 0, int(n_layers)
    if (not isinstance(layers, (list, tuple)) or len(layers) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in layers)):
        raise ConfigError(f"refine.layers must be null or [start, stop], not {layers!r}")
    start, stop = int(layers[0]), int(layers[1])
    if not 0 <= start < stop <= n_layers:
        raise ConfigError(f"refine.layers [{start}, {stop}] must satisfy 0 <= start < stop <= {n_layers} "
                          "(the model's decoder layers)")
    return start, stop


def calibration_total(cfg: dict) -> int:
    return sum(max(int(s.get("n") or 0), 0) for s in cfg["calibrate"].get("sources") or ())


def check_config(cfg: dict, n_layers: int | None = None, calib_blocks: int | None = None) -> None:
    """Refuse refine settings the stage cannot run (ConfigError), before any stage runs."""
    r = cfg["refine"]
    for key, allowed in (("target", TARGETS), ("assign", ASSIGNS), ("init", INITS), ("cache", CACHES)):
        if r.get(key) not in allowed:
            raise ConfigError(f"refine.{key} must be one of {list(allowed)}, not {r.get(key)!r}")
    if r.get("checkpoint") not in CHECKPOINT:
        raise ConfigError(f"refine.checkpoint must be auto, true or false, not {r.get('checkpoint')!r}")
    if not isinstance(r.get("rollback"), bool):
        raise ConfigError(f"refine.rollback must be true or false, not {r.get('rollback')!r}")
    _number(r, "window", lo=1, integer=True)
    batch = _number(r, "batch", lo=1, integer=True)
    blocks = _number(r, "blocks", lo=1, integer=True)
    val = _number(r, "val_blocks", lo=0, integer=True)
    _number(r, "epochs", lo=0, integer=True)
    _number(r, "seed", integer=True)
    for key in ("lr_codes", "lr_scales"):
        _number(r, key, lo=0)
    _number(r, "eps", lo=0, lo_open=True)
    _number(r, "warmup", lo=0, hi=1, hi_open=True)
    for key in ("mix", "zero_anneal", "freeze_codes_after"):
        _number(r, key, lo=0, hi=1)
    _number(r, "zeros", lo=0, hi=int(cfg["rotation"]["group"]), integer=True)
    if blocks % batch or val % batch:
        raise ConfigError(f"refine.blocks ({blocks}) and refine.val_blocks ({val}) must be multiples of refine.batch "
                          f"({batch}): the captured rotary/mask tensors are tied to one batch size")
    if r["rollback"] and val == 0:
        raise ConfigError("refine.rollback needs held-out blocks (refine.val_blocks > 0); set refine.rollback=false "
                          "to train without them")
    if r["assign"] == "quantile":
        fca, za = float(r["freeze_codes_after"]), float(r["zero_anneal"])
        if 0 < fca < za:
            raise ConfigError(f"refine.zero_anneal ({za}) must not exceed refine.freeze_codes_after ({fca}) under "
                              "assign quantile: the codes would freeze before every group reaches refine.zeros")
    if n_layers is not None:
        layer_range(cfg, n_layers)
    elif r.get("layers") is not None:
        layer_range(cfg, 1 << 30)
    total = calibration_total(cfg) if calib_blocks is None else int(calib_blocks)
    if blocks + val > total:
        raise ConfigError(f"refine.blocks + refine.val_blocks = {blocks + val} exceeds the {total} calibration "
                          "blocks (calibrate.sources); lower them or add calibration blocks")


def checkpointing(r: dict) -> bool:
    mode = r.get("checkpoint", "auto")
    if mode == "auto":
        return int(r["window"]) >= 2 or int(r["batch"]) > 2
    return bool(mode)


# -- calibration subset ------------------------------------------------------------------------------------

def source_labels(calib_dir: Path, n: int) -> list[int]:
    """The calibration source of each block of calib/tokens.npy (calibrate concatenates the sources in
    order, then permutes the blocks with its seed). All 0 when sources.json does not describe the file."""
    try:
        info = json.loads((Path(calib_dir) / "sources.json").read_text(encoding="utf-8"))
        counts = [int(s.get("blocks") or 0) for s in info.get("sources") or ()]
        seed = int(info.get("seed", 0))
    except (OSError, ValueError, TypeError, AttributeError):
        return [0] * n
    if sum(counts) != n:
        return [0] * n
    order = [k for k, c in enumerate(counts) for _ in range(c)]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return [order[p] for p in perm.tolist()]


def select_blocks(labels, n_train: int, n_val: int, seed: int) -> tuple[list[int], list[int]]:
    """(training, held-out) block indices, stratified by source: each source's blocks in a seeded order,
    interleaved so every prefix keeps the sources' shares; the held-out blocks come first, so they do not
    change with ``n_train``."""
    labels = list(labels)
    if n_train + n_val > len(labels):
        raise ConfigError(f"refine needs {n_train + n_val} calibration blocks, the run has {len(labels)}")
    g = torch.Generator().manual_seed(int(seed))
    keyed = []
    for k in sorted(set(labels)):
        idx = [i for i, lab in enumerate(labels) if lab == k]
        perm = torch.randperm(len(idx), generator=g).tolist()
        keyed += [((rank + 0.5) / len(idx), k, idx[p]) for rank, p in enumerate(perm)]
    order = [i for _, _, i in sorted(keyed)]
    return order[n_val:n_val + n_train], order[:n_val]


# -- activation streams ------------------------------------------------------------------------------------

class Stream:
    """Activation blocks, one tensor [batch, seq, hidden] per element, kept on the compute device
    (``cuda``), in CPU memory (``cpu``, pinned when the device is a GPU) or as files (``disk``)."""

    def __init__(self, where: str, device, directory: Path | None = None):
        self.where, self.device, self.directory = where, torch.device(device), directory
        self.items: list = []
        if where == "disk":
            if directory is None:
                raise ValueError("a disk stream needs a directory")
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir(parents=True)

    def _store(self, x):
        x = x.detach()
        if self.where == "cuda":
            return x.to(self.device)
        x = x.to("cpu")
        if self.device.type == "cuda":
            try:
                x = x.pin_memory()
            except RuntimeError:
                pass
        return x

    def append(self, x) -> None:
        if self.where == "disk":
            path = self.directory / f"{len(self.items):06d}.pt"
            torch.save(x.detach().cpu().clone(), path)
            self.items.append(path)
        else:
            self.items.append(self._store(x))

    def __getitem__(self, j):
        item = self.items[j]
        if self.where == "disk":
            return torch.load(item, weights_only=True).to(self.device)
        return item.to(self.device, non_blocking=True)

    def __setitem__(self, j, x) -> None:
        if self.where == "disk":
            torch.save(x.detach().cpu().clone(), self.items[j])
        else:
            self.items[j] = self._store(x)

    def __len__(self) -> int:
        return len(self.items)

    def relocate(self, where: str, device) -> None:
        """Move every element to ``where`` (``cuda`` or ``cpu``) on ``device``: the streams are captured in
        CPU memory while the whole model is on the GPU, and move there only once it has left."""
        if self.where == "disk" or where == "disk":
            raise ValueError("a disk stream stays on disk")
        self.where, self.device = where, torch.device(device)
        self.items = [self._store(x) for x in self.items]

    def close(self) -> None:
        self.items = []
        if self.where == "disk" and self.directory is not None:
            shutil.rmtree(self.directory, ignore_errors=True)


def stream_placement(r: dict, device, element_bytes: int, n_elements: int) -> str:
    """Where the streams live: ``refine.cache``, or for ``auto`` the GPU while they fit
    ``GPU_STREAM_BUDGET``, else CPU memory while they fit half the free RAM, else disk."""
    mode = r.get("cache", "auto")
    device = torch.device(device)
    if mode == "cuda" and device.type != "cuda":
        return "cpu"
    if mode != "auto":
        return mode
    total = element_bytes * n_elements
    if device.type == "cuda" and total <= GPU_STREAM_BUDGET:
        return "cuda"
    try:
        import psutil
        free_ram = psutil.virtual_memory().available
    except Exception:          # psutil is optional: assume the RAM is there
        return "cpu"
    return "cpu" if total <= free_ram // 2 else "disk"


# -- losses and statistics ---------------------------------------------------------------------------------

def token_rel_error(y, t):
    """Per-token ||y - t||^2 / max(||t||^2, 1e-6 * mean ||t||^2) in FP32, shape [batch, seq]."""
    y, t = y.float(), t.float()
    num = (y - t).pow(2).sum(-1)
    den = t.pow(2).sum(-1)
    floor = (REL_FLOOR * den.mean()).clamp_min(torch.finfo(torch.float32).tiny)
    return num / torch.maximum(den, floor)


def rel_loss(y, t):
    return token_rel_error(y, t).mean()


def module_counts(c0, s0, c, s) -> dict:
    """Code and scale changes of one tensor against its input (counts, summed over modules)."""
    c0, c = c0.to(torch.int8), c.to(torch.int8)
    pos = s0 > 0
    drift = (s[pos].double() / s0[pos].double()).log().abs() if bool(pos.any()) else torch.zeros(1).double()
    return {"weights": c.numel(), "flips": int((c != c0).sum()),
            "sign_flips": int((c.to(torch.int16) * c0.to(torch.int16) == -1).sum()),
            "zero_flips": int(((c == 0) != (c0 == 0)).sum()),
            "zeros_before": int((c0 == 0).sum()), "zeros_after": int((c == 0).sum()),
            "groups": int(pos.sum()), "drift_sum": float(drift.sum()), "drift_max": float(drift.max()),
            "scales_changed": int((s != s0).sum())}


def summarize(counts: list[dict]) -> dict:
    n = sum(c["weights"] for c in counts)
    groups = sum(c["groups"] for c in counts)
    return {"flips": sum(c["flips"] for c in counts) / n,
            "sign_flips": sum(c["sign_flips"] for c in counts) / n,
            "zero_flips": sum(c["zero_flips"] for c in counts) / n,
            "zeros_before": sum(c["zeros_before"] for c in counts) / n,
            "zeros_after": sum(c["zeros_after"] for c in counts) / n,
            "scale_drift_mean": sum(c["drift_sum"] for c in counts) / max(groups, 1),
            "scale_drift_max": max(c["drift_max"] for c in counts),
            "scales_changed": sum(c["scales_changed"] for c in counts) / max(groups, 1)}


def _finite(x):
    return float(x) if x is not None and math.isfinite(float(x)) else None


# -- layers ------------------------------------------------------------------------------------------------

def ternary_layer(layer, adapter, li: int, tensors: dict, signs: dict, block: int, h, device, latent: dict | None = None):
    """A copy of decoder layer ``li`` on ``device`` whose planned linears are TernaryLinear modules with
    ``tensors`` (keyed ``layers.{li}.{name}``), or LatentTernaryLinear modules with the ``latent`` options
    (``assign``, ``init``, ``zeros``). Every other tensor is the original's, frozen. The planned linears'
    weights are not copied (only the rotated weight is read when an option needs it)."""
    names = adapter.layer_plan(li).quantized
    linears = {n: layer.get_submodule(n) for n in names}
    memo = {id(m.weight): m.weight for m in linears.values()}    # shared, then replaced: never copied or moved
    new = copy.deepcopy(layer, memo)
    for n in names:
        t = tensors[f"layers.{li}.{n}"]
        lin = linears[n]
        width = int(t.shape[1])
        args = (t.codes.to(device), t.scales.to(device), signs[width].to(device), block, h)
        bias = None if lin.bias is None else lin.bias.detach().to(device)
        if latent is None:
            module = TernaryLinear(*args, bias=bias)
        else:
            w_rot = None
            if latent["init"] == "original" or latent["assign"] == "quantile":
                w_rot = rotate(lin.weight.detach().to(device=device, dtype=torch.float32), signs[width].to(device),
                               block)
            module = LatentTernaryLinear(*args, bias=bias, assign=latent["assign"], init=latent["init"],
                                         w_rot=w_rot, zeros=int(latent["zeros"]))
            del w_rot
        parent_name, _, attr = n.rpartition(".")
        setattr(new.get_submodule(parent_name) if parent_name else new, attr, module)
    new.to(device)       # the rest of the layer (norms, conv1d, kept linears); the ternary modules are there
    new.eval()
    return new


def latents_of(layers) -> list[tuple[str, LatentTernaryLinear]]:
    return [(n, m) for layer in layers for n, m in layer.named_modules() if isinstance(m, LatentTernaryLinear)]


def window_forward(layers, x, kwargs, ckpt: bool = False):
    for layer, kw in zip(layers, kwargs):
        if ckpt:
            x = grad_checkpoint(lambda h, _l=layer, _k=kw: layer_forward(_l, h, _k), x, use_reentrant=False)
        else:
            x = layer_forward(layer, x, kw)
    return x


@torch.no_grad()
def advance(stream: Stream, layers, kwargs) -> None:
    """Push every element of ``stream`` through ``layers``, in place."""
    for j in range(len(stream)):
        stream[j] = window_forward(layers, stream[j], kwargs)


@torch.no_grad()
def outputs(stream: Stream, layers, kwargs, where: str, device, directory=None) -> Stream:
    """A new stream: every element of ``stream`` through ``layers``."""
    out = Stream(where, device, directory)
    for j in range(len(stream)):
        out.append(window_forward(layers, stream[j], kwargs))
    return out


@torch.no_grad()
def val_errors(layers, xq: Stream, t_orig: Stream, t_quant: Stream, kwargs) -> dict | None:
    """Mean per-token relative error of ``layers`` on the held-out inputs against both targets (FP64 sums)."""
    if not len(xq):
        return None
    sums, n = {"original": 0.0, "quantized": 0.0}, 0
    for j in range(len(xq)):
        y = window_forward(layers, xq[j], kwargs)
        sums["original"] += float(token_rel_error(y, t_orig[j]).double().sum())
        sums["quantized"] += float(token_rel_error(y, t_quant[j]).double().sum())
        n += y.shape[0] * y.shape[1]
    return {k: v / n for k, v in sums.items()}


def objective(errors: dict | None, target: str, mix: float) -> float | None:
    if errors is None:
        return None
    if target == "original":
        return errors["original"]
    if target == "quantized":
        return errors["quantized"]
    return (1 - mix) * errors["original"] + mix * errors["quantized"]


def eps_damped(opt, latents, eps: float) -> tuple[float | None, float | None]:
    """(fraction of latent coordinates whose bias-corrected Adam sqrt(v) is below 10 * eps, where eps damps
    the update rather than the gradient scale setting it; the RMS of the latent gradients, sqrt(mean v),
    to compare with eps). (None, None) before any latent step."""
    damped = total = 0
    vsum = 0.0
    for _, m in latents:
        st = opt.state.get(m.latent)
        if not st or "exp_avg_sq" not in st:
            continue
        step = float(st["step"])
        vhat = st["exp_avg_sq"] / (1 - BETAS[1] ** step)
        damped += int((vhat.sqrt() < 10 * eps).sum())
        vsum += float(vhat.double().sum())
        total += vhat.numel()
    return (damped / total, math.sqrt(vsum / total)) if total else (None, None)


def anneal_frac(t: int, anneal_steps: int) -> float:
    """quantile: progress of the zero-count anneal at step ``t`` (0 at step 0, so step 0 is exact)."""
    if t <= 0:
        return 0.0
    return 1.0 if anneal_steps <= 0 else min(1.0, t / anneal_steps)


def train_window(students, xs: Stream, targets: list[tuple[Stream, float]], kwargs, r: dict, *,
                 order_seed: int) -> dict:
    """Train the latent codes and log multipliers of ``students`` on (input, target) elements; returns the
    training statistics. The codes freeze after ``freeze_codes_after`` of the steps."""
    latents = latents_of(students)
    n = len(xs)
    epochs = int(r["epochs"])
    total = epochs * n
    eps = float(r["eps"])
    freeze_step = int(round(float(r["freeze_codes_after"]) * total))
    anneal_steps = int(round(float(r["zero_anneal"]) * total))
    stats = {"steps": total, "code_steps": min(freeze_step, total), "train_loss_first": None,
             "train_loss_last": None, "eps_damped": None, "latent_grad_rms": None}
    if total == 0:
        return stats
    v_params = [m.latent for _, m in latents]
    u_params = [m.log_mult for _, m in latents]
    opt = torch.optim.AdamW([{"params": v_params, "lr": float(r["lr_codes"])},
                             {"params": u_params, "lr": float(r["lr_scales"])}],
                            betas=BETAS, weight_decay=0.0, eps=eps)
    warm = max(1, int(round(float(r["warmup"]) * total)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    ckpt = checkpointing(r)
    losses = []
    t = 0
    for epoch in range(epochs):
        order = torch.randperm(n, generator=torch.Generator().manual_seed(order_seed * 1_000_003 + epoch)).tolist()
        for j in order:
            if t == freeze_step:
                for _, m in latents:
                    m.set_anneal(anneal_frac(t, anneal_steps))
                stats["eps_damped"], stats["latent_grad_rms"] = eps_damped(opt, latents, eps)
                for _, m in latents:
                    m.freeze()
                    opt.state.pop(m.latent, None)
            elif t < freeze_step:
                for _, m in latents:
                    m.set_anneal(anneal_frac(t, anneal_steps))
            x = xs[j]
            with torch.enable_grad():
                y = window_forward(students, x, kwargs, ckpt)
                loss = sum(w * rel_loss(y, tgt[j]) for tgt, w in targets if w)
                loss.backward()
            value = float(loss.detach())
            if not math.isfinite(value):
                raise RuntimeError(f"refine: non-finite loss {value} at step {t}; lower refine.lr_codes/lr_scales")
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            for _, m in latents:
                if not m.frozen:
                    m.clamp_latent()
            losses.append(value)
            t += 1
    if freeze_step >= total:      # never frozen: the codes are the ones the latent gives at the end
        for _, m in latents:
            m.set_anneal(anneal_frac(total, anneal_steps))
        stats["eps_damped"], stats["latent_grad_rms"] = eps_damped(opt, latents, eps)
    stats["train_loss_first"] = sum(losses[:n]) / len(losses[:n])
    stats["train_loss_last"] = sum(losses[-n:]) / len(losses[-n:])
    return stats


# -- the stage ---------------------------------------------------------------------------------------------

@torch.no_grad()
def capture_inputs(model, adapter, ids, batch: int, device, sink) -> None:
    """``sink(hidden states entering layer 0)`` for every batch of ``ids``."""
    layers = adapter.layers(model)

    def hook(mod, args, kwargs):
        sink(args[0] if args else kwargs["hidden_states"])
        raise _Stop

    handle = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for b in range(0, len(ids), batch):
            try:
                model(input_ids=ids[b:b + batch].to(device), use_cache=False)
            except _Stop:
                pass
    finally:
        handle.remove()


def layer_tensors(state: TernaryState, li: int) -> dict:
    prefix = f"layers.{li}."
    return {k: t for k, t in state.tensors.items() if k.startswith(prefix)}


def _layer_file(layers_dir: Path, li: int) -> Path:
    return layers_dir / f"{li:04d}.safetensors"


def _peak_gb(device, reserved: bool = False):
    """Peak CUDA memory since the last reset, GiB: allocated tensors, or with ``reserved`` what the caching
    allocator holds from the driver (the figure that spills to system RAM on Windows)."""
    if torch.device(device).type != "cuda":
        return None
    return (torch.cuda.max_memory_reserved if reserved else torch.cuda.max_memory_allocated)(device) / 2**30


def _fmt(x, spec=".4g"):
    return "n/a" if x is None else format(x, spec)


def refine_model(model, adapter, train_ids, val_ids, base: TernaryState, *, settings: dict, emb=None,
                 layers_dir: Path | None = None, streams_dir: Path | None = None, device=None, log=print,
                 stop_after: int | None = None):
    """Refine the decoder layers in ``refine.layers`` of ``model`` (the original BF16 model) starting from
    ``base`` (quantize's state); ``emb``: the ternary embedding (TernaryTensor) when endpoints made one.

    Returns (state, report). With ``layers_dir`` every finished layer is saved and reused on rerun;
    ``stop_after`` (tests) raises KeyboardInterrupt once the window holding that layer is saved."""
    r = settings
    device = torch.device(device or next(model.parameters()).device)
    batch, window = int(r["batch"]), int(r["window"])
    target, mix = r["target"], float(r["mix"])
    n_layers = len(adapter.layers(model))
    start, stop = layer_range({"refine": r}, n_layers)
    if len(train_ids) % batch or len(val_ids) % batch or not len(train_ids):
        raise ValueError(f"refine: {len(train_ids)} training and {len(val_ids)} held-out blocks must be nonzero "
                         f"multiples of refine.batch ({batch})")
    signs, block, group = base.signs, base.block, base.group
    h = shared_hadamard(block, device)
    keep_fp_train = target in ("original", "mix")
    kwargs = capture_layer_kwargs(model, adapter, train_ids[:batch], device)
    hidden = adapter.hidden_size
    seq = int(train_ids.shape[1])
    dtype = next(adapter.layers(model).parameters()).dtype
    element_bytes = batch * seq * hidden * torch.empty((), dtype=dtype).element_size()
    n_train, n_val = len(train_ids) // batch, len(val_ids) // batch
    resident = n_train * (2 if target == "original" else 3 if target == "mix" else 2) + n_val * 3
    where = stream_placement(r, device, element_bytes, resident)
    sdir = Path(streams_dir) if streams_dir is not None else None
    if where == "disk" and sdir is None:
        raise ValueError("refine.cache disk needs a streams directory")

    def sub(name):
        return sdir / name if sdir is not None else None

    t_setup = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # GPU streams are captured in (unpinned) CPU memory and moved to the GPU after the model has left it:
    # the BF16 model (about 17 GiB at 9B) and up to GPU_STREAM_BUDGET of streams would not both fit.
    cap_where, cap_device = ("cpu", "cpu") if where == "cuda" else (where, device)
    xq = Stream(cap_where, cap_device, sub("xq"))
    xfp = Stream(cap_where, cap_device, sub("xfp")) if keep_fp_train else None
    vq, vfp = Stream(cap_where, cap_device, sub("vq")), Stream(cap_where, cap_device, sub("vfp"))
    if keep_fp_train:
        capture_inputs(model, adapter, train_ids, batch, device, xfp.append)
    capture_inputs(model, adapter, val_ids, batch, device, vfp.append)
    if emb is not None:      # the student sees the ternary embedding (the runtime arithmetic)
        adapter.set_embedding(model, TernaryEmbedding(emb.codes.to(device), emb.scales.to(device),
                                                      signs[hidden].to(device), block, h))
    capture_inputs(model, adapter, train_ids, batch, device, xq.append)
    capture_inputs(model, adapter, val_ids, batch, device, vq.append)
    layers = adapter.layers(model)
    setup_peak = (_peak_gb(device), _peak_gb(device, reserved=True))
    model.to("cpu")
    free(device)
    if where == "cuda":
        for s in (xq, xfp, vq, vfp):
            if s is not None:
                s.relocate("cuda", device)
    log(f"captured {n_train} training and {n_val} held-out elements of {(batch, seq, hidden)} "
        f"(streams: {where}, {resident * element_bytes / 2**30:.2f} GiB); layers [{start}, {stop}), window {window}, "
        f"target {target}, {time.time() - t_setup:.0f}s"
        + (f"; setup peak VRAM {setup_peak[0]:.2f} GiB (reserved {setup_peak[1]:.2f})" if setup_peak[0] is not None
           else ""))

    refined, report, rolled_back = {}, [], []
    li = 0
    while li < stop:
        if li < start:
            j = li + 1
        else:
            j = min(li + window, stop)
        span = list(range(li, j))
        orig = [layers[k].to(device) for k in span]
        kw = kwargs[li:j]
        files = [_layer_file(layers_dir, k) for k in span] if layers_dir is not None else []
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if li < start:
            tensors = {k: t for k_ in span for k, t in layer_tensors(base, k_).items()}
            mode = "input"
            if li == 0:
                log(f"layers 0..{start - 1} are outside refine.layers: the streams pass through quantize's codes")
        elif files and all(f.exists() for f in files):
            states = [load_state(f) for f in files]
            tensors = {k: t for st in states for k, t in st.tensors.items()}
            entry = dict(states[0].meta.get("report") or {})
            if entry.get("rollback"):
                rolled_back += span
            refined.update(tensors)
            report.append(entry)
            mode = "reused"
        else:
            mode = "trained"
        if mode != "trained":
            if xfp is not None:
                advance(xfp, orig, kw)
            advance(vfp, orig, kw)
            hard = [ternary_layer(layers[k], adapter, k, tensors, signs, block, h, device) for k in span]
            advance(xq, hard, kw)
            advance(vq, hard, kw)
            del hard
            if mode == "reused":
                log(f"layers {span} reused from {', '.join(f.name for f in files)}")
        else:
            inputs = {k: t for k_ in span for k, t in layer_tensors(base, k_).items()}
            # targets: the original layers on the original-prefix stream (in place: it becomes the next
            # layer's input) and/or on the student's stream
            t_quant = None
            if keep_fp_train:
                advance(xfp, orig, kw)
            if target in ("quantized", "mix"):
                t_quant = outputs(xq, orig, kw, where, device, sub("tq"))
            advance(vfp, orig, kw)
            vt_quant = outputs(vq, orig, kw, where, device, sub("vtq"))
            latent = {"assign": r["assign"], "init": r["init"], "zeros": r["zeros"]}
            students = [ternary_layer(layers[k], adapter, k, inputs, signs, block, h, device, latent=latent)
                        for k in span]
            before = val_errors(students, vq, vfp, vt_quant, kw)
            targets = [(xfp, 1.0)] if target == "original" else [(t_quant, 1.0)] if target == "quantized" \
                else [(xfp, 1.0 - mix), (t_quant, mix)]
            stats = train_window(students, xq, targets, kw, r, order_seed=int(r["seed"]) * 1009 + li)
            if t_quant is not None:
                t_quant.close()
            new = {}
            for idx, k in enumerate(span):
                for n, m in students[idx].named_modules():
                    if isinstance(m, LatentTernaryLinear):
                        codes, scales = m.hard()
                        new[f"layers.{k}.{n}"] = TernaryTensor(codes, scales)
            del students
            free(device)
            changed = any(not (torch.equal(new[k].codes, inputs[k].codes)
                               and torch.equal(new[k].scales, inputs[k].scales)) for k in new)
            hard = [ternary_layer(layers[k], adapter, k, new, signs, block, h, device) for k in span]
            after = val_errors(hard, vq, vfp, vt_quant, kw) if changed else before
            b_obj, a_obj = objective(before, target, mix), objective(after, target, mix)
            rollback = bool(r["rollback"] and changed and b_obj is not None and not a_obj < b_obj)
            counts = {k: module_counts(inputs[k].codes, inputs[k].scales, new[k].codes, new[k].scales) for k in new}
            if rollback:
                new = dict(inputs)
                hard = [ternary_layer(layers[k], adapter, k, new, signs, block, h, device) for k in span]
                rolled_back += span
            vt_quant.close()
            advance(xq, hard, kw)
            advance(vq, hard, kw)
            del hard
            entry = {"layers": span, "kinds": [adapter.layer_plan(k).kind for k in span],
                     **{k: (_finite(v) if isinstance(v, float) else v) for k, v in stats.items()},
                     "val_before": {k: _finite(v) for k, v in (before or {}).items()} or None,
                     "val_after": {k: _finite(v) for k, v in (after or {}).items()} or None,
                     "objective_before": _finite(b_obj), "objective_after": _finite(a_obj),
                     "rollback": rollback, "changed": changed, **summarize(list(counts.values())),
                     "modules": {k: summarize([c]) for k, c in counts.items()}}
            timing = {"seconds": time.time() - t0, "peak_vram_gb": _peak_gb(device),
                      "peak_reserved_gb": _peak_gb(device, reserved=True)}
            refined.update(new)
            report.append({**entry, **timing})
            for k, f in zip(span, files):
                lt = {key: t for key, t in new.items() if key.startswith(f"layers.{k}.")}
                save_state(f, TernaryState(lt, {w: signs[w] for w in {t.shape[1] for t in lt.values()}},
                                           block, group, {"stage": "refine", "report": {**entry, **timing}}))
            log(f"layers {span} ({'/'.join(entry['kinds'])}) {timing['seconds']:.1f}s: held-out rel err vs target "
                f"{_fmt(b_obj)} -> {_fmt(a_obj)} (original {_fmt((before or {}).get('original'))} -> "
                f"{_fmt((after or {}).get('original'))}){' ROLLED BACK' if rollback else ''}; train loss "
                f"{_fmt(entry['train_loss_first'])} -> {_fmt(entry['train_loss_last'])} over {stats['steps']} steps; "
                f"code flips {entry['flips']:.3%} (sign {entry['sign_flips']:.3%}, zero<->nonzero "
                f"{entry['zero_flips']:.3%}); zeros {entry['zeros_before']:.2%} -> {entry['zeros_after']:.2%}; "
                f"scale drift mean |log| {entry['scale_drift_mean']:.4f}, max {entry['scale_drift_max']:.4f}; "
                f"eps-damped {_fmt(entry['eps_damped'], '.2%')} (latent grad rms "
                f"{_fmt(entry['latent_grad_rms'], '.2e')}, eps {float(r['eps']):.0e})"
                + (f"; peak VRAM {timing['peak_vram_gb']:.2f} GiB (reserved {timing['peak_reserved_gb']:.2f})"
                   if timing["peak_vram_gb"] is not None else ""))
        for k in span:
            layers[k].to("cpu")
        free(device)
        li = j
        if stop_after is not None and li > stop_after and mode == "trained":
            raise KeyboardInterrupt(f"stopped after layer {li - 1}")
    for s in (xq, xfp, vq, vfp):
        if s is not None:
            s.close()
    tensors = {k: refined.get(k, t) for k, t in base.tensors.items()}
    meta = {"stage": "refine", "geometry": base.meta.get("geometry"), "layers": [start, stop],
            "settings": refine_hashed(r), "rolled_back": sorted(set(rolled_back))}
    state = TernaryState(tensors, dict(signs), block, group, meta).validate()
    return state, {"layers": [start, stop], "streams": where, "windows": report, "rolled_back": meta["rolled_back"]}


def run(ctx) -> dict:
    r = ctx.cfg["refine"]
    log = ctx.log("refine")
    adapter = ctx.adapter()
    calib = ctx.calib_tokens()
    check_config(ctx.cfg, adapter.n_layers, len(calib))
    out = ctx.run.stage_dir("refine")
    layers_dir = out / "layers"
    layers_dir.mkdir(exist_ok=True)
    streams_dir = out / "streams"
    shutil.rmtree(streams_dir, ignore_errors=True)
    base = ctx.quantize_state()
    emb = None
    if ctx.enabled("endpoints"):
        ep = ctx.endpoints_state()
        if ep is not None and "embed_tokens" in ep.tensors:
            emb = ep.tensors["embed_tokens"]
    labels = source_labels(ctx.run.stage_dir("calibrate", create=False), len(calib))
    train_idx, val_idx = select_blocks(labels, int(r["blocks"]), int(r["val_blocks"]), int(r["seed"]))
    shares = {}
    for i in train_idx:
        shares[labels[i]] = shares.get(labels[i], 0) + 1
    log(f"{len(train_idx)} training and {len(val_idx)} held-out calibration blocks (training blocks per source "
        f"{dict(sorted(shares.items()))}); embedding {'ternary' if emb is not None else 'high precision'}")
    free(ctx.device)
    cap = cap_vram(ctx.device, log)
    try:
        model, _ = ctx.load_model()
        t0 = time.time()
        state, report = refine_model(model, adapter, calib[train_idx], calib[val_idx], base, settings=r, emb=emb,
                                     layers_dir=layers_dir, streams_dir=streams_dir, device=ctx.device, log=log)
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(f"refine needs more than the {cap:.1f} GiB of VRAM that were free; lower refine.blocks, "
                           "set refine.cache=cpu, or use refine.window=1 (finished layers are kept)") from e
    finally:
        model = None
        free(ctx.device)
        if cap is not None:
            torch.cuda.set_per_process_memory_fraction(1.0, cuda_index(ctx.device))
    shutil.rmtree(streams_dir, ignore_errors=True)
    save_state(out / "state.safetensors", state)
    atomic_write_json(out / "report.json", {"seconds": time.time() - t0, "train_blocks": train_idx,
                                            "val_blocks": val_idx, **report})
    log(f"refined layers [{report['layers'][0]}, {report['layers'][1]}) in {time.time() - t0:.0f}s; "
        f"rolled back {report['rolled_back'] or 'none'}")
    return {"layers": report["layers"], "rolled_back": report["rolled_back"], "modules": len(state.tensors)}

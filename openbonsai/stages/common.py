"""Shared stage plumbing: config access, model loading, signs, state install."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import gc
from pathlib import Path

import numpy as np
import torch

from ..adapters import ModelAdapter, load_adapter
from ..config import (canonical_hash, distill_uses_endpoints, input_paths, stage_enabled, stage_inputs,
                      upstream)
from ..core.geometry import Geometry, check_signs, derive_geometry, make_signs
from ..core.ternary import TernaryEmbedding, TernaryLinear, dense_weight, shared_hadamard
from ..digest import DigestCache
from ..formats.state import (TernaryState, TernaryTensor, load_signs, load_state, read_scales, read_signs,
                             read_small, save_signs)
from ..rundir import RunDir, StageNotReady

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
VRAM_MARGIN = 3 * 2**29            # left free under the allocator cap (CUDA context growth, other processes)


@dataclass
class StageContext:
    cfg: dict
    run: RunDir
    _adapter: ModelAdapter | None = field(default=None, repr=False)
    _digests: DigestCache | None = field(default=None, repr=False)

    # -- config ----------------------------------------------------------
    @property
    def device(self) -> torch.device:
        dev = self.cfg["model"].get("device") or "auto"
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(dev)

    @property
    def dtype(self) -> torch.dtype:
        return _DTYPES[self.cfg["model"].get("dtype") or "bfloat16"]

    # -- stage identity ----------------------------------------------------
    def enabled(self, stage: str) -> bool:
        return stage_enabled(self.cfg, stage)

    def digests(self) -> DigestCache:
        if self._digests is None:
            self._digests = DigestCache(self.run.digests_path)
        return self._digests

    def model_digest(self) -> str | None:
        """Content digest of a local model directory (None for a hub id, which the revision pins)."""
        path = Path(self.model_path())
        return self.digests().tree(path) if path.is_dir() else None

    def file_digests(self, paths) -> dict:
        """SHA-256 of each file (of each folder's files, for a folder ``path``)."""
        return {p: self.digests().tree(p) if Path(p).is_dir() else self.digests().file(p) for p in paths}

    def stage_hash(self, stage: str) -> str:
        """Config sections, named file contents, model contents, upstream [config hash, output digest]."""
        ups = {u: self.run.done_key(u) for u in upstream(self.cfg, stage)}
        return canonical_hash({"inputs": stage_inputs(self.cfg, stage), "model_sha256": self.model_digest(),
                               "files": self.file_digests(input_paths(self.cfg, stage)), "upstream": ups})

    def stale_reason(self, stage: str) -> str | None:
        """Why ``stage``'s output cannot be used now (None when it and everything it reads are current)."""
        if not self.enabled(stage):
            return f"{stage} is disabled in the config"
        for u in upstream(self.cfg, stage):
            reason = self.stale_reason(u)
            if reason is not None:
                return reason
        info = self.run.done_info(stage)
        if info is None:
            return f"{stage} has not completed"
        if info.get("config_hash") != self.stage_hash(stage):
            return f"{stage} output is stale (its config, input files or an upstream stage changed since it ran)"
        return None

    def require_current(self, stage: str) -> None:
        reason = self.stale_reason(stage)
        if reason is not None:
            raise StageNotReady(f"{reason}; rerun it: openbonsai run --out {self.run.root}")

    def require_inputs(self, stage: str) -> None:
        """Every stage this one reads is enabled, complete and current."""
        for u in upstream(self.cfg, stage):
            if not self.enabled(u):
                raise StageNotReady(f"{stage} reads the {u} stage's output, but {u} is disabled; "
                                    f"enable it or give {stage} its own path: sources")
            self.require_current(u)

    def log(self, stage: str):
        return self.run.logger(stage)

    # -- model -------------------------------------------------------------
    def model_path(self) -> str:
        path = self.cfg["model"].get("path")
        if not path:
            raise ValueError("model.path is not set (use --model or the config's model.path)")
        return str(path)

    def adapter(self) -> ModelAdapter:
        if self._adapter is None:
            m = self.cfg["model"]
            self._adapter = load_adapter(self.model_path(), revision=m.get("revision"),
                                         local_files_only=bool(m.get("local_files_only")),
                                         name=m.get("adapter"))
        return self._adapter

    def load_model(self):
        m = self.cfg["model"]
        return self.adapter().load(self.model_path(), revision=m.get("revision"), device=self.device,
                                   dtype=self.dtype, attn=m.get("attn") or "sdpa",
                                   local_files_only=bool(m.get("local_files_only")),
                                   keep_vision=bool(m.get("keep_vision")))

    def model_dir(self, *, weights: bool = True) -> Path:
        """Local directory of the model's files (config, tokenizer, safetensors); resolves a hub id
        through the Hugging Face cache (downloads only when local_files_only is off). ``weights=False``
        fetches only the config, tokenizer and generation files (a GGUF supplies the weights)."""
        path = Path(self.model_path())
        if path.is_dir():
            return path
        from huggingface_hub import snapshot_download
        m = self.cfg["model"]
        patterns = ["*.json", "*.jinja", "*.txt"] + (["*.safetensors"] if weights else [])
        return Path(snapshot_download(self.model_path(), revision=m.get("revision"),
                                      local_files_only=bool(m.get("local_files_only")),
                                      allow_patterns=patterns))

    def load_tokenizer(self):
        m = self.cfg["model"]
        return self.adapter().load_tokenizer(self.model_path(), revision=m.get("revision"),
                                             local_files_only=bool(m.get("local_files_only")))

    # -- rotation ----------------------------------------------------------
    def geometry(self, model) -> Geometry:
        a = self.adapter()
        r = self.cfg["rotation"]
        return derive_geometry(a.quantized_widths(model), a.hidden_size,
                               group=int(r["group"]), max_block=int(r["max_block"]))

    def signs(self, model) -> tuple[dict, Geometry]:
        """Per-width signs for this run in run/signs.safetensors, made from ``rotation.signs``.

        The file records the sign spec and geometry it was made from; when
        either changes it is rewritten (the stages that use signs hash the same
        spec, so they rerun)."""
        geo = self.geometry(model)
        spec = self.cfg["rotation"]["signs"] or {}
        ident = {"mode": spec.get("mode", "seeded"), "seed": spec.get("seed", "width"),
                 "path_sha256": self.digests().file(spec["path"]) if spec.get("path") else None,
                 "geometry": geo.to_json()}
        if self.run.signs_path.exists():
            signs, block, meta = read_signs(self.run.signs_path)
            if meta.get("spec") == ident and block == geo.block:
                check_signs(signs, geo)
                return signs, geo
        if spec.get("path"):
            signs, _ = load_signs(spec["path"])
            signs = {w: signs[w] for w in geo.widths if w in signs}
        else:
            signs = make_signs(geo, ident["mode"], ident["seed"])
        check_signs(signs, geo)
        save_signs(self.run.signs_path, signs, block=geo.block, meta={"spec": ident})
        return signs, geo

    # -- stage outputs -----------------------------------------------------
    def calib_tokens(self) -> torch.Tensor:
        path = self.run.stage_dir("calibrate", create=False) / "tokens.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run the calibrate stage first")
        return torch.from_numpy(np.load(path).astype(np.int64))

    def quantize_state(self) -> TernaryState:
        return load_state(self.run.stage_dir("quantize", create=False) / "state.safetensors")

    def endpoints_state(self) -> TernaryState | None:
        path = self.run.stage_dir("endpoints", create=False) / "state.safetensors"
        return load_state(path) if path.exists() else None

    def refine_state(self) -> TernaryState:
        path = self.run.stage_dir("refine", create=False) / "state.safetensors"
        if not path.exists():
            raise StageNotReady(f"{path} is missing; run the refine stage")
        return load_state(path)

    def decoder_state(self) -> TernaryState:
        """The decoder codes and scales later stages build on: refine's state when refine is enabled,
        otherwise quantize's."""
        return self.refine_state() if self.enabled("refine") else self.quantize_state()

    def distill_base(self) -> dict:
        """Output digests of the states distillation trains on (recorded with its scales)."""
        base = {"quantize": (self.run.done_key("quantize") or [None, None])[1],
                "endpoints": (self.run.done_key("endpoints") or [None, None])[1]
                if distill_uses_endpoints(self.cfg) else None}
        if self.enabled("refine"):    # only then: scales distilled before refine existed stay valid
            base["refine"] = (self.run.done_key("refine") or [None, None])[1]
        return base

    def state_through(self, stage: str) -> TernaryState:
        """Decoder state, plus endpoints, plus distilled scales, up to ``stage``.

        ``refine`` and ``distill`` build on ``decoder_state()`` (refine's codes when refine is enabled).
        Refuses outputs that are missing, disabled or stale, so the codes and
        scales always come from one consistent pass of every stage."""
        if stage not in ("quantize", "endpoints", "refine", "distill"):
            raise ValueError(f"no ternary state for stage {stage!r}")
        self.require_current(stage)
        state = self.quantize_state() if stage in ("quantize", "endpoints") else self.decoder_state()
        # Distillation trains (and so evaluates) the endpoints only when it installed them.
        if stage == "endpoints" or (stage == "refine" and self.enabled("endpoints"))                 or (stage == "distill" and distill_uses_endpoints(self.cfg)):
            ep = self.endpoints_state()
            if ep is None:
                raise StageNotReady("endpoints/state.safetensors is missing")
            if ep.tensors:
                state = state.merged(ep)
        if stage == "distill":
            scales, meta = read_scales(self.run.stage_dir("distill", create=False) / "scales.safetensors")
            if meta.get("base") != self.distill_base():
                raise StageNotReady("distill/scales.safetensors was trained on a different quantize/endpoints"
                                    "/refine state than the current one; rerun distill")
            state = state.with_scales(scales)
        return state

    def distill_small(self) -> dict:
        """Trained high-precision tensors of the distill stage ({} unless distill.train.small), as the
        values the distilled model computed with (FP32 tensors). Call after ``state_through("distill")``."""
        return read_small(self.run.stage_dir("distill", create=False) / "scales.safetensors")


def resolve_module(adapter: ModelAdapter, model, key: str):
    """(parent, attribute, module) for a relative key."""
    if key == "lm_head":
        return None, "lm_head", adapter.head(model)
    if key == "embed_tokens":
        return None, "embed_tokens", adapter.embedding(model)
    head, li, sub = key.split(".", 2)
    if head != "layers":
        raise KeyError(f"unknown module key {key!r}")
    layer = adapter.layers(model)[int(li)]
    parent_name, _, attr = sub.rpartition(".")
    parent = layer.get_submodule(parent_name) if parent_name else layer
    return parent, attr, getattr(parent, attr)


@torch.no_grad()
def apply_dense(adapter: ModelAdapter, model, records) -> int:
    """Write (C * s) R^T into ordinary Linear/Embedding weights (``apply_state``)."""
    n = 0
    for key, tensor, signs, block in records:
        _, _, module = resolve_module(adapter, model, key)
        w = dense_weight(tensor.codes, tensor.scales, signs, block, device=module.weight.device,
                         dtype=module.weight.dtype)
        if w.shape != module.weight.shape:
            raise ValueError(f"{key}: state shape {tuple(w.shape)} != module {tuple(module.weight.shape)}")
        module.weight.copy_(w)
        n += 1
    return n


def has_ternary(model) -> bool:
    return any(isinstance(m, (TernaryLinear, TernaryEmbedding)) for m in model.modules())


@torch.no_grad()
def densify(model, dtype) -> int:
    """Replace every TernaryLinear/TernaryEmbedding in ``model`` by an ordinary Linear/Embedding
    holding its dequantized weight (C * s) R^T in ``dtype``: ``dense_weight``, the weights
    ``apply_dense`` writes (any trained scale multiplier included). Generation then runs plain
    GEMMs. One module at a time, so the ternary buffers are freed as the dense weights appear.
    Returns the number of modules replaced."""
    names = [n for n, m in model.named_modules() if isinstance(m, (TernaryLinear, TernaryEmbedding))]
    shared = {}
    for name in names:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        module = getattr(parent, attr)
        identity = id(module.log_mult)
        saved = shared.get(identity)
        weight = None if saved is None else saved[1]
        if weight is None:
            w = dense_weight(module.codes, module.scales(), module.signs, module.block, device=module.codes.device,
                             dtype=dtype)
            weight = torch.nn.Parameter(w, requires_grad=False)
            shared[identity] = (module.log_mult, weight)  # keep identity alive until all modules are replaced
        if isinstance(module, TernaryLinear):
            new = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None,
                                  device="meta", dtype=dtype)
            if module.bias is not None:
                new.bias = torch.nn.Parameter(module.bias.to(dtype), requires_grad=False)
        else:
            new = torch.nn.Embedding(module.num, module.dim, device="meta", dtype=dtype)
        new.weight = weight
        setattr(parent, attr, new)
        del module
    return len(names)


def install_ternary(adapter: ModelAdapter, model, records, device) -> list:
    """Replace modules with TernaryLinear/TernaryEmbedding (``install_ternary``/``install_endpoints``)."""
    hs, installed = {}, []
    dtype = next(model.parameters()).dtype
    dtype = torch.float16 if dtype == torch.float16 else torch.bfloat16
    for key, tensor, signs, block in records:
        if getattr(adapter, "tied", False) and key == "lm_head":
            raise ValueError("tied ternary endpoints must use one canonical embed_tokens record")
        if block not in hs:
            hs[block] = shared_hadamard(block, device, dtype=dtype)
        parent, attr, module = resolve_module(adapter, model, key)
        args = (tensor.codes.to(device), tensor.scales.to(device), signs.to(device), block, hs[block])
        if key == "embed_tokens":
            new = TernaryEmbedding(*args)
            adapter.set_embedding(model, new)
            if getattr(adapter, "tied", False):
                head = TernaryLinear(*args)
                head.codes, head.s0, head.signs = new.codes, new.s0, new.signs
                head.log_mult = new.log_mult
                adapter.set_head(model, head)
        else:
            bias = getattr(module, "bias", None)
            new = TernaryLinear(*args, bias=None if bias is None else bias.to(device))
            if key == "lm_head":
                adapter.set_head(model, new)
            else:
                setattr(parent, attr, new)
        del module
        installed.append((key, new))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return installed


@torch.no_grad()
def apply_small(adapter: ModelAdapter, model, values: dict, *, device=None, dtype=None,
                keep_fp32_vectors: bool = False) -> int:
    """Set high-precision tensors named relative to the text model (``norm.weight``,
    ``layers.0.linear_attn.A_log``), or the head's ``lm_head.weight``.

    A real parameter keeps its dtype and device and gets the values copied in.
    A meta parameter (a model built for a GGUF) is replaced by one in ``dtype``
    on ``device``; with ``keep_fp32_vectors`` 1-D tensors stay FP32."""
    lm = adapter.text_model(model)
    for key, value in values.items():
        parent_name, _, attr = key.rpartition(".")
        if parent_name == "lm_head":
            parent = adapter.head(model)
        else:
            parent = lm.get_submodule(parent_name) if parent_name else lm
        current = getattr(parent, attr, None)
        if not isinstance(current, torch.nn.Parameter):
            raise KeyError(f"{key}: no such parameter in the text model")
        if tuple(current.shape) != tuple(value.shape):
            raise ValueError(f"{key}: shape {tuple(value.shape)} != model {tuple(current.shape)}")
        if current.is_meta:
            want = torch.float32 if keep_fp32_vectors and value.ndim == 1 else (dtype or value.dtype)
            parent._parameters[attr] = torch.nn.Parameter(value.to(device=device, dtype=want), requires_grad=False)
        else:
            current.copy_(value.to(device=current.device, dtype=current.dtype))
    return len(values)


def legacy_records(legacy: dict):
    """Records from ``load_legacy_pt``: each module keeps its own signs and block."""
    for key in sorted(legacy):
        t = legacy[key]
        yield key, TernaryTensor(t.codes, t.scales), t.signs, t.block


def free(device=None) -> None:
    """Collect dropped models, then return cached CUDA memory (call after ``del model``)."""
    gc.collect()
    if getattr(device, "type", device) == "cuda" or (device is None and torch.cuda.is_initialized()):
        torch.cuda.empty_cache()


def cuda_index(device) -> torch.device:
    """``device`` with an explicit index: ``set_per_process_memory_fraction`` refuses a bare ``cuda``."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def cap_vram(device, log) -> float | None:
    """Cap this process's CUDA caching allocator at the memory free now (plus what it already holds) less
    ``VRAM_MARGIN``; returns the cap in GiB (None off CUDA). On Windows the driver pages an allocation that
    does not fit to system RAM instead of failing, and the stage then runs many times slower while it holds
    the GPU; with the cap it stops with out-of-memory instead. The cap also makes the allocator return the
    blocks it cached before it allocates past the cap: without one it never meets out-of-memory, so it never
    does, and a KV cache that grows a token per step leaves a trail of blocks too small for the next step
    (Qwen3-8B trace generation reserved 37 GB, 6.6 GB of it in system RAM, for weights and a KV cache of
    about 21 GB). ``set_per_process_memory_fraction(1.0)`` lifts it."""
    device = cuda_index(device)
    if device.type != "cuda":
        return None
    free_b, total_b = torch.cuda.mem_get_info(device)
    held = torch.cuda.memory_reserved(device)      # this process's own cache counts against the cap too
    fraction = min(1.0, max(0.05, (free_b + held - VRAM_MARGIN) / total_b))
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    cap = fraction * total_b / 2**30
    log(f"VRAM: {free_b / 2**30:.1f} of {total_b / 2**30:.1f} GiB free; this process is capped at {cap:.1f} GiB "
        "(an overflow stops the stage instead of spilling to system RAM)")
    return cap


@contextmanager
def vram_cap(device, log):
    """``cap_vram`` for the duration of the block, lifted after; yields the cap in GiB (None off CUDA)."""
    cap = cap_vram(device, log)
    try:
        yield cap
    finally:
        if cap is not None:
            torch.cuda.set_per_process_memory_fraction(1.0, cuda_index(device))

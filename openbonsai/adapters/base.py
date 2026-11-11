"""What a model architecture must provide to the framework.

The core only sees relative keys (``layers.{i}.{sub}``, ``lm_head``,
``embed_tokens``) and the per-layer plan. Rotation geometry (sign widths,
Hadamard block) is derived by the core from the planned linears
(openbonsai.core.geometry), never declared by the adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import ClassVar

import numpy as np
import torch
from torch import nn


# The date every chat template sees. Templates that print today's date (Llama 3.2's
# ``strftime_now("%d %b %Y")`` system header, Mistral Small's) would otherwise change their
# prompts, traces, calibration text and generation-cache keys every day.
CHAT_DATE = (2024, 7, 26)     # "26 Jul 2024", Llama 3.1's own default date_string


def _fixed_strftime_now(fmt: str) -> str:
    import datetime
    return datetime.datetime(*CHAT_DATE).strftime(fmt)


def template_vars(chat_kwargs: dict) -> dict:
    """Variables for one chat-template render: the adapter's ``chat_kwargs`` plus a fixed clock
    (``strftime_now``, which transformers binds to the real time, and ``date_string``, which Llama
    3.x templates read first). Templates that use neither ignore them."""
    import datetime
    return {"strftime_now": _fixed_strftime_now,
            "date_string": datetime.datetime(*CHAT_DATE).strftime("%d %b %Y"), **chat_kwargs}


@dataclass(frozen=True)
class LayerPlan:
    kind: str                                  # e.g. "linear_attention", "full_attention", "dense"
    groups: tuple[tuple[str, ...], ...]        # sequential GPTQ order; one group shares one input
    keep: tuple[str, ...] = ()                 # Linear submodules kept high precision

    @property
    def quantized(self) -> tuple[str, ...]:
        return tuple(n for g in self.groups for n in g)


class PlanError(ValueError):
    """A layer's Linear modules are not covered exactly once by its plan."""


# Optional CUDA-only kernel packages transformers binds into a model's module when it first imports it
# (use_kernel_func_from_hub_with_fallback); they cannot take CPU tensors.
CUDA_KERNEL_PACKAGES = ("fla", "causal_conv1d")
_TORCH_BOUND: set[str] = set()   # modeling modules imported here with those packages hidden


def _importable(package: str) -> bool:
    import importlib.util
    if package in sys.modules:
        return sys.modules[package] is not None
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False


def import_for_device(module: str, device, packages=CUDA_KERNEL_PACKAGES) -> None:
    """Import a transformers modeling ``module`` so it runs on ``device``.

    On the CPU the module is imported with the CUDA kernel packages hidden, so
    transformers binds its PyTorch paths (as the measured runs and the CPU tests
    use); on CUDA nothing changes. If the module was already imported with an
    installed kernel package, the CPU cannot run it: refuse with a clear message."""
    import importlib
    if torch.device(device).type != "cpu" or module in _TORCH_BOUND:
        return
    if module in sys.modules:
        present = [p for p in packages if _importable(p)]
        if present:
            raise RuntimeError(f"{module} was imported with {present} installed, so transformers bound their "
                               "CUDA kernels, which cannot run on the CPU. Use model.device: cuda, or run on the "
                               "CPU in a fresh process (openbonsai hides the kernels when it imports the model)")
        return
    missing = object()
    saved = {p: sys.modules.get(p, missing) for p in packages}
    for p in packages:
        sys.modules[p] = None   # import fails; transformers falls back to its PyTorch functions
    try:
        importlib.import_module(module)
        _TORCH_BOUND.add(module)
    finally:
        for p, value in saved.items():
            if value is missing:
                sys.modules.pop(p, None)
            else:
                sys.modules[p] = value


class ModelAdapter:
    name: ClassVar[str] = "base"
    model_types: ClassVar[tuple[str, ...]] = ()
    gguf_arch: ClassVar[str | None] = None     # Prism GGUF architecture; None = not exportable
    gguf_endpoints_optional: ClassVar[bool] = False  # the GGUF layout can hold high-precision head/embedding
    add_bos: ClassVar[bool] = False            # start every packed block / eval sequence with BOS
    zero_centered_norm: ClassVar[bool] = False  # norm is (1 + w) * x_hat instead of w * x_hat
    modeling_module: ClassVar[str | None] = None  # transformers module of the model (see import_for_device)
    think_end: ClassVar[str | None] = None     # text that closes the thinking part of a generation

    def __init__(self, config):
        self.config = config
        self.text_config = getattr(config, "text_config", None) or config

    # -- loading ---------------------------------------------------------
    def load(self, path, *, revision=None, device="cpu", dtype=torch.bfloat16, attn="sdpa",
             local_files_only=False, **options):
        """Return (model, tokenizer); the model is in eval mode with grads off."""
        raise NotImplementedError

    def prepare_device(self, device) -> None:
        """Call before building the model for ``device`` (keeps CUDA-only kernels off the CPU)."""
        if self.modeling_module:
            import_for_device(self.modeling_module, device)

    def load_tokenizer(self, path, *, revision=None, local_files_only=False):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path, revision=revision, local_files_only=local_files_only,
                                             trust_remote_code=False)

    # -- structure -------------------------------------------------------
    def text_model(self, model) -> nn.Module:
        """Module with .embed_tokens, .layers and .norm whose forward returns last_hidden_state."""
        raise NotImplementedError

    def head(self, model) -> nn.Module:
        return model.lm_head

    def set_head(self, model, module) -> None:
        model.lm_head = module

    def embedding(self, model) -> nn.Module:
        return self.text_model(model).embed_tokens

    def set_embedding(self, model, module) -> None:
        self.text_model(model).embed_tokens = module

    def layers(self, model) -> nn.ModuleList:
        return self.text_model(model).layers

    @property
    def n_layers(self) -> int:
        return int(self.text_config.num_hidden_layers)

    @property
    def hidden_size(self) -> int:
        return int(self.text_config.hidden_size)

    @property
    def vocab_size(self) -> int:
        return int(self.text_config.vocab_size)

    @property
    def tied(self) -> bool:
        return bool(getattr(self.config, "tie_word_embeddings", False)
                    or getattr(self.text_config, "tie_word_embeddings", False))

    def layer_plan(self, i: int) -> LayerPlan:
        raise NotImplementedError

    # -- numerics --------------------------------------------------------
    def embedding_gain(self, model) -> torch.Tensor:
        """FP32 gain of the first layer's entry norm: what an embedding row is
        multiplied by before layer 0 sees it. GPTQ for the embedding uses
        H = diag(gain^2)."""
        w = self.layers(model)[0].input_layernorm.weight.float()
        return 1.0 + w if self.zero_centered_norm else w

    # -- chat and generation ---------------------------------------------
    def chat_kwargs(self, *, thinking: bool) -> dict:
        return {}

    def eos_ids(self, tokenizer, model) -> list[int]:
        """Model generation-config EOS ids plus the tokenizer's EOS."""
        configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        values = list(configured) if isinstance(configured, (list, tuple)) else (
            [] if configured is None else [configured])
        tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
        if tokenizer_eos is not None:
            values.append(tokenizer_eos)
        return [int(v) for v in dict.fromkeys(values)]

    def bos_id(self, tokenizer) -> int | None:
        if not self.add_bos:
            return None
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is None:
            raise ValueError(f"{self.name} adds BOS but the tokenizer defines none")
        return int(bos)

    # -- plan helpers (enforced by the core) -----------------------------
    def planned_linears(self, model, i: int) -> dict[str, nn.Linear]:
        """Validate layer i's plan against its actual Linear modules."""
        plan = self.layer_plan(i)
        layer = self.layers(model)[i]
        linears = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
        planned = list(plan.quantized) + list(plan.keep)
        dup = {n for n in planned if planned.count(n) > 1}
        missing = [n for n in planned if n not in linears]
        uncovered = [n for n in linears if n not in planned]
        if dup or missing or uncovered:
            raise PlanError(f"layer {i} ({plan.kind}): duplicated {sorted(dup)}, missing {missing}, "
                            f"unplanned linears {uncovered}")
        return {n: linears[n] for n in plan.quantized}

    def quantized_widths(self, model) -> set[int]:
        widths = set()
        for i in range(self.n_layers):
            widths.update(m.in_features for m in self.planned_linears(model, i).values())
        return widths

    # -- checkpoint names --------------------------------------------------
    def checkpoint_name(self, key: str) -> str:
        """Checkpoint tensor name of a parameter named relative to the text model
        (``layers.0.input_layernorm.weight``, ``norm.weight``) or ``lm_head.weight``."""
        raise NotImplementedError

    # -- Prism PTQ1_0 GGUF -------------------------------------------------
    # An exportable adapter sets ``gguf_arch`` and describes its file layout;
    # writing, reading and verifying are architecture-agnostic
    # (openbonsai.stages.export, openbonsai.formats.gguf).
    def gguf_unsupported(self, model_dir=None) -> str | None:
        """Why this config cannot be written as a Prism PTQ1_0 GGUF (None: it can). With
        ``model_dir`` the tokenizer files are checked too. The export then writes a latent
        checkpoint instead (``export.format: auto``) or refuses (``ptq1_0-gguf``)."""
        return None if self.gguf_arch else f"the {self.name} adapter has no PTQ1_0 GGUF layout"

    def gguf_tensors(self) -> list:
        """The file's tensors in file order (``formats.gguf.TensorMap``), derived from the config."""
        raise NotImplementedError(f"{self.name}: no PTQ1_0 GGUF layout")

    def gguf_plan(self, state_keys) -> list:
        """The layout for a state holding ``state_keys`` (an adapter whose layout can keep the head or
        embedding in high precision picks those from the state; the default ignores it)."""
        return self.gguf_tensors()

    def gguf_plan_for_file(self, reader) -> list:
        """The layout a file to be read was written with (default: the config's one layout)."""
        return self.gguf_tensors()

    def gguf_kvs(self, model_dir, *, name: str, signs: dict, block: int, tensors: list,
                 size_label: str | None = None) -> tuple[list[tuple[str, str, object]], list[str]]:
        """(every KV in file order as (key, type, value), notes); ``signs``: int8 vector per width;
        ``model_dir`` holds the tokenizer and config files."""
        raise NotImplementedError(f"{self.name}: no PTQ1_0 GGUF layout")

    def gguf_reference(self) -> dict | None:
        """A published file's header dump when this config is that model's (the export then asserts
        an identical tensor layout and the runtime KVs)."""
        return None

    def gguf_source_rows(self, name: str, n: int) -> np.ndarray | None:
        """Export verification: the source row (dim 0) each of the ``n`` rows of GGUF tensor ``name``
        holds, stated independently of ``gguf_tensors()`` (from the tensor's name, by arithmetic), so a
        wrong permutation in the layout fails verification. None: no independent rule (the verifier
        then can only check the codec and I/O against the layout's own permutation)."""
        return None

    def gguf_f32_op(self, name: str) -> str | None:
        """Export verification: the F32 transform of GGUF tensor ``name`` (``formats.gguf.F32_OPS``),
        stated independently of ``gguf_tensors()``; None: no independent rule."""
        return None

    def gguf_check(self, kvd: dict) -> None:
        """Refuse a GGUF whose metadata this adapter cannot map back (reading)."""

    def empty_model(self, *, dtype=torch.bfloat16, attn: str = "sdpa"):
        """The native text model with parameters on the meta device (buffers real), for ``from_gguf``."""
        raise NotImplementedError(f"{self.name}: cannot build a model from a GGUF")

    def read_gguf(self, path, *, log=None):
        """(TernaryState, {text-model tensor name: FP32 tensor}, kv dict) in this adapter's conventions.

        The reusable loader: rows and heads are permuted back, F32 transforms
        undone, signs and block taken from ``prism.hadamard.*``. The state can
        seed distillation (start from a published file) or be installed into
        a model (``from_gguf``)."""
        from ..formats.gguf import GGUFError, GGUFReader, f32_inverse, prism_signs
        from ..formats.ptq1_0 import QK
        from ..formats.state import TernaryState, TernaryTensor
        with GGUFReader(path) as r:
            arch = r.value("general.architecture")
            if arch != self.gguf_arch:
                raise GGUFError(f"{path}: architecture {arch!r}, this adapter reads {self.gguf_arch!r}")
            self.gguf_check(r.kvd)
            plan = self.gguf_plan_for_file(r)
            got = {t.name: (t.gtype, tuple(t.ne)) for t in r.tensors}
            want = {t.name: (t.gtype, tuple(t.ne)) for t in plan}
            if got != want:
                diff = sorted(set(got.items()) ^ set(want.items()))[:4]
                raise GGUFError(f"{path}: tensors do not match this model's config "
                                f"({len(got)} in the file, {len(want)} expected; e.g. {diff})")
            block = int(r.value("prism.hadamard.block_size"))
            signs = {w: torch.from_numpy(s.copy()) for w, s in prism_signs(r).items()}
            tensors, floats = {}, {}
            for t in plan:
                inv = t.inverse_perm()
                if t.ternary:
                    codes, scales = r.ptq1_0(t.name)
                    if inv is not None:
                        codes, scales = codes[inv], scales[inv]
                    tensors[t.source] = TernaryTensor(torch.from_numpy(np.ascontiguousarray(codes)),
                                                      torch.from_numpy(np.ascontiguousarray(scales)))
                elif t.data is None:   # generated constants (rope_freqs) have no model tensor
                    a = r.array(t.name)
                    if inv is not None:
                        a = a[inv]
                    floats[t.source] = f32_inverse(t.op, torch.from_numpy(np.ascontiguousarray(a)), t.shape)
            kv = {k: v for k, (_, v) in r.kvd.items() if not k.startswith("tokenizer.ggml.")}
        used = {t.shape[1] for t in tensors.values()}
        missing = sorted(used - set(signs))
        if missing:
            raise GGUFError(f"{path}: no prism.hadamard sign vector for widths {missing}")
        state = TernaryState(tensors, {w: s for w, s in signs.items() if w in used}, block, QK,
                             {"source": "gguf", "path": str(path), "general.name": kv.get("general.name")})
        if log is not None:
            log(f"read {path}: {len(tensors)} PTQ1_0 tensors, {len(floats)} F32 tensors, block {block}")
        return state.validate(), floats, kv

    def from_gguf(self, path, *, model_dir=None, dtype=torch.bfloat16, device="cpu", attn: str = "sdpa",
                  small_fp32: bool = False, log=None):
        """A native model whose every weight comes from a Prism PTQ1_0 GGUF.

        Ternary tensors become TernaryLinear/TernaryEmbedding (codes stay in
        the rotated basis, exactly as ``eval --from distill`` installs them);
        F32 tensors go back to the checkpoint conventions in ``dtype``, or in
        FP32 for 1-D tensors (norms, A_log, dt_bias) with ``small_fp32``, as
        Prism's runtime keeps them. ``model_dir`` supplies generation_config.json."""
        from ..stages.common import apply_small, install_ternary
        from ..core.ternary import TernaryEmbedding
        device = torch.device(device)
        state, floats, _ = self.read_gguf(path, log=log)
        self.prepare_device(device)
        model = self.empty_model(dtype=dtype, attn=attn)
        # floats first: a Linear's bias (F32 in the file) must be real before TernaryLinear takes it over
        apply_small(self, model, floats, device=device, dtype=dtype, keep_fp32_vectors=small_fp32)
        install_ternary(self, model, state.records(), device)
        if self.tied and not isinstance(self.embedding(model), TernaryEmbedding):
            model.tie_weights()   # the head shares the table the file supplied for the embedding
        meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
        if meta:
            raise ValueError(f"{path}: the GGUF leaves {len(meta)} tensors unset, e.g. {meta[:4]}")
        model.to(device)
        if model_dir is not None and (Path(model_dir) / "generation_config.json").exists():
            from transformers import GenerationConfig
            model.generation_config = GenerationConfig.from_pretrained(model_dir)
        model.eval()
        model.requires_grad_(False)
        self.text_model(model).config.use_cache = False
        return model

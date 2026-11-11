"""Qwen3.5 dense hybrid (Gated DeltaNet + full attention), text path.

Supports ``Qwen3_5ForConditionalGeneration`` checkpoints (``qwen3_5``; the
vision tower is dropped after loading unless ``keep_vision``) and text-only
``Qwen3_5ForCausalLM`` (``qwen3_5_text``). MTP weights are never loaded.

- Norms are zero-centered: (1 + w) * x_hat (the gated ``linear_attn.norm`` is
  plain w * x_hat but never touches an embedding row).
- GatedDeltaNet ``in_proj_a``/``in_proj_b`` stay high precision (Prism keeps
  them F32 as ``ssm_alpha``/``ssm_beta``); conv1d, A_log, dt_bias and norms
  are not Linear and are untouched.
- Sequential GPTQ order matches OpenBonsai research/experiments/gptq_ternary.py.
- Prism's ``qwen35`` PTQ1_0 GGUF layout: adapters/qwen3_5_gguf.py.
"""
from __future__ import annotations

import torch

from .base import LayerPlan, ModelAdapter
from .qwen3_5_gguf import Qwen35GGUF

_ORDER = {
    "linear_attention": (("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"), ("linear_attn.out_proj",),
                         ("mlp.gate_proj", "mlp.up_proj"), ("mlp.down_proj",)),
    "full_attention": (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), ("self_attn.o_proj",),
                       ("mlp.gate_proj", "mlp.up_proj"), ("mlp.down_proj",)),
}
_KEEP = {"linear_attention": ("linear_attn.in_proj_a", "linear_attn.in_proj_b"), "full_attention": ()}


class Qwen35Adapter(Qwen35GGUF, ModelAdapter):
    name = "qwen3_5"
    model_types = ("qwen3_5", "qwen3_5_text")
    gguf_arch = "qwen35"
    add_bos = False
    zero_centered_norm = True
    modeling_module = "transformers.models.qwen3_5.modeling_qwen3_5"
    think_end = "</think>"

    @property
    def gguf_endpoints_optional(self):
        return self.tied  # ordinary shared float table or an explicitly opted-in version-2 table

    def __init__(self, config):
        super().__init__(config)
        if config.model_type not in self.model_types:
            raise ValueError(f"Qwen35Adapter cannot handle model_type {config.model_type!r}")
        if getattr(config, "quantization_config", None) is not None:
            raise ValueError("start from unquantized weights (config has quantization_config)")
        types = list(self.text_config.layer_types)
        unknown = sorted(set(types) - set(_ORDER))
        if unknown or len(types) != self.n_layers:
            raise ValueError(f"unsupported Qwen3.5 layer types {unknown or types}")

    def load(self, path, *, revision=None, device="cpu", dtype=torch.bfloat16, attn="sdpa",
             local_files_only=False, keep_vision=False, **options):
        self.prepare_device(device)   # before transformers imports the model's module
        from transformers import Qwen3_5ForConditionalGeneration
        cls = self._model_class()
        model = cls.from_pretrained(path, revision=revision, dtype=dtype, device_map=str(device),
                                    attn_implementation=attn, local_files_only=local_files_only,
                                    trust_remote_code=False)
        if cls is Qwen3_5ForConditionalGeneration and not keep_vision:
            model.model.visual = None   # text-only path never reaches it
        model.eval()
        model.requires_grad_(False)
        self.text_model(model).config.use_cache = False
        tokenizer = self.load_tokenizer(path, revision=revision, local_files_only=local_files_only)
        return model, tokenizer

    def _model_class(self):
        from transformers import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
        return Qwen3_5ForConditionalGeneration if self.config.model_type == "qwen3_5" else Qwen3_5ForCausalLM

    def empty_model(self, *, dtype=torch.bfloat16, attn="sdpa"):
        from accelerate import init_empty_weights
        cls = self._model_class()
        with init_empty_weights(include_buffers=False):
            model = cls._from_config(self.config, dtype=dtype, attn_implementation=attn)
        if self.config.model_type == "qwen3_5":
            model.model.visual = None   # the GGUF is text only
        return model

    def text_model(self, model):
        inner = model.model
        return inner.language_model if hasattr(inner, "language_model") else inner

    def checkpoint_name(self, key: str) -> str:
        if key == "lm_head.weight":
            return key
        return ("model.language_model." if self.config.model_type == "qwen3_5" else "model.") + key

    def layer_plan(self, i: int) -> LayerPlan:
        kind = self.text_config.layer_types[i]
        return LayerPlan(kind=kind, groups=_ORDER[kind], keep=_KEEP[kind])

    def chat_kwargs(self, *, thinking: bool) -> dict:
        return {"enable_thinking": bool(thinking)}

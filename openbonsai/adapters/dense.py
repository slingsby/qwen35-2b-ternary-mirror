"""Dense decoder-only transformers behind one adapter and a small family table.

Covers Llama (and Llama-architecture checkpoints such as Nanbeige4-style
models published as ``LlamaForCausalLM``), Mistral, Qwen2/Qwen2.5 and Qwen3:
every layer is pre-norm attention + SwiGLU MLP with the same seven Linear
modules, so one layer plan fits all of them:

    [q_proj, k_proj, v_proj] -> [o_proj] -> [gate_proj, up_proj] -> [down_proj]

Nothing is kept high precision inside a layer (norms and Qwen3's q/k norms are
not Linear); Linear biases (Qwen2 q/k/v, Llama ``attention_bias`` /
``mlp_bias``) stay in their checkpoint dtype inside TernaryLinear and are
written as F32 in a GGUF.

A family row says what differs between model types (``FAMILIES``):

- the llama.cpp architecture Prism's converter writes and whether Prism's
  loader accepts ``prism.hadamard`` metadata for it (llama-model.cpp verifies
  llama, qwen3, qwen3moe, qwen35, qwen35moe, qwen3next and dspark only; e.g.
  ``qwen2`` and ``nanbeige`` are refused), so whether a PTQ1_0 GGUF can be
  written at all; the export otherwise writes a latent checkpoint;
- BOS on every block, the chat template's thinking switch, the norm
  convention (all four use w * x_hat), the q/k row permutation llama.cpp's
  ``llama`` rope needs, the pre-tokenizer name and which biases have a slot.

Tied embedding/head (Qwen3 0.6B-4B, Llama 3.2 1B/3B): the shared table stays
in high precision and gets no GPTQ (``endpoints`` ``auto``); ternary for a
tied table (one joint GPTQ) is refused. In a GGUF the table is
``token_embd.weight`` in the checkpoint's float type with no ``output.weight``
(llama.cpp then reuses the embedding), outside ``prism.hadamard``.

A new dense model type is one row here plus a tiny test config
(tests/test_dense.py); a new architecture is its own adapter
(docs/adding-a-model.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..core.geometry import block_note
from ..formats.gguf import GGML_BF16, GGML_F16, GGML_F32, GGML_PTQ1_0, GGUFError, TensorMap
from ..formats.ptq1_0 import LLAMA_FTYPE_MOSTLY_PTQ1_0
from . import gguf_common as G
from .base import LayerPlan, ModelAdapter

PLAN = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), ("self_attn.o_proj",),
        ("mlp.gate_proj", "mlp.up_proj"), ("mlp.down_proj",))
ATTN = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
MLP = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
_GGUF_NAME = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
              "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
              "mlp.down_proj": "ffn_down"}
ENDPOINTS = ("lm_head", "embed_tokens")

# Architectures whose prism.hadamard metadata Prism's loader accepts (llama-model.cpp, the switch after
# "the activation-side transform is applied only by build_lora_mm"; Prism c9c34bc).
PRISM_HADAMARD_ARCHS = ("llama", "qwen3", "qwen3moe", "qwen35", "qwen35moe", "qwen3next", "dspark")

# Pre-tokenizer probe hashes (gguf_common.CHKTXT) Prism's converter maps to a name (get_vocab_base_pre).
KNOWN_PRE = {
    "0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5": "llama-bpe",   # meta-llama/Meta-Llama-3-8B
    "e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea": "qwen2",       # Qwen/Qwen1.5-7B
    "d4540891389ea895b53b399da6ac824becc30f2fba0e9ddbb98f92e55ca0e97c": "qwen2",       # Qwen/Qwen3-Embedding-0.6B
    "63b97e4253352e6f357cc59ea5b583e3a680eaeaf2632188c2b952de2588485e": "tekken",      # mistralai/Mistral-Nemo-Base-2407
    "855059429035d75a914d1eda9f10a876752e281a054a7a3d421ef0533e5b6249": "smollm",      # HuggingFaceTB/SmolLM-135M
}
ROPE_TYPES = ("default", "linear", "llama3")   # rope scalings the GGUF export writes (others: latent export)


@dataclass(frozen=True)
class Family:
    model_type: str
    gguf_arch: str                 # llama.cpp architecture Prism's converter writes for this model type
    add_bos: bool                  # every packed block / eval sequence starts with BOS
    thinking: bool                 # the chat template has an ``enable_thinking`` switch
    rope_permute: bool             # q/k rows permuted (HF rotate-half -> llama.cpp's interleaved NORM rope)
    pre: str | None                # tokenizer.ggml.pre when the tokenizer's probe hash is not in KNOWN_PRE
    zero_centered_norm: bool = False   # RMSNorm (1 + w) * x_hat; False: w * x_hat
    qk_norm: bool = False          # per-head q_norm / k_norm (Qwen3)
    biases: tuple[tuple[str | None, tuple[str, ...]], ...] = ()   # (config flag or None = always, modules)
    gguf_bias_slots: tuple[str, ...] = ATTN[:3] + ("self_attn.o_proj",) + MLP   # biases the arch's loader has
    sliding_ok: bool = False       # the GGUF arch implements sliding-window layers
    notes: tuple[str, ...] = field(default=())

    @property
    def prism_hadamard(self) -> bool:
        return self.gguf_arch in PRISM_HADAMARD_ARCHS


FAMILIES: dict[str, Family] = {f.model_type: f for f in (
    Family("llama", "llama", add_bos=True, thinking=False, rope_permute=True, pre="llama-bpe",
           biases=(("attention_bias", ATTN), ("mlp_bias", MLP)),
           notes=("Llama 2/3.x and Llama-architecture checkpoints (e.g. Nanbeige4-style models published as "
                  "LlamaForCausalLM); Llama 3.2 1B/3B tie the embedding and head",
                  "NanbeigeForCausalLM (looped Nanbeige4.2) is Prism's 'nanbeige' arch: not hadamard-verified "
                  "and needs remote code, so it is not covered")),
    Family("mistral", "llama", add_bos=True, thinking=False, rope_permute=True, pre=None,
           notes=("Prism's converter writes MistralForCausalLM as 'llama', which has no sliding window: the GGUF "
                  "matches the model only within config.sliding_window tokens",
                  "Mistral 7B v0.x tokenizers are SentencePiece: the GGUF export needs a byte-level BPE "
                  "tokenizer.json, so those export a latent checkpoint")),
    Family("qwen2", "qwen2", add_bos=False, thinking=False, rope_permute=False, pre="qwen2",
           biases=((None, ATTN[:3]),), gguf_bias_slots=ATTN[:3],
           notes=("Qwen2 / Qwen2.5: Prism refuses prism.hadamard for 'qwen2', so the export writes a latent "
                  "checkpoint",)),
    Family("qwen3", "qwen3", add_bos=False, thinking=True, rope_permute=False, pre="qwen2", qk_norm=True,
           biases=(("attention_bias", ATTN),), gguf_bias_slots=ATTN[:3],
           notes=("Qwen3 0.6B/1.7B/4B tie the embedding and head",)),
)}


def llama_rope_perm(rows: int, n_head: int) -> np.ndarray:
    """Prism's LlamaModel.permute as a row index: gguf[i] = hf[perm[i]] (per head, (2, d/2) -> (d/2, 2))."""
    if rows % n_head or (rows // n_head) % 2:
        raise GGUFError(f"{rows} rows do not split into {n_head} heads of even size")
    hd = rows // n_head
    return np.arange(rows).reshape(n_head, 2, hd // 2).swapaxes(1, 2).reshape(-1)


def llama3_rope_factors(rp: dict, dim: int) -> np.ndarray:
    """``rope_freqs.weight`` exactly as Prism's converter generates it (LlamaModel.generate_extra_tensors)."""
    base = rp.get("rope_theta", 10000.0)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    factor = rp.get("factor", 8.0)
    low_freq_factor = rp.get("low_freq_factor", 1.0)
    high_freq_factor = rp.get("high_freq_factor", 4.0)
    old_context_len = rp.get("original_max_position_embeddings", 8192)
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    out = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            out.append(1)
        elif wavelen > low_freq_wavelen:
            out.append(factor)
        else:
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            out.append(1 / ((1 - smooth) / factor + smooth))
    return torch.tensor(out, dtype=torch.float32).numpy()


def byte_level_problem(model_dir) -> str | None:
    """Why the tokenizer cannot be written as GGUF ``gpt2`` (byte-level BPE), or None."""
    root = Path(model_dir)
    path = root / "tokenizer.json"
    if not path.is_file():
        return "no tokenizer.json (the GGUF export writes byte-level BPE vocabularies only)"
    tj = json.loads(path.read_text(encoding="utf-8"))
    model = tj.get("model") or {}
    text = json.dumps([tj.get("pre_tokenizer"), tj.get("decoder")])
    if model.get("type") != "BPE" or model.get("byte_fallback") or "ByteLevel" not in text:
        return ("the tokenizer is not byte-level BPE (SentencePiece-style vocabularies need the llama/SPM "
                "GGUF vocabulary, which the export does not write)")
    return None


class DenseAdapter(ModelAdapter):
    name = "dense"
    model_types = tuple(FAMILIES)
    gguf_endpoints_optional = True    # the layout holds high-precision head/embedding tables too

    def __init__(self, config):
        super().__init__(config)
        if config.model_type not in FAMILIES:
            raise ValueError(f"DenseAdapter cannot handle model_type {config.model_type!r}; "
                             f"known: {sorted(FAMILIES)}")
        if getattr(config, "quantization_config", None) is not None:
            raise ValueError("start from unquantized weights (config has quantization_config)")
        self.family = FAMILIES[config.model_type]
        self.add_bos = self.family.add_bos
        self.zero_centered_norm = self.family.zero_centered_norm
        self.think_end = "</think>" if self.family.thinking else None
        self.gguf_arch = self.family.gguf_arch if self.family.prism_hadamard else None
        types = list(getattr(self.text_config, "layer_types", None) or ["full_attention"] * self.n_layers)
        unknown = sorted(set(types) - {"full_attention", "sliding_attention"})
        if unknown or len(types) != self.n_layers:
            raise ValueError(f"unsupported layer types {unknown or types}")
        self.layer_types = types

    # -- loading -----------------------------------------------------------
    def load(self, path, *, revision=None, device="cpu", dtype=torch.bfloat16, attn="sdpa",
             local_files_only=False, **options):
        from transformers import AutoModelForCausalLM
        self.prepare_device(device)
        model = AutoModelForCausalLM.from_pretrained(path, revision=revision, dtype=dtype, device_map=str(device),
                                                     attn_implementation=attn, local_files_only=local_files_only,
                                                     trust_remote_code=False)
        model.eval()
        model.requires_grad_(False)
        self.text_model(model).config.use_cache = False
        tokenizer = self.load_tokenizer(path, revision=revision, local_files_only=local_files_only)
        return model, tokenizer

    def empty_model(self, *, dtype=torch.bfloat16, attn="sdpa"):
        from accelerate import init_empty_weights
        from transformers import AutoModelForCausalLM
        with init_empty_weights(include_buffers=False):
            return AutoModelForCausalLM.from_config(self.config, dtype=dtype, attn_implementation=attn)

    # -- structure -----------------------------------------------------------
    def text_model(self, model):
        return model.model

    def checkpoint_name(self, key: str) -> str:
        return key if key.startswith("lm_head.") else "model." + key

    def layer_plan(self, i: int) -> LayerPlan:
        return LayerPlan(kind=self.layer_types[i], groups=PLAN, keep=())

    def bias_modules(self) -> tuple[str, ...]:
        """Linear submodules with a bias under this config (the family's rule)."""
        out = []
        for flag, mods in self.family.biases:
            if flag is None or getattr(self.text_config, flag, False):
                out += [m for m in mods if m not in out]
        return tuple(m for m in ATTN + MLP if m in out)

    def chat_kwargs(self, *, thinking: bool) -> dict:
        if self.family.thinking:
            return {"enable_thinking": bool(thinking)}
        if thinking:
            raise ValueError(f"{self.config.model_type}: the chat template has no thinking switch; "
                             "use thinking off (traces.thinking, eval.reasoning.thinking)")
        return {}

    # -- Prism PTQ1_0 GGUF -----------------------------------------------------
    def _rope(self) -> dict:
        tc = self.text_config
        rp = dict(getattr(tc, "rope_parameters", None) or getattr(tc, "rope_scaling", None) or {})
        rp.setdefault("rope_theta", getattr(tc, "rope_theta", 10000.0))
        return rp

    @property
    def head_dim(self) -> int:
        tc = self.text_config
        return int(getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads)

    def gguf_unsupported(self, model_dir=None) -> str | None:
        f, tc = self.family, self.text_config
        problems = []
        if not f.prism_hadamard:
            problems.append(f"Prism's loader refuses prism.hadamard metadata for the {f.gguf_arch!r} architecture "
                            f"({self.config.model_type}); it accepts {', '.join(PRISM_HADAMARD_ARCHS)}")
        rope_type = self._rope().get("rope_type", "default")
        if rope_type not in ROPE_TYPES:
            problems.append(f"rope_type {rope_type!r} (the export writes {', '.join(ROPE_TYPES)})")
        if float(getattr(tc, "partial_rotary_factor", None) or self._rope().get("partial_rotary_factor") or 1.0) != 1:
            problems.append("partial rotary embeddings")
        if "sliding_attention" in self.layer_types and not f.sliding_ok:
            problems.append(f"sliding-window layers: the {f.gguf_arch!r} GGUF arch has none")
        homeless = [m for m in self.bias_modules() if m not in f.gguf_bias_slots]
        if homeless:
            problems.append(f"biases on {homeless} have no slot in the {f.gguf_arch!r} GGUF loader")
        if model_dir is not None:
            tok = byte_level_problem(model_dir)
            if tok:
                problems.append(tok)
            elif f.pre is None:
                chk = G.gpt2_vocab(model_dir, self.vocab_size)["chkhsh"]
                if chk not in KNOWN_PRE:
                    problems.append(f"pre-tokenizer hash {str(chk)[:16]}... is unknown and the {f.model_type} "
                                    "family has no default pre-tokenizer")
        return "; ".join(problems) or None

    def _float_type(self) -> int:
        """GGUF type of a high-precision head/embedding table: the checkpoint's float type."""
        dt = getattr(self.config, "dtype", None) or getattr(self.config, "torch_dtype", None)
        dt = str(dt).removeprefix("torch.") if dt is not None else "bfloat16"
        return {"float16": GGML_F16, "float32": GGML_F32}.get(dt, GGML_BF16)

    def gguf_plan(self, state_keys) -> list[TensorMap]:
        return self.gguf_tensors(endpoints=tuple(k for k in ENDPOINTS if k in set(state_keys)))

    def gguf_plan_for_file(self, reader) -> list[TensorMap]:
        types = {t.name: t.gtype for t in reader.tensors}
        ends = tuple(k for k, n in (("lm_head", "output.weight"), ("embed_tokens", "token_embd.weight"))
                     if types.get(n) == GGML_PTQ1_0)
        return self.gguf_tensors(endpoints=ends)

    def gguf_tensors(self, endpoints=None) -> list[TensorMap]:
        """The file's tensors. ``endpoints``: which of lm_head / embed_tokens are ternary (the others are
        written in the checkpoint's float type; a tied model has no ``output.weight``). Default: both
        ternary, none for a tied model."""
        reason = self.gguf_unsupported()
        if reason:
            raise GGUFError(f"no PTQ1_0 GGUF layout for this {self.config.model_type} config: {reason}")
        tc, f = self.text_config, self.family
        endpoints = set(() if self.tied else ENDPOINTS) if endpoints is None else set(endpoints)
        if self.tied and endpoints:
            raise GGUFError("tied embedding/head: the shared table stays high precision (no ternary tied "
                            "tables yet)")
        hid, ffn, vocab = int(tc.hidden_size), int(tc.intermediate_size), int(tc.vocab_size)
        nh, nkv, hd = int(tc.num_attention_heads), int(tc.num_key_value_heads), self.head_dim
        ftype = self._float_type()

        def tern(name, key, out_f, in_f, perm=None, role="fold"):
            return TensorMap(name, GGML_PTQ1_0, (in_f, out_f), key, perm=perm, role=role)

        def f32(name, key, shape, perm=None, op="copy", data=None):
            return TensorMap(name, GGML_F32, tuple(reversed(shape)), key, op=op, perm=perm, shape=tuple(shape),
                             data=data)

        def table(name, key):
            return TensorMap(name, ftype, (hid, vocab), key, shape=(vocab, hid))

        rows = {"self_attn.q_proj": nh * hd, "self_attn.k_proj": nkv * hd, "self_attn.v_proj": nkv * hd,
                "self_attn.o_proj": hid, "mlp.gate_proj": ffn, "mlp.up_proj": ffn, "mlp.down_proj": hid}
        cols = {"self_attn.o_proj": nh * hd, "mlp.down_proj": ffn}
        perms = {}
        if f.rope_permute:
            perms = {"self_attn.q_proj": llama_rope_perm(nh * hd, nh), "self_attn.k_proj": llama_rope_perm(nkv * hd, nkv)}
        plan = []
        if not self.tied:
            plan.append(tern("output.weight", "lm_head", vocab, hid) if "lm_head" in endpoints
                        else table("output.weight", "lm_head.weight"))
        plan.append(f32("output_norm.weight", "norm.weight", (hid,)))
        rp = self._rope()
        if rp.get("rope_type") == "llama3":
            plan.append(f32("rope_freqs.weight", "", (hd // 2,), op="const", data=llama3_rope_factors(rp, hd)))
        plan.append(tern("token_embd.weight", "embed_tokens", vocab, hid, role="inverse") if "embed_tokens" in endpoints
                    else table("token_embd.weight", "embed_tokens.weight"))
        biased = set(self.bias_modules())
        for li in range(self.n_layers):
            b, k = f"blk.{li}.", f"layers.{li}."
            blk = [f32(b + "attn_norm.weight", k + "input_layernorm.weight", (hid,)),
                   f32(b + "ffn_norm.weight", k + "post_attention_layernorm.weight", (hid,))]
            for sub in ATTN + MLP:
                stem = b + _GGUF_NAME[sub]
                blk.append(tern(stem + ".weight", k + sub, rows[sub], cols.get(sub, hid), perms.get(sub)))
                if sub in biased:
                    blk.append(f32(stem + ".bias", k + sub + ".bias", (rows[sub],), perms.get(sub)))
            if f.qk_norm:
                blk += [f32(b + "attn_q_norm.weight", k + "self_attn.q_norm.weight", (hd,)),
                        f32(b + "attn_k_norm.weight", k + "self_attn.k_norm.weight", (hd,))]
            plan += sorted(blk, key=lambda t: t.name)
        return plan

    def gguf_source_rows(self, name: str, n: int) -> np.ndarray:
        """GGUF row i holds HF row out[i]: identity, except llama-rope q/k (weights and biases), where
        head h's row 2a + b holds HF row h * d + b * d/2 + a."""
        out = np.arange(n)
        stem = name.split(".")[2] if name.startswith("blk.") else ""
        if not self.family.rope_permute or stem not in ("attn_q", "attn_k"):
            return out
        tc = self.text_config
        heads = int(tc.num_attention_heads if stem == "attn_q" else tc.num_key_value_heads)
        d = n // heads
        h, j = out // d, out % d
        return h * d + (j % 2) * (d // 2) + j // 2

    def gguf_f32_op(self, name: str) -> str | None:
        return "const" if name == "rope_freqs.weight" else "copy"

    def _hparam_kvs(self) -> list[tuple[str, str, object]]:
        tc, arch, rp = self.text_config, self.family.gguf_arch, self._rope()
        kvs = [
            (f"{arch}.block_count", "u32", int(tc.num_hidden_layers)),
            (f"{arch}.context_length", "u32", int(tc.max_position_embeddings)),
            (f"{arch}.embedding_length", "u32", int(tc.hidden_size)),
            (f"{arch}.feed_forward_length", "u32", int(tc.intermediate_size)),
            (f"{arch}.attention.head_count", "u32", int(tc.num_attention_heads)),
            (f"{arch}.attention.head_count_kv", "u32", int(tc.num_key_value_heads)),
        ]
        if rp.get("rope_type") == "linear":
            kvs += [(f"{arch}.rope.scaling.type", "str", "linear"),
                    (f"{arch}.rope.scaling.factor", "f32", float(rp["factor"]))]
        kvs += [(f"{arch}.rope.freq_base", "f32", float(rp["rope_theta"])),
                (f"{arch}.attention.layer_norm_rms_epsilon", "f32", float(tc.rms_norm_eps)),
                (f"{arch}.attention.key_length", "u32", self.head_dim),
                (f"{arch}.attention.value_length", "u32", self.head_dim)]
        if arch == "llama":   # LlamaModel.set_gguf_parameters
            kvs += [(f"{arch}.vocab_size", "u32", int(tc.vocab_size)),
                    (f"{arch}.rope.dimension_count", "u32", self.head_dim)]
        return kvs

    def gguf_kvs(self, model_dir, *, name, signs, block, tensors, size_label=None):
        f = self.family
        reason = self.gguf_unsupported(model_dir)
        if reason:
            raise GGUFError(f"no PTQ1_0 GGUF for this model: {reason}")
        vocab = G.gpt2_vocab(model_dir, self.vocab_size)
        notes = list(vocab["notes"])
        pre = KNOWN_PRE.get(vocab["chkhsh"])
        if pre is None:
            pre = f.pre
            notes.append(f"pre-tokenizer hash {str(vocab['chkhsh'])[:16]}... is not one Prism's converter knows; "
                         f"wrote the {f.model_type} family default {pre!r} (check tokenization in Prism)")
        folded = [t.name for t in tensors if t.role == "fold"]
        inverse = [t.name for t in tensors if t.role == "inverse"]
        widths = sorted({int(t.ne[0]) for t in tensors if t.ternary})
        missing = [w for w in widths if w not in signs]
        if missing:
            raise GGUFError(f"no sign vector for input widths {missing}")
        n_params = sum(int(np.prod(t.ne)) for t in tensors)
        label = size_label or G.size_label(G.model_basename(model_dir), n_params)
        kvs = [("general.architecture", "str", f.gguf_arch),
               ("general.type", "str", "model"),
               ("general.name", "str", name),
               ("general.size_label", "str", label),
               *self._hparam_kvs(),
               ("prism.hadamard.version", "u32", 1),
               ("prism.hadamard.block_size", "u32", int(block)),
               ("prism.hadamard.transform", "str", "normalized-sylvester-walsh-hadamard"),
               ("prism.hadamard.axis", "str", "input-last-dimension"),
               ("prism.hadamard.sign_mode", "str", "explicit"),
               ("prism.hadamard.weight_names", "arr[str]", folded),
               ("prism.hadamard.sign_widths", "arr[i32]", widths),
               ("prism.hadamard.sign_values", "arr[i32]", [int(v) for w in widths for v in signs[w].tolist()])]
        if inverse:
            kvs.append(("prism.hadamard.inverse_weight_names", "arr[str]", inverse))
        kvs += [("tokenizer.ggml.model", "str", "gpt2"),
                ("tokenizer.ggml.pre", "str", pre),
                ("tokenizer.ggml.tokens", "arr[str]", vocab["tokens"]),
                ("tokenizer.ggml.token_type", "arr[i32]", vocab["token_type"]),
                ("tokenizer.ggml.merges", "arr[str]", vocab["merges"])]
        kvs += [(f"tokenizer.ggml.{G.SPECIAL_KV[typ]}", "u32", int(tid)) for typ, tid in vocab["ids"].items()]
        add = dict(vocab["add"])
        add.setdefault("bos", f.add_bos)   # the HF tokenizer adds BOS through its post-processor
        kvs += [(f"tokenizer.ggml.add_{typ}_token", "bool", bool(flag)) for typ, flag in add.items()]
        kvs += G.chat_template_kvs(vocab["chat_template"])
        kvs += [("general.quantization_version", "u32", 2),
                ("general.file_type", "u32", LLAMA_FTYPE_MOSTLY_PTQ1_0)]
        window = getattr(self.text_config, "sliding_window", None)
        if window and "sliding_attention" not in self.layer_types and f.gguf_arch == "llama":
            notes.append(f"config.sliding_window {window}: the llama GGUF arch attends to the whole context, so "
                         f"it matches the model only within {window} tokens")
        return kvs, notes

    def gguf_check(self, kvd: dict) -> None:
        if int(kvd.get("prism.hadamard.version", (None, 0))[1]) != 1:
            raise GGUFError("only prism.hadamard version 1 files are supported (no ternary tied tables)")
        for key, value in (("transform", "normalized-sylvester-walsh-hadamard"), ("axis", "input-last-dimension")):
            if kvd.get(f"prism.hadamard.{key}", (None, None))[1] != value:
                raise GGUFError(f"prism.hadamard.{key} must be {value!r}")
        inverse = kvd.get("prism.hadamard.inverse_weight_names", (None, []))[1]
        if not set(inverse) <= {"token_embd.weight"}:
            raise GGUFError(f"unexpected inverse-lookup tensors {inverse}")


def family_table() -> list[dict]:
    """The family table as rows (docs, ``openbonsai inspect``)."""
    return [{"model_type": f.model_type, "gguf_arch": f.gguf_arch, "prism_ptq1_0_gguf": f.prism_hadamard,
             "add_bos": f.add_bos, "thinking": f.thinking,
             "norm": "(1 + w) * x_hat" if f.zero_centered_norm else "w * x_hat",
             "rope_permute": f.rope_permute, "qk_norm": f.qk_norm, "notes": list(f.notes)}
            for f in FAMILIES.values()]

"""Qwen3.5 in Prism's ``qwen35`` PTQ1_0 GGUF layout (the Ternary-Bonsai-2 file layout).

Everything here was verified against Prism llama.cpp c9c34bc and the official
Ternary-Bonsai-2-9B-PTQ1_0.gguf (rev 5b2abb0) by OpenBonsai research/experiments/export_gguf.py,
which wrote a 9B file Prism's CPU runtime loads (97/97 checks). The layout is
derived from the config, so every Qwen3.5 size gets its own; for the 9B config
it equals the official header byte for byte (tests/test_export_parity.py).

Tied models store one ordinary float table with version 1, or, explicitly
quantized, one latent PTQ1_0 table with version 2 and tied_output=true.
The output tensor is absent: Prism binds token_embd as the output too.

Tensors (``ne`` = [in, out]): ``output`` (PTQ1_0, folded), ``output_norm``,
``token_embd`` (PTQ1_0, inverse lookup), then ``blk.N`` in layer order with
names ASCII-sorted inside a block (llama.cpp's weight_name_comparer).

- Gated DeltaNet value heads are grouped by key head in HF (g = k*rep + r)
  and tiled in GGUF (t = r*n_k + k): in_proj_qkv V rows, in_proj_z rows,
  conv1d V channels, in_proj_a/b rows, A_log, dt_bias. ``out_proj``
  (ssm_out) columns stay grouped (``prism.hadamard.gdn_v_grouped = true``).
- F32: zero-centered norms store 1 + w (input/post-attention/final norm,
  q_norm, k_norm); linear_attn.norm stores w; ssm_a = -exp(A_log); conv1d is
  squeezed to [C, K].
- ``prism.hadamard.weight_names``: output.weight, then the folded decoder
  weights in sorted HF-name order; ``inverse_weight_names``: token_embd.weight.
- KVs in the official order; ``general.type``, ``general.sampling.*`` and the
  trailing ``tokenizer.ggml.bos_token_id`` (``<|endoftext|>``) are not in the
  HF snapshot and follow the official file (sampling: generation_config.json
  when it sets them).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..formats.gguf import GGML_BF16, GGML_F16, GGML_F32, GGML_PTQ1_0, GGUFError, TensorMap
from ..formats.ptq1_0 import LLAMA_FTYPE_MOSTLY_PTQ1_0
from . import gguf_common as G

REFERENCE_9B = Path(__file__).resolve().parents[1] / "formats" / "assets" / "ternary-bonsai-2-9b.header.json"

_TERNARY = {"linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate",
            "linear_attn.out_proj": "ssm_out", "mlp.down_proj": "ffn_down", "mlp.gate_proj": "ffn_gate",
            "mlp.up_proj": "ffn_up", "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
            "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output"}
# Pre-tokenizer probe hashes Prism's converter maps to "qwen35".
QWEN35_PRE_HASHES = {"d30d75d9059f1aa2c19359de71047b3ae408c70875e8a3ccf8c5fba56c9d8af4",   # Qwen/Qwen3.5-9B-Instruct
                     "1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f"}   # openbmb/MiniCPM-V-4_6
# Values the official file carries that the HF snapshot does not (Qwen3.5's recommended sampling).
GENERAL_TYPE = "model"
SAMPLING = {"min_p": 0.0, "temp": 1.0, "top_p": 0.95, "top_k": 20, "penalty_repeat": 1.0}
_GEN_CONFIG_SAMPLING = {"min_p": "min_p", "temp": "temperature", "top_p": "top_p", "top_k": "top_k",
                        "penalty_repeat": "repetition_penalty"}
# transformers 5.20.2 Qwen3_5TextRotaryEmbedding: rope_parameters.get("mrope_section", [11, 11, 10]). It
# always recomposes the sections interleaved (mrope_interleaved is not read), as the qwen35 runtime does.
MROPE_SECTION_DEFAULT = (11, 11, 10)

# Export verification (independent of gguf_tensors): GGUF tensor stem -> how its rows map to the HF tensor
# ("qkv": value heads of the q|k|v channels, "v": value heads, "head": one row per value head) and the F32
# transform. The value-head rule, in arithmetic: tiled index t = r * n_k + k holds grouped g = k * rep + r.
_ROW_KIND = {"attn_qkv": "qkv", "attn_gate": "v", "ssm_conv1d": "qkv", "ssm_a": "head", "ssm_alpha": "head",
             "ssm_beta": "head", "ssm_dt": "head"}
_F32_OP = {"output_norm": "plus1", "attn_norm": "plus1", "post_attention_norm": "plus1", "attn_q_norm": "plus1",
           "attn_k_norm": "plus1", "ssm_norm": "copy", "ssm_a": "neg_exp", "ssm_alpha": "copy", "ssm_beta": "copy",
           "ssm_dt": "copy", "ssm_conv1d": "squeeze"}


def v_head_perm(n_k: int, n_v: int, unit: int) -> np.ndarray:
    """Grouped -> tiled value heads as a row index: gguf[i] = hf[perm[i]] (Prism's _reorder_v_heads)."""
    if n_v % n_k:
        raise GGUFError(f"linear_num_value_heads {n_v} is not a multiple of linear_num_key_heads {n_k}")
    rep = n_v // n_k
    return np.arange(n_v * unit).reshape(n_k, rep, unit).transpose(1, 0, 2).reshape(-1)


class Qwen35GGUF:
    """GGUF methods of Qwen35Adapter (mixin; uses ``self.text_config`` and ``self.tied``)."""

    gguf_arch = "qwen35"

    def _gguf_dims(self) -> dict:
        tc = self.text_config
        if getattr(tc, "attention_bias", False):
            raise GGUFError("attention_bias: the qwen35 GGUF layout has no bias tensors for q/k/v/o_proj")
        d = {"hid": int(tc.hidden_size), "ffn": int(tc.intermediate_size), "vocab": int(tc.vocab_size),
             "n_k": int(tc.linear_num_key_heads), "n_v": int(tc.linear_num_value_heads),
             "dk": int(tc.linear_key_head_dim), "dv": int(tc.linear_value_head_dim),
             "nh": int(tc.num_attention_heads), "nkv": int(tc.num_key_value_heads), "hd": int(tc.head_dim),
             "ksz": int(tc.linear_conv_kernel_dim), "types": list(tc.layer_types)}
        d["qk"] = 2 * d["n_k"] * d["dk"]
        d["interval"] = self._full_attention_interval(d["types"])
        return d

    def _full_attention_interval(self, types) -> int:
        """Prism derives layer types from one interval; refuse a schedule it cannot express."""
        full = [i for i, t in enumerate(types) if t == "full_attention"]
        k = full[0] + 1 if full else len(types) + 1
        if types != ["full_attention" if (i + 1) % k == 0 else "linear_attention" for i in range(len(types))]:
            raise GGUFError(f"layer_types {types} are not 'full attention every k-th layer'; "
                            "the qwen35 GGUF stores only full_attention_interval")
        configured = getattr(self.text_config, "full_attention_interval", None)
        if configured is not None and full and int(configured) != k:
            raise GGUFError(f"full_attention_interval {configured} disagrees with layer_types (every {k}th)")
        return k

    # -- layout ------------------------------------------------------------
    def gguf_unsupported(self, model_dir=None) -> str | None:
        try:
            self._gguf_dims()
        except GGUFError as error:
            return str(error)
        return None

    def gguf_plan(self, state_keys):
        return self.gguf_tensors(tied_ternary=True) if self.tied and "embed_tokens" in state_keys else self.gguf_tensors()

    def gguf_plan_for_file(self, reader):
        return (self.gguf_tensors(tied_ternary=True)
                if self.tied and reader.info("token_embd.weight").gtype == GGML_PTQ1_0 else self.gguf_tensors())

    def gguf_tensors(self, *, tied_ternary=False) -> list[TensorMap]:
        d = self._gguf_dims()
        hid, ffn, vocab, n_k, n_v, dv, qk = d["hid"], d["ffn"], d["vocab"], d["n_k"], d["n_v"], d["dv"], d["qk"]
        nh, nkv, hd = d["nh"], d["nkv"], d["hd"]
        vperm = v_head_perm(n_k, n_v, dv)
        qkv_perm = np.concatenate([np.arange(qk), qk + vperm])
        head_perm = v_head_perm(n_k, n_v, 1)
        conv_dim = qk + n_v * dv

        def tern(name, key, out_f, in_f, perm=None, role="fold"):
            return TensorMap(name, GGML_PTQ1_0, (in_f, out_f), key, perm=perm, role=role)

        def f32(name, key, shape, op, perm=None):
            stored = (shape[0], int(np.prod(shape[1:]))) if op == "squeeze" else tuple(shape)
            return TensorMap(name, GGML_F32, tuple(reversed(stored)), key, op=op, perm=perm, shape=tuple(shape))

        plan = [tern("output.weight", "lm_head", vocab, hid),
                f32("output_norm.weight", "norm.weight", (hid,), "plus1"),
                tern("token_embd.weight", "embed_tokens", vocab, hid, role="inverse")]
        if self.tied:
            # An ordinary shared table needs neither folded nor inverse rotation.
            # Version 2 tied_output is for a *rotated* shared table, not this layout.
            dt = getattr(self.text_config, "dtype", None) or getattr(self.config, "dtype", None)
            ftype = {"float16": GGML_F16, "float32": GGML_F32}.get(
                str(dt).removeprefix("torch."), GGML_BF16)
            plan = [plan[1], TensorMap("token_embd.weight", ftype, (hid, vocab),
                                      "embed_tokens.weight", shape=(vocab, hid))]
            if tied_ternary:
                plan[1] = tern("token_embd.weight", "embed_tokens", vocab, hid, role="inverse")
        for li, typ in enumerate(d["types"]):
            b, k = f"blk.{li}.", f"layers.{li}."
            blk = [f32(b + "attn_norm.weight", k + "input_layernorm.weight", (hid,), "plus1"),
                   f32(b + "post_attention_norm.weight", k + "post_attention_layernorm.weight", (hid,), "plus1"),
                   tern(b + "ffn_down.weight", k + "mlp.down_proj", hid, ffn),
                   tern(b + "ffn_gate.weight", k + "mlp.gate_proj", ffn, hid),
                   tern(b + "ffn_up.weight", k + "mlp.up_proj", ffn, hid)]
            if typ == "linear_attention":
                blk += [tern(b + "attn_qkv.weight", k + "linear_attn.in_proj_qkv", conv_dim, hid, qkv_perm),
                        tern(b + "attn_gate.weight", k + "linear_attn.in_proj_z", n_v * dv, hid, vperm),
                        tern(b + "ssm_out.weight", k + "linear_attn.out_proj", hid, n_v * dv),  # columns stay grouped
                        f32(b + "ssm_a", k + "linear_attn.A_log", (n_v,), "neg_exp", head_perm),
                        f32(b + "ssm_alpha.weight", k + "linear_attn.in_proj_a.weight", (n_v, hid), "copy", head_perm),
                        f32(b + "ssm_beta.weight", k + "linear_attn.in_proj_b.weight", (n_v, hid), "copy", head_perm),
                        f32(b + "ssm_conv1d.weight", k + "linear_attn.conv1d.weight", (conv_dim, 1, d["ksz"]),
                            "squeeze", qkv_perm),
                        f32(b + "ssm_dt.bias", k + "linear_attn.dt_bias", (n_v,), "copy", head_perm),
                        f32(b + "ssm_norm.weight", k + "linear_attn.norm.weight", (dv,), "copy")]
            elif typ == "full_attention":
                blk += [tern(b + "attn_q.weight", k + "self_attn.q_proj", nh * hd * 2, hid),  # query + output gate
                        tern(b + "attn_k.weight", k + "self_attn.k_proj", nkv * hd, hid),
                        tern(b + "attn_v.weight", k + "self_attn.v_proj", nkv * hd, hid),
                        tern(b + "attn_output.weight", k + "self_attn.o_proj", hid, nh * hd),
                        f32(b + "attn_q_norm.weight", k + "self_attn.q_norm.weight", (hd,), "plus1"),
                        f32(b + "attn_k_norm.weight", k + "self_attn.k_norm.weight", (hd,), "plus1")]
            else:
                raise GGUFError(f"unknown layer type {typ}")
            plan += sorted(blk, key=lambda t: t.name)
        return plan

    @staticmethod
    def gguf_weight_names(tensors) -> list[str]:
        """output.weight first, then the folded decoder weights in sorted HF-name order (the official order)."""
        folded = [t for t in tensors if t.role == "fold"]
        head = [t.name for t in folded if t.source == "lm_head"]
        rest = sorted((t.source + ".weight", t.name) for t in folded if t.source != "lm_head")
        return head + [name for _, name in rest]

    @staticmethod
    def _stem(name: str) -> str:
        """'blk.3.ssm_a' -> 'ssm_a'; 'blk.0.attn_qkv.weight' -> 'attn_qkv'; 'output_norm.weight' -> 'output_norm'."""
        parts = name.split(".")
        return parts[2] if parts[0] == "blk" else parts[0]

    def gguf_source_rows(self, name: str, n: int) -> np.ndarray:
        """GGUF row i holds HF row out[i] (OpenBonsai research/experiments/export_gguf.py Verifier.src_index)."""
        kind = _ROW_KIND.get(self._stem(name))
        out = np.arange(n)
        if kind is None:
            return out
        tc = self.text_config
        n_k, n_v = int(tc.linear_num_key_heads), int(tc.linear_num_value_heads)
        dv = int(tc.linear_value_head_dim)
        unit, base = {"head": (1, 0), "v": (dv, 0),
                      "qkv": (dv, 2 * n_k * int(tc.linear_key_head_dim))}[kind]
        j = out[base:] - base
        t, e = j // unit, j % unit
        r, k = t // n_k, t % n_k
        out[base:] = base + (k * (n_v // n_k) + r) * unit + e
        return out

    def gguf_f32_op(self, name: str) -> str | None:
        if self.tied and self._stem(name) == "token_embd":
            return "copy"
        return _F32_OP.get(self._stem(name))

    # -- metadata ----------------------------------------------------------
    def _sampling(self, model_dir) -> dict:
        out = dict(SAMPLING)
        path = Path(model_dir) / "generation_config.json"
        if path.exists():
            gc = json.loads(path.read_text(encoding="utf-8"))
            for kv, hf in _GEN_CONFIG_SAMPLING.items():
                if gc.get(hf) is not None:
                    out[kv] = gc[hf]
        return out

    def _hparam_kvs(self) -> list[tuple[str, str, object]]:
        """The ``qwen35.*`` model KVs, in file order."""
        tc = self.text_config
        d = self._gguf_dims()
        arch = self.gguf_arch
        rp = dict(getattr(tc, "rope_parameters", None) or {})
        # transformers' default when the config has none (Qwen3_5TextRotaryEmbedding); the runtime needs
        # four sections (the fourth, unused for text, is 0) whose sum is not 0.
        sections = [int(x) for x in rp.get("mrope_section") or MROPE_SECTION_DEFAULT]
        if len(sections) > 4 or not sum(sections) or min(sections) < 0:
            raise GGUFError(f"rope_parameters.mrope_section {sections} cannot be written as the qwen35 runtime's "
                            "four rope.dimension_sections")
        sections += [0] * (4 - len(sections))
        return [
            (f"{arch}.block_count", "u32", int(tc.num_hidden_layers)),          # no MTP block
            (f"{arch}.context_length", "u32", int(tc.max_position_embeddings)),
            (f"{arch}.embedding_length", "u32", d["hid"]),
            (f"{arch}.feed_forward_length", "u32", d["ffn"]),
            (f"{arch}.attention.head_count", "u32", d["nh"]),
            (f"{arch}.attention.head_count_kv", "u32", d["nkv"]),
            (f"{arch}.rope.dimension_sections", "arr[i32]", sections),
            (f"{arch}.rope.freq_base", "f32", float(rp["rope_theta"])),
            (f"{arch}.attention.layer_norm_rms_epsilon", "f32", float(tc.rms_norm_eps)),
            (f"{arch}.attention.key_length", "u32", d["hd"]),
            (f"{arch}.attention.value_length", "u32", d["hd"]),
            (f"{arch}.ssm.conv_kernel", "u32", d["ksz"]),
            (f"{arch}.ssm.state_size", "u32", d["dk"]),
            (f"{arch}.ssm.group_count", "u32", d["n_k"]),
            (f"{arch}.ssm.time_step_rank", "u32", d["n_v"]),
            (f"{arch}.ssm.inner_size", "u32", d["dv"] * d["n_v"]),
            (f"{arch}.full_attention_interval", "u32", d["interval"]),
            (f"{arch}.rope.dimension_count", "u32", int(d["hd"] * float(rp.get("partial_rotary_factor", 1.0)))),
        ]

    def gguf_kvs(self, model_dir, *, name, signs, block, tensors, size_label=None):
        d = self._gguf_dims()
        arch = self.gguf_arch
        vocab = G.gpt2_vocab(model_dir, d["vocab"])
        if vocab["chkhsh"] is not None and vocab["chkhsh"] not in QWEN35_PRE_HASHES:
            vocab["notes"].append(f"pre-tokenizer hash {vocab['chkhsh'][:16]}... is not a known qwen35 hash; "
                                  "writing 'qwen35' (the qwen35 runtime's pre-tokenizer)")
        widths = sorted({int(t.ne[0]) for t in tensors if t.ternary})
        missing = [w for w in widths if w not in signs]
        if missing:
            raise GGUFError(f"no sign vector for input widths {missing}")
        sign_values = [int(v) for w in widths for v in signs[w].tolist()]
        n_params = sum(int(np.prod(t.ne)) for t in tensors)
        label = size_label or G.size_label(G.model_basename(model_dir), n_params)
        sampling = self._sampling(model_dir)
        kvs = [
            ("general.architecture", "str", arch),
            ("general.type", "str", GENERAL_TYPE),
            ("general.sampling.min_p", "f32", float(sampling["min_p"])),
            ("general.name", "str", name),
            ("general.size_label", "str", label),
            *self._hparam_kvs(),
            ("prism.hadamard.version", "u32", 2 if self.tied and any(t.role == "inverse" for t in tensors) else 1),
            ("prism.hadamard.block_size", "u32", int(block)),
            ("prism.hadamard.transform", "str", "normalized-sylvester-walsh-hadamard"),
            ("prism.hadamard.axis", "str", "input-last-dimension"),
            ("prism.hadamard.sign_mode", "str", "explicit"),
            ("prism.hadamard.weight_names", "arr[str]", self.gguf_weight_names(tensors)),
            ("prism.hadamard.sign_widths", "arr[i32]", widths),
            ("prism.hadamard.sign_values", "arr[i32]", sign_values),
            ("prism.hadamard.inverse_weight_names", "arr[str]", [t.name for t in tensors if t.role == "inverse"]),
            ("prism.hadamard.gdn_v_grouped", "bool", True),
            ("tokenizer.ggml.model", "str", "gpt2"),
            ("tokenizer.ggml.pre", "str", "qwen35"),
            ("tokenizer.ggml.tokens", "arr[str]", vocab["tokens"]),
            ("tokenizer.ggml.token_type", "arr[i32]", vocab["token_type"]),
            ("tokenizer.ggml.merges", "arr[str]", vocab["merges"]),
        ]
        if self.tied and any(t.role == "inverse" for t in tensors):
            kvs.append(("prism.hadamard.tied_output", "bool", True))
        kvs += [(f"tokenizer.ggml.{G.SPECIAL_KV[typ]}", "u32", int(tid)) for typ, tid in vocab["ids"].items()]
        kvs += [(f"tokenizer.ggml.add_{typ}_token", "bool", bool(flag)) for typ, flag in vocab["add"].items()]
        kvs += G.chat_template_kvs(vocab["chat_template"])
        kvs += [("general.quantization_version", "u32", 2),
                ("general.file_type", "u32", LLAMA_FTYPE_MOSTLY_PTQ1_0)]
        if "bos" not in vocab["ids"]:
            # The official file adds BOS = <|endoftext|> after conversion (tokenizer_config has bos_token null).
            bos = next((i for i, t in vocab["vocab"].items() if t == "<|endoftext|>"), vocab["ids"].get("pad"))
            if bos is not None:
                kvs.append(("tokenizer.ggml.bos_token_id", "u32", int(bos)))
        kvs += [("general.sampling.temp", "f32", float(sampling["temp"])),
                ("general.sampling.top_p", "f32", float(sampling["top_p"])),
                ("general.sampling.top_k", "i32", int(sampling["top_k"])),
                ("general.sampling.penalty_repeat", "f32", float(sampling["penalty_repeat"]))]
        return kvs, list(vocab["notes"])

    def gguf_reference(self) -> dict | None:
        """The official Ternary-Bonsai-2-9B header dump when this config has its hyperparameters and
        vocabulary size (the export then asserts the identical tensor layout and KV keys)."""
        if not REFERENCE_9B.exists():
            return None
        ref = json.loads(REFERENCE_9B.read_text(encoding="utf-8"))
        theirs = {h["key"]: h for h in ref["kv"]}
        for key, typ, val in self._hparam_kvs():
            h = theirs.get(key)
            if h is None or h["type"] != typ or not G.reference_kv_equal(h, typ, val)[0]:
                return None
        tokens = theirs.get("tokenizer.ggml.tokens", {})
        return ref if tokens.get("len") == int(self.text_config.vocab_size) else None

    def gguf_check(self, kvd: dict) -> None:
        grouped = kvd.get("prism.hadamard.gdn_v_grouped", (None, None))[1]
        identity = self.text_config.linear_num_key_heads == self.text_config.linear_num_value_heads
        if grouped is not True and not (grouped is None and identity):
            raise GGUFError("the GGUF does not keep ssm_out columns grouped (prism.hadamard.gdn_v_grouped); "
                            "only that layout maps back to Qwen3.5")
        version = int(kvd.get("prism.hadamard.version", (None, 0))[1])
        tied_output = kvd.get("prism.hadamard.tied_output", (None, False))[1]
        if version not in (1, 2) or (version == 2) != bool(tied_output) or (version == 2 and not self.tied):
            raise GGUFError("invalid prism.hadamard version/tied_output combination")
        expected = [] if self.tied and version == 1 else ["token_embd.weight"]
        if kvd.get("prism.hadamard.inverse_weight_names", (None, []))[1] != expected:
            raise GGUFError(f"expected inverse-lookup tensors {expected}")
        for key, value in (("transform", "normalized-sylvester-walsh-hadamard"), ("axis", "input-last-dimension")):
            if kvd.get(f"prism.hadamard.{key}", (None, None))[1] != value:
                raise GGUFError(f"prism.hadamard.{key} must be {value!r}")

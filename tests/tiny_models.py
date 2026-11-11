"""Tiny random Qwen3.5 fixtures for CPU tests; outputs say nothing about 9B quality.

The text model has three Gated DeltaNet layers and one full-attention layer.
Every rotated width (hidden 128, MLP 256, value/head dims 128) is a multiple
of 128, so the derived Hadamard block is 128.
"""
import torch
from transformers import (Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration,
                          Qwen3_5TextConfig)

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
)


def tiny_text_config(vocab_size=256, eos_token_id=2, pad_token_id=0, attn="eager", **overrides):
    params = dict(
        vocab_size=vocab_size, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=32, linear_value_head_dim=32,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1],
                         "mrope_interleaved": True},
        use_cache=False, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
    )
    text = Qwen3_5TextConfig(**{**params, **overrides})
    text._attn_implementation = attn
    return text


def tiny_qwen3_5(seed=0, multimodal=False, *, vocab_size=256, eos_token_id=2, pad_token_id=0, attn="eager",
                 random_norms=False):
    text = tiny_text_config(vocab_size, eos_token_id, pad_token_id, attn)
    config, constructor = text, Qwen3_5ForCausalLM
    if multimodal:
        config = Qwen3_5Config(
            text_config=text,
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                           "num_heads": 4, "out_hidden_size": 128, "patch_size": 2,
                           "spatial_merge_size": 2, "temporal_patch_size": 2,
                           "num_position_embeddings": 16},
            image_token_id=vocab_size - 4, video_token_id=vocab_size - 3, vision_start_token_id=vocab_size - 2,
            vision_end_token_id=vocab_size - 1,
        )
        config._attn_implementation = attn
        config.text_config._attn_implementation = attn
        config.vision_config._attn_implementation = attn
        config.text_config.use_cache = False
        constructor = Qwen3_5ForConditionalGeneration
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = constructor(config)
        if random_norms:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name.endswith("layernorm.weight") or name.endswith("model.norm.weight") \
                            or name.endswith("language_model.norm.weight"):
                        p.normal_(0.0, 0.3)
    model.eval()
    model.requires_grad_(False)
    return model


def byte_tokenizer():
    """Byte-level BPE with no merges (ids 0..255 = bytes) plus Qwen-style special tokens."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from openbonsai.eval.backends import _byte_decoder
    vocab = {char: i for i, char in enumerate(_byte_decoder())}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=[t for t in SPECIAL if t not in ("<|im_end|>", "<|endoftext|>")])
    tok.chat_template = CHAT_TEMPLATE
    return tok


def save_tiny_qwen3_5(path, *, seed=0, multimodal=False, vocab_size=320):
    """Write a loadable tiny model directory (config, weights, tokenizer, chat template)."""
    tok = byte_tokenizer()
    eos, pad = tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")
    model = tiny_qwen3_5(seed, multimodal, vocab_size=vocab_size, eos_token_id=eos, pad_token_id=pad,
                         attn="sdpa", random_norms=True)
    model.generation_config.eos_token_id = eos
    model.generation_config.pad_token_id = pad
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path


# -- dense models (adapters/dense.py) -------------------------------------------

LLAMA_SPECIAL = ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
LLAMA_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n"
    "{{ m['content'] }}<|eot_id|>{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)


def llama_byte_tokenizer():
    """Byte-level BPE with no merges plus Llama-3-style special tokens (BOS, <|eot_id|>)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from openbonsai.eval.backends import _byte_decoder
    vocab = {char: i for i, char in enumerate(_byte_decoder())}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>",
                                  additional_special_tokens=[t for t in LLAMA_SPECIAL
                                                             if t not in ("<|begin_of_text|>", "<|eot_id|>")])
    tok.chat_template = LLAMA_TEMPLATE
    return tok


def tiny_dense_config(model_type, *, tied=False, vocab_size=320, eos_token_id=2, pad_token_id=None, bos_token_id=None,
                      attn="sdpa", **overrides):
    """Two layers, hidden 128, MLP 256, 4 heads of 32 (2 KV heads): every rotated width is 128 or 256."""
    from transformers import LlamaConfig, MistralConfig, Qwen2Config, Qwen3Config
    common = dict(vocab_size=vocab_size, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=512,
                  rms_norm_eps=1e-6, tie_word_embeddings=tied, use_cache=False, eos_token_id=eos_token_id,
                  pad_token_id=pad_token_id, bos_token_id=bos_token_id)
    if model_type == "llama":   # llama3 rope (rope_freqs.weight) and every bias slot the llama GGUF has
        common.update(rope_parameters={"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0,
                                       "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                       "original_max_position_embeddings": 64},
                      attention_bias=True)
        cls = LlamaConfig
    else:
        cls = {"mistral": MistralConfig, "qwen2": Qwen2Config, "qwen3": Qwen3Config}[model_type]
        if model_type == "mistral":
            common["sliding_window"] = None
    config = cls(**{**common, **overrides})
    config._attn_implementation = attn
    return config


def tiny_dense(model_type, seed=0, *, random_norms=True, sharp=True, **config):
    """A random dense model. ``sharp``: q/k weights x6 and the head x10 (x3 tied), so attention depends on
    position (a wrong rope layout shows) and next-token distributions are peaked, as in a trained model."""
    from transformers import AutoModelForCausalLM
    cfg = tiny_dense_config(model_type, **config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_config(cfg)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith("norm.weight") and random_norms:
                    p.normal_(1.0, 0.3)
                elif name.endswith(".bias"):
                    p.normal_(0.0, 0.1)
                if sharp and (name.endswith("q_proj.weight") or name.endswith("k_proj.weight")):
                    p.mul_(6.0)
            if sharp:   # a tied table is also the embedding: a smaller factor keeps the model from echoing its input
                model.lm_head.weight.mul_(3.0 if cfg.tie_word_embeddings else 10.0)
    model.eval()
    model.requires_grad_(False)
    return model


def save_tiny_dense(path, model_type, *, tied=False, seed=0, vocab_size=320, **config):
    """A loadable tiny dense model directory (config, BF16 weights, tokenizer, chat template)."""
    tok = llama_byte_tokenizer() if model_type in ("llama", "mistral") else byte_tokenizer()
    eos = tok.eos_token_id
    pad = tok.pad_token_id
    model = tiny_dense(model_type, seed, tied=tied, vocab_size=vocab_size, eos_token_id=eos, pad_token_id=pad,
                       bos_token_id=tok.bos_token_id, **config).to(torch.bfloat16)
    model.generation_config.eos_token_id = eos
    model.generation_config.pad_token_id = pad
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path

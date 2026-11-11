"""A real shared table through GPTQ, GGUF and native reload; no mocked tying."""
import pytest
import torch
from transformers import Qwen3_5ForCausalLM

from tiny_models import tiny_text_config, byte_tokenizer
from openbonsai.adapters import adapter_for_config
from openbonsai.config import load_config
from openbonsai.core.geometry import derive_geometry, make_signs
from openbonsai.formats.gguf import GGUFReader, GGML_BF16
from openbonsai.stages.endpoints import ternary_endpoints
from openbonsai.stages.export import export_gguf
from openbonsai.stages.quantize import quantize_model


@pytest.mark.parametrize("ternary_table", [False, True])
def test_shared_table_survives_decoder_compression_and_gguf(tmp_path, ternary_table):
    torch.set_num_threads(2)
    torch.manual_seed(19)
    cfg = tiny_text_config(vocab_size=320, tie_word_embeddings=True)
    cfg._attn_implementation = "sdpa"
    model = Qwen3_5ForCausalLM._from_config(cfg, dtype=torch.bfloat16).eval().requires_grad_(False)
    adapter = adapter_for_config(cfg)
    assert adapter.tied
    table = model.model.embed_tokens.weight
    assert model.lm_head.weight is table
    before = table.clone()
    assert ternary_endpoints(load_config(None), adapter) == (False, False)
    with pytest.raises(NotImplementedError, match="tied"):
        ternary_endpoints(load_config(None, ["endpoints.head=ternary"]), adapter)
    folder = tmp_path / "model"
    model.save_pretrained(folder)
    byte_tokenizer().save_pretrained(folder)
    geo = derive_geometry(adapter.quantized_widths(model), adapter.hidden_size)
    ids = torch.randint(10, 250, (2, 32))
    with torch.no_grad():
        teacher = model(input_ids=ids, use_cache=False).logits.float().log_softmax(-1)
    state, stats = quantize_model(model, adapter, ids, make_signs(geo), geo, batch=1, log=lambda _: None)
    assert not {"lm_head", "embed_tokens"} & set(state.tensors)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert torch.equal(table, before)
    assert all(torch.isfinite(torch.tensor(v["rel_mse"])) for v in stats.values())
    path = tmp_path / "tied.gguf"
    if ternary_table:
        from openbonsai.core.gptq import quantize_rows
        from openbonsai.formats.state import TernaryState, TernaryTensor
        from openbonsai.stages.endpoints import joint_hessian, head_hessian
        h = head_hessian(model.model, ids, 1, "cpu")
        metric = joint_hessian(h, adapter.embedding_gain(model))
        signs = make_signs(geo)
        c, s, _ = quantize_rows(before, metric, signs[128], geo.block, chunk=64)
        state = state.merged(TernaryState({"embed_tokens": TernaryTensor(c, s)}, signs, geo.block, geo.group))
    export_gguf(adapter, state, folder, path, name="tied", log=lambda _: None)
    with GGUFReader(path) as r:
        names = {t.name: t for t in r.tensors}
        assert "output.weight" not in names
        from openbonsai.formats.gguf import GGML_PTQ1_0
        assert names["token_embd.weight"].gtype == (GGML_PTQ1_0 if ternary_table else GGML_BF16)
        assert r.value("prism.hadamard.inverse_weight_names") == (["token_embd.weight"] if ternary_table else [])
        assert r.value("prism.hadamard.version") == (2 if ternary_table else 1)
        assert "token_embd.weight" not in r.value("prism.hadamard.weight_names")
    # The expected logits use the serialized ternary modules, matching the loader.
    from openbonsai.stages.common import install_ternary
    install_ternary(adapter, model, state.records(), torch.device("cpu"))
    back = adapter.from_gguf(path, dtype=torch.bfloat16)
    if ternary_table:
        assert back.lm_head.log_mult is back.model.embed_tokens.log_mult
        assert back.lm_head.codes is back.model.embed_tokens.codes
    else:
        assert back.lm_head.weight is back.model.embed_tokens.weight
        assert torch.equal(back.model.embed_tokens.weight, before)
    with torch.no_grad():
        expected = model(input_ids=ids[:1], use_cache=False).logits
        got = back(input_ids=ids[:1], use_cache=False).logits
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert back.generate(ids[:1, :4], max_new_tokens=3, do_sample=False).shape[1] == 7
    if ternary_table:
        from openbonsai.stages.distill import train
        import numpy as np
        lp, idx = teacher.topk(16, -1)
        def kl():
            with torch.no_grad():
                student = back(input_ids=ids, use_cache=False).logits.float().log_softmax(-1)
                return (teacher.exp() * (teacher - student)).sum(-1).mean().item()
        initial = kl()
        codes_before = back.model.embed_tokens.codes.clone()
        out = tmp_path / "distill"
        out.mkdir()
        history = train(back, adapter, ids.numpy(), idx.numpy().astype(np.int32), lp.numpy(), out,
                        lr=0.005, epochs=2, accum=1, warmup=1, max_steps=4, chunk=16,
                        ckpt_every=1, log_every=1, device=torch.device("cpu"), log=lambda _: None)
        assert len(history) == 4
        assert torch.equal(codes_before, back.model.embed_tokens.codes)
        assert kl() < initial
        from openbonsai.stages.common import densify
        from openbonsai.stages.distill import trained_scales
        assert "lm_head" not in trained_scales(back, adapter)
        densify(back, torch.bfloat16)
        assert back.lm_head.weight is back.model.embed_tokens.weight


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_shared_table_gradient_is_sum_of_both_uses(dtype):
    from openbonsai.core.ternary import TernaryEmbedding, TernaryLinear, shared_hadamard
    torch.manual_seed(4)
    codes = torch.randint(-1, 2, (16, 128), dtype=torch.int8)
    scales = torch.ones(16, 1) * 0.03
    signs = torch.ones(128, dtype=torch.int8)
    h = shared_hadamard(128, "cpu", dtype=dtype)
    emb = TernaryEmbedding(codes, scales, signs, 128, h)
    head = TernaryLinear(codes, scales, signs, 128, h)
    head.log_mult = emb.log_mult
    ids = torch.tensor([[1, 3]])
    x = torch.randn(1, 2, 128).to(dtype)
    a = emb(ids).float().square().sum()
    b = head(x).float().square().sum()
    ga = torch.autograd.grad(a, emb.log_mult, retain_graph=True)[0]
    gb = torch.autograd.grad(b, emb.log_mult, retain_graph=True)[0]
    total = torch.autograd.grad(a + b, emb.log_mult)[0]
    torch.testing.assert_close(total, ga + gb)
    assert ga.abs().sum() > 0 and gb.abs().sum() > 0
    # Both lookup and projection use the same effective table, within BF16 rounding.
    table = emb(torch.arange(16)).float()
    torch.testing.assert_close(head(x).float(), x.float() @ table.T, atol=0.02, rtol=0.03)


def test_joint_endpoint_stage_and_scale_distillation(tmp_path, monkeypatch):
    from pathlib import Path
    from openbonsai import cli
    from openbonsai.formats.state import load_state, read_scales
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    torch.set_num_threads(2)
    cfg = tiny_text_config(vocab_size=320, tie_word_embeddings=True)
    cfg._attn_implementation = "sdpa"
    folder = tmp_path / "model"
    Qwen3_5ForCausalLM._from_config(cfg, dtype=torch.bfloat16).save_pretrained(folder)
    byte_tokenizer().save_pretrained(folder)
    out = tmp_path / "run"
    assert cli.main(["run", "--model", str(folder), "--config", "examples/tiny-qwen3_5.yaml",
                     "--out", str(out), "--until", "export", "--set", "endpoints.tied=ternary",
                     "--set", "distill.train.small=true"]) == 0
    endpoints = load_state(out / "endpoints/state.safetensors")
    assert set(endpoints.tensors) == {"embed_tokens"}
    assert endpoints.meta["head"] == endpoints.meta["embedding"] == "ternary"
    scales, _ = read_scales(out / "distill/scales.safetensors")
    assert "embed_tokens" in scales and "lm_head" not in scales
    from openbonsai.formats.state import read_small
    small = read_small(out / "distill/scales.safetensors")
    assert "norm.weight" in small and "lm_head.weight" not in small


def test_prism_omitted_grouping_flag_requires_identity_permutation():
    from openbonsai.formats.gguf import GGUFError
    kv = {"prism.hadamard.version": ("u32", 2), "prism.hadamard.tied_output": ("bool", True),
          "prism.hadamard.inverse_weight_names": ("arr[str]", ["token_embd.weight"]),
          "prism.hadamard.transform": ("str", "normalized-sylvester-walsh-hadamard"),
          "prism.hadamard.axis": ("str", "input-last-dimension")}
    adapter = adapter_for_config(tiny_text_config(tie_word_embeddings=True, linear_num_value_heads=2))
    adapter.gguf_check(kv)
    other = adapter_for_config(tiny_text_config(tie_word_embeddings=True))
    with pytest.raises(GGUFError, match="grouped"):
        other.gguf_check(kv)
    with pytest.raises(GGUFError, match="grouped"):
        adapter.gguf_check({**kv, "prism.hadamard.gdn_v_grouped": ("bool", False)})

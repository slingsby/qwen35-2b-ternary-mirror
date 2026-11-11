"""Ternary modules beyond the scripts' Qwen3.5: a Linear bias is kept."""
import torch

from openbonsai.core.hadamard import seeded_signs
from openbonsai.core.ternary import TernaryLinear, dense_weight, shared_hadamard


def test_ternary_linear_keeps_a_bias():
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(-1, 2, (8, 256), generator=g).to(torch.int8)
    scales = (torch.rand(8, 2, generator=g) + 0.5).half()
    signs = seeded_signs(256, 256)
    bias = torch.randn(8, generator=g).to(torch.bfloat16)
    lin = torch.nn.Linear(256, 8, bias=True, dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_(dense_weight(codes, scales, signs, 128, device="cpu", dtype=torch.bfloat16))
        lin.bias.copy_(bias)
    tern = TernaryLinear(codes, scales, signs, 128, shared_hadamard(128, "cpu"), bias=lin.bias)
    x = torch.randn(3, 256, generator=g).to(torch.bfloat16)
    with torch.no_grad():
        assert torch.allclose(tern(x).float(), lin(x).float(), atol=0.05, rtol=0.02)
        plain = TernaryLinear(codes, scales, signs, 128, shared_hadamard(128, "cpu"))(x).float()
        assert not torch.allclose(plain, lin(x).float(), atol=0.05, rtol=0.02)   # the bias matters here
    assert [n for n, _ in tern.named_parameters()] == ["log_mult"]


def test_install_ternary_carries_a_linear_bias():
    from openbonsai.formats.state import TernaryTensor
    from openbonsai.stages.common import install_ternary

    class Adapter:
        def layers(self, model):
            return model.layers

    g = torch.Generator().manual_seed(1)
    lin = torch.nn.Linear(128, 4, bias=True, dtype=torch.bfloat16)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.ModuleDict({"proj": lin})])
    t = TernaryTensor(torch.randint(-1, 2, (4, 128), generator=g).to(torch.int8), torch.rand(4, 1, generator=g).half())
    (key, new), = install_ternary(Adapter(), model, [("layers.0.proj", t, seeded_signs(128, 128), 128)],
                                  torch.device("cpu"))
    assert key == "layers.0.proj" and model.layers[0]["proj"] is new
    assert torch.equal(new.bias, lin.bias)

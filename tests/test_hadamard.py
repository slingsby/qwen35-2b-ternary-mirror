"""Rotation algebra only: random tensors, no ternary model quality claims."""
import math

import pytest
import torch
from torch.nn import functional as F

from openbonsai.core.geometry import UnsupportedGeometry, derive_block, derive_geometry, make_signs
from openbonsai.core.hadamard import (hadamard_matrix, inverse_rotate, normalized_hadamard, rotate, rotate_hessian,
                                      rotation_matrix, seeded_signs)


def dense_hadamard(block):
    result = torch.ones(1, 1, dtype=torch.float64)
    while result.shape[0] < block:
        result = torch.cat((torch.cat((result, result), 1), torch.cat((result, -result), 1)), 0)
    return result / math.sqrt(block)


@pytest.mark.parametrize("block,blocks", [(1, 1), (8, 1), (16, 3)])
def test_independent_dense_oracle_and_wrong_inverse_negative_control(block, blocks):
    generator = torch.Generator().manual_seed(99)
    width = block * blocks
    signs = torch.randint(0, 2, (width,), generator=generator) * 2 - 1
    h = torch.block_diag(*([dense_hadamard(block)] * blocks))
    r = torch.diag(signs.double()) @ h
    x = torch.randn(2, 3, width, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(rotate(x, signs, block), x @ r, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(inverse_rotate(x, signs, block), x @ r.T, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(inverse_rotate(rotate(x, signs, block), signs, block), x, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(r @ r.T, torch.eye(width, dtype=torch.float64), atol=2e-13, rtol=2e-13)
    if block > 1:
        assert (rotate(rotate(x, signs, block), signs, block) - x).square().mean().sqrt() > .1
        assert (rotate(x, signs, block) * math.sqrt(block) - x @ r).square().mean().sqrt() > .1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_4096_block_linear_and_embedding_values_and_gradients(dtype):
    generator = torch.Generator().manual_seed(17)
    width = 8192
    signs = torch.randint(0, 2, (width,), generator=generator) * 2 - 1
    x = torch.randn(2, width, generator=generator, dtype=dtype).requires_grad_()
    weight = torch.randn(7, width, generator=generator, dtype=dtype) / math.sqrt(width)
    bias = torch.randn(7, generator=generator, dtype=dtype)
    original = F.linear(x, weight, bias)
    rotated = F.linear(rotate(x, signs, 4096), rotate(weight, signs, 4096), bias)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-11
    torch.testing.assert_close(rotated, original, atol=tolerance, rtol=tolerance)
    original_grad, = torch.autograd.grad(original.sum(), x)
    rotated_grad, = torch.autograd.grad(rotated.sum(), x)
    torch.testing.assert_close(rotated_grad, original_grad, atol=tolerance, rtol=tolerance)
    table = torch.randn(9, width, generator=generator, dtype=dtype).requires_grad_()
    ids = torch.tensor([[8, 0, 2, 2], [1, 8, 2, 0]])
    expected = F.embedding(ids, table)
    actual = inverse_rotate(F.embedding(ids, rotate(table, signs, 4096)), signs, 4096)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    original_grad, = torch.autograd.grad(expected.square().sum(), table)
    rotated_grad, = torch.autograd.grad(actual.square().sum(), table)
    torch.testing.assert_close(rotated_grad, original_grad, atol=tolerance, rtol=tolerance)


def test_half_precision_accumulates_fp32_and_rounds_once():
    x = torch.linspace(-4, 5, 32).reshape(2, 16).bfloat16()
    expected = normalized_hadamard(x.float(), 8).bfloat16()
    assert torch.equal(normalized_hadamard(x, 8), expected)
    assert normalized_hadamard(torch.empty(2, 0, 16), 8).shape == (2, 0, 16)


@pytest.mark.parametrize("block,width", [(3, 12), (0, 16), (True, 16), (8, 15)])
def test_invalid_block_contract_rejected(block, width):
    with pytest.raises(ValueError):
        rotate(torch.ones(2, width), torch.ones(width), block)


def test_invalid_sign_contract_rejected():
    with pytest.raises(ValueError): rotate(torch.ones(2, 16), torch.ones(15), 8)
    with pytest.raises(ValueError): rotate(torch.ones(2, 16), torch.zeros(16), 8)


# Dense parity path, Hessian rotation and model geometry.

@pytest.mark.parametrize("width,block", [(128, 128), (384, 128), (1024, 512)])
def test_fwht_equals_dense_rotation_and_hessian(width, block):
    g = torch.Generator().manual_seed(width)
    signs = seeded_signs(width, width)
    r = rotation_matrix(signs, block).double()
    w = torch.randn(5, width, generator=g, dtype=torch.float64)
    x = torch.randn(64, width, generator=g, dtype=torch.float64)
    h = x.T @ x
    assert torch.allclose(rotate(w, signs, block), w @ r, atol=1e-12)
    assert torch.allclose(inverse_rotate(w @ r, signs, block), w, atol=1e-12)
    assert torch.allclose(rotate_hessian(h, signs, block), r.T @ h @ r, atol=1e-9)
    assert torch.allclose(r @ r.T, torch.eye(width, dtype=torch.float64), atol=1e-12)


def test_hadamard_matrix_is_sylvester_natural_order():
    h = (hadamard_matrix(8) * math.sqrt(8)).round()
    for i in range(8):
        for j in range(8):
            assert h[i, j] == (-1) ** bin(i & j).count("1")


def test_seeded_signs_are_deterministic_int8_and_width_seeded():
    s = seeded_signs(4096, 4096)
    assert s.dtype == torch.int8 and torch.equal(s, seeded_signs(4096, 4096))
    assert set(s.unique().tolist()) == {-1, 1} and not torch.equal(s[:128], seeded_signs(128, 128))


@pytest.mark.parametrize("widths,block", [
    ((4096, 12288), 4096),             # Qwen3.5-9B
    ((2048, 6144), 2048),              # Qwen3.5-2B
    ((2560, 4096, 9216), 512),         # Qwen3.5-4B
    ((5120, 6144, 17408), 1024),       # Qwen3.5-27B
    ((2560, 10496), 256),              # Nanbeige4-3B (llama)
    ((8192, 28672), 4096),             # capped at 4096
    ((128, 256), 128),                 # tiny test model
])
def test_global_block_is_largest_power_of_two_dividing_all_widths(widths, block):
    assert derive_block(widths) == block
    geo = derive_geometry(widths, widths[0])
    assert geo.block == block and set(make_signs(geo)) == set(widths)


@pytest.mark.parametrize("widths", [(576, 1536), (4096, 100), (0,)])
def test_unsupported_geometry_is_refused(widths):
    with pytest.raises(UnsupportedGeometry):
        derive_block(widths)


def test_fwht_quantization_agrees_with_dense_path():
    from openbonsai.core.gptq import quantize_linear
    g = torch.Generator().manual_seed(0)
    w = torch.randn(32, 256, generator=g)
    x = torch.randn(1024, 256, generator=g)
    h = x.T @ x / 1024
    signs = seeded_signs(256, 256)
    dense = quantize_linear(w, h, signs, 128, impl="dense")
    fast = quantize_linear(w, h, signs, 128, impl="fwht")
    assert (dense[0] == fast[0]).float().mean() > 0.99
    assert torch.allclose(dense[2], fast[2], atol=2e-3)

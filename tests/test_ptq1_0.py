"""PTQ1_0 block codec against two independent oracles.

- ``spec_pack``: the NumPy packer verified against Prism's
  ``quantize_row_ptq1_0_ref`` (export spec, section 1);
- ``audit_decode``: the decoder of OpenBonsai research/tools/audit_bonsai_ternary.py,
  verified there against Prism's native ``dequantize_row_ptq1_0``.
Both are copied here; tests never import research/.
"""
import itertools

import numpy as np
import pytest

from openbonsai.formats import ptq1_0 as P

# ------------------------------------------------------------- oracles ----
POW = np.array([81, 27, 9, 3, 1], np.int32)


def spec_pack(codes, scales):
    r, w = codes.shape
    d = codes.reshape(-1, 128).astype(np.int32) + 1
    o = np.empty((d.shape[0], 28), np.uint8)
    o[:, 0:16] = (np.einsum('bnm,n->bm', d[:, :80].reshape(-1, 5, 16), POW) * 256 + 242) // 243
    o[:, 16:24] = (np.einsum('bnm,n->bm', d[:, 80:120].reshape(-1, 5, 8), POW) * 256 + 242) // 243
    o[:, 24:26] = (np.einsum('bmh,m->bh', d[:, 120:].reshape(-1, 4, 2), POW[1:]) * 3 * 256 + 242) // 243
    o[:, 26:28] = scales.astype('<f2').reshape(-1, 1).view(np.uint8)
    return o.reshape(r, w // 128 * 28)


POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)
LUT = ((((np.arange(256, dtype=np.uint16)[:, None] * POW3) & 255) * 3) >> 8).astype(np.int8) - 1


def audit_decode(blocks):
    return np.concatenate((LUT[blocks[:, :16]].transpose(0, 2, 1).reshape(-1, 80),
                           LUT[blocks[:, 16:24]].transpose(0, 2, 1).reshape(-1, 40),
                           LUT[blocks[:, 24:26], :4].transpose(0, 2, 1).reshape(-1, 8)), axis=1)


def scale_bits(blocks):
    return blocks[:, 26].astype(np.uint16) | (blocks[:, 27].astype(np.uint16) << 8)
# --------------------------------------------------------- end oracles ----


def random_problem(rows, cols, seed):
    rng = np.random.default_rng(seed)
    codes = rng.integers(-1, 2, size=(rows, cols)).astype(np.int8)
    scales = rng.uniform(1e-4, 2.0, size=(rows, cols // 128)).astype(np.float16)
    return codes, scales


@pytest.mark.parametrize("rows,cols", [(1, 128), (3, 384), (17, 1024)])
def test_round_trip_matches_both_oracles(rows, cols):
    codes, scales = random_problem(rows, cols, rows * cols)
    blocks = P.encode(codes, scales)
    assert blocks.shape == (rows, cols // 128 * 28) and blocks.dtype == np.uint8
    assert np.array_equal(blocks, spec_pack(codes, scales))
    flat = blocks.reshape(-1, 28)
    assert np.array_equal(audit_decode(flat).reshape(rows, cols), codes)
    assert np.array_equal(scale_bits(flat).view(np.float16).reshape(rows, -1), scales)
    c, s = P.decode(blocks, cols)
    assert np.array_equal(c, codes) and np.array_equal(s.view(np.uint16), scales.view(np.uint16))
    deq = P.dequantize(blocks)
    assert np.array_equal(deq, (codes.reshape(rows, -1, 128) * scales.astype(np.float32)[..., None]).reshape(rows, cols))
    assert P.nbytes(rows, cols) == blocks.nbytes


def test_all_zero_all_negative_and_all_positive_blocks():
    s = np.array([[0.5]], np.float16)
    zero = P.encode(np.zeros((1, 128), np.int8), s)[0]
    assert (zero[:24] == 128).all() and (zero[24:26] == 127).all()
    assert (P.encode(-np.ones((1, 128), np.int8), s)[0, :26] == 0).all()
    plus = P.encode(np.ones((1, 128), np.int8), s)[0]
    assert (plus[:24] == 255).all() and (plus[24:26] == 253).all()  # qh: q=80, *3 -> ceil(240*256/243)
    assert np.array_equal(zero[26:], np.array([0.5], "<f2").view(np.uint8))


def test_every_five_and_four_trit_pattern_round_trips():
    patterns = np.array(list(itertools.product((-1, 0, 1), repeat=5)), np.int8)   # 243
    codes = np.zeros((len(patterns), 128), np.int8)
    for n in range(5):
        codes[:, 0 + 16 * n] = patterns[:, n]          # qs[0]
        codes[:, 80 + 8 * n] = patterns[:, n]          # qs[16]
    for m in range(4):
        codes[:, 120 + 2 * m] = patterns[:, m]         # qh[0] (first 81 patterns are all distinct)
    scales = np.full((len(patterns), 1), 1.0, np.float16)
    blocks = P.encode(codes, scales)
    assert len(set(blocks[:, 0].tolist())) == 243 and len(set(blocks[:, 24].tolist())) == 81
    assert np.array_equal(P.decode(blocks)[0], codes)
    assert np.array_equal(audit_decode(blocks), codes)


def test_every_byte_value_decodes_like_the_audit_decoder():
    blocks = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 28, axis=1)
    blocks[:, 26:] = np.array([.5], "<f2").view(np.uint8)
    codes, scales = P.decode(blocks)
    assert np.array_equal(codes, audit_decode(blocks)) and (scales == np.float16(.5)).all()


def test_negative_and_negative_zero_scales_are_normalized():
    codes, scales = random_problem(2, 256, 4)
    scales[0, 1] = -scales[0, 1]
    scales[1, 0] = np.float16(-0.0)
    blocks = P.encode(codes, scales)
    c, s = P.decode(blocks)
    assert not np.signbit(s).any()
    assert np.array_equal(c[0, 128:], -codes[0, 128:]) and np.array_equal(c[0, :128], codes[0, :128])
    assert s.view(np.uint16)[1, 0] == 0
    deq = P.dequantize(blocks)
    ref = (codes.reshape(2, -1, 128) * scales.astype(np.float32)[..., None]).reshape(2, 256)
    assert np.array_equal(deq, ref + 0.0)


@pytest.mark.parametrize("codes,scales,match", [
    (np.zeros((1, 100), np.int8), np.ones((1, 1), np.float16), "multiple of 128"),
    (np.full((1, 128), 2, np.int8), np.ones((1, 1), np.float16), "-1, 0 or"),
    (np.zeros((1, 128), np.int8), np.ones((1, 2), np.float16), "scales must be"),
    (np.zeros((1, 128), np.int8), np.array([[np.nan]], np.float16), "finite"),
])
def test_invalid_inputs_are_rejected(codes, scales, match):
    with pytest.raises(ValueError, match=match):
        P.encode(codes, scales)


def test_decode_rejects_partial_blocks():
    with pytest.raises(ValueError):
        P.decode(np.zeros((1, 30), np.uint8))
    with pytest.raises(ValueError):
        P.decode(np.zeros((1, 56), np.uint8), cols=128)


def test_torch_state_tensors_encode_directly():
    import torch
    codes = torch.randint(-1, 2, (4, 256), dtype=torch.int8, generator=torch.Generator().manual_seed(0))
    scales = torch.rand(4, 2, generator=torch.Generator().manual_seed(1)).half()
    c, s = P.decode(P.encode(codes, scales))
    assert np.array_equal(c, codes.numpy()) and np.array_equal(s, scales.numpy())

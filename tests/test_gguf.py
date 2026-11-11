"""GGUF container and the PTQ1_0 writer codec against the research exporter's packer.

The oracle below is a verbatim copy of OpenBonsai research/experiments/export_gguf.py's
``pack_ptq1_0``, ``normalize_scale_signs`` and ``_c_ref_quant_block`` (a port
of Prism's ``quantize_row_ptq1_0_ref``). Tests never import research/.
"""
import math
import struct

import numpy as np
import pytest
import torch

from openbonsai.formats import gguf as G
from openbonsai.formats import ptq1_0 as P

# ---------------------------------------------------------------- oracle ----
QK, BLOCK_BYTES = 128, 28


def pack_ptq1_0(codes: np.ndarray, sbits: np.ndarray) -> np.ndarray:
    rows, width = codes.shape
    assert width % QK == 0 and sbits.shape == (rows, width // QK)
    nb = rows * (width // QK)
    d = (codes.reshape(nb, QK) + np.int8(1)).astype(np.uint16)
    out = np.empty((nb, BLOCK_BYTES), np.uint8)
    q = 81 * d[:, 0:16] + 27 * d[:, 16:32] + 9 * d[:, 32:48] + 3 * d[:, 48:64] + d[:, 64:80]
    out[:, 0:16] = (q * 256 + 242) // 243
    q = 81 * d[:, 80:88] + 27 * d[:, 88:96] + 9 * d[:, 96:104] + 3 * d[:, 104:112] + d[:, 112:120]
    out[:, 16:24] = (q * 256 + 242) // 243
    q = (27 * d[:, 120:122] + 9 * d[:, 122:124] + 3 * d[:, 124:126] + d[:, 126:128]) * 3
    out[:, 24:26] = (q * 256 + 242) // 243
    out[:, 26:28] = np.ascontiguousarray(sbits, dtype="<u2").reshape(nb, 1).view(np.uint8)
    return out.reshape(rows, (width // QK) * BLOCK_BYTES)


def normalize_scale_signs(codes: np.ndarray, sbits: np.ndarray):
    rows, g = sbits.shape
    cv = codes.reshape(rows, g, QK)
    assert np.shares_memory(cv, codes), "codes must be contiguous (in-place normalization)"
    neg = (sbits & 0x8000) != 0
    n_neg = int(neg.sum())
    if n_neg:
        cv[neg] = -cv[neg]
        sbits &= 0x7FFF
    dead = (sbits == 0) & cv.any(-1)
    n_dead = int(dead.sum())
    if n_dead:
        cv[dead] = 0
    return n_neg, n_dead


def _c_ref_quant_block(x):
    x = x.astype(np.float32)
    d = np.float32(np.max(np.abs(x)))
    idv = np.float32(1.0) / d if d else np.float32(0.0)

    def lround(v):
        v = np.float32(v)
        return int(math.floor(abs(float(v)) + 0.5)) * (1 if v >= 0 else -1)

    qs, p, j = [0] * 24, 0, 0
    for c in (32, 16, 8):
        while j + c <= 24:
            for m in range(c):
                q = 0
                for n in range(5):
                    q = (q * 3 + lround(x[p + m + n * c] * idv) + 1) & 0xFF
                qs[j + m] = ((q * 256 + 242) // 243) & 0xFF
            p += 5 * c
            j += c
    qh = []
    for h in range(2):
        q = 0
        for m in range(4):
            q = (q * 3 + lround(x[p + h + m * 2] * idv) + 1) & 0xFF
        q = (q * 3) & 0xFF
        qh.append(((q * 256 + 242) // 243) & 0xFF)
    return bytes(qs + qh) + np.float16(d).tobytes()
# ------------------------------------------------------------ end oracle ----


def edge_case_problem(rows=12, cols=512, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(-1, 2, size=(rows, cols)).astype(np.int8)
    scales = rng.uniform(1e-3, 0.5, size=(rows, cols // 128)).astype(np.float16)
    codes[0, :128] = 0                       # all-zero group, nonzero scale
    scales[1, 1] = -scales[1, 1]             # negative scale
    scales[2, 2] = np.float16(-0.0)          # -0.0 with live codes
    scales[3, 0] = np.float16(0.0)           # +0 with live codes
    return codes, scales


def test_writer_codec_is_the_research_exporters_bytes():
    codes, scales = edge_case_problem()
    want_c, want_s = codes.copy(), scales.view(np.uint16).copy()
    assert normalize_scale_signs(want_c, want_s) == (2, 2)
    want = pack_ptq1_0(want_c, want_s)
    assert np.array_equal(P.encode(codes, scales), want)
    got_c, got_s = codes.copy(), scales.view(np.uint16).copy()
    assert P.canonicalize(got_c, got_s) == (2, 2)
    assert np.array_equal(P.pack(got_c, got_s), want)
    # the canonical bytes are quantize_row_ptq1_0_ref's for the same function (nonzero groups),
    # and its all-zero block for the dead groups
    deq = (codes.reshape(12, -1, 128).astype(np.float32) * scales.astype(np.float32)[..., None]).reshape(12, 512)
    for r in range(4):
        for g in range(4):
            block = bytes(want[r, g * 28:(g + 1) * 28])
            ref = _c_ref_quant_block(deq[r, g * 128:(g + 1) * 128])
            if want_c[r, g * 128:(g + 1) * 128].any():
                assert block == ref, (r, g)
            else:
                assert block[:26] == ref[:26], (r, g)
    c, s = P.decode(want)
    assert np.array_equal(c, want_c) and np.array_equal(s.view(np.uint16), want_s)


def test_every_kv_type_and_tensor_kind_round_trips(tmp_path):
    kvs = [("a.u8", "u8", 7), ("a.i8", "i8", -7), ("a.u16", "u16", 65535), ("a.i16", "i16", -3),
           ("a.u32", "u32", 2 ** 32 - 1), ("a.i32", "i32", -2 ** 31), ("a.f32", "f32", 0.95),
           ("a.bool", "bool", True), ("a.str", "str", "héllo \U0001f680"), ("a.u64", "u64", 2 ** 63),
           ("a.i64", "i64", -5), ("a.f64", "f64", math.pi), ("a.ai32", "arr[i32]", [1, -1, 3]),
           ("a.astr", "arr[str]", ["x", "", "Ġ Ġ"]), ("a.af32", "arr[f32]", [0.5, -2.0]),
           ("a.abool", "arr[bool]", [True, False]), ("a.empty", "arr[str]", [])]
    codes = torch.randint(-1, 2, (3, 256), dtype=torch.int8, generator=torch.Generator().manual_seed(0))
    scales = torch.rand(3, 2, generator=torch.Generator().manual_seed(1)).half()
    f = np.arange(12, dtype=np.float32).reshape(3, 4)
    infos = [G.TensorInfo("t.ptq", G.GGML_PTQ1_0, (256, 3)), G.TensorInfo("t.f32", G.GGML_F32, (4, 3)),
             G.TensorInfo("t.vec", G.GGML_F32, (5,))]
    size = G.layout(infos)
    assert [t.offset for t in infos] == [0, 192, 256] and size == 288
    payload = {"t.ptq": P.encode(codes, scales), "t.f32": f, "t.vec": np.ones(5, np.float32)}
    path = tmp_path / "x.gguf"
    G.write_gguf(path, kvs, infos, lambda t: [payload[t.name]])
    with G.GGUFReader(path) as r:
        assert [(k, t) for k, t, _ in r.kv] == [(k, t) for k, t, _ in kvs]
        for (k, t, v), (_, _, got) in zip(kvs, r.kv):
            if t in ("f32", "arr[f32]"):
                assert np.array_equal(np.float32(v), np.float32(got)), k
            else:
                assert got == v, k
        assert r.data_start % 32 == 0 and r.size == r.data_start + 288
        c, s = r.ptq1_0("t.ptq")
        assert np.array_equal(c, codes.numpy()) and np.array_equal(s, scales.numpy())
        c1, s1 = r.ptq1_0("t.ptq", 1, 3, chunk_rows=1)
        assert np.array_equal(c1, codes.numpy()[1:]) and np.array_equal(s1, scales.numpy()[1:])
        assert np.array_equal(r.array("t.f32"), f) and r.array("t.vec").shape == (5,)
        with pytest.raises(G.GGUFError, match="not PTQ1_0"):
            r.ptq1_0("t.f32")
        with pytest.raises(G.GGUFError, match="no tensor"):
            r.read("nope")
    path.replace(tmp_path / "renamed.gguf")     # the reader holds no handle once closed


def test_writer_checks_payload_sizes_and_reader_refuses_other_files(tmp_path):
    infos = [G.TensorInfo("t", G.GGML_F32, (4,))]
    G.layout(infos)
    with pytest.raises(G.GGUFError, match="wrote 12 B"):
        G.write_gguf(tmp_path / "bad.gguf", [], infos, lambda t: [np.zeros(3, np.float32)])
    (tmp_path / "not.gguf").write_bytes(b"GGML" + bytes(64))
    with pytest.raises(G.GGUFError, match="not a GGUF"):
        G.GGUFReader(tmp_path / "not.gguf")
    (tmp_path / "v2.gguf").write_bytes(b"GGUF" + struct.pack("<IQQ", 2, 0, 0))
    with pytest.raises(G.GGUFError, match="version 2"):
        G.GGUFReader(tmp_path / "v2.gguf")


def test_reader_refuses_unknown_value_types_and_rows_outside_a_tensor(tmp_path):
    kvs = [("general.architecture", "str", "x"), ("a.list", "arr[i32]", [1, 2, 3])]
    infos = [G.TensorInfo("w", G.GGML_F32, (4, 2)), G.TensorInfo("v", G.GGML_F32, (4, 2))]
    G.layout(infos)
    path = tmp_path / "t.gguf"
    G.write_gguf(path, kvs, infos, lambda t: [np.arange(8, dtype="<f4")])
    raw = path.read_bytes()
    for key, at in (("general.architecture", 0), ("a.list", 4)):   # scalar type id, array element type id
        bad = bytearray(raw)
        i = raw.index(key.encode()) + len(key) + at
        bad[i:i + 4] = struct.pack("<I", 13)
        (tmp_path / "bad.gguf").write_bytes(bad)
        with pytest.raises(G.GGUFError, match="unknown GGUF value type 13"):
            G.GGUFReader(tmp_path / "bad.gguf")
    with G.GGUFReader(path) as r:
        assert r.read("w", 1, 2).shape == (1, 16) and r.read("w", 2, 2).shape == (0, 16)
        for r0, r1 in ((0, 3), (-1, 1), (2, 1)):   # "w" is not last: rows 2.. would be "v"'s bytes
            with pytest.raises(G.GGUFError, match="outside"):
                r.read("w", r0, r1)


def test_f32_transforms_invert():
    x = torch.randn(6, 1, 4)
    assert torch.equal(G.f32_inverse("squeeze", G.f32_forward("squeeze", x), (6, 1, 4)), x)
    w = torch.randn(8).to(torch.bfloat16)
    assert torch.equal(G.f32_inverse("plus1", G.f32_forward("plus1", w), (8,)).to(torch.bfloat16), w)
    a = torch.randn(8)
    assert torch.allclose(G.f32_inverse("neg_exp", G.f32_forward("neg_exp", a), (8,)), a, atol=1e-6)
    with pytest.raises(G.GGUFError, match="non-negative"):
        G.f32_inverse("neg_exp", torch.ones(3), (3,))
    with pytest.raises(G.GGUFError, match="unknown"):
        G.f32_forward("log", a)

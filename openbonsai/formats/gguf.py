"""Minimal GGUF v3 container: a streaming writer and a reader (no gguf-py dependency).

Only what Prism's PTQ1_0 files need: typed KVs (scalars, strings, arrays),
tensor infos with ``ne`` in GGUF order (``ne[0]`` = the contiguous input
width), alignment 32 (no ``general.alignment`` key), and F32 / F16 / BF16 /
PTQ1_0 (type 143) payloads. The writer streams each tensor's bytes from a producer,
so a 2 GB file never sits in memory; the reader parses the header and reads
tensor rows on demand through an ordinary file handle (no memory map, so the
file can be renamed or replaced as soon as the reader is closed).

Architecture knowledge is not here: an adapter describes its layout as a
list of ``TensorMap`` entries (GGUF name, type, ``ne``, the state key or
text-model tensor it comes from, an F32 transform and a row permutation),
and this module applies it in both directions (``f32_forward`` /
``f32_inverse``). The Prism runtime metadata (``prism.hadamard.*``) is read
by ``prism_signs``.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct
import time

import numpy as np

from . import ptq1_0

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGN = 32
GGML_F32 = 0
GGML_F16 = 1
GGML_BF16 = 30
FLOAT_TYPES = {GGML_F32: "<f4", GGML_F16: "<f2", GGML_BF16: "<u2"}   # numpy element of each float payload
GGML_PTQ1_0 = ptq1_0.GGML_TYPE_PTQ1_0

VT = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6, "bool": 7, "str": 8, "arr": 9,
      "u64": 10, "i64": 11, "f64": 12}
VT_NAME = {v: k for k, v in VT.items()}
SCALAR_FMT = {"u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I", "i32": "<i", "f32": "<f",
              "bool": "<?", "u64": "<Q", "i64": "<q", "f64": "<d"}
NP_ELEM = {"u8": "<u1", "i8": "<i1", "u16": "<u2", "i16": "<i2", "u32": "<u4", "i32": "<i4", "f32": "<f4",
           "bool": "<u1", "u64": "<u8", "i64": "<i8", "f64": "<f8"}
F32_OPS = ("copy", "plus1", "neg_exp", "squeeze", "const")


class GGUFError(ValueError):
    """A file or layout that does not match what the reader or writer expects."""


def align_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def tensor_nbytes(gtype: int, ne) -> int:
    n = int(np.prod(ne))
    if gtype == GGML_PTQ1_0:
        if ne[0] % ptq1_0.QK:
            raise GGUFError(f"PTQ1_0 rows must be a multiple of {ptq1_0.QK} wide, got ne {tuple(ne)}")
        return n // ptq1_0.QK * ptq1_0.BLOCK_BYTES
    if gtype == GGML_F32:
        return 4 * n
    if gtype in (GGML_F16, GGML_BF16):
        return 2 * n
    raise GGUFError(f"unsupported ggml type {gtype}")


@dataclass(eq=False)
class TensorInfo:
    name: str
    gtype: int
    ne: tuple[int, ...]
    offset: int = 0        # relative to the start of the data section

    @property
    def nbytes(self) -> int:
        return tensor_nbytes(self.gtype, self.ne)

    @property
    def rows(self) -> int:
        return int(np.prod(self.ne[1:])) if len(self.ne) > 1 else 1

    @property
    def row_bytes(self) -> int:
        return self.nbytes // self.rows


@dataclass(eq=False)
class TensorMap:
    """One GGUF tensor and where it comes from (an adapter's layout entry).

    ``source``: for PTQ1_0, the state key (``layers.{i}.{sub}``, ``lm_head``,
    ``embed_tokens``); for F32, the tensor name relative to the text model
    (``layers.0.linear_attn.A_log``, ``norm.weight``). ``perm`` reorders dim 0
    (rows / channels / heads): GGUF row i holds source row ``perm[i]``.
    ``op`` transforms an F32 source (see ``f32_forward``); ``shape`` is the
    source tensor's shape. ``role`` names the Prism Hadamard list the tensor
    belongs to: ``fold`` (weight_names) or ``inverse`` (inverse_weight_names).
    A float tensor may be F32, F16 or BF16 (a high-precision embedding table
    in the checkpoint's own type). ``data``: a generated constant (op
    ``const``, ``source`` empty), e.g. llama3 ``rope_freqs``.
    """
    name: str
    gtype: int
    ne: tuple[int, ...]
    source: str
    op: str = "copy"
    perm: np.ndarray | None = None
    role: str = ""
    shape: tuple[int, ...] | None = None
    data: np.ndarray | None = None

    @property
    def ternary(self) -> bool:
        return self.gtype == GGML_PTQ1_0

    def info(self) -> TensorInfo:
        return TensorInfo(self.name, self.gtype, tuple(int(x) for x in self.ne))

    def inverse_perm(self) -> np.ndarray | None:
        return None if self.perm is None else np.argsort(self.perm, kind="stable")


def layout(infos: list[TensorInfo]) -> int:
    """Assign aligned data offsets in order; returns the data section size."""
    off = 0
    for t in infos:
        t.offset = off
        off = align_up(off + t.nbytes)
    return off


# -- F32 transforms -------------------------------------------------------

def f32_forward(op: str, x):
    """Source tensor (torch, any float dtype) -> the FP32 tensor GGUF stores (before ``perm``).

    copy; plus1 = 1 + w (zero-centered norms); neg_exp = -exp(x) (A_log -> A);
    squeeze = [C, 1, K] -> [C, K] (depthwise conv); const = a generated tensor, as is. FP32
    arithmetic, as Prism's converter."""
    import torch
    x = x.to(torch.float32)
    if op in ("copy", "const"):
        return x
    if op == "plus1":
        return x + 1
    if op == "neg_exp":
        return -torch.exp(x)
    if op == "squeeze":
        return x.reshape(x.shape[0], -1)
    raise GGUFError(f"unknown F32 op {op!r}")


def f32_inverse(op: str, x, shape):
    """GGUF FP32 tensor (after undoing ``perm``) -> the source convention, FP32, ``shape``."""
    import torch
    x = x.to(torch.float32)
    if op == "plus1":
        x = x - 1
    elif op == "neg_exp":
        if bool((x >= 0).any()):
            raise GGUFError("neg_exp tensor holds non-negative values; expected -exp(A_log) < 0")
        x = torch.log(-x.double()).float()
    elif op not in ("copy", "squeeze", "const"):
        raise GGUFError(f"unknown F32 op {op!r}")
    return x.reshape(tuple(shape))


# -- writer -----------------------------------------------------------------

def _enc_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _enc_value(typ: str, val) -> bytes:
    if typ == "str":
        return _enc_str(val)
    if typ.startswith("arr["):
        et = typ[4:-1]
        head = struct.pack("<IQ", VT[et], len(val))
        if et == "str":
            return head + b"".join(_enc_str(s) for s in val)
        return head + np.asarray(val, dtype=NP_ELEM[et]).tobytes()
    return struct.pack(SCALAR_FMT[typ], val)


def encode_kv(key: str, typ: str, val) -> bytes:
    vt = VT["arr"] if typ.startswith("arr[") else VT[typ]
    return _enc_str(key) + struct.pack("<I", vt) + _enc_value(typ, val)


def header_bytes(kvs, infos) -> bytes:
    """Magic, version, counts, KVs and tensor infos (offsets must be assigned)."""
    parts = [GGUF_MAGIC + struct.pack("<IQQ", GGUF_VERSION, len(infos), len(kvs))]
    parts += [encode_kv(k, t, v) for k, t, v in kvs]
    for t in infos:
        parts.append(_enc_str(t.name) + struct.pack("<I", len(t.ne)) + struct.pack(f"<{len(t.ne)}Q", *t.ne)
                     + struct.pack("<IQ", t.gtype, t.offset))
    return b"".join(parts)


def write_gguf(path, kvs, infos, produce, *, log=None) -> dict:
    """Write ``path`` (not atomically: the caller writes a temp name and renames).

    ``produce(info)`` yields numpy arrays whose bytes, concatenated, are the
    tensor's payload. Offsets come from ``layout``; every payload size is
    checked, padding is zeros, and the file is fsynced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(path, "wb", buffering=32 << 20) as f:
        f.write(header_bytes(kvs, infos))
        header_end = f.tell()
        f.write(b"\0" * (align_up(header_end) - header_end))
        data_start = f.tell()
        for i, t in enumerate(infos):
            if f.tell() - data_start != t.offset:
                raise GGUFError(f"{t.name}: data offset {f.tell() - data_start} != planned {t.offset}")
            n = 0
            for chunk in produce(t):
                buf = np.ascontiguousarray(chunk).reshape(-1).view(np.uint8)
                f.write(buf)
                n += buf.nbytes
            if n != t.nbytes:
                raise GGUFError(f"{t.name}: wrote {n} B, expected {t.nbytes}")
            f.write(b"\0" * (align_up(n) - n))
            if log is not None and (t.nbytes > 50_000_000 or i % 64 == 0):
                log(f"  [{i + 1}/{len(infos)}] {t.name} {t.nbytes / 1e6:.1f} MB t={time.time() - t0:.0f}s")
        f.flush()
        os.fsync(f.fileno())
        end = f.tell()
    return {"header_end": header_end, "data_start": data_start, "file_size": end, "seconds": time.time() - t0}


# -- reader -----------------------------------------------------------------

class GGUFReader:
    """Parse a GGUF header; read tensor rows on demand. Use as a context manager."""

    def __init__(self, path):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        self._f = open(self.path, "rb")
        want = 16 << 20
        while True:
            self._f.seek(0)
            self.buf = self._f.read(min(want, self.size))
            try:
                self._parse()
                break
            except (struct.error, IndexError, ValueError) as error:
                if isinstance(error, GGUFError) or want >= self.size:
                    self.close()
                    raise
                want *= 4
        del self.buf
        self.by_name = {t.name: t for t in self.tensors}
        if len(self.by_name) != len(self.tensors):
            self.close()
            raise GGUFError(f"{self.path}: duplicate tensor names")

    # header parsing ----------------------------------------------------
    def _u(self, fmt):
        v = struct.unpack_from(fmt, self.buf, self.pos)[0]
        self.pos += struct.calcsize(fmt)
        return v

    def _s(self):
        n = self._u("<Q")
        b = self.buf[self.pos:self.pos + n]
        if len(b) != n:
            raise ValueError("short read")
        self.pos += n
        return b.decode("utf-8")

    def _type(self, vt) -> str:
        if vt not in VT_NAME:
            raise GGUFError(f"{self.path}: unknown GGUF value type {vt} at byte {self.pos}")
        return VT_NAME[vt]

    def _val(self, vt):
        if vt == VT["str"]:
            return "str", self._s()
        if vt == VT["arr"]:
            et, n = self._u("<I"), self._u("<Q")
            name = self._type(et)
            if name == "str":
                return "arr[str]", [self._s() for _ in range(n)]
            if name == "arr":
                raise GGUFError("nested GGUF arrays are not supported")
            dt = np.dtype(NP_ELEM[name])
            if self.pos + n * dt.itemsize > len(self.buf):
                raise ValueError("short read")
            a = np.frombuffer(self.buf, dtype=dt, count=n, offset=self.pos)
            self.pos += n * dt.itemsize
            return f"arr[{name}]", (a.astype(bool) if name == "bool" else a).tolist()
        name = self._type(vt)
        return name, self._u(SCALAR_FMT[name])

    def _parse(self):
        if self.buf[:4] != GGUF_MAGIC:
            raise GGUFError(f"{self.path}: not a GGUF file")
        self.pos = 4
        self.version, self.n_tensors, self.n_kv = self._u("<I"), self._u("<Q"), self._u("<Q")
        if self.version != GGUF_VERSION:
            raise GGUFError(f"{self.path}: GGUF version {self.version}, expected {GGUF_VERSION}")
        self.kv = []
        for _ in range(self.n_kv):
            key = self._s()
            typ, val = self._val(self._u("<I"))
            self.kv.append((key, typ, val))
        self.kvd = {k: (t, v) for k, t, v in self.kv}
        self.tensors = []
        for _ in range(self.n_tensors):
            name = self._s()
            nd = self._u("<I")
            ne = tuple(self._u("<Q") for _ in range(nd))
            gtype, offset = self._u("<I"), self._u("<Q")
            self.tensors.append(TensorInfo(name, gtype, ne, offset))
        self.header_end = self.pos
        self.alignment = int(self.kvd.get("general.alignment", ("u32", ALIGN))[1])
        self.data_start = align_up(self.header_end, self.alignment)

    # access ------------------------------------------------------------
    def value(self, key, default=None):
        return self.kvd[key][1] if key in self.kvd else default

    def info(self, name) -> TensorInfo:
        if name not in self.by_name:
            raise GGUFError(f"{self.path}: no tensor {name!r}")
        return self.by_name[name]

    def read(self, name, r0: int = 0, r1: int | None = None) -> np.ndarray:
        """Raw payload rows r0..r1 as uint8 [rows, row_bytes]."""
        t = self.info(name)
        r1 = t.rows if r1 is None else r1
        if not 0 <= r0 <= r1 <= t.rows:
            raise GGUFError(f"{name}: rows {r0}..{r1} outside 0..{t.rows}")
        rb = t.row_bytes
        start = self.data_start + t.offset + r0 * rb
        n = (r1 - r0) * rb
        if start + n > self.size:
            raise GGUFError(f"{name}: payload runs past the end of the file")
        self._f.seek(start)
        data = self._f.read(n)
        if len(data) != n:
            raise GGUFError(f"{name}: short read")
        return np.frombuffer(data, np.uint8).reshape(r1 - r0, rb)

    def array(self, name) -> np.ndarray:
        """An F32/F16/BF16 tensor as float32 (exact) with numpy shape ``reversed(ne)``."""
        t = self.info(name)
        if t.gtype not in FLOAT_TYPES:
            raise GGUFError(f"{name}: type {t.gtype} is not a float tensor")
        raw = self.read(name).reshape(-1).view(FLOAT_TYPES[t.gtype])
        a = (raw.astype(np.uint32) << 16).view("<f4") if t.gtype == GGML_BF16 else raw.astype(np.float32)
        return a.reshape(tuple(reversed(t.ne)))

    def ptq1_0(self, name, r0: int = 0, r1: int | None = None, chunk_rows: int = 16384):
        """Decode PTQ1_0 rows -> (codes int8 [rows, ne0], scales fp16 [rows, ne0/128])."""
        t = self.info(name)
        if t.gtype != GGML_PTQ1_0:
            raise GGUFError(f"{name}: type {t.gtype} is not PTQ1_0")
        r1 = t.rows if r1 is None else r1
        if not 0 <= r0 <= r1 <= t.rows:
            raise GGUFError(f"{name}: rows {r0}..{r1} outside 0..{t.rows}")
        codes = np.empty((r1 - r0, t.ne[0]), np.int8)
        scales = np.empty((r1 - r0, t.ne[0] // ptq1_0.QK), np.float16)
        for a in range(r0, r1, chunk_rows):
            b = min(r1, a + chunk_rows)
            codes[a - r0:b - r0], scales[a - r0:b - r0] = ptq1_0.decode(self.read(name, a, b), t.ne[0])
        return codes, scales

    def close(self):
        f, self._f = getattr(self, "_f", None), None
        if f is not None:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# -- Prism runtime metadata ----------------------------------------------------

def prism_signs(reader: GGUFReader) -> dict[int, np.ndarray]:
    """``prism.hadamard.sign_widths`` / ``sign_values`` -> int8 sign vector per width."""
    widths = reader.value("prism.hadamard.sign_widths")
    values = reader.value("prism.hadamard.sign_values")
    if widths is None or values is None:
        raise GGUFError(f"{reader.path}: no prism.hadamard sign table (not a Prism PTQ1_0 file?)")
    if sum(widths) != len(values) or not set(values) <= {-1, 1}:
        raise GGUFError(f"{reader.path}: malformed prism.hadamard sign table")
    out, off = {}, 0
    for w in widths:
        out[int(w)] = np.asarray(values[off:off + int(w)], np.int8)
        off += int(w)
    return out


def float_payload(x, gtype: int) -> np.ndarray:
    """A float tensor (torch, FP32 values) as the little-endian payload elements of ``gtype``
    (F32, F16, or BF16 rounded to nearest even as torch rounds)."""
    import torch
    if gtype == GGML_F32:
        return np.ascontiguousarray(x.to(torch.float32).numpy(), dtype="<f4")
    if gtype == GGML_F16:
        return np.ascontiguousarray(x.to(torch.float16).numpy(), dtype="<f2")
    if gtype == GGML_BF16:
        return np.ascontiguousarray(x.to(torch.bfloat16).view(torch.int16).numpy()).view("<u2")
    raise GGUFError(f"type {gtype} is not a float type")

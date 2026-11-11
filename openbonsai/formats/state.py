"""Ternary run state: codes, fp16 group scales, shared signs per width.

One safetensors file per state (or per resumable unit, e.g. one decoder
layer). Tensors:

- ``codes::<key>``  uint8 [ceil(rows*cols/4)] (2-bit, formats/packing.py) or int8 [rows, cols]
- ``scales::<key>`` fp16 [rows, cols/group]
- ``signs::<width>`` int8 [width], one vector per rotated input width

Keys are relative module names: ``layers.{i}.{sub}``, ``lm_head`` and
``embed_tokens``. The safetensors header holds one metadata entry,
``openbonsai``: canonical JSON with the format id, block, group, code
encoding, shapes and a ``meta`` dict (one entry, because safetensors writes
its metadata map in hash order). Nothing in a state depends on time or host,
so equal inputs give byte-identical files.

Old ``.pt`` states from research/experiments load read-only through
``load_legacy_pt`` (for evaluation). Only a legacy state whose signs are
shared per width converts to this format; per-module signs cannot be
expressed in Prism's layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor

from .atomic import atomic_write_bytes
from .packing import pack_ternary_codes, unpack_ternary_codes

STATE_FORMAT = "openbonsai.ternary-state.v1"
SCALES_FORMAT = "openbonsai.ternary-scales.v1"
SIGNS_FORMAT = "openbonsai.signs.v1"


@dataclass
class TernaryTensor:
    codes: Tensor   # int8 {-1, 0, 1} [rows, cols], rotated basis
    scales: Tensor  # fp16 [rows, cols // group]

    def __post_init__(self):
        if self.codes.ndim != 2 or self.codes.dtype != torch.int8:
            raise ValueError("codes must be a 2-D int8 tensor")
        if self.scales.ndim != 2 or self.scales.dtype != torch.float16:
            raise ValueError("scales must be a 2-D fp16 tensor")
        rows, cols = self.codes.shape
        if self.scales.shape[0] != rows or not self.scales.shape[1] or cols % self.scales.shape[1]:
            raise ValueError(f"scales {tuple(self.scales.shape)} do not tile codes {tuple(self.codes.shape)}")

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.codes.shape)

    @property
    def group(self) -> int:
        return self.codes.shape[1] // self.scales.shape[1]


@dataclass
class TernaryState:
    tensors: dict[str, TernaryTensor]
    signs: dict[int, Tensor]
    block: int
    group: int
    meta: dict = field(default_factory=dict)

    def validate(self) -> "TernaryState":
        if type(self.block) is not int or self.block <= 0 or self.block & (self.block - 1):
            raise ValueError("block must be a positive power of two")
        for w, s in self.signs.items():
            if s.dtype != torch.int8 or s.ndim != 1 or s.numel() != w or w % self.block:
                raise ValueError(f"invalid sign vector for width {w}")
            if not bool(((s == 1) | (s == -1)).all()):
                raise ValueError(f"sign vector for width {w} is not +/-1")
        for key, t in self.tensors.items():
            if t.group != self.group:
                raise ValueError(f"{key}: group {t.group} != state group {self.group}")
            if t.shape[1] not in self.signs:
                raise ValueError(f"{key}: no sign vector for width {t.shape[1]}")
        return self

    def merged(self, other: "TernaryState") -> "TernaryState":
        if self.block != other.block or self.group != other.group:
            raise ValueError("cannot merge states with different block or group")
        for w in set(self.signs) & set(other.signs):
            if not torch.equal(self.signs[w], other.signs[w]):
                raise ValueError(f"cannot merge states with different signs for width {w}")
        overlap = set(self.tensors) & set(other.tensors)
        if overlap:
            raise ValueError(f"states overlap on {sorted(overlap)[:3]}")
        return TernaryState({**self.tensors, **other.tensors}, {**self.signs, **other.signs},
                            self.block, self.group, {**self.meta, **other.meta}).validate()

    def with_scales(self, scales: dict[str, Tensor]) -> "TernaryState":
        unknown = set(scales) - set(self.tensors)
        if unknown:
            raise ValueError(f"scales for unknown tensors {sorted(unknown)[:3]}")
        for k, v in scales.items():
            if tuple(v.shape) != tuple(self.tensors[k].scales.shape):
                raise ValueError(f"{k}: scales {tuple(v.shape)} do not match {tuple(self.tensors[k].scales.shape)}")
        tensors = {k: TernaryTensor(t.codes, scales[k].to(torch.float16).contiguous()) if k in scales else t
                   for k, t in self.tensors.items()}
        return TernaryState(tensors, self.signs, self.block, self.group, dict(self.meta)).validate()

    def records(self) -> Iterator[tuple[str, TernaryTensor, Tensor, int]]:
        """(key, tensor, signs for its width, block), sorted by key."""
        for key in sorted(self.tensors):
            t = self.tensors[key]
            yield key, t, self.signs[t.shape[1]], self.block


def _metadata(**fields) -> dict[str, str]:
    return {"openbonsai": json.dumps(fields, sort_keys=True, separators=(",", ":"))}


def _read_metadata(f, expected: str, path) -> dict:
    raw = (f.metadata() or {}).get("openbonsai")
    md = json.loads(raw) if raw else {}
    if md.get("format") != expected:
        raise ValueError(f"{path}: not an {expected} file")
    return md


def _sorted_keys(keys):
    def order(k):
        parts = k.split(".")
        return [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in parts]
    return sorted(keys, key=order)


def state_to_bytes(state: TernaryState, *, packed: bool = True) -> bytes:
    from safetensors.torch import save
    state.validate()
    tensors, shapes = {}, {}
    for key in _sorted_keys(state.tensors):
        t = state.tensors[key]
        shapes[key] = list(t.shape)
        codes = t.codes.detach().cpu().contiguous()
        if packed:
            codes = pack_ternary_codes(codes.reshape(-1)).contiguous()
        tensors[f"codes::{key}"] = codes
        tensors[f"scales::{key}"] = t.scales.detach().cpu().contiguous()
    for w in sorted(state.signs):
        tensors[f"signs::{w}"] = state.signs[w].detach().cpu().contiguous()
    return save(tensors, _metadata(format=STATE_FORMAT, block=state.block, group=state.group,
                                   codes="2bit" if packed else "int8", shapes=shapes, meta=state.meta))


def save_state(path, state: TernaryState, *, packed: bool = True) -> Path:
    return atomic_write_bytes(path, state_to_bytes(state, packed=packed))


def load_state(path) -> TernaryState:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        md = _read_metadata(f, STATE_FORMAT, path)
        shapes = md["shapes"]
        packed = md["codes"] == "2bit"
        tensors = {}
        for key, (rows, cols) in shapes.items():
            codes = f.get_tensor(f"codes::{key}")
            if packed:
                codes = unpack_ternary_codes(codes.reshape(-1), rows * cols).reshape(rows, cols)
            tensors[key] = TernaryTensor(codes.to(torch.int8).contiguous(), f.get_tensor(f"scales::{key}"))
        signs = {int(name.split("::", 1)[1]): f.get_tensor(name)
                 for name in f.keys() if name.startswith("signs::")}
        return TernaryState(tensors, signs, int(md["block"]), int(md["group"]), md["meta"]).validate()


def save_signs(path, signs: dict[int, Tensor], *, block: int, meta: dict | None = None) -> Path:
    from safetensors.torch import save
    tensors = {f"signs::{w}": signs[w].to(torch.int8).contiguous() for w in sorted(signs)}
    return atomic_write_bytes(path, save(tensors, _metadata(format=SIGNS_FORMAT, block=block, meta=meta or {})))


def read_signs(path) -> tuple[dict[int, Tensor], int, dict]:
    """(signs per width, block, meta)."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        md = _read_metadata(f, SIGNS_FORMAT, path)
        return {int(n.split("::", 1)[1]): f.get_tensor(n) for n in f.keys()}, int(md["block"]), md.get("meta") or {}


def load_signs(path) -> tuple[dict[int, Tensor], int]:
    signs, block, _ = read_signs(path)
    return signs, block


def save_scales(path, scales: dict[str, Tensor], meta: dict | None = None,
                small: dict[str, Tensor] | None = None, small_dtypes: dict[str, str] | None = None) -> Path:
    """Distilled fp16 scales; optionally the trained high-precision tensors (``small::<name>``, FP32
    masters, names relative to the text model) with the dtype each was computed in."""
    from safetensors.torch import save
    tensors = {f"scales::{k}": scales[k].to(torch.float16).contiguous() for k in _sorted_keys(scales)}
    small = small or {}
    dtypes = dict(small_dtypes or {})
    if set(dtypes) != set(small):
        raise ValueError("small_dtypes must name the compute dtype of every small tensor")
    for k in _sorted_keys(small):
        tensors[f"small::{k}"] = small[k].detach().to(torch.float32).cpu().contiguous()
    fields = {"format": SCALES_FORMAT, "meta": meta or {}}
    if small:
        fields["small_dtypes"] = {k: dtypes[k] for k in _sorted_keys(dtypes)}
    return atomic_write_bytes(path, save(tensors, _metadata(**fields)))


def read_scales(path) -> tuple[dict[str, Tensor], dict]:
    """(scales per key, meta)."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        md = _read_metadata(f, SCALES_FORMAT, path)
        return ({n.split("::", 1)[1]: f.get_tensor(n) for n in f.keys() if n.startswith("scales::")},
                md.get("meta") or {})


def read_small(path, *, cast: bool = True) -> dict[str, Tensor]:
    """Trained high-precision tensors saved with the scales (empty if none).

    ``cast``: FP32 values of what the model computed with (the master rounded
    to its compute dtype, e.g. BF16); False returns the FP32 masters."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        md = _read_metadata(f, SCALES_FORMAT, path)
        dtypes = md.get("small_dtypes") or {}
        out = {}
        for n in f.keys():
            if n.startswith("small::"):
                key = n.split("::", 1)[1]
                t = f.get_tensor(n)
                out[key] = t.to(getattr(torch, dtypes[key])).to(torch.float32) if cast else t
        return out


def read_small_masters(path) -> tuple[dict[str, Tensor], dict[str, str]]:
    """(FP32 masters, the dtype each was computed in) of the trained high-precision tensors, as
    ``save_scales`` takes them (``small``, ``small_dtypes``)."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        md = _read_metadata(f, SCALES_FORMAT, path)
        masters = {n.split("::", 1)[1]: f.get_tensor(n) for n in f.keys() if n.startswith("small::")}
        return masters, {k: (md.get("small_dtypes") or {})[k] for k in masters}


def load_scales(path) -> dict[str, Tensor]:
    return read_scales(path)[0]


# Legacy research/experiments states (read-only).

@dataclass
class LegacyTensor:
    codes: Tensor
    scales: Tensor
    signs: Tensor   # int8, per module
    block: int


def load_legacy_pt(path) -> dict[str, LegacyTensor]:
    """Read a gptq_ternary.py ``ternary_state.pt`` or endpoints.py ``endpoint_state.pt``.

    Each module keeps its own signs and block, so per-module seed states
    evaluate exactly as ``apply_state`` did. States written before the scripts
    stored ``signs`` (for example the 256-sample per-module run) carry only
    ``seed``; their signs are regenerated with the scripts' ``signs_from_seed``.
    A module with both must have ``signs == signs_from_seed(seed)``: the scripts
    (``apply_state``, e2e_kd.py) rebuilt the rotation from the seed, so other
    stored signs were never the model that was trained or evaluated.
    """
    from ..core.hadamard import seeded_signs
    obj = torch.load(path, map_location="cpu", weights_only=True)
    entries = obj["modules"] if isinstance(obj, dict) and "modules" in obj else obj
    out, seeded = {}, {}
    for key, st in entries.items():
        if not isinstance(st, dict) or "codes" not in st:
            continue
        codes = st["codes"].to(torch.int8)
        width = int(codes.shape[1])
        if "seed" in st:
            ck = (int(st["seed"]), width)
            if ck not in seeded:
                seeded[ck] = seeded_signs(*ck)
        if "signs" in st:
            signs = st["signs"].to(torch.int8).reshape(-1)
        elif "seed" in st:
            signs = seeded[ck]
        else:
            raise ValueError(f"{path}: {key} has neither signs nor seed")
        if signs.numel() != width:
            raise ValueError(f"{path}: {key} signs do not match its input width")
        if "seed" in st and not torch.equal(signs, seeded[ck]):
            raise ValueError(f"{path}: {key} stored signs differ from signs_from_seed(seed={ck[0]}, {width}); "
                             "the research scripts rebuilt the rotation from the seed, so the state does not "
                             "describe the model they trained or evaluated")
        out[key] = LegacyTensor(codes, st["scales"].to(torch.float16), signs, int(st["block"]))
    if not out:
        raise ValueError(f"{path}: no ternary modules found")
    return out


def legacy_to_state(legacy: dict[str, LegacyTensor], meta: dict | None = None) -> TernaryState:
    """Convert a shared-sign legacy state; refuse per-module signs."""
    blocks = {t.block for t in legacy.values()}
    if len(blocks) != 1:
        raise ValueError(f"legacy state mixes Hadamard blocks {sorted(blocks)}; not representable")
    signs: dict[int, Tensor] = {}
    for key, t in legacy.items():
        w = t.codes.shape[1]
        if w in signs and not torch.equal(signs[w], t.signs):
            raise ValueError(f"legacy state uses per-module signs (width {w}, {key}); "
                             "it can be evaluated but not converted or exported")
        signs.setdefault(w, t.signs)
    groups = {t.codes.shape[1] // t.scales.shape[1] for t in legacy.values()}
    if len(groups) != 1:
        raise ValueError("legacy state mixes group sizes")
    tensors = {k: TernaryTensor(t.codes, t.scales) for k, t in legacy.items()}
    return TernaryState(tensors, signs, blocks.pop(), groups.pop(), dict(meta or {})).validate()

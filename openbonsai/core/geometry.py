"""Rotation geometry derived from the model, never from the adapter.

One global Hadamard block per model: the largest power of two <= max_block
that divides every rotated input width. One sign vector per width. Prism's
PTQ1_0 runtime stores exactly one ``block_size`` and one sign vector per width,
so a model whose widths would need mixed blocks is not representable.

Prism's runtime builds its rotation for any power-of-two block. It has run
real models only at block 4096 (the official Ternary-Bonsai-2-9B and our 9B
export), and tiny random models at block 128 in our tests; other blocks, such
as 2048 for Llama 3.1 8B, are untested on real weights. ``block_note`` says so
for any block other than 4096 (quantize logs it, the export records it).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .hadamard import seeded_signs, validate_rotation

MAX_BLOCK = 4096
GROUP = 128
PRISM_VERIFIED_BLOCK = 4096   # the only Hadamard block Prism's runtime has run real models at


class UnsupportedGeometry(ValueError):
    """The model's widths cannot be expressed in the PTQ1_0 rotation layout."""


@dataclass(frozen=True)
class Geometry:
    widths: tuple[int, ...]      # every rotated input width, sorted (includes hidden)
    block: int
    group: int
    hidden: int

    def to_json(self) -> dict:
        return {"widths": list(self.widths), "block": self.block, "group": self.group, "hidden": self.hidden}


def derive_block(widths, *, group: int = GROUP, max_block: int = MAX_BLOCK) -> int:
    widths = sorted(set(int(w) for w in widths))
    if not widths:
        raise UnsupportedGeometry("no rotated widths")
    if type(group) is not int or group <= 0 or group & (group - 1):
        raise UnsupportedGeometry("group must be a positive power of two")
    if type(max_block) is not int or max_block <= 0 or max_block & (max_block - 1):
        raise UnsupportedGeometry("max_block must be a positive power of two")
    bad = [w for w in widths if w <= 0 or w % group]
    if bad:
        raise UnsupportedGeometry(
            f"rotated widths {bad} are not multiples of the group size {group}; "
            "PTQ1_0 needs every quantized input width to be a multiple of 128")
    g = 0
    for w in widths:
        g = math.gcd(g, w)
    block = min(g & -g, max_block)   # largest power of two dividing gcd, capped
    if block < group:
        raise UnsupportedGeometry(f"derived Hadamard block {block} is smaller than group {group}")
    return block


def block_note(block: int) -> str | None:
    """A note for a block Prism's runtime has not run a real model at (None for 4096)."""
    if int(block) == PRISM_VERIFIED_BLOCK:
        return None
    return (f"Hadamard block {block}: Prism's runtime accepts any power-of-two block, but it has run real models "
            f"only at block {PRISM_VERIFIED_BLOCK} (and tiny random models at 128 in our tests); block {block} is "
            "untested on real weights, so check the exported model in Prism")


def derive_geometry(widths, hidden: int, *, group: int = GROUP, max_block: int = MAX_BLOCK) -> Geometry:
    widths = tuple(sorted(set(int(w) for w in widths) | {int(hidden)}))
    return Geometry(widths=widths, block=derive_block(widths, group=group, max_block=max_block),
                    group=group, hidden=int(hidden))


def make_signs(geometry: Geometry, mode: str = "seeded", seed="width") -> dict[int, Tensor]:
    """Sign vectors per width. ``seed: width`` reproduces the experiments'
    shared-sign GPTQ (seed = in_features); an int seeds every width with
    ``seed + width``."""
    if mode != "seeded":
        raise ValueError(f"unknown sign mode {mode!r}; use 'seeded' or load explicit signs")
    out = {}
    for w in geometry.widths:
        s = w if seed == "width" else int(seed) + w
        out[w] = seeded_signs(s, w)
    return out


def check_signs(signs: dict[int, Tensor], geometry: Geometry) -> None:
    missing = [w for w in geometry.widths if w not in signs]
    if missing:
        raise UnsupportedGeometry(f"no sign vector for widths {missing}")
    for w, s in signs.items():
        validate_rotation(s, geometry.block, int(w))
        if s.dtype != torch.int8:
            raise ValueError("sign vectors are stored as int8")

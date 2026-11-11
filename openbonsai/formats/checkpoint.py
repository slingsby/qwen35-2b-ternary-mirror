"""Read single tensors from a Hugging Face safetensors checkpoint directory.

Sharded (``model.safetensors.index.json``) or single-file (``model.safetensors``).
Tensors keep their stored dtype; handles stay open until ``close``.
"""
from __future__ import annotations

import json
from pathlib import Path


class Checkpoint:
    def __init__(self, root):
        self.root = Path(root)
        index = self.root / "model.safetensors.index.json"
        if index.exists():
            self.map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        elif (self.root / "model.safetensors").exists():
            from safetensors import safe_open
            with safe_open(str(self.root / "model.safetensors"), framework="pt") as f:
                self.map = {k: "model.safetensors" for k in f.keys()}
        else:
            raise FileNotFoundError(f"{self.root}: no model.safetensors or model.safetensors.index.json")
        self._handles = {}

    def __contains__(self, name: str) -> bool:
        return name in self.map

    def _handle(self, name):
        if name not in self.map:
            raise KeyError(f"{self.root}: checkpoint has no tensor {name!r}")
        f = self.map[name]
        if f not in self._handles:
            from safetensors import safe_open
            self._handles[f] = safe_open(str(self.root / f), framework="pt")
        return self._handles[f]

    def get(self, name):
        return self._handle(name).get_tensor(name)

    def close(self):
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

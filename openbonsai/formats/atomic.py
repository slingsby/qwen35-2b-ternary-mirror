"""Crash-safe writes: temp file in the same directory, fsync, then os.replace."""
from __future__ import annotations

import json
import os
from pathlib import Path


def atomic_write_bytes(path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_json(path, value) -> Path:
    text = json.dumps(value, indent=1, sort_keys=False, ensure_ascii=False, allow_nan=False) + "\n"
    return atomic_write_bytes(path, text.encode("utf-8"))


def durable_save(obj, path, saver=None) -> Path:
    """``saver(obj, file)`` into a temp file, fsync, then rename (torch.save by default)."""
    if saver is None:
        import torch
        saver = torch.save
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        saver(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path

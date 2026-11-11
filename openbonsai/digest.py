"""SHA-256 of input files and model directories, cached by size and mtime.

Stage hashes include the contents of the files a config names, so an edited
prompt file or a fine-tune retrained in place reruns the stages that read
it. A file is rehashed only when its size or mtime changes; the cache lives
in the run directory (``digests.json``), so a copied model directory is
rehashed once and, with equal contents, keeps every stage current.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .formats.atomic import atomic_write_json


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DigestCache:
    def __init__(self, store=None):
        self.store = Path(store) if store is not None else None
        self._entries: dict = {}
        if self.store is not None and self.store.exists():
            try:
                self._entries = json.loads(self.store.read_text(encoding="utf-8"))
            except ValueError:
                self._entries = {}

    def file(self, path) -> str:
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"input file not found: {path}")
        st = path.stat()
        key = str(path)
        entry = self._entries.get(key)
        if entry and entry.get("size") == st.st_size and entry.get("mtime_ns") == st.st_mtime_ns:
            return entry["sha256"]
        sha = file_sha256(path)
        self._entries[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha}
        if self.store is not None:
            atomic_write_json(self.store, self._entries)
        return sha

    def tree(self, root) -> str:
        """One digest over every file below ``root`` (hidden entries skipped)."""
        root = Path(root).resolve()
        files = {}
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if any(part.startswith(".") for part in rel.parts) or not path.is_file():
                continue
            files[rel.as_posix()] = self.file(path)
        return hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()

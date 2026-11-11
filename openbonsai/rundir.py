"""One run directory holds every stage's output.

    runs/<name>/
      run.yaml            resolved config
      signs.safetensors   per-width sign vectors (rewritten only when rotation.signs changes)
      digests.json        cached SHA-256 of input files (by size and mtime)
      traces/ calib/ quantize/ endpoints/ refine/ distill/ export/ eval/
      export/<name>.gguf  the PTQ1_0 GGUF (or export.out) and its .export.json report
      logs/<stage>.log

A stage is complete when ``<stage>/DONE.json`` exists; it records the stage's
config hash and the SHA-256 of its output file (``OUTPUTS``, or the export's GGUF). Downstream
hashes include both, so a stage that reran with a different result makes
every later stage stale. Rerunning skips a stage whose hash is unchanged and
whose output is still there with the recorded SHA-256 (a deleted, moved or
edited output reruns the stage).
Within a stage, units of work (a decoder layer, a teacher shard, a
distillation checkpoint) are written atomically, so a crash loses at most the
unit in progress.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import yaml

from .config import dump
from .formats.atomic import atomic_write_bytes, atomic_write_json

_STAGE_DIRS = {"calibrate": "calib"}
# The file each stage produces for later stages (relative to its directory).
OUTPUTS = {"traces": "traces.jsonl", "calibrate": "tokens.npy", "quantize": "state.safetensors",
           "endpoints": "state.safetensors", "refine": "state.safetensors", "distill": "scales.safetensors"}


class StageNotReady(RuntimeError):
    """A stage's input is missing, stale (config or upstream changed) or disabled."""


class RunDir:
    def __init__(self, root):
        self.root = Path(root)

    @property
    def config_path(self) -> Path:
        return self.root / "run.yaml"

    @property
    def signs_path(self) -> Path:
        return self.root / "signs.safetensors"

    @property
    def digests_path(self) -> Path:
        return self.root / "digests.json"

    def output_path(self, stage: str) -> Path | None:
        name = OUTPUTS.get(stage)
        return self.stage_dir(stage, create=False) / name if name else None

    def stage_dir(self, stage: str, create: bool = True) -> Path:
        path = self.root / _STAGE_DIRS.get(stage, stage)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    # -- config ----------------------------------------------------------
    def save_config(self, cfg: dict) -> None:
        atomic_write_bytes(self.config_path, dump(cfg).encode("utf-8"))

    def load_config(self) -> dict | None:
        if not self.config_path.exists():
            return None
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    # -- completion --------------------------------------------------------
    def done_info(self, stage: str) -> dict | None:
        path = self.stage_dir(stage, create=False) / "DONE.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def is_done(self, stage: str, config_hash: str) -> bool:
        info = self.done_info(stage)
        return info is not None and info.get("config_hash") == config_hash

    def mark_done(self, stage: str, config_hash: str, *, output_sha256: str | None = None, **info) -> None:
        atomic_write_json(self.stage_dir(stage) / "DONE.json",
                          {"stage": stage, "config_hash": config_hash, "output_sha256": output_sha256, **info})

    def done_key(self, stage: str) -> list | None:
        """What a downstream stage hashes for this one: [config hash, output digest]."""
        info = self.done_info(stage)
        return None if info is None else [info.get("config_hash"), info.get("output_sha256")]

    def clear_done(self, stage: str) -> None:
        path = self.stage_dir(stage, create=False) / "DONE.json"
        if path.exists():
            path.unlink()

    # -- logging -----------------------------------------------------------
    def logger(self, stage: str, echo: bool = True):
        path = self.root / "logs" / f"{stage}.log"
        path.parent.mkdir(parents=True, exist_ok=True)

        def log(msg: str) -> None:
            line = f"[{time.strftime('%H:%M:%S')}] {stage}: {msg}"
            if echo:
                print(line, flush=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return log

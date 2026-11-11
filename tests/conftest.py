"""Shared CPU fixtures: a saved tiny Qwen3.5 directory and a run context factory."""
import os

# The suite is CPU-only and must never touch a GPU. An empty value would be
# dropped on Windows (unsetting the variable), so hide every device with -1
# before torch can initialize CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys  # noqa: E402

# Keep transformers on its PyTorch Gated DeltaNet and causal-conv paths even where the optional
# CUDA kernels are installed (scripts/setup_linux.sh installs flash-linear-attention): those
# kernels cannot take CPU tensors, and the parity tests pin the reference arithmetic.
for _kernels in ("fla", "causal_conv1d"):
    sys.modules.setdefault(_kernels, None)

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    from tiny_models import save_tiny_qwen3_5
    torch.set_num_threads(2)
    return save_tiny_qwen3_5(tmp_path_factory.mktemp("tiny-qwen3_5"))


@pytest.fixture(scope="session")
def tiny_mm_dir(tmp_path_factory):
    from tiny_models import save_tiny_qwen3_5
    return save_tiny_qwen3_5(tmp_path_factory.mktemp("tiny-qwen3_5-mm"), multimodal=True)


@pytest.fixture
def make_ctx(tiny_dir, monkeypatch):
    """StageContext on the tiny model with examples/tiny-qwen3_5.yaml (+ overrides)."""
    from openbonsai.config import load_config
    from openbonsai.rundir import RunDir
    from openbonsai.stages.common import StageContext
    monkeypatch.chdir(ROOT)

    def make(run_dir, *sets):
        cfg = load_config(ROOT / "examples" / "tiny-qwen3_5.yaml", sets, model=str(tiny_dir))
        run = RunDir(run_dir)
        run.root.mkdir(parents=True, exist_ok=True)
        run.save_config(cfg)
        return StageContext(cfg, run)
    return make


@pytest.fixture
def calibrated(make_ctx, tmp_path):
    """A run directory with traces and calibration done."""
    from openbonsai.cli import execute
    ctx = make_ctx(tmp_path / "run")
    execute(ctx, "traces")
    execute(ctx, "calibrate")
    return ctx

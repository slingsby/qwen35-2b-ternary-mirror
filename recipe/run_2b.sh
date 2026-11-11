#!/usr/bin/env bash
# Train, export and evaluate Qwen3.5-2B with the mirror-descent ternary recipe on one Linux GPU box.
# Resumable: each finished step writes $OUT/state/<step>.done and is skipped on re-run.
#
# Inputs (environment variables; see README "Inputs"):
#   START      GPTQ ternary start state (.safetensors); built with recipe/build_start.py if missing and WIKI is set
#   WIKI       WikiText-2 raw train parquet (only needed to build START)
#   TRAIN      directory with ids.npy and mask.npy (rows x 512 tokens)
#   VAL_DOC    .npy of 512-token document blocks for validation
#   VAL_CHAT   directory with chat ids.npy and mask.npy for validation
#   SEALED     optional: held-out evaluation directory (manifest.json, doc-wiki.npy, doc-python.npy, chat-*.npy, specs.json)
#   STEPS      updates (default 1000)
# Usage: START=... TRAIN=... VAL_DOC=... VAL_CHAT=... nohup bash recipe/run_2b.sh > run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD/recipe" HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
OUT=${OUT:-runs/2b}; ST=$OUT/state; mkdir -p "$ST" "$OUT/logs"
MODEL_REPO=Qwen/Qwen3.5-2B; MODEL_REV=15852e8c16360a2fea060d615a32b45270f8a8fc; MODEL=${MODEL:-runs/qwen2b/model}
REF_REPO=prism-ml/Ternary-Bonsai-2-2B-gguf; REF_REV=a6032a5089ac27e96be3a80c7b42d1ad0bd89887
REF=runs/bonsai2b/Ternary-Bonsai-2-2B-PTQ1_0.gguf; REF_SHA=36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa
START=${START:-$OUT/start/initial.safetensors}; STEPS=${STEPS:-1000}
step() { [ -f "$ST/$1.done" ] && { echo "skip $1"; return 1; }; echo "=== $1 $(date -Is)"; return 0; }
done_() { date -Is > "$ST/$1.done"; }

if step setup; then
  python3 -m venv .venv && . .venv/bin/activate
  CUDA_MAJOR=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' | head -1)
  INDEX=https://download.pytorch.org/whl/cu130; [ "${CUDA_MAJOR:-13}" -lt 13 ] && INDEX=https://download.pytorch.org/whl/cu128
  pip install -q --upgrade pip && pip install -q "torch>=2.14,<3" --index-url "$INDEX"
  pip install -q -e ".[test]" psutil
  done_ setup
fi
. .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$OUT/logs/gpu.txt"
python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__)" | tee -a "$OUT/logs/gpu.txt"

if step download; then
  python - <<PY
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("$MODEL_REPO", revision="$MODEL_REV", local_dir="$MODEL")
hf_hub_download("$REF_REPO", "Ternary-Bonsai-2-2B-PTQ1_0.gguf", revision="$REF_REV", local_dir="runs/bonsai2b")
PY
  echo "$REF_SHA  $REF" | sha256sum -c -
  done_ download
fi

if step tests; then python -m pytest -q -p no:cacheprovider tests/test_mirror.py tests/test_recipe.py | tail -2 | tee "$OUT/logs/tests.txt"; done_ tests; fi

if [ ! -f "$START" ]; then
  [ -n "${WIKI:-}" ] || { echo "START not found and WIKI not set"; exit 1; }
  python recipe/build_start.py --model-dir "$MODEL" --wiki "$WIKI" --out "$(dirname "$START")"
fi

if step train; then
  python recipe/train.py --model-dir "$MODEL" --initial "$START" --train "$TRAIN" --val-doc "$VAL_DOC" --val-chat "$VAL_CHAT" \
    --run-dir "$OUT/train" --steps "$STEPS" --validate-every 100 --patience 3 --timeout 43200
  python recipe/export.py --model-dir "$MODEL" --initial "$START" --run-dir "$OUT/train" --name "Qwen3.5-2B-ternary-mirror"
  done_ train
fi

if [ -n "${SEALED:-}" ]; then
  for pair in "bf16 bf16" "start gptq:$START" "bonsai-2b gguf:$REF" "trained gguf:$OUT/train/trained.gguf"; do
    set -- $pair
    if step "eval-$1"; then
      python recipe/evaluate.py --model-dir "$MODEL" --candidate "$2" --label "$1" --sealed "$SEALED" --out "$OUT/eval" > "$OUT/logs/eval-$1.log" 2>&1
      tail -1 "$OUT/logs/eval-$1.log"; done_ "eval-$1"
    fi
  done
fi
echo "ALL DONE $(date -Is): $OUT/train/trained.gguf"

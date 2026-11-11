# Qwen3.5-2B ternary: a mirror-descent recipe (research)

> **This is a research project, not a product.** It records one independent experiment: the method, the code and the measured results, including where it falls short. The models it produced are research artifacts and are not released. Nothing here is meant for production use.

**The research question:** how close can Qwen3.5-2B get to Prism ML's Ternary Bonsai 2B on a consumer budget? Bonsai showed what a ternary 2B model can do. This experiment tried to recover as much of that as possible without a training cluster.
- The recipe was developed and validated on a home PC with an RTX 4070 SUPER (12 GB), using Qwen3.5-0.8B. The GPTQ start for 2B was built on the same PC.
- The 2B training run needs about 24 GB, so it ran on one rented A100 40GB: about 4 hours and $3.22 in total, evaluations included.

**Finding:** the recipe is a clear improvement over a GPTQ ternary start, but it does **not** reach Bonsai's quality. The numbers below show the gap.

**Limits of the evidence:**
- One seed per arm.
- One held-out split, evaluated once per model.
- Generation checks test format and final values, not full correctness.
- No benchmark suites were run.

This is an independent project. It is not affiliated with or endorsed by Prism ML or the Qwen team. No model weights are included.

## Results at 2B

Each model was evaluated once on a held-out split, against the BF16 Qwen3.5-2B as teacher. The split is WikiText-2 test, Python source and chat responses, plus 80 generation prompts. **Lower is better** for PPL and KL.

| Model | WikiText-2 PPL / KL | Python PPL / KL | Chat PPL / KL | Generation (of 80) |
|---|---|---|---|---|
| BF16 Qwen3.5-2B (ceiling) | 15.73 / 0 | 3.36 / 0 | 1.34 / 0 | 76 |
| GPTQ ternary start | 94.38 / 1.769 | 12.24 / 1.288 | 9.10 / 2.033 | 0 |
| **This recipe** | **36.73 / 0.870** | **6.75 / 0.694** | **1.93 / 0.521** | **48** |
| This recipe + a broader data mixture | 37.38 / 0.911 | 6.82 / 0.702 | 2.07 / 0.584 | 31 |
| Ternary Bonsai 2B (reference) | 28.66 / 0.527 | 4.71 / 0.321 | 1.57 / 0.229 | 59 |

KL is full-vocabulary KL(teacher‖model) per token. As a multiple of BF16 perplexity, the recipe is at 2.33× / 2.01× / 1.44× on the three splits, and Bonsai at 1.82× / 1.40× / 1.17×.

Generation checks per domain (passes of 16):

| Model | code | debug | tool call | instruction | reasoning |
|---|---|---|---|---|---|
| BF16 | 16 | 16 | 16 | 12 | 16 |
| This recipe | 15 | 16 | 16 | 1 | 0 |
| Ternary Bonsai 2B | 16 | 16 | 0 | 11 | 16 |

The checks are format and value checks, not full functional tests (`recipe/checks.py`):
- **code / debug:** the reply parses as Python and defines the requested function with a `return`.
- **tool call:** the JSON call matches exactly.
- **instruction:** exact-output prompts.
- **reasoning:** two-step arithmetic word problems, final number checked.

How to read the generation table:
- **Bonsai's 0/16 on tool calls is a checker artifact.** Bonsai writes the correct call, then a stray `</think>` and the call again, which the strict parser rejects. The checker was not changed after seeing this.
- **The recipe's 0/16 on reasoning is real.** For "a tank holds X litres and drains b litres per hour for a hours", it answers X − b, ignoring the hours.
- **Instruction-following and multi-step arithmetic are where the gap to Bonsai is clearest.**

The broader mixture (FineWeb-Edu, code/instruction/tool/chat sets and GSM8K train, decontaminated against every evaluation array) did not help here. Its validation document KL rose after 200 updates and early stopping ended it at 500 of 2,298 updates. One run, one seed.

Full results, training histories and the recorded recipes are in [`results/`](results/).

### At 0.8B (the home PC)

The same recipe on Qwen3.5-0.8B, trained and evaluated on the RTX 4070 SUPER, on the matching 0.8B held-out split:

| Model | WikiText-2 PPL / KL | Python PPL / KL | Chat PPL / KL | Generation (of 80) |
|---|---|---|---|---|
| BF16 Qwen3.5-0.8B | 22.77 / 0 | 3.98 / 0 | 1.18 / 0 | 70 |
| GPTQ ternary start | 144.38 / 1.824 | 21.97 / 1.708 | 21.30 / 2.814 | 0 |
| This recipe | 56.17 / 0.908 | 9.29 / 0.849 | 2.46 / 0.726 | 33 |

## The recipe (as researched)

1. **Ternary start (`recipe/build_start.py`).**
   - Hadamard rotation (blocks of at most 512, seeded signs).
   - Sequential GPTQ to {−1, 0, +1} with one FP16 scale per 128 weights, layer by layer in data-flow order.
   - Calibration: 64 blocks of 256 tokens from WikiText-2 train and Python standard-library source.
   - The tied embedding/head is quantized once, with a joint head/embedding Hessian.
2. **Mirror-descent training on the codes (`recipe/train.py`).**
   - **Parameterization:** each ternary table keeps an FP32 *dual* variable. The weights are the mirror map of the dual under the prune+quantize potential ψ(w) = Σ max(|w|^(1+ε), β|w|^q), with ε = 0.05, q = 8, β = 1 (`openbonsai/core/mirror.py`). The dual starts at ∇ψ(codes), so training begins exactly at the GPTQ codes. The forward pass uses the rounded ternary codes.
   - **Code updates:** two-step stochastic mirror descent. The dual moves along an Adafactor direction with code LR 0.02, 32 warmup updates, then cosine annealing to 0 over the run. Annealing is what stopped late-run drift at 0.8B.
   - **Frozen and trained parts:** the shared embedding/head codes stay frozen. Group scales and the small high-precision tensors (norms, short convolutions, DeltaNet gate parameters) train with AdamW: LRs 1e-3 / 1e-4 / 2e-5, cosine to 0.1×.
   - **Loss:** full-vocabulary KL from the BF16 model, run online as teacher, on 2,044 scored tokens per update.
   - **Training data:** 128 rows of 512 tokens. Half are WikiText-2/Python-source documents. Half are chat responses written by the BF16 0.8B model to synthetic code, debugging, tool-call, instruction and arithmetic prompts.
   - **Checkpoint selection:** validation every 100 updates (mean of document and chat KL). The best checkpoint is kept, and training stops after 3 validations without a new best.
   - At 2B, 1,000 updates took 1.93 h at 6.3 s/update, peaking at about 24 GB. 3.79M of 1.88B codes changed, all from 0 to ±1.
3. **Export (`recipe/export.py`).** Writes a PTQ1_0 GGUF (Prism's ternary format) and verifies it against the trained state. The block-512 rotation has been checked through this repo's own loader. It has not been run in Prism's llama.cpp runtime on real weights.
4. **Evaluation (`recipe/evaluate.py`).** Perplexity, full-vocabulary KL and top-1 agreement against the online BF16 teacher, plus greedy generation on the held-out prompts. Each label can be evaluated only once, and every evaluation is logged to `access-log.jsonl`.

## Running it

You need a Linux box with an NVIDIA GPU with about 24 GB for 2B training. 12 GB is enough for the start builder and for 0.8B.

```
pip install torch --index-url https://download.pytorch.org/whl/cu130   # a CUDA build of torch >= 2.14
pip install -e ".[test]" psutil
python -m pytest -q                                                   # CPU-only suite, ~7 min

START=runs/start/initial.safetensors WIKI=<wikitext-2-raw-v1 train parquet> \
TRAIN=<dir with ids.npy, mask.npy> VAL_DOC=<val.npy> VAL_CHAT=<dir with ids.npy, mask.npy> \
SEALED=<held-out dir> bash recipe/run_2b.sh
```

`run_2b.sh` does the following:
- downloads Qwen/Qwen3.5-2B at revision `15852e8c` and the Bonsai 2B GGUF (hash-checked) for comparison;
- builds the start if it is missing;
- trains, exports and, if `SEALED` is set, evaluates.

It can be resumed: rerun it, and finished steps are skipped. Each script also runs on its own; see `--help`.

### Inputs

- **Training and validation rows** are `int64` arrays of shape rows × 512. `mask.npy` marks real tokens; padding is false, and whole conversations, prompt included, are scored. Any data in this format works.
- **The exact arrays behind the results** aren't in this repo yet. Their SHA-256s are in [`results/2b-a/recipe.json`](results/2b-a/recipe.json).
- **The start state:** the published runs used SHA-256 `32b338a2…` (in that recipe file).
- **Rebuilding the start:** `build_start.py` gives the same file only on the same Python version, because the calibration includes standard-library source.
- **Software for the 2B run:** torch 2.15.0+cu132 and transformers 5.20.2 ([`results/environment.txt`](results/environment.txt)).

The scripts here behave the same as the ones that produced the results. Comments were tidied, and an unused hidden-state loss option was removed.

## Layout

| Path | What |
|---|---|
| `recipe/` | Start builder, training, export, evaluation, generation checks, one-command runner |
| `openbonsai/` | Ternary core: Hadamard rotation, GPTQ, PTQ1_0 and GGUF formats, Qwen3.5 adapter, mirror-descent maths |
| `tests/` | CPU tests on tiny random models |
| `results/` | Evaluation outputs (with every generated reply), training histories, recipes |

## Credits and licence

The mirror-descent parameterization follows patent application US20260220467A1 (Caltech).

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The `openbonsai/` package comes from the OpenBonsai project, under the same licence.

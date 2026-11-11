# Reasoning test subsets (frozen)

TEST problems for the `gsm8k` and `humaneval` evaluation suites (`openbonsai/eval/reasoning.py`).

| File | Dataset | Problems | File SHA256 |
|---|---|---|---|
| `gsm8k-test-100.json` | `openai/gsm8k` `main` test (1,319 rows) @ `740312add88f781978c0658806c59bc2815b9866`, `main/test-00000-of-00001.parquet` | 100: question, final number | `f1783b93eb7a7d14be338cb36545441b6572ce65ffeecc7103616caf3dd59d0a` |
| `humaneval-test-50.json` | `openai/openai_humaneval` test (164 rows) @ `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`, `openai_humaneval/test-00000-of-00001.parquet` | 50: prompt, test, entry point, canonical solution | `392f9271c8c43989ea096efd6817579064c821516f0e18a4d7ee5fe30887e9ad` |

**Selector `sha256-rank/1`, seed 2026.** Rank every row index `r` (0-based, in the parquet file's order) by the hex SHA256 of `"{repo}:{split}:{seed}:{r}"`, ascending, and keep the first `n` rows in that order. The rule depends only on the row count, so `row_ids` can be checked without the data. The first `k` problems of a file are what the selector picks for `n = k`, so a smaller `eval.reasoning.<suite>.n` uses a prefix.

Each manifest records the dataset (repo, config, split, revision, file, row count and file SHA256), the selector and seed, the row ids, the problems, and `content_sha256`: the SHA256 of the manifest's canonical JSON without that key. The loader checks both hashes. The package pins the file SHA256s, and `.gitattributes` keeps these bytes unchanged on Windows checkouts.

To rebuild the files from the pinned datasets and compare them byte for byte (offline, with the files in the HF cache, or with local parquet paths):

```
HF_HUB_OFFLINE=1 python -m openbonsai.eval.reasoning check [--gsm8k PATH] [--humaneval PATH]
```

**These are test data.** The calibrate and traces stages refuse sources that name these test splits, local copies of the test files, and any calibration stream or trace prompt that contains a test problem: every problem of both splits when they are in the Hugging Face cache, else these frozen ones (`openbonsai/eval/heldout.py`). Before scoring a stage, eval also checks the run's stored calibration and distillation-teacher tokens for the problems it is about to score. Future math or code calibration data must come from training splits, for example `openai/gsm8k` `main/train-00000-of-00001.parquet`. HumanEval has no training split, so code calibration data must come from another dataset's training split.

The problem texts are from the GSM8K (MIT license) and HumanEval (MIT license) datasets.

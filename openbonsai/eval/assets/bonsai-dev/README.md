# Bonsai development evaluation inputs

[frozen-inputs.json](frozen-inputs.json) is the exact reviewed conversion input
supplied to the completed official Bonsai CPU run (OpenBonsai repository: `research/docs/official-bonsai-development-result.md`).
The run saved and evaluated a semantically identical JSON copy with different
serialization bytes; both hashes are recorded in the validation receipt. This
file is included for reproducible evaluation and transfer to the separate 5090
checkout. It contains no weights or tokenizer asset files.

This is **reused development data**, with role `search_validation`, not a fresh
held-out confirmation set. Its questions and answer keys are for scoring only;
do not use them for candidate fitting, teacher-trace preparation, or allocation
optimization. Historical `heldout` item names do not change that status.

| Identity | Value |
|---|---|
| Exact file bytes | 637466 |
| Exact file SHA256 | `9936a5382e9cc9921ac21ccae0ceb84fab8a9a7681d46fd0f3efbbc05e6b70be` |
| Canonical `frozen_sha256` | `8021d9cc8f54943a3b816af74cbc41d588173d385f51b73a681e979393ef5fdf` |
| Tokenizer | `Qwen/Qwen3.5-9B` at `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Workload | 88 texts: four NLL inputs, 80 ARC option inputs, four generation prompts |
| Scoring | 2048 causal NLL targets; 20 ARC-Easy option-letter choices |
| Generation | Greedy; 128 tokens maximum per prompt; EOS `[248044, 248046]` |

The file is copied byte-for-byte from the reviewed conversion. Git text
conversion is disabled for this path so a Windows checkout preserves its hash.
Do not reformat it, render another template, retokenize answer boundaries, or
change caps for a matched comparison. Both backends must still pass the exact
native token-ID and raw-piece checks before scoring.

Public dataset provenance is recorded in the manifest:

- [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3),
  test split, four already selected token slices. Text fields are reconstructed
  from authenticated token pieces, not independently recovered source prose.
- [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453),
  20 already selected validation questions. The score is conditional likelihood
  of the A/B/C/D label, not the whole answer text.
- Four manually authored instruction, reasoning, coding and Turkish rewriting
  prompts already used in the original development diagnostics.

Dataset and tokenizer material retain their respective licenses. The original
transport, rendering helper hashes, tokenizer file hashes, selected source IDs,
and explicit normalization policy are recorded in the manifest and the
validation receipt (OpenBonsai repository: `research/validation/validation-official-bonsai-development.json`).

Score it in torch with `openbonsai eval --suite dev` (openbonsai/eval/suites.py),
or a GGUF in Prism's llama.cpp with
[tools/prism/evaluate_prism_manifest.py](../../../../tools/prism/evaluate_prism_manifest.py),
which applies the full token-ID and raw-piece checks of openbonsai/eval/manifest.py.
The history of the matched comparison is in
research/docs/matched-evaluation.md (OpenBonsai repository: `research/docs/matched-evaluation.md`).

The official result uses centered FP64 normalization of FP32 logits. Earlier
OpenBonsai FP32 cross-entropy scalars cannot be converted into an exact matched
baseline; rescore the original and selected completed artifact. These common
inputs and probability rules do not make different backend kernels identical.

## Extended suite inputs

`openbonsai eval --suite extended` reads two more files, copied byte for byte
from local runs and pinned by SHA256 in openbonsai/eval/suites.py:

| File | Bytes | SHA256 | Contents |
|---|---|---|---|
| `legacy-eval-inputs.json` | 689368 | `2952cf072dff2e9b00b09a59a5f2345a02b3e509725854af07ea8478f8a7a64c` | From `runs/2026-10-26-5090-pilot/eval-inputs.json`: 32 x 513-token WikiText-2 test windows (16,384 scored tokens), 100 ARC-Easy letter questions, 10 prompts (unused). |
| `chat-traces16.jsonl` | 77139 | `443aaba5e5b41c641c27cdcd45f34fd5a2e770fa466321c9bbab84436ff40aaf` | From `runs/2026-10-26-5090-activation-diagnostic-v1/data/search-consumed16.jsonl`: 16 Qwen3.5-9B greedy answers to UltraChat 200k first-turn prompts, disjoint from calibration. |

All three files are tokenized for Qwen/Qwen3.5-9B; the suites refuse other
tokenizers (use `--suite compare` for a model with its own tokenizer).


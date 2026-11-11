"""Calibration sources: parity with OpenBonsai research/experiments/calib_data.py, BOS, user JSONL.

The oracle functions are a verbatim copy of calib_data.py with its module
constants turned into parameters. Tests never import research/.
"""
import json
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from openbonsai.stages import calibrate as C
from tiny_models import byte_tokenizer


# ---------------------------------------------------------------- oracle ----
def _pack(stream, seq_len, n_blocks):
    blocks, buf = [], []
    for ids in stream:
        buf.extend(ids)
        while len(buf) >= seq_len and len(blocks) < n_blocks:
            blocks.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(blocks) >= n_blocks:
            break
    return blocks


def wiki_blocks(tok, seq_len, n, seed, WIKI_TRAIN):
    text = "".join(pq.read_table(WIKI_TRAIN).column("text").to_pylist())
    ids = tok(text, add_special_tokens=False)["input_ids"]
    rng = random.Random(seed)
    off = rng.randrange(seq_len)
    starts = sorted(rng.sample(range(off, len(ids) - seq_len, seq_len), n))
    return [ids[s:s + seq_len] for s in starts]


def ultrachat_blocks(tok, seq_len, n, seed, ULTRACHAT, skip_rows=6000):
    rows = pq.read_table(ULTRACHAT, columns=["messages"]).column("messages").to_pylist()[skip_rows:]
    random.Random(seed).shuffle(rows)

    def gen():
        for msgs in rows:
            text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
            yield tok(text, add_special_tokens=False)["input_ids"]

    return _pack(gen(), seq_len, n)


def ayot_blocks(seq_len, n, AYOT):
    def gen():
        for line in AYOT.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            yield [t for t, m in zip(r["input_ids"], r["attention_mask"]) if m]

    blocks = _pack(gen(), seq_len, n)
    assert len(blocks) == n
    return blocks


def oracle_build(tok, seq_len, n_wiki, n_chat, n_ayot, seed, paths):
    blocks = wiki_blocks(tok, seq_len, n_wiki, seed, paths["wiki"]) + \
        ultrachat_blocks(tok, seq_len, n_chat, seed, paths["chat"], skip_rows=paths["skip"]) + \
        ayot_blocks(seq_len, n_ayot, paths["ayot"])
    out = torch.tensor(blocks, dtype=torch.long)
    perm = torch.randperm(len(out), generator=torch.Generator().manual_seed(seed))
    return out[perm]
# ------------------------------------------------------------ end oracle ----

WORDS = ("river", "meadow", "compiler", "token", "scale", "ternary", "comet", "bread", "library", "rotation")


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    d = tmp_path_factory.mktemp("sources")
    rng = random.Random(3)
    wiki = [" ".join(rng.choice(WORDS) for _ in range(rng.randrange(5, 40))) + "\n" for _ in range(200)]
    pq.write_table(pa.table({"text": wiki}), d / "wiki.parquet")
    chats = [[{"role": "user", "content": " ".join(rng.choice(WORDS) for _ in range(6))},
              {"role": "assistant", "content": " ".join(rng.choice(WORDS) for _ in range(30))}] for _ in range(60)]
    pq.write_table(pa.table({"messages": chats}), d / "chat.parquet")
    with open(d / "traces.jsonl", "w", encoding="utf-8") as f:
        for k in range(6):
            ids = [rng.randrange(256) for _ in range(150)]
            mask = [1] * 140 + [0] * 10 if k % 2 else [1] * 150
            f.write(json.dumps({"input_ids": ids, "attention_mask": mask}) + "\n")
    return {"wiki": d / "wiki.parquet", "chat": d / "chat.parquet", "ayot": d / "traces.jsonl", "skip": 10}


@pytest.mark.parametrize("seed", [0, 5])
def test_blocks_match_calib_data_exactly(sources, seed):
    tok = byte_tokenizer()
    ref = oracle_build(tok, 64, 4, 3, 2, seed, sources)
    got, report = C.build(tok, [{"kind": "wikitext2", "path": str(sources["wiki"]), "n": 4},
                                {"kind": "ultrachat", "path": str(sources["chat"]), "n": 3, "min_row": 10},
                                {"kind": "traces", "path": str(sources["ayot"]), "n": 2}],
                          seq_len=64, seed=seed, chat_kwargs={"enable_thinking": False})
    assert torch.equal(got, ref)
    assert [r["kind"] for r in report] == ["wikitext2", "ultrachat", "traces"]
    assert all(len(r["sha256"]) == 64 for r in report)


def test_bos_starts_every_block(sources):
    tok = byte_tokenizer()
    got, _ = C.build(tok, [{"kind": "wikitext2", "path": str(sources["wiki"]), "n": 3},
                           {"kind": "traces", "path": str(sources["ayot"]), "n": 2}], seq_len=32, seed=1, bos=300)
    assert got.shape == (5, 32) and (got[:, 0] == 300).all() and not (got[:, 1:] == 300).any()


def test_user_jsonl_chat_text_and_ids(tmp_path):
    tok = byte_tokenizer()
    path = tmp_path / "mine.jsonl"
    rows = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}]},
            {"text": "plain words " * 4}, {"input_ids": list(range(40))}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    got, _ = C.build(tok, [{"kind": "jsonl", "path": str(path), "n": 2}], seq_len=40, seed=0)
    stream = (tok(tok.apply_chat_template(rows[0]["messages"], tokenize=False), add_special_tokens=False)["input_ids"]
              + tok(rows[1]["text"], add_special_tokens=False)["input_ids"] + rows[2]["input_ids"])
    expected = torch.tensor([stream[:40], stream[40:80]])
    perm = torch.randperm(2, generator=torch.Generator().manual_seed(0))
    assert torch.equal(got, expected[perm])
    path.write_text('{"other": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="messages, text or input_ids"):
        C.build(tok, [{"kind": "jsonl", "path": str(path), "n": 1}], seq_len=8)


def test_base_model_without_chat_template_falls_back_to_text():
    tok = byte_tokenizer()
    tok.chat_template = None
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert C.chat_text(tok, msgs, {}) == "a\n\nb"


def test_too_little_data_and_bad_sources_fail_clearly(sources, tmp_path):
    tok = byte_tokenizer()
    with pytest.raises(ValueError, match="only"):
        C.build(tok, [{"kind": "traces", "path": str(sources["ayot"]), "n": 100}], seq_len=64)
    with pytest.raises(ValueError, match="unknown calibration source"):
        C.build(tok, [{"kind": "books", "n": 1}], seq_len=64)
    with pytest.raises(FileNotFoundError):
        C.build(tok, [{"kind": "jsonl", "path": str(tmp_path / "missing.jsonl"), "n": 1}], seq_len=64)
    with pytest.raises(FileNotFoundError, match="traces"):
        C.build(tok, [{"kind": "traces", "n": 1}], seq_len=64, traces_path=tmp_path / "none.jsonl")


def test_stage_writes_tokens_and_sources(calibrated):
    import numpy as np
    d = calibrated.run.stage_dir("calibrate")
    tokens = np.load(d / "tokens.npy")
    summary = json.loads((d / "sources.json").read_text())
    assert tokens.shape == (8, 64) and tokens.dtype == np.int32
    assert summary["shape"] == [8, 64] and [s["kind"] for s in summary["sources"]] == ["jsonl", "jsonl", "traces"]

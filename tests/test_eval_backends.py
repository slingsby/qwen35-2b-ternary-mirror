"""Reference contract tests; random native fixtures are not quality evidence."""
import hashlib
import errno
import json
import math
import subprocess
import sys

import pytest
import torch
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, decoders, normalizers
from transformers import PreTrainedTokenizerFast

from tiny_models import tiny_qwen3_5
from openbonsai.eval.backends import (TransformersBackend, PrismBackend,
    PRISM_RUNTIME_SHA, PROTOCOL, REDUCTION, _byte_decoder, local_directory_identity)


IDENTITY = {'sha256': 'a' * 64, 'kind': 'declared-random-test-fixture'}


def tokenizer_fixture():
    vocabulary = {char: i for i, char in enumerate(_byte_decoder())}
    tokens = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    tokens.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokens.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=tokens,
                                   additional_special_tokens=['<|test|>'])


def native_fixture(*, multimodal=False, chunk=2):
    torch.set_num_threads(2)
    model = tiny_qwen3_5(seed=91, multimodal=multimodal)
    # Actual Qwen also has padded model rows with no tokenizer entry.
    model.resize_token_embeddings(260, mean_resizing=False)
    model.eval()
    return TransformersBackend(model, tokenizer_fixture(), context_length=32,
        source_identity=IDENTITY, tokenizer_identity=IDENTITY, logits_chunk_size=chunk)


@pytest.mark.parametrize('multimodal', [False, True])
def test_native_selected_position_scores_match_direct_full_vocab(multimodal):
    backend = native_fixture(multimodal=multimodal)
    ids, selected = [10, 20, 10, 40, 50, 259], [1, 3, 4, 5]
    with torch.no_grad():
        logits = backend.model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float().double()
    expected = [float(logits[j-1, ids[j]] - torch.logsumexp(logits[j-1], dim=0)) for j in selected]
    calls = []
    hook = backend.model.lm_head.register_forward_pre_hook(lambda _, args: calls.append(args[0].shape[1]))
    result = backend.score_tokens(ids, selected)
    hook.remove()
    assert calls == [2, 2]
    assert result['score_positions'] == selected
    assert result['count'] == 4
    # The head runs on 2-row chunks here and on all rows above: different GEMM
    # shapes, so FP32 rounding differs by a few ulp (abs 1e-7 was flaky).
    assert result['target_logprobs'] == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert result['nll_sum'] == pytest.approx(-math.fsum(expected), rel=1e-6, abs=1e-6)
    backend.score_tokens([2, 3, 4], [1, 2])
    assert backend.score_tokens(ids, selected) == result
    metadata = backend.metadata()
    assert metadata['vocab_size'] == 260 > metadata['tokenizer_vocab_size']
    assert metadata['cache_mode'] == 'full-prefix-reference'
    metadata['source_identity']['sha256'] = 'changed'
    assert backend.metadata()['source_identity'] == IDENTITY


def test_generation_matches_manual_prefix_and_exact_tie_eos():
    backend = native_fixture()
    ids, expected = [4, 5], []
    with torch.no_grad():
        for _ in range(3):
            output = backend.model(input_ids=torch.tensor([ids + expected]), use_cache=False)
            expected.append(int(output.logits[0, -1].float().argmax()))
    eos = next(i for i in range(260) if i not in expected)
    assert backend.generate(ids, 3, [eos]) == {'generated_ids': expected, 'stop_reason': 'length'}
    with torch.no_grad():
        backend.model.lm_head.weight.zero_()
    assert backend.generate(ids, 4, [0]) == {'generated_ids': [0], 'stop_reason': 'eos'}
    assert backend.generate(ids, 4, [1]) == {'generated_ids': [0] * 4, 'stop_reason': 'length'}


def test_native_rejects_nonfinite_and_restores_precision():
    backend = native_fixture()
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    with torch.no_grad():
        backend.model.lm_head.weight[1].fill_(float('nan'))
    with pytest.raises(RuntimeError, match='invalid full-vocabulary logits'):
        backend.score_tokens([4, 5], [1])
    assert before == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def test_bytelevel_raw_pieces_special_handling_and_padded_rows():
    backend = native_fixture()
    text = 'hi <|test|> ığ 😄\n'
    plain = backend.tokenize(text)
    special = backend.tokenize(text, parse_special=True)
    assert 256 not in plain and 256 in special
    assert b''.join(backend.token_pieces(plain)) == text.encode()
    assert b''.join(backend.token_pieces(special)) == text.encode()
    assert any(len(piece) == 1 and piece[0] >= 128 for piece in backend.token_pieces(plain))
    assert backend.tokenize('') == [] and backend.token_pieces([]) == []
    with pytest.raises(ValueError, match='no tokenizer piece'):
        backend.token_pieces([259])
    backend.tokenizer.backend_tokenizer.decoder = decoders.WordPiece()
    with pytest.raises(ValueError, match='Unsupported tokenizer decoder'):
        TransformersBackend(backend.model, backend.tokenizer, context_length=32,
                            source_identity=IDENTITY, tokenizer_identity=IDENTITY)


def test_normalizer_provenance_is_explicit_without_hidden_preprocessing():
    native = native_fixture()
    native.tokenizer.backend_tokenizer.normalizer = normalizers.NFC()
    backend = TransformersBackend(native.model, native.tokenizer, context_length=32,
                                  source_identity=IDENTITY, tokenizer_identity=IDENTITY)
    assert backend.metadata()['tokenization']['normalizer'] == {'type': 'NFC'}
    assert backend.tokenize('e\u0301') == backend.tokenize('é')
    assert b''.join(backend.token_pieces(backend.tokenize('e\u0301'))) == 'é'.encode()


@pytest.mark.parametrize('spelling', ['Ġtest', 'çö'])
def test_nonascii_added_tokens_fail_closed_instead_of_claiming_raw_utf8(spelling):
    native = native_fixture()
    native.tokenizer.add_tokens([AddedToken(spelling, normalized=False)])
    backend = TransformersBackend(native.model, native.tokenizer, context_length=32,
                                  source_identity=IDENTITY, tokenizer_identity=IDENTITY)
    token = native.tokenizer.convert_tokens_to_ids(spelling)
    # This is why spelling.encode() is not a valid general ByteLevel inverse.
    assert native.tokenizer.decode([token]) != spelling
    with pytest.raises(ValueError, match='Non-ASCII added-token raw bytes are unsupported'):
        backend.token_pieces([token])


@pytest.mark.parametrize('ids,positions', [([], [1]), ([1, 2], []), ([1, 2], [0]),
    ([1, 2], [2]), ([1, 2, 3], [2, 1]), ([1, 2], [1, 1]), ([1, 2], [True]),
    ([True, 2], [1]), ([260, 2], [1]), ([1]*33, [1])])
def test_native_invalid_score_inputs(ids, positions):
    with pytest.raises(ValueError):
        native_fixture().score_tokens(ids, positions)


@pytest.mark.parametrize('ids,maximum,eos', [([], 1, [2]), ([1], 0, [2]),
    ([1], True, [2]), ([1], 1, []), ([1], 1, [2, 2]), ([1], 1, [260]), ([1]*31, 2, [2])])
def test_native_invalid_generation_inputs(ids, maximum, eos):
    with pytest.raises(ValueError):
        native_fixture().generate(ids, maximum, eos)


def test_hashes_are_deterministic_and_checked_before_loading(tmp_path, monkeypatch):
    root = tmp_path / 'original'
    root.mkdir()
    (root / 'weights').write_bytes(b'original')
    first = local_directory_identity(root)
    assert first == local_directory_identity(root)
    called = []
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', lambda *a, **kw: called.append(kw))
    with pytest.raises(ValueError, match='Model directory SHA256 mismatch'):
        TransformersBackend.from_local(root, context_length=32, expected_model_sha256='b'*64)
    assert not called
    (root / 'weights').write_bytes(b'altered')
    assert local_directory_identity(root)['sha256'] != first['sha256']


def test_directory_identity_rejects_symlinks_when_os_permits_them(tmp_path):
    (tmp_path / 'weights').write_bytes(b'original')
    try:
        (tmp_path / 'link').symlink_to(tmp_path / 'weights')
    except OSError as error:
        if getattr(error, 'winerror', None) == 1314 or error.errno == errno.EPERM:
            pytest.skip('OS account lacks symlink creation privilege')
        raise
    with pytest.raises(ValueError, match='symlinks'):
        local_directory_identity(tmp_path)


def test_from_local_forbids_downloads_and_keeps_saved_source(tmp_path, monkeypatch):
    from openbonsai.eval import backends as eval_backends
    native = native_fixture()
    root = tmp_path / 'saved'
    root.mkdir()
    native.tokenizer.save_pretrained(root)
    (root / 'fixture-weights').write_bytes(b'model declaration')
    calls = []
    def load(path, **kwargs):
        calls.append((path, kwargs))
        return native.model
    monkeypatch.setattr(eval_backends, '_load_native', load)
    backend = TransformersBackend.from_local(root, context_length=32, dtype='float32')
    assert calls == [(str(root), {'device': 'cpu', 'dtype': 'float32', 'local_files_only': True})]
    assert backend.tokenize('hello') == native.tokenize('hello')
    assert backend.metadata()['source_identity'] == {'kind': 'local-directory', 'path': str(root),
                                                     'model_kind': 'original'}
    # With an expected SHA-256 the directory is hashed and checked first.
    sha = local_directory_identity(root)['sha256']
    checked = TransformersBackend.from_local(root, context_length=32, dtype='float32', expected_model_sha256=sha)
    assert checked.metadata()['source_identity']['files']['fixture-weights']['sha256'] == hashlib.sha256(b'model declaration').hexdigest()


FAKE_WORKER = r'''
import argparse, json, math, sys, time
p = argparse.ArgumentParser()
p.add_argument('--model'); p.add_argument('--context-length',type=int)
p.add_argument('--threads',type=int); p.add_argument('--gpu-layers',type=int)
a = p.parse_args()
mode = MODE
for line in sys.stdin:
    q = json.loads(line)
    op = q['op']
    if op == 'metadata':
        if mode == 'stderr':
            sys.stderr.write('x'*100000); sys.stderr.flush()
        result = {'protocol':PROTOCOL,'reduction':REDUCTION,'runtime_sha':SHA,
                  'vocab_size':260,'context_length':a.context_length,'threads':a.threads,
                  'gpu_layers':a.gpu_layers,'cache_mode':'full-prefix-reference',
                  'reset_policy':'per-record','logits_dtype':'float32'}
        if mode == 'metadata-contract': result['cache_mode'] = 'cached'
        if mode == 'metadata-bool': result['gpu_layers'] = False
        if mode == 'metadata-extra': result['unknown'] = 1
    elif op == 'tokenize':
        assert q['add_special'] is False
        result = {'ids':list(q['text'].encode())}
        if mode == 'token-id': result['ids'] = [260]
    elif op == 'token_pieces':
        result = {'pieces_hex':[bytes([i]).hex() for i in q['ids']]}
        if mode == 'pieces': result['pieces_hex'] = ['xz']
    elif op == 'score':
        positions = q['score_positions']
        result = {'score_positions':positions,'target_logprobs':[-math.log(260)]*len(positions),
                  'nll_sum':math.log(260)*len(positions),'count':len(positions)}
        if mode == 'wrong-count': result['count'] += 1
        if mode == 'positive-logprob': result['target_logprobs'][0] = 1.
        if mode == 'wrong-sum': result['nll_sum'] += 1.
        if mode == 'bool-position': result['score_positions'][0] = True
        if mode == 'ambiguous': result['positions'] = result['score_positions']
    elif op == 'generate':
        result = {'generated_ids':[0] if 0 in q['eos_ids'] else [0]*q['max_new_tokens'],
                  'stop_reason':'eos' if 0 in q['eos_ids'] else 'length'}
        if mode == 'short-generation': result['generated_ids'] = []
        if mode == 'early-eos': result = {'generated_ids':[q['eos_ids'][0],0], 'stop_reason':'length'}
        if mode == 'bad-generated-id': result['generated_ids'] = [260]
    else: raise RuntimeError(op)
    response = {'id':q['id'],'ok':True,**result}
    if op != 'metadata':
        if mode == 'timeout': time.sleep(10)
        if mode == 'eof': sys.exit(0)
        if mode == 'oversize': print('x'*10000,flush=True); continue
        if mode == 'noise': print('diagnostic noise',flush=True); continue
        if mode == 'duplicate': print('{"id":1,"id":2,"ok":true}',flush=True); continue
        if mode == 'nonfinite': print('{"id":'+str(q['id'])+',"ok":true,"count":NaN}',flush=True); continue
        if mode == 'wrong-id': response['id'] += 1
        if mode == 'error': response = {'id':q['id'],'ok':False,'error':'intentional'}
    print(json.dumps(response), flush=True)
    if op == 'metadata' and mode == 'blocked-reader': time.sleep(10)
'''


@pytest.fixture
def worker_factory(tmp_path, monkeypatch):
    from openbonsai.eval import backends as eval_backends
    # Explicit interpreter launch makes these pipe tests runnable on Windows,
    # where an executable Python shebang is not recognized by CreateProcess.
    original_popen = subprocess.Popen
    monkeypatch.setattr(eval_backends.subprocess, 'Popen',
                        lambda argv, **kw: original_popen([sys.executable, *argv], **kw))
    created = []
    def factory(mode='normal', **kwargs):
        directory = tmp_path / str(len(created))
        directory.mkdir()
        created.append(None)
        worker, model = directory/'worker', directory/'model.gguf'
        worker.write_text('#!'+sys.executable+'\n'+FAKE_WORKER.replace('MODE', repr(mode))
            .replace('PROTOCOL', repr(PROTOCOL)).replace('REDUCTION', repr(REDUCTION)).replace('SHA', repr(PRISM_RUNTIME_SHA)))
        worker.chmod(0o755)
        model.write_bytes(b'not a real GGUF; protocol fixture only')
        result = PrismBackend(worker, model, context_length=32,
            stderr_path=directory/'stderr.log', request_timeout=kwargs.pop('request_timeout', 3.), **kwargs)
        created[-1] = result
        return result
    yield factory
    for worker in created:
        if worker is not None:
            worker.close()


def test_persistent_worker_roundtrip_contract_and_bounded_stderr(worker_factory):
    worker = worker_factory('stderr', stderr_max_bytes=128)
    assert worker.tokenize('a😄') == list('a😄'.encode())
    assert worker.token_pieces([97, 240, 159]) == [b'a', b'\xf0', b'\x9f']
    score = worker.score_tokens([1, 2, 3, 4], [1, 3])
    assert score['score_positions'] == [1, 3]
    assert score['nll_sum'] == pytest.approx(2*math.log(260))
    assert worker.generate([1], 3, [2]) == {'generated_ids':[0, 0, 0], 'stop_reason':'length'}
    assert worker.generate([1], 3, [0]) == {'generated_ids':[0], 'stop_reason':'eos'}
    assert worker.metadata()['stderr']['truncated']
    assert worker.stderr_path.stat().st_size == 128
    assert worker.metadata()['argv'][-6:] == ['--context-length','32','--threads','1','--gpu-layers','0']
    process = worker._process
    worker.close()
    assert process.poll() is not None
    assert not worker._stdout_thread.is_alive() and not worker._stderr_thread.is_alive()
    with pytest.raises(RuntimeError, match='closed'):
        worker.score_tokens([1, 2], [1])


@pytest.mark.parametrize('mode', ['timeout', 'eof', 'oversize', 'noise', 'duplicate', 'nonfinite',
    'wrong-id', 'error', 'wrong-count', 'positive-logprob', 'wrong-sum', 'bool-position', 'ambiguous'])
def test_portable_worker_failure_closes_without_retry(worker_factory, mode):
    worker = worker_factory(mode, max_response_bytes=2048)
    worker._timeout = .3  # Exclude interpreter cold startup from this timeout probe.
    with pytest.raises((RuntimeError, ValueError, TimeoutError)):
        worker.score_tokens([1, 2], [1])
    assert worker._closed and worker._process.poll() is not None
    assert worker._counter == 2  # Metadata and exactly one failed operation.
    assert not worker._stdout_thread.is_alive()


@pytest.mark.parametrize('mode', ['metadata-contract', 'metadata-bool', 'metadata-extra'])
def test_worker_metadata_must_match_contract(worker_factory, mode):
    with pytest.raises(RuntimeError):
        worker_factory(mode)


@pytest.mark.parametrize('mode,operation', [('pieces','pieces'), ('token-id','tokenize'),
    ('short-generation','generate'), ('early-eos','generate'), ('bad-generated-id','generate')])
def test_worker_output_validation(worker_factory, mode, operation):
    worker = worker_factory(mode)
    with pytest.raises(RuntimeError):
        if operation == 'pieces': worker.token_pieces([1])
        elif operation == 'tokenize': worker.tokenize('a')
        else: worker.generate([1], 2, [3])
    assert worker._closed


def test_client_validation_does_not_send_invalid_request(worker_factory):
    with worker_factory() as worker:
        with pytest.raises(ValueError): worker.score_tokens([1, 2], [0])
        with pytest.raises(ValueError): worker.generate([1], 1, [])
        with pytest.raises(ValueError): worker.tokenize('a', parse_special=1)
        assert worker._counter == 1 and not worker._closed
    assert worker._closed


def test_stalled_worker_request_write_has_a_portable_timeout(worker_factory):
    worker = worker_factory('blocked-reader')
    worker._timeout = .2
    # Exceed OS pipe capacity so the writer, not just response reading, stalls.
    with pytest.raises(TimeoutError, match='request write timed out'):
        worker.tokenize('x' * (2 << 20))
    assert worker._closed and worker._process.poll() is not None
    assert not worker._writer_thread.is_alive() and not worker._stdout_thread.is_alive()


def test_worker_built_from_another_runtime_commit_is_refused(worker_factory):
    with pytest.raises(RuntimeError, match='disagrees'):
        worker_factory(runtime_sha='0' * 40)


def test_files_are_hashed_only_to_check_an_expected_sha(worker_factory):
    worker = worker_factory()
    assert 'sha256' not in worker.metadata()['source_identity']
    worker.close()
    with pytest.raises(ValueError, match='File SHA256 mismatch'):
        worker_factory(expected_model_sha256='0'*64)

"""Independent-record Transformers and pinned Prism backends for eval/manifest.py.

No downloads, templates, BOS insertion, sampling, logits processors or implicit
retries. These adapters establish an evaluation contract, not model equivalence.
Files are hashed only when the caller passes an expected SHA-256 to check; the
Prism worker reports the runtime commit it was built from, which must match
the pin.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import queue
import subprocess
import threading
import time

import torch

PRISM_RUNTIME_SHA = 'c9c34bc9c4042c13a31598ebf43461cd0ad8f00d'
PROTOCOL = 'openbonsai.frozen-evaluation.v1'
REDUCTION = 'fp32-logits-fp64-logsumexp-v1'
_PRECISION_LOCK = threading.RLock()


def _positive(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f'{name} must be a {"nonnegative" if zero else "positive"} integer')
    return value


def _ids(values, vocabulary, *, empty=False):
    if (not isinstance(values, (list, tuple)) or (not values and not empty)
            or any(type(value) is not int or not 0 <= value < vocabulary for value in values)):
        raise ValueError('Expected token IDs within the model vocabulary')
    return list(values)


def _score_inputs(ids, positions, vocabulary, context):
    ids = _ids(ids, vocabulary)
    if len(ids) > context:
        raise ValueError('Input exceeds the declared context length')
    if (not isinstance(positions, (list, tuple)) or not positions
            or any(type(j) is not int or not 1 <= j < len(ids) for j in positions)
            or list(positions) != sorted(set(positions))):
        raise ValueError('score_positions must be nonempty strictly increasing target indices j >= 1')
    return ids, list(positions)


def _generation_inputs(ids, maximum, eos, vocabulary, context):
    ids = _ids(ids, vocabulary)
    _positive(maximum, 'max_new_tokens')
    eos = _ids(eos, vocabulary)
    if len(set(eos)) != len(eos):
        raise ValueError('EOS IDs must be unique')
    if len(ids) + maximum > context:
        raise ValueError('Prompt plus generation cap exceeds the declared context length')
    return ids, eos


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _identity(path, expected_sha256=None, *, hashed=None):
    """Path and size, plus the SHA-256 when ``hashed`` (default: when an expected SHA-256 is given to check)."""
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected a materialized regular file: {path}')
    if not (expected_sha256 is not None if hashed is None else hashed):
        return {'path': str(path), 'bytes': path.stat().st_size}
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError('File changed while hashing')
    result = {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': after.st_size}
    if expected_sha256 is not None and result['sha256'] != expected_sha256:
        raise ValueError(f'File SHA256 mismatch: {path.name}')
    return result


def local_directory_identity(directory):
    """Hash every local artifact file (for an expected-SHA check); never resolve a hub ID."""
    root = Path(directory).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Model/tokenizer must be a materialized local directory')
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Source directory cannot contain symlinks')
        if path.is_file():
            info = _identity(path, hashed=True)
            files[path.relative_to(root).as_posix()] = {'sha256': info['sha256'], 'bytes': info['bytes']}
    if not files:
        raise ValueError('Source directory contains no files')
    return {'kind': 'local-files-sha256', 'path': str(root), 'files': files,
            'sha256': hashlib.sha256(_canonical(files)).hexdigest(),
            'bytes': sum(info['bytes'] for info in files.values())}


def _byte_decoder():
    original = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codes = original[:]
    extra = 0
    for byte in range(256):
        if byte not in original:
            original.append(byte)
            codes.append(256 + extra)
            extra += 1
    return {chr(code): byte for byte, code in zip(original, codes)}


def _load_native(model_id, device='cpu', dtype='auto', *, revision=None, local_files_only=False):
    """Load an original (unquantized) model through its ModelAdapter; no remote code."""
    from ..adapters import load_adapter
    adapter = load_adapter(model_id, revision=revision, local_files_only=local_files_only)
    if isinstance(dtype, str):
        choices = {'auto': torch.bfloat16, 'float32': torch.float32, 'float16': torch.float16,
                   'bfloat16': torch.bfloat16}
        if dtype not in choices:
            raise ValueError('dtype must be auto, float32, float16, bfloat16, or a torch.dtype')
        dtype = choices[dtype]
    model, _ = adapter.load(model_id, revision=revision, device=device, dtype=dtype, attn='eager',
                            local_files_only=local_files_only)
    return model


def _directory(path, expected_sha256, label):
    """Path only, or the hashed identity when an expected SHA-256 is given to check."""
    root = Path(path).absolute()
    if expected_sha256 is None:
        if root.is_symlink() or not root.is_dir():
            raise ValueError('Model/tokenizer must be a materialized local directory')
        return {'kind': 'local-directory', 'path': str(root)}
    identity = local_directory_identity(root)
    if identity['sha256'] != expected_sha256:
        raise ValueError(f'{label} directory SHA256 mismatch')
    return identity


class TransformersBackend:
    """Native dense Qwen3.5 (modules may be swapped in place); caller owns model.

    One native decoder forward computes hidden rows, then the unchanged native
    head is called in bounded output-position batches. Full vocabulary logits
    for every requested position enter the normalizer. No state is reused across
    calls. In-memory identities are declarations; use from_local for file hashes.
    """
    def __init__(self, model, tokenizer, *, context_length, source_identity,
                 tokenizer_identity, logits_chunk_size=16):
        from ..adapters import adapter_for_model
        adapter_for_model(model)  # Refuse architectures without an adapter.
        self.model, self.tokenizer = model, tokenizer
        self.context_length = _positive(context_length, 'context_length')
        self.logits_chunk_size = _positive(logits_chunk_size, 'logits_chunk_size')
        config = getattr(model.config, 'text_config', model.config)
        self.vocab_size = _positive(config.vocab_size, 'vocab_size')
        maximum = getattr(config, 'max_position_embeddings', None)
        if isinstance(maximum, int) and self.context_length > maximum:
            raise ValueError('Declared context exceeds native model configuration')
        if not getattr(tokenizer, 'is_fast', False) or not hasattr(tokenizer, 'backend_tokenizer'):
            raise ValueError('Exact token pieces require a ByteLevel fast tokenizer')
        decoder = tokenizer.backend_tokenizer.decoder
        state = json.loads(decoder.__getstate__()) if decoder is not None else {}
        if state.get('type') != 'ByteLevel':
            raise ValueError('Unsupported tokenizer decoder; exact raw pieces are not established')
        normalizer = tokenizer.backend_tokenizer.normalizer
        self._tokenization = {'normalizer': json.loads(normalizer.__getstate__()) if normalizer else None,
                              'decoder': state, 'add_special_tokens': False,
                              'parse_special': 'explicit per request',
                              'raw_piece_policy': 'ByteLevel bytes; ASCII added tokens only'}
        self._byte_decoder = _byte_decoder()
        self._added = set(tokenizer.get_added_vocab())
        self._source, self._tokenizer_identity = copy.deepcopy(source_identity), copy.deepcopy(tokenizer_identity)
        _canonical(self._source); _canonical(self._tokenizer_identity)
        self._lock = threading.RLock()
        self.model.eval()

    @classmethod
    def from_local(cls, model_path, *, tokenizer_path=None, context_length,
                   model_kind='original', device='cpu', dtype='auto', logits_chunk_size=16,
                   expected_model_sha256=None, expected_tokenizer_sha256=None):
        from transformers import AutoTokenizer
        model_path = Path(model_path).absolute()
        if model_kind != 'original':
            # Mixed artifacts are a legacy format (research/legacy; tag legacy-staged-pipeline).
            raise ValueError('model_kind must be original')
        tokenizer_path = Path(tokenizer_path).absolute() if tokenizer_path is not None else model_path
        source = _directory(model_path, expected_model_sha256, 'Model')
        tokens = (copy.deepcopy(source) if tokenizer_path == model_path and expected_tokenizer_sha256 is None
                  else _directory(tokenizer_path, expected_tokenizer_sha256, 'Tokenizer'))
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
        model = _load_native(str(model_path), device=device, dtype=dtype, local_files_only=True)
        source['model_kind'] = model_kind
        return cls(model, tokenizer, context_length=context_length, source_identity=source,
                   tokenizer_identity=tokens, logits_chunk_size=logits_chunk_size)

    def metadata(self):
        import transformers
        return copy.deepcopy({'backend': 'transformers', 'protocol': PROTOCOL, 'reduction': REDUCTION,
            'context_length': self.context_length, 'vocab_size': self.vocab_size,
            'tokenizer_vocab_size': len(self.tokenizer), 'cache_mode': 'full-prefix-reference',
            'reset_policy': 'per-record', 'reset_method': 'native use_cache=False; no past_key_values',
            'source_identity': self._source, 'tokenizer_identity': self._tokenizer_identity,
            'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__},
            'tokenization': self._tokenization,
            'settings': {'logits_chunk_size': self.logits_chunk_size, 'logits_dtype': 'float32',
                         'reduction_dtype': 'float64', 'tf32': False,
                         'logits_strategy': 'native-hidden-once-head-position-chunks',
                         'dtypes': sorted({str(p.dtype) for p in self.model.parameters()})}})

    def tokenize(self, text, parse_special=False):
        if not isinstance(text, str) or type(parse_special) is not bool:
            raise ValueError('tokenize requires text and a boolean parse_special')
        text.encode('utf-8')  # Reject invalid surrogate strings.
        with self._lock:
            result = self.tokenizer.encode(text, add_special_tokens=False, truncation=False,
                                           split_special_tokens=not parse_special)
        return _ids(result, self.vocab_size, empty=True)

    def token_pieces(self, ids):
        ids = _ids(ids, self.vocab_size, empty=True)
        pieces = self.tokenizer.convert_ids_to_tokens(ids)
        result = []
        for piece in pieces:
            if not isinstance(piece, str):
                raise ValueError('Model token ID has no tokenizer piece (possibly an unused padded row)')
            if piece in self._added:
                # ByteLevel may reinterpret non-ASCII added-token characters
                # as byte markers (e.g. Ġ -> space), even when normalization
                # is disabled. Refuse those unestablished semantics rather
                # than treating the displayed spelling as raw UTF-8 bytes.
                if not piece.isascii():
                    raise ValueError('Non-ASCII added-token raw bytes are unsupported')
                result.append(piece.encode('utf-8'))
            else:
                try:
                    result.append(bytes(self._byte_decoder[c] for c in piece))
                except KeyError as error:
                    raise ValueError('Tokenizer piece is incompatible with ByteLevel raw-byte decoding') from error
        return result

    def _hidden(self, ids):
        embedding = self.model.get_input_embeddings()
        # A TernaryEmbedding keeps its table in buffers: find the device from
        # whatever storage the module has.
        reference = next(embedding.parameters(), None)
        reference = next(embedding.buffers(), None) if reference is None else reference
        if reference is None:
            raise RuntimeError('Native embedding has no discoverable storage device')
        device = reference.device
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        output = self.model.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                  past_key_values=None, use_cache=False, return_dict=True)
        hidden = output.last_hidden_state
        if hidden.shape[:2] != (1, len(ids)) or not bool(torch.isfinite(hidden).all()):
            raise RuntimeError('Native decoder returned invalid hidden states')
        return hidden

    def _logits(self, hidden, prediction_positions):
        positions = torch.tensor(prediction_positions, dtype=torch.long, device=hidden.device)
        output = self.model.lm_head(hidden.index_select(1, positions)).float().detach().cpu()[0]
        if output.shape != (len(prediction_positions), self.vocab_size) or not bool(torch.isfinite(output).all()):
            raise RuntimeError('Native head returned invalid full-vocabulary logits')
        return output

    @contextmanager
    def _precision(self):
        # Scope rather than permanently mutate the caller's CUDA policy.
        with _PRECISION_LOCK:
            matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                yield
            finally:
                torch.backends.cuda.matmul.allow_tf32 = matmul
                torch.backends.cudnn.allow_tf32 = cudnn

    @torch.no_grad()
    def score_tokens(self, input_ids, score_positions):
        from .manifest import reduce_target_logprobs
        ids, positions = _score_inputs(input_ids, score_positions, self.vocab_size, self.context_length)
        with self._lock, self._precision():
            hidden = self._hidden(ids)
            probabilities = []
            for start in range(0, len(positions), self.logits_chunk_size):
                chunk = positions[start:start + self.logits_chunk_size]
                logits = self._logits(hidden, [j - 1 for j in chunk])
                probabilities.extend(reduce_target_logprobs(logits, [ids[j] for j in chunk]))
        return {'score_positions': positions, 'target_logprobs': probabilities,
                'nll_sum': -math.fsum(probabilities), 'count': len(positions)}

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, eos_ids):
        ids, eos = _generation_inputs(input_ids, max_new_tokens, eos_ids, self.vocab_size, self.context_length)
        generated = []
        with self._lock, self._precision():
            for _ in range(max_new_tokens):
                hidden = self._hidden(ids + generated)
                logits = self._logits(hidden, [hidden.shape[1] - 1])[0]
                token = int(logits.argmax())  # PyTorch returns the lowest index in a tie.
                generated.append(token)
                if token in eos:
                    return {'generated_ids': generated, 'stop_reason': 'eos'}
        return {'generated_ids': generated, 'stop_reason': 'length'}

    def close(self):
        """The caller owns the in-memory native model; no process to close."""


def _unique_json(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('Worker response contains duplicate JSON keys')
        result[name] = value
    return result


class PrismBackend:
    """Persistent strict JSONL subprocess; failures close it without retry."""
    def __init__(self, executable, model_path, *, context_length,
                 stderr_path, threads=1, gpu_layers=0, runtime_sha=PRISM_RUNTIME_SHA,
                 expected_executable_sha256=None, expected_model_sha256=None,
                 request_timeout=300., max_response_bytes=4 << 20, stderr_max_bytes=1 << 20):
        self.context_length = _positive(context_length, 'context_length')
        self.threads = _positive(threads, 'threads')
        self.gpu_layers = _positive(gpu_layers, 'gpu_layers', zero=True)
        self._maximum = _positive(max_response_bytes, 'max_response_bytes')
        self._stderr_max = _positive(stderr_max_bytes, 'stderr_max_bytes')
        if isinstance(request_timeout, bool) or not isinstance(request_timeout, (int, float)) or not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError('request_timeout must be finite and positive')
        self._timeout, self._counter = request_timeout, 0
        self._stdout_queue, self._stdout_error = queue.Queue(maxsize=1), None
        self._lock, self._closed = threading.RLock(), False
        self._runtime = {'sha': runtime_sha}   # the worker must report this commit (_validate_metadata)
        self._executable = _identity(executable, expected_executable_sha256)
        self._model = _identity(model_path, expected_model_sha256)
        self.stderr_path = Path(stderr_path).absolute()
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_handle = self.stderr_path.open('xb')
        self._stderr_seen = 0
        argv = [self._executable['path'], '--model', self._model['path'],
                '--context-length', str(context_length), '--threads', str(threads), '--gpu-layers', str(gpu_layers)]
        self._argv = argv
        try:
            self._process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, bufsize=0)
        except BaseException:
            self._stderr_handle.close()
            self._closed = True
            raise
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stdout_thread.start()
        try:
            self._worker = self._request('metadata')
            self._validate_metadata(runtime_sha)
            self.vocab_size = self._worker['vocab_size']
        except BaseException:
            self.close()
            raise

    def _drain_stderr(self):
        try:
            while True:
                block = self._process.stderr.read(8192)
                if not block:
                    break
                keep = max(0, self._stderr_max - self._stderr_seen)
                if keep:
                    self._stderr_handle.write(block[:keep])
                    self._stderr_handle.flush()
                self._stderr_seen += len(block)
        finally:
            self._stderr_handle.close()

    def _read_stdout(self):
        # select()/selectors cannot wait on Windows subprocess pipes. One
        # bounded reader thread plus a single-slot queue is portable, and
        # neither an overlong line nor unsolicited output can grow memory.
        try:
            while True:
                line = self._process.stdout.readline(self._maximum + 1)
                item = ('line', line)
                if not line:
                    item = ('error', 'Prism worker closed stdout before completing a response')
                elif len(line) > self._maximum:
                    item = ('error', 'Prism worker response exceeds the byte limit')
                elif not line.endswith(b'\n'):
                    item = ('error', 'Prism worker returned an unterminated response')
                try:
                    self._stdout_queue.put_nowait(item)
                except queue.Full:
                    self._stdout_error = 'Unsolicited extra worker output'
                    return
                if item[0] == 'error':
                    return
        except (OSError, ValueError) as error:
            if not self._closed:
                self._stdout_error = f'Worker stdout read failed: {error}'

    def _line(self, timeout):
        try:
            kind, value = self._stdout_queue.get(timeout=max(0., timeout))
        except queue.Empty:
            raise TimeoutError('Prism worker response timed out') from None
        if self._stdout_error is not None:
            raise RuntimeError(self._stdout_error)
        if kind == 'error':
            raise RuntimeError(value)
        return value

    def _write_request(self, data):
        """One bounded writer per request; stalled pipe writes also time out."""
        completed = queue.Queue(maxsize=1)
        def write():
            try:
                remaining = memoryview(data)
                while remaining:
                    amount = self._process.stdin.write(remaining)
                    if not amount:
                        raise RuntimeError('Prism worker stdin closed')
                    remaining = remaining[amount:]
                self._process.stdin.flush()
                completed.put_nowait(None)
            except BaseException as error:
                completed.put_nowait(error)
        self._writer_thread = threading.Thread(target=write, daemon=True)
        self._writer_thread.start()
        try:
            error = completed.get(timeout=self._timeout)
        except queue.Empty:
            raise TimeoutError('Prism worker request write timed out') from None
        self._writer_thread.join()
        if error is not None:
            raise error

    def _request(self, operation, **fields):
        with self._lock:
            if self._closed:
                raise RuntimeError('Prism backend is closed')
            self._counter += 1
            request = {'id': self._counter, 'op': operation, **fields}
            try:
                if self._stdout_error is not None or not self._stdout_queue.empty():
                    raise RuntimeError(self._stdout_error or 'Unsolicited worker output before request')
                data = _canonical(request) + b'\n'
                if len(data) > 16 << 20:
                    raise ValueError('Worker request exceeds the line limit')
                deadline = time.monotonic() + self._timeout
                self._write_request(data)
                response = json.loads(self._line(deadline - time.monotonic()), object_pairs_hook=_unique_json,
                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite worker JSON')))
                if (not isinstance(response, dict) or type(response.get('id')) is not int
                        or response['id'] != self._counter or type(response.get('ok')) is not bool):
                    raise RuntimeError('Invalid worker response ID or envelope')
                if not response['ok']:
                    if set(response) != {'id', 'ok', 'error'} or not isinstance(response['error'], str):
                        raise RuntimeError('Invalid worker error schema')
                    raise RuntimeError(f'Prism worker rejected request: {response["error"]}')
                return {key: value for key, value in response.items() if key not in ('id', 'ok')}
            except BaseException:
                self.close()
                raise

    def _validate_metadata(self, runtime_sha):
        required = {'protocol', 'reduction', 'cache_mode', 'reset_policy', 'logits_dtype',
                    'runtime_sha', 'vocab_size', 'context_length', 'threads', 'gpu_layers'}
        optional = {'allocated_context_length', 'model_path', 'cuda_build', 'batch_size', 'ubatch_size',
                    'kv_cache_dtype', 'flash_attention', 'load_mtp', 'native_eog_ids'}
        if not required.issubset(self._worker) or set(self._worker) - required - optional:
            raise RuntimeError('Worker metadata is missing required contract fields')
        expected = {'protocol': PROTOCOL, 'reduction': REDUCTION, 'cache_mode': 'full-prefix-reference',
                    'reset_policy': 'per-record', 'logits_dtype': 'float32', 'runtime_sha': runtime_sha,
                    'context_length': self.context_length, 'threads': self.threads, 'gpu_layers': self.gpu_layers}
        if any(type(self._worker[name]) is not type(value) or self._worker[name] != value for name, value in expected.items()):
            raise RuntimeError('Worker metadata disagrees with the requested evaluation contract')
        _positive(self._worker['vocab_size'], 'worker vocab_size')

    def metadata(self):
        return copy.deepcopy({**self._worker, 'backend': 'prism', 'source_identity': self._model,
            'tokenizer_identity': {'kind': 'embedded-gguf', 'model': self._model},
            'runtime': {**self._runtime, 'executable': self._executable}, 'argv': self._argv,
            'stderr': {'path': str(self.stderr_path), 'max_bytes': self._stderr_max,
                       'truncated': self._stderr_seen > self._stderr_max}})

    def _fields(self, result, names):
        if set(result) != set(names):
            self.close()
            raise RuntimeError('Worker returned unexpected response fields')

    def tokenize(self, text, parse_special=False):
        if not isinstance(text, str) or type(parse_special) is not bool:
            raise ValueError('tokenize requires text and a boolean parse_special')
        text.encode('utf-8')
        result = self._request('tokenize', text=text, parse_special=parse_special, add_special=False)
        self._fields(result, ['ids'])
        try:
            return _ids(result['ids'], self.vocab_size, empty=True)
        except ValueError:
            self.close()
            raise RuntimeError('Worker returned invalid token IDs') from None

    def token_pieces(self, ids):
        ids = _ids(ids, self.vocab_size, empty=True)
        result = self._request('token_pieces', ids=ids)
        self._fields(result, ['pieces_hex'])
        if not isinstance(result['pieces_hex'], list) or len(result['pieces_hex']) != len(ids):
            self.close(); raise RuntimeError('Worker token-piece count mismatch')
        pieces = []
        for value in result['pieces_hex']:
            if not isinstance(value, str) or len(value) % 2 or any(c not in '0123456789abcdef' for c in value):
                self.close(); raise RuntimeError('Worker returned invalid hexadecimal token pieces')
            pieces.append(bytes.fromhex(value))
        return pieces

    def score_tokens(self, input_ids, score_positions):
        ids, positions = _score_inputs(input_ids, score_positions, self.vocab_size, self.context_length)
        result = self._request('score', input_ids=ids, score_positions=positions)
        self._fields(result, ['score_positions', 'target_logprobs', 'nll_sum', 'count'])
        values = result['target_logprobs']
        if (not isinstance(result['score_positions'], list)
                or any(type(j) is not int for j in result['score_positions'])
                or result['score_positions'] != positions or type(result['count']) is not int
                or result['count'] != len(positions) or not isinstance(values, list)
                or len(values) != len(positions)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v > 0 for v in values)
                or type(result['nll_sum']) not in (int, float) or not math.isfinite(result['nll_sum'])
                or not math.isclose(result['nll_sum'], -math.fsum(values), rel_tol=1e-12, abs_tol=1e-10)):
            self.close(); raise RuntimeError('Worker score result violates the normalization contract')
        return result

    def generate(self, input_ids, max_new_tokens, eos_ids):
        ids, eos = _generation_inputs(input_ids, max_new_tokens, eos_ids, self.vocab_size, self.context_length)
        result = self._request('generate', input_ids=ids, max_new_tokens=max_new_tokens, eos_ids=eos)
        self._fields(result, ['generated_ids', 'stop_reason'])
        try:
            generated = _ids(result['generated_ids'], self.vocab_size, empty=True)
        except ValueError:
            self.close()
            raise RuntimeError('Worker returned invalid generated IDs') from None
        reason = result['stop_reason']
        if (reason not in ('eos', 'length') or len(generated) > max_new_tokens
                or any(value in eos for value in generated[:-1])
                or (reason == 'eos' and (not generated or generated[-1] not in eos))
                or (reason == 'length' and (len(generated) != max_new_tokens
                                          or (generated and generated[-1] in eos)))):
            self.close(); raise RuntimeError('Worker generation violates EOS or exact-cap semantics')
        return result

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if hasattr(self, '_process'):
                # Release a blocked FileIO.write before closing its descriptor.
                if getattr(self, '_writer_thread', None) is not None and self._writer_thread.is_alive():
                    if self._process.poll() is None:
                        self._process.terminate()
                    self._writer_thread.join(timeout=1.)
                    if self._writer_thread.is_alive() and self._process.poll() is None:
                        self._process.kill()
                        self._writer_thread.join(timeout=1.)
                try:
                    self._process.stdin.close()
                except (OSError, BrokenPipeError):
                    pass
                try:
                    self._process.wait(timeout=1.)
                except subprocess.TimeoutExpired:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=1.)
                    except subprocess.TimeoutExpired:
                        self._process.kill(); self._process.wait(timeout=1.)
                if hasattr(self, '_stderr_thread'):
                    self._stderr_thread.join(timeout=2.)
                if hasattr(self, '_stdout_thread'):
                    self._stdout_thread.join(timeout=2.)
                self._process.stdout.close()
                self._process.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

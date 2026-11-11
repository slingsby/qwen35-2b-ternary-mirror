"""Generation checks used by evaluate.py: a repetition guard and per-prompt format validators.

These are format/syntax/value checks, not functional correctness tests.
"""
import ast
import json
import re


def looping_tail(ids):
    """True when the last 96 generated tokens repeat with a period of 1..32 tokens."""
    if len(ids) < 96: return False
    tail = ids[-96:]
    return any(all(tail[j] == tail[j % period] for j in range(96)) for period in range(1, 33))


def stripped_block(text, kind):
    blocks = re.findall(r'```(?:' + kind + r')?\s*\n(.*?)```', text, re.S | re.I)
    return blocks[0].strip() if blocks else text.strip()


def validate_response(spec, text):
    """Return (passed, reason) for one prompt spec: python | json | exact | integer."""
    try:
        kind = spec['validator']
        if kind == 'python':
            tree = ast.parse(stripped_block(text, 'python'))
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == spec['function']]
            return (len(functions) == 1 and any(isinstance(n, ast.Return) for n in ast.walk(functions[0]))), 'Python syntax/function/return check'
        if kind == 'json':
            actual = json.loads(stripped_block(text, 'json'))
            correct = actual == spec['expected']
            if correct:
                args = actual['arguments']; correct = all(type(args[k]) is type(v) for k, v in spec['expected']['arguments'].items())
            return correct, 'JSON schema/argument check'
        if kind == 'exact': return text.strip() == spec['expected'], 'exact instruction check'
        if kind == 'integer':
            numbers = re.findall(r'-?\d+(?:\.\d+)?', text.replace(',', ''))
            return bool(numbers) and float(numbers[-1]) == spec['expected'], 'arithmetic final-value check'
        return False, 'unknown validator'
    except (SyntaxError, ValueError, KeyError, TypeError, OverflowError): return False, 'response parse failure'

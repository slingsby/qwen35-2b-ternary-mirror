"""Scoring for the reasoning suites: GSM8K final-number exact match, HumanEval pass@1.

GSM8K: ``extract_answer`` takes the model's final number, preferring an
explicit marker: the last ``\\boxed{...}`` holding a number, else the number
after the last ``####``, else the answer after the last "answer is" /
"answer:" (the clause up to the end of that sentence or line; when it states
an equation, the number after its last ``=``, else its first number), else
the last number in the text. ``normalize_number`` drops thousands separators,
``$``, a ``+`` sign and trailing zeros (``1,080.00`` -> ``1080``, ``-0`` ->
``0``), so the match is on the value, not its spelling; a run of more than
``MAX_DIGITS`` digits is not a number (a looping generation, never an answer).

HumanEval: ``extract_code`` takes the code from fenced blocks (the last one
that defines the entry point, else the first) or the raw text; ``prepare_program``
prepends the problem's prompt (so a continued function body completes the
stub, and a full redefinition keeps the prompt's imports and helpers) and
appends the official ``test`` and ``check(entry_point)``. Code indented as a
whole (a fence under a markdown list item) is dedented first. The model's own
top-level statements are then cleaned as EvalPlus's sanitizer does: only
imports, function and class definitions and the module-level assignments
those definitions read are kept, so the model's example calls, its own
asserts, ``print`` and ``input()`` calls never run; ``from __future__``
imports move to the top of the program. Code that does not parse is left as
written (it fails). ``run_program`` executes the program.

Execution is NOT a security sandbox. Every program runs in a separate Python
process (``python -I -X utf8``: no user site, no PYTHON* variables, script
directory not on sys.path, UTF-8 standard streams on every OS) in a fresh
temporary directory, with stdin closed, a reduced environment (temp and home
pointed at that directory, no GPU), stdout and stderr written to files, and
limits: a wall-clock timeout, ``OUTPUT_LIMIT`` bytes of output and
``MEMORY_LIMIT`` bytes of memory (address space through ``prlimit`` on Linux,
committed memory of the whole process tree through a Job Object on Windows;
no memory limit on other systems). When the program ends or breaks a limit,
it and every process it started are killed (its process group on POSIX, its
Job Object on Windows). That protects the evaluator from crashes, hangs,
floods, runaway allocations, leftover child processes and ``exit()`` in model
code, which never runs in the evaluator's process. It does not stop the code
from reading or writing your files or opening network connections: the suite
has no network intent, and nothing enforces that. Evaluate untrusted models in
a container or VM.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import textwrap
import time

# -- GSM8K -----------------------------------------------------------------

# A number not glued to a preceding word, digit or dot; thousands groups of exactly 3 digits.
_NUMBER = re.compile(r"(?<![\w.])[-+\u2212]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_FRAC = re.compile(r"\\[dt]?frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}")
_ANSWER_IS = re.compile(r"answer\s*(?:is\b|:)", re.I)
_CLAUSE_END = re.compile(r"\n|[.!?;](?!\d)")   # the end of a sentence or line (not a decimal point)
MAX_DIGITS = 64   # a longer digit run is no answer (a looping generation); str(int) refuses > 4300 digits


def normalize_number(text: str) -> str | None:
    """Canonical string of a number (``"1,080.00"`` -> ``"1080"``, ``"-0.50"`` -> ``"-0.5"``); None if not one."""
    s = str(text).strip().replace(",", "").replace("$", "").replace("\u2212", "-").replace(" ", "")
    s = s.rstrip(".")
    if s.startswith("+"):
        s = s[1:]
    if sum(c.isdigit() for c in s) > MAX_DIGITS:
        return None
    try:
        value = Decimal(s)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def _numbers(text: str) -> list[str]:
    return [m.group(0) for m in _NUMBER.finditer(text)]


def _braced(text: str, start: int) -> tuple[str, int] | None:
    """Content of the brace group opening at ``text[start] == '{'`` (nesting allowed), and its end."""
    depth = 0
    for k in range(start, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:k], k + 1
    return None


def boxed_contents(text: str) -> list[str]:
    """Contents of every ``\\boxed{...}`` / ``\\fbox{...}``, in order."""
    out = []
    for m in re.finditer(r"\\(?:boxed|fbox)\s*(?=\{)", text):
        got = _braced(text, m.end())
        if got is not None:
            out.append(got[0])
    return out


def _latex_number(content: str) -> str | None:
    """The value of a boxed answer: a ``\\frac{a}{b}``, else its last number."""
    c = content.replace("{,}", ",").replace("\\,", "").replace("\\!", "").replace("\\$", "$")
    frac = _FRAC.search(c)
    if frac:
        a, b = normalize_number(frac.group(1)), normalize_number(frac.group(2))
        if a is not None and b is not None and Decimal(b) != 0:
            return normalize_number(str(Decimal(a) / Decimal(b)))
    nums = _numbers(c)
    return normalize_number(nums[-1]) if nums else None


def answer_after_marker(tail: str) -> str | None:
    """The number an "answer is" / "answer:" marker introduces, from the text after it: in the clause
    up to the end of its sentence or line, the number after the last ``=`` when it states an equation
    (``Answer: 16 - 3 - 4 = 9``), else the clause's first number (``18 (6 boxes of 3)``); else the
    first number of the tail."""
    tail = tail.replace("\\$", "$").lstrip(" \t\r\n*:_`")      # "**Answer:**\n\n18"
    end = _CLAUSE_END.search(tail)
    clause = tail[:end.start()] if end else tail
    for part in (clause.rsplit("=", 1)[1] if "=" in clause else "", clause, tail):
        nums = _numbers(part)
        if nums:
            return nums[0]
    return None


def extract_answer(text: str) -> tuple[str | None, str]:
    """(normalized final number or None, method): boxed, ####, answer is, last number, none."""
    for content in reversed(boxed_contents(text)):
        value = _latex_number(content)
        if value is not None:
            return value, "boxed"
    if "####" in text:
        nums = _numbers(text.rsplit("####", 1)[1])
        if nums:
            return normalize_number(nums[0]), "####"
    hits = list(_ANSWER_IS.finditer(text))
    if hits:
        number = answer_after_marker(text[hits[-1].end():hits[-1].end() + 200])
        if number is not None:
            return normalize_number(number), "answer is"
    nums = _numbers(text)
    if nums:
        return normalize_number(nums[-1]), "last number"
    return None, "none"


def gsm8k_reference(solution: str) -> str:
    """The reference answer of a GSM8K solution: the number after ``####``."""
    if "####" not in solution:
        raise ValueError("GSM8K solution without ####")
    value = normalize_number(solution.rsplit("####", 1)[1])
    if value is None:
        raise ValueError(f"GSM8K reference is not a number: {solution.rsplit('####', 1)[1]!r}")
    return value


# -- HumanEval ---------------------------------------------------------------

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[^\n]*\n(.*?)(?:```|\Z)", re.S)
_CODE_LANGS = {"", "python", "py", "python3", "py3"}
_MAIN = re.compile(r"""^if\s+__name__\s*==\s*['"]__main__['"]\s*:""")


def _defines(code: str, name: str) -> bool:
    return re.search(rf"^def\s+{re.escape(name)}\s*\(", code, re.M) is not None


def strip_main_block(code: str) -> str:
    """Drop a top-level ``if __name__ == "__main__":`` block (example usage that would run, or wait on input)."""
    lines, out, skipping = code.split("\n"), [], False
    for line in lines:
        if _MAIN.match(line):
            skipping = True
            continue
        if skipping and (not line.strip() or line[:1] in (" ", "\t")):
            continue
        skipping = False
        out.append(line)
    return "\n".join(out)


_CODE_START = re.compile(r"^(?:from\s+\S+\s+import\s|import\s|def\s|class\s|async\s+def\s|@)", re.M)


def extract_code(text: str, entry_point: str) -> tuple[str, str]:
    """(code, how): from the last fenced block that defines ``entry_point``, else the first fenced
    Python block, else the raw text (from its first top-level import/def when it defines the entry
    point, so a line of prose before unfenced code is dropped). A fence the generation cap cut off
    still counts; a block indented as a whole (under a markdown list item) counts as written."""
    blocks = [m.group(2) for m in _FENCE.finditer(text) if m.group(1).lower() in _CODE_LANGS]
    defining = [b for b in blocks if _defines(textwrap.dedent(b), entry_point)]
    if defining:
        return defining[-1], "fenced"
    if blocks:
        return blocks[0], "fenced"
    start = _CODE_START.search(text) if _defines(text, entry_point) else None
    return (text[start.start():] if start else text), "raw"


_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _free_names(node) -> set[str]:
    """Names a definition reads from the module: loaded and not bound inside it (or declared global)."""
    loads, binds, declared = set(), set(), set()
    for x in ast.walk(node):
        if isinstance(x, ast.Name):
            (loads if isinstance(x.ctx, ast.Load) else binds).add(x.id)
        elif isinstance(x, ast.arg):
            binds.add(x.arg)
        elif isinstance(x, (ast.Global, ast.Nonlocal)):
            declared.update(x.names)
    return (loads - binds) | declared


def _loads(node) -> set[str]:
    return {x.id for x in ast.walk(node) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}


def clean_program(body: str, first_line: int) -> tuple[str, int]:
    """(source, statements dropped). The model's top-level statements, those starting at line
    ``first_line`` (1-based) or later, keep only imports, function and class definitions and the
    assignments the module's definitions read (transitively); every other top-level statement
    (calls, prints, asserts, loops, ``if __name__ == "__main__":``) is dropped, as EvalPlus's
    sanitizer does. ``from __future__`` imports move to the first line. Source that does not parse
    is returned unchanged."""
    try:
        tree = ast.parse(body)
    except (SyntaxError, ValueError):
        return body, 0

    def first(node):
        return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", ())])

    used = set().union(*(_free_names(n) for n in tree.body if isinstance(n, _DEFINITIONS)))
    assigns = [n for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None]
    kept_assigns: set[int] = set()
    grew = True
    while grew:
        grew = False
        for n in assigns:
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            names = {x.id for t in targets for x in ast.walk(t) if isinstance(x, ast.Name)}
            if id(n) not in kept_assigns and names & used:
                kept_assigns.add(id(n))
                used |= _loads(n.value)
                grew = True
    lines = body.split("\n")
    keep_lines, drop_lines, future, dropped = set(), set(), [], 0
    for n in tree.body:
        span = set(range(first(n), n.end_lineno + 1))
        if isinstance(n, ast.ImportFrom) and n.module == "__future__" and first(n) >= first_line:
            future += lines[first(n) - 1:n.end_lineno]
            drop_lines |= span
        elif first(n) < first_line or isinstance(n, (ast.Import, ast.ImportFrom) + _DEFINITIONS) \
                or id(n) in kept_assigns:
            keep_lines |= span
        else:
            drop_lines |= span
            dropped += 1
    drop_lines -= keep_lines     # a line shared with a kept statement (``f(); x = 1``) stays
    if not drop_lines:
        return body, 0
    kept = [ln for k, ln in enumerate(lines, 1) if k not in drop_lines]
    return "\n".join(future + kept), dropped


def _parses(source: str) -> bool:
    try:
        ast.parse(source)
        return True
    except (SyntaxError, ValueError):
        return False


def _indent(code: str) -> str:
    return "".join(("    " + ln if ln.strip() else ln) + "\n" for ln in code.rstrip("\n").split("\n"))


def prepare_program(prompt: str, code: str, test: str, entry_point: str, *, clean: bool = True) -> dict:
    """{"program", "mode", "dropped"}. Modes: ``function`` (the code defines the entry point; it follows
    the prompt, so the prompt's imports and helpers stay), ``continuation`` (indented code continues the
    prompt's function body), ``body`` (an unindented body without definitions; indented under the
    prompt), ``module`` (other definitions). Code that defines the entry point while indented as a
    whole (a fence under a markdown list item) is dedented; an indented body stays as written when
    it parses after the prompt (it may open with a nested helper ``def``), else it is re-indented
    under the prompt (a body indented by 3 spaces). With ``clean``, ``clean_program`` drops the
    model's other top-level statements (``dropped`` counts them)."""
    code = code.replace("\r\n", "\n")
    first = next((ln for ln in code.split("\n") if ln.strip()), "")
    head = prompt if prompt.endswith("\n") else prompt + "\n"
    flat = strip_main_block(textwrap.dedent(code)).rstrip() + "\n"
    code = strip_main_block(code).rstrip() + "\n"
    if _defines(flat, entry_point):
        code, lead, mode = flat, head + "\n", "function"
    elif first[:1] in (" ", "\t"):
        lead, mode = head, "continuation"
        if not _parses(head + code) and _parses(head + _indent(flat)):
            code = _indent(flat)
    elif not re.search(r"^(def|class|async\s+def)\s", code, re.M):
        lead, mode, code = head, "body", _indent(code)
    else:
        lead, mode = head + "\n", "module"
    body, dropped = lead + code, 0
    if clean:
        body, dropped = clean_program(body, lead.count("\n") + 1)
    return {"program": f"{body}\n\n{test}\n\ncheck({entry_point})\n", "mode": mode, "dropped": dropped}


def build_program(prompt: str, code: str, test: str, entry_point: str, *, clean: bool = True) -> tuple[str, str]:
    """(program, mode) of ``prepare_program``."""
    p = prepare_program(prompt, code, test, entry_point, clean=clean)
    return p["program"], p["mode"]


# -- execution -------------------------------------------------------------------

OUTPUT_LIMIT = 8 * 2**20     # bytes of stdout + stderr before the program is killed (and fails)
MEMORY_LIMIT = 2 * 2**30     # bytes of memory per program (and everything it starts)


def _child_env(tmp: str) -> dict:
    keep = ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "LANG", "LC_ALL")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update({"TMP": tmp, "TEMP": tmp, "TMPDIR": tmp, "HOME": tmp, "USERPROFILE": tmp,
                "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1"})
    return env


_KERNEL32 = None


def _kernel32():
    """kernel32 with the Job Object calls typed (Windows)."""
    global _KERNEL32
    if _KERNEL32 is None:
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.restype = wintypes.HANDLE
        k.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        k.SetInformationJobObject.restype = wintypes.BOOL
        k.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        k.QueryInformationJobObject.restype = wintypes.BOOL
        k.QueryInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                                ctypes.c_void_p)
        k.OpenProcess.restype = wintypes.HANDLE
        k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k.AssignProcessToJobObject.restype = wintypes.BOOL
        k.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        k.TerminateJobObject.restype = wintypes.BOOL
        k.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        k.CloseHandle.restype = wintypes.BOOL
        k.CloseHandle.argtypes = (wintypes.HANDLE,)
        k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        k.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
        k.Thread32First.restype = wintypes.BOOL
        k.Thread32First.argtypes = (wintypes.HANDLE, ctypes.c_void_p)
        k.Thread32Next.restype = wintypes.BOOL
        k.Thread32Next.argtypes = (wintypes.HANDLE, ctypes.c_void_p)
        k.OpenThread.restype = wintypes.HANDLE
        k.OpenThread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k.ResumeThread.restype = wintypes.DWORD
        k.ResumeThread.argtypes = (wintypes.HANDLE,)
        _KERNEL32 = k
    return _KERNEL32


def _resume_windows(pid: int) -> int:
    """Resume the threads of a process started with CREATE_SUSPENDED; returns how many."""
    import ctypes
    from ctypes import wintypes

    class ThreadEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ThreadID", wintypes.DWORD),
                    ("th32OwnerProcessID", wintypes.DWORD), ("tpBasePri", wintypes.LONG),
                    ("tpDeltaPri", wintypes.LONG), ("dwFlags", wintypes.DWORD)]

    k = _kernel32()
    snap = k.CreateToolhelp32Snapshot(0x4, 0)          # TH32CS_SNAPTHREAD
    if not snap or snap == wintypes.HANDLE(-1).value:
        return 0
    resumed = 0
    try:
        entry = ThreadEntry()
        entry.dwSize = ctypes.sizeof(entry)
        ok = k.Thread32First(snap, ctypes.byref(entry))
        while ok:
            if entry.th32OwnerProcessID == pid:
                thread = k.OpenThread(0x0002, False, entry.th32ThreadID)   # THREAD_SUSPEND_RESUME
                if thread:
                    resumed += k.ResumeThread(thread) != 0xFFFFFFFF
                    k.CloseHandle(thread)
            ok = k.Thread32Next(snap, ctypes.byref(entry))
    finally:
        k.CloseHandle(snap)
    return resumed


_JOB_STRUCTS = None


def _job_structs():
    """(JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOBOBJECT_BASIC_ACCOUNTING_INFORMATION) as ctypes structures."""
    global _JOB_STRUCTS
    if _JOB_STRUCTS is None:
        import ctypes
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                                                          "OtherOperationCount", "ReadTransferCount",
                                                          "WriteTransferCount", "OtherTransferCount")]

        class BasicLimit(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class ExtendedLimit(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimit), ("IoInfo", IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        class Accounting(ctypes.Structure):
            _fields_ = [("TotalUserTime", ctypes.c_longlong), ("TotalKernelTime", ctypes.c_longlong),
                        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                        ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                        ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]

        _JOB_STRUCTS = (ExtendedLimit, Accounting)
    return _JOB_STRUCTS


_KILL_ON_JOB_CLOSE, _JOB_MEMORY = 0x2000, 0x200
_PROCESS_SET_QUOTA, _PROCESS_TERMINATE = 0x0100, 0x0001
_EXTENDED_LIMIT_INFORMATION, _BASIC_ACCOUNTING_INFORMATION = 9, 1


class _Tree:
    """A started program and every process it starts: a Job Object on Windows (killed as one, with a
    limit on the committed memory of the whole job), the program's own process group on POSIX (with
    an address-space limit through ``prlimit`` on Linux).

    On Windows the program must be started with ``CREATE_SUSPENDED``: it joins the job before it
    runs, then is resumed here. (A Windows venv's interpreter is a launcher that starts the real
    interpreter at once, in a job of its own that lets children break away; assigned after it
    started, that interpreter and everything it starts would be outside the job.)"""

    def __init__(self, proc: subprocess.Popen, memory_limit: int | None):
        self.proc, self.job = proc, None
        if os.name == "nt":
            try:
                self.job = self._windows_job(proc.pid, memory_limit)
            finally:
                if not _resume_windows(proc.pid):
                    proc.kill()
                    raise OSError(f"could not resume the suspended program (pid {proc.pid})")
        elif memory_limit:
            try:
                import resource
                if hasattr(resource, "prlimit"):
                    resource.prlimit(proc.pid, resource.RLIMIT_AS, (int(memory_limit), int(memory_limit)))
            except (OSError, ValueError):
                pass

    @staticmethod
    def _windows_job(pid: int, memory_limit: int | None):
        import ctypes
        try:
            k = _kernel32()
            ExtendedLimit, _ = _job_structs()
            job = k.CreateJobObjectW(None, None)
            if not job:
                return None
            info = ExtendedLimit()
            info.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE | (_JOB_MEMORY if memory_limit else 0)
            info.JobMemoryLimit = int(memory_limit or 0)
            handle = k.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
            ok = bool(handle) and bool(k.SetInformationJobObject(job, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                                                                 ctypes.sizeof(info))) \
                and bool(k.AssignProcessToJobObject(job, handle))
            if handle:
                k.CloseHandle(handle)
            if not ok:
                k.CloseHandle(job)
                return None
            return job
        except (OSError, AttributeError):
            return None

    def exited(self, wait: float) -> bool:
        """Whether the program has exited (waiting up to ``wait`` seconds). On POSIX the exited
        program is not reaped yet, so its process group id stays reserved until ``kill``."""
        if os.name == "posix" and hasattr(os, "waitid"):
            try:
                if os.waitid(os.P_PID, self.proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                    return True
            except ChildProcessError:
                return True
            time.sleep(wait)
            return False
        try:
            self.proc.wait(timeout=wait)
            return True
        except subprocess.TimeoutExpired:
            return False

    def kill(self) -> None:
        """Kill the program (if still running) and every process it started; reap it."""
        if os.name == "posix":
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        elif self.job is not None:
            import ctypes
            k = _kernel32()
            _, Accounting = _job_structs()
            k.TerminateJobObject(self.job, 1)
            acct, deadline = Accounting(), time.monotonic() + 5
            while time.monotonic() < deadline:     # the processes release their files (the temp dir) on exit
                if not k.QueryInformationJobObject(self.job, _BASIC_ACCOUNTING_INFORMATION, ctypes.byref(acct),
                                                   ctypes.sizeof(acct), None) or acct.ActiveProcesses == 0:
                    break
                time.sleep(0.01)
            k.CloseHandle(self.job)
            self.job = None
        else:   # no Job Object: kill the tree by parent links (misses orphaned grandchildren)
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)
        try:
            self.proc.kill()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _tail(text: str, limit: int = 2000) -> str:
    return text if len(text) <= limit else "..." + text[-limit:]


def _read_tail(path: Path, size: int = 65536) -> bytes:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - size))
        return f.read()


def run_program(source: str, *, timeout: float = 10.0, python: str | None = None,
                memory_limit: int | None = MEMORY_LIMIT, output_limit: int = OUTPUT_LIMIT) -> dict:
    """Run ``source`` in a separate ``python -I -X utf8`` process in a fresh temporary directory.

    Returns {"status": "passed" | "failed" | "timeout", "returncode", "seconds", "stderr"}.
    ``passed`` needs exit code 0 AND a random end marker printed after the program's last
    line, so code that exits early (``sys.exit(0)``, ``os._exit``) fails. More than
    ``output_limit`` bytes of output fail; ``memory_limit`` bounds its memory (Linux and
    Windows). Afterwards nothing it started is left running. Not a security sandbox (see the
    module docstring)."""
    marker = secrets.token_hex(16)
    with tempfile.TemporaryDirectory(prefix="openbonsai-exec-", ignore_cleanup_errors=True) as root:
        work = Path(root) / "work"            # the program's directory: nothing else in it
        work.mkdir()
        path = work / "program.py"
        path.write_text(f"{source}\n\nprint({marker!r}, flush=True)\n", encoding="utf-8")
        out_path, err_path = Path(root) / "stdout", Path(root) / "stderr"
        with open(out_path, "wb") as out_f, open(err_path, "wb") as err_f:
            kwargs = {"cwd": str(work), "stdin": subprocess.DEVNULL, "stdout": out_f, "stderr": err_f,
                      "env": _child_env(str(work))}
            if os.name == "posix":
                kwargs["start_new_session"] = True
            else:     # suspended until it is in its Job Object (_Tree)
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | 0x4   # CREATE_SUSPENDED
            t0 = time.monotonic()
            proc = subprocess.Popen([python or sys.executable, "-I", "-X", "utf8", path.name], **kwargs)
            tree = _Tree(proc, memory_limit)
            stopped = None
            try:
                while not tree.exited(0.02):
                    if time.monotonic() - t0 > timeout:
                        stopped = "timeout"
                        break
                    if os.fstat(out_f.fileno()).st_size + os.fstat(err_f.fileno()).st_size > output_limit:
                        stopped = "output"
                        break
            finally:
                tree.kill()
            seconds = time.monotonic() - t0
        out = _read_tail(out_path).decode("utf-8", errors="replace")
        message = _read_tail(err_path).decode("utf-8", errors="replace").replace("\r\n", "\n")
        for prefix in {str(work), os.path.realpath(work)}:   # tracebacks name the absolute script path
            message = message.replace(prefix + os.sep, "")
    if stopped == "timeout":
        status = "timeout"
    elif stopped is None and proc.returncode == 0 and marker in out:
        status = "passed"
    else:
        status = "failed"
        if stopped == "output":
            message += f"\n[openbonsai] killed: more than {output_limit} bytes of output"
        elif proc.returncode == 0:
            message += "\n[openbonsai] the program exited before the test finished"
    return {"status": status, "returncode": proc.returncode, "seconds": round(seconds, 3), "stderr": _tail(message)}


def run_programs(sources: list[str], *, timeout: float = 10.0, workers: int = 4,
                 memory_limit: int | None = MEMORY_LIMIT) -> list[dict]:
    """``run_program`` for each source, ``workers`` processes at a time; results in input order."""
    if not sources:
        return []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        return list(pool.map(lambda s: run_program(s, timeout=timeout, memory_limit=memory_limit), sources))

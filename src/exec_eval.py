"""Execute model-written Python against assert-style tests in isolated subprocesses.

Each program runs in its own interpreter (`python -I`), in an empty temporary directory, with CPU-time
and memory limits and a wall-clock timeout. This is the usual HumanEval/MBPP-style harness, not a
security sandbox.
"""
import re
import resource
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

TIMEOUT_S = 10


def extract_code(text: str) -> str:
    """Take the first fenced code block if present, else the raw text."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:python|py)?\s*\n(.*)", text, flags=re.S)   # unterminated block (truncation)
    return (m.group(1) if m else text).strip()


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S))
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))


def run_program(program: str) -> str:
    """Return 'pass', 'timeout', or the exception class name / 'error'."""
    with tempfile.TemporaryDirectory() as d:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", program], cwd=d, capture_output=True, text=True,
                               timeout=TIMEOUT_S, preexec_fn=_limits, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return "timeout"
    if p.returncode == 0:
        return "pass"
    last = [l for l in p.stderr.strip().splitlines() if l.strip()]
    m = re.match(r"(\w+(?:Error|Exception))", last[-1]) if last else None
    return m.group(1) if m else ("timeout" if p.returncode in (-9, -24) else "error")


def check_many(codes: list, tests: list, setups: list = None, workers: int = 12) -> list:
    """codes[i] is run followed by tests[i] (list of assert strings). Returns a status string per item."""
    setups = setups or [""] * len(codes)
    progs = [f"{s}\n{c}\n\n" + "\n".join(t) + "\n" for c, t, s in zip(codes, tests, setups)]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(run_program, progs))

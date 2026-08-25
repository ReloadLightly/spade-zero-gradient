"""Environment contract, safe loading, and the V1 executability battery.

The env contract (frozen in PREREG.md, mirrored verbatim in the designer
strategy contract block):

  * one Python class whose name ends in ``Env``
  * ``reset(self, seed=None) -> (obs: str, info: dict)``
  * ``step(self, action: str) -> (obs: str, reward: float, terminated: bool,
    truncated: bool, info: dict)``
  * all randomness derived from ``random.Random(seed)`` captured at reset
    (same seed => identical episode dynamics)
  * per-step rewards in [0, 1]; an episode's return is the clipped sum in
    [0, 1]; success == return >= 0.99 unless partial rewards are used
  * imports restricted to: random, math, itertools, re, collections, string,
    functools, heapq
  * env must terminate or truncate within MAX_STEPS_CONTRACT steps
"""

from __future__ import annotations

import ast
import builtins as _builtins
import signal
import time
from dataclasses import dataclass, field
from typing import Any

MAX_STEPS_CONTRACT = 10
MAX_ENV_LINES = 400
STEP_TIMEOUT_S = 2.0

ALLOWED_IMPORTS = {
    "random", "math", "itertools", "re", "collections", "string",
    "functools", "heapq",
}

_FORBIDDEN_NAMES = {
    "open", "eval", "exec", "compile", "globals", "locals", "vars",
    "__import__", "input", "exit", "quit", "breakpoint", "getattr",
    "setattr", "delattr", "memoryview", "bytearray",
}

_SAFE_BUILTIN_NAMES = [
    "range", "len", "enumerate", "zip", "min", "max", "sum", "abs", "sorted",
    "list", "dict", "set", "tuple", "str", "int", "float", "bool", "print",
    "isinstance", "issubclass", "round", "any", "all", "map", "filter",
    "reversed", "divmod", "pow", "chr", "ord", "repr", "hash", "frozenset",
    "iter", "next", "slice", "type", "object", "super", "staticmethod",
    "classmethod", "property", "ValueError", "TypeError", "KeyError",
    "IndexError", "StopIteration", "Exception", "ZeroDivisionError",
    "AttributeError", "NotImplementedError", "RuntimeError", "AssertionError",
    "True", "False", "None",
]


class EnvContractError(Exception):
    """Raised when generated environment code violates the contract."""


class _Timeout(Exception):
    pass


class time_limit:
    """SIGALRM-based wall-clock limit (Linux, main thread)."""

    def __init__(self, seconds: float):
        self.seconds = seconds

    def __enter__(self):
        def handler(signum, frame):
            raise _Timeout()
        self._old = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, exc_type, exc, tb):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self._old)
        return False


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
    import importlib
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of '{name}' is not allowed")
    return importlib.import_module(name)


def _safe_builtins() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in _SAFE_BUILTIN_NAMES:
        if hasattr(_builtins, name):
            out[name] = getattr(_builtins, name)
    # class definitions need these two; imports go through the allowlist guard
    out["__build_class__"] = _builtins.__build_class__
    out["__name__"] = "szg_generated_env"
    out["__import__"] = _guarded_import
    return out


def validate_source(code: str) -> list[str]:
    """Static contract checks. Returns a list of issues (empty == pass)."""
    issues: list[str] = []
    if len(code.splitlines()) > MAX_ENV_LINES:
        issues.append(f"env exceeds {MAX_ENV_LINES} lines")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    env_classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    issues.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                issues.append(f"forbidden import-from: {node.module}")
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            issues.append(f"forbidden name: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in ("__init__",):
            issues.append(f"forbidden dunder attribute access: {node.attr}")
        elif isinstance(node, ast.ClassDef) and node.name.endswith("Env"):
            env_classes.append(node)

    if len(env_classes) != 1:
        issues.append(f"expected exactly one *Env class, found {len(env_classes)}")
    else:
        methods = {n.name for n in env_classes[0].body if isinstance(n, ast.FunctionDef)}
        for required in ("reset", "step"):
            if required not in methods:
                issues.append(f"class missing required method: {required}")
    return issues


def load_env_class(code: str):
    """Exec env code in a restricted namespace and return the *Env class."""
    issues = validate_source(code)
    if issues:
        raise EnvContractError("; ".join(issues))
    import random as _random
    import math as _math
    import itertools as _itertools
    import re as _re
    import collections as _collections
    import string as _string
    import functools as _functools
    import heapq as _heapq
    ns: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "random": _random, "math": _math, "itertools": _itertools,
        "re": _re, "collections": _collections, "string": _string,
        "functools": _functools, "heapq": _heapq,
    }
    with time_limit(5.0):
        exec(code, ns)  # noqa: S102 — statically screened, restricted builtins
    classes = [v for k, v in ns.items()
               if isinstance(v, type) and k.endswith("Env")]
    if len(classes) != 1:
        raise EnvContractError(f"expected exactly one *Env class after exec, found {len(classes)}")
    return classes[0]


@dataclass
class BatteryReport:
    passed: bool
    issues: list[str] = field(default_factory=list)
    steps_run: int = 0
    wall_s: float = 0.0


_BATTERY_ACTIONS = ["0", "1", "2", "yes", "no", "a", "3 4", "guess x", "help", "b"]


def _run_trace(env_cls, seed: int, n_steps: int) -> list[tuple]:
    env = env_cls()
    with time_limit(STEP_TIMEOUT_S):
        obs, info = env.reset(seed=seed)
    trace = [(str(obs),)]
    for i in range(n_steps):
        action = _BATTERY_ACTIONS[(seed + i) % len(_BATTERY_ACTIONS)]
        with time_limit(STEP_TIMEOUT_S):
            obs, r, term, trunc, info = env.step(action)
        if not isinstance(r, (int, float)):
            raise EnvContractError(f"reward not numeric: {type(r)}")
        if not (-1e-9 <= float(r) <= 1.0 + 1e-9):
            raise EnvContractError(f"per-step reward outside [0,1]: {r}")
        if not isinstance(term, bool) or not isinstance(trunc, bool):
            raise EnvContractError("terminated/truncated must be bool")
        trace.append((str(obs), float(r), term, trunc))
        if term or trunc:
            break
    else:
        # battery random-walk hit n_steps without termination — acceptable,
        # runner enforces MAX_STEPS_CONTRACT truncation anyway
        pass
    return trace


def run_battery(code: str, seeds=(0, 1, 2)) -> BatteryReport:
    """V1 executability battery: load, seeded random-walks, determinism."""
    t0 = time.monotonic()
    issues: list[str] = []
    steps = 0
    try:
        env_cls = load_env_class(code)
    except (EnvContractError, _Timeout, Exception) as e:  # noqa: BLE001
        return BatteryReport(False, [f"load: {e!r}"], 0, time.monotonic() - t0)
    for seed in seeds:
        try:
            trace = _run_trace(env_cls, seed, MAX_STEPS_CONTRACT)
            steps += len(trace) - 1
        except (_Timeout,) as e:
            issues.append(f"seed {seed}: step timeout")
            continue
        except Exception as e:  # noqa: BLE001
            issues.append(f"seed {seed}: {e!r}")
            continue
        # determinism: identical trace on identical seed
        try:
            trace2 = _run_trace(env_cls, seed, MAX_STEPS_CONTRACT)
            if trace != trace2:
                issues.append(f"seed {seed}: nondeterministic under same seed")
        except Exception as e:  # noqa: BLE001
            issues.append(f"seed {seed} (repeat): {e!r}")
    return BatteryReport(len(issues) == 0, issues, steps, time.monotonic() - t0)

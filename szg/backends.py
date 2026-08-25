"""Model backends. One protocol, three implementations.

* ClaudeCLIBackend — headless ``claude -p`` (subscription-backed on Roland's
  machine; the route sandbox-verified against ShinkaEvolve on 2026-08-18).
* MockBackend — deterministic scripted outputs for tests and machinery smoke.
* Transcript logging is the runner's job, not the backend's.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field


class BackendError(Exception):
    pass


@dataclass
class CallStats:
    calls: int = 0
    wall_s: float = 0.0
    by_model: dict = field(default_factory=dict)


class ClaudeCLIBackend:
    """Non-interactive ``claude -p`` per completion. Stateless per call."""

    def __init__(self, timeout_s: float = 240.0, retries: int = 1):
        self.timeout_s = timeout_s
        self.retries = retries
        self.stats = CallStats()

    def complete(self, prompt: str, model: str, max_tokens: int | None = None) -> str:
        last_err: Exception | None = None
        for _attempt in range(self.retries + 1):
            t0 = time.monotonic()
            try:
                proc = subprocess.run(
                    ["claude", "-p", "--model", model, "--output-format", "text"],
                    input=prompt, capture_output=True, text=True,
                    timeout=self.timeout_s,
                )
                dt = time.monotonic() - t0
                self.stats.calls += 1
                self.stats.wall_s += dt
                self.stats.by_model[model] = self.stats.by_model.get(model, 0) + 1
                if proc.returncode != 0:
                    raise BackendError(
                        f"claude -p rc={proc.returncode}: {proc.stderr.strip()[:500]}")
                out = proc.stdout.strip()
                if not out:
                    raise BackendError("claude -p returned empty output")
                return out
            except (subprocess.TimeoutExpired, BackendError) as e:
                last_err = e
                continue
        raise BackendError(f"backend failed after retries: {last_err}")


class MockBackend:
    """Scripted outputs, or a policy function ``fn(prompt, model) -> str``."""

    def __init__(self, outputs: list[str] | None = None, policy=None):
        self.outputs = list(outputs or [])
        self.policy = policy
        self.stats = CallStats()
        self.prompts: list[str] = []

    def complete(self, prompt: str, model: str, max_tokens: int | None = None) -> str:
        self.stats.calls += 1
        self.stats.by_model[model] = self.stats.by_model.get(model, 0) + 1
        self.prompts.append(prompt)
        if self.policy is not None:
            return self.policy(prompt, model)
        if not self.outputs:
            raise BackendError("MockBackend exhausted")
        return self.outputs.pop(0)


def make_backend(name: str, **kwargs):
    if name in ("claude-cli", "cli", "claude"):
        return ClaudeCLIBackend(**kwargs)
    if name == "mock":
        return MockBackend(**kwargs)
    raise ValueError(f"unknown backend: {name}")

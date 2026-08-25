"""Designer call: strategy + memory digest + skill + topic -> env code + hint.

The designer model is FROZEN (no weight updates). Everything it can 'learn'
arrives through this prompt: its evolved strategy text (C2), the environment
memory digest (C1/C2), and the fixed contract block (information parity).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .env_api import ALLOWED_IMPORTS, MAX_ENV_LINES, MAX_STEPS_CONTRACT

CONTRACT_BLOCK = f"""ENVIRONMENT CONTRACT (hard requirements — violations are discarded):
- Output exactly one Python class whose name ends in `Env`.
- `reset(self, seed=None)` returns `(observation: str, info: dict)`.
- `step(self, action: str)` returns `(observation: str, reward: float, terminated: bool, truncated: bool, info: dict)`.
- Derive ALL randomness from `random.Random(seed)` stored at reset — the same seed must reproduce the identical episode.
- Per-step rewards in [0, 1]; total achievable return exactly 1.0 for a perfect episode; partial rewards for verifiable progress are encouraged.
- The episode must terminate (success) or truncate within {MAX_STEPS_CONTRACT} steps; enforce your own step counter.
- Allowed imports ONLY: {", ".join(sorted(ALLOWED_IMPORTS))}. No I/O, no printing, at most {MAX_ENV_LINES} lines.
- The solver is a language model that sends plain-string actions; the opening observation must state the goal, the action format, and the step limit.
- The environment must be self-verifying: rewards computed from internal state, never from trusting the agent's claims.

OUTPUT FORMAT (exactly this structure):
CONCEPT: <one line: name + what makes it challenging>
```python
<the environment class>
```
HINT: <at most 4 sentences and 600 characters of privileged strategic insight that
makes the environment easier WITHOUT stating the literal solution or any literal
action string. Never include 'ACTION:' or code in the hint.>"""


@dataclass
class DesignerOutput:
    concept: str
    code: str
    hint: str
    raw: str


class DesignerParseError(Exception):
    pass


_CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def build_designer_prompt(strategy_text: str, skill: str, topic: str,
                          memory_digest: str | None) -> str:
    parts = [
        "You are the Environment Designer in a SPADE-style self-play loop "
        "(you write training environments; a separate frozen solver model plays them).",
        "== YOUR CURRENT DESIGN STRATEGY ==\n" + strategy_text.strip(),
        "== TARGET SKILL ==\n" + skill,
        "== GROUNDING TOPIC (domain flavor; adapt freely) ==\n" + topic,
    ]
    if memory_digest:
        parts.append("== MEMORY ==\n" + memory_digest)
    parts.append(CONTRACT_BLOCK)
    parts.append("Design one environment now. Follow the OUTPUT FORMAT exactly.")
    return "\n\n".join(parts)


def parse_designer_output(text: str) -> DesignerOutput:
    concept_m = re.search(r"^CONCEPT:\s*(.+)$", text, re.MULTILINE)
    code_m = _CODE_RE.search(text)
    hint_m = re.search(r"^HINT:\s*(.+)\Z", text, re.MULTILINE | re.DOTALL)
    if not code_m:
        raise DesignerParseError("no fenced python block found")
    if not hint_m:
        raise DesignerParseError("no HINT: block found")
    return DesignerOutput(
        concept=(concept_m.group(1).strip() if concept_m else "(unnamed)"),
        code=code_m.group(1).strip() + "\n",
        hint=" ".join(hint_m.group(1).split()),
        raw=text,
    )


def generate(backend, model: str, strategy_text: str, skill: str, topic: str,
             memory_digest: str | None = None, retries: int = 1) -> DesignerOutput:
    prompt = build_designer_prompt(strategy_text, skill, topic, memory_digest)
    last: Exception | None = None
    for _ in range(retries + 1):
        text = backend.complete(prompt, model=model)
        try:
            return parse_designer_output(text)
        except DesignerParseError as e:
            last = e
            prompt = prompt + ("\n\nYour previous output could not be parsed "
                               f"({e}). Follow the OUTPUT FORMAT exactly.")
    raise last  # type: ignore[misc]

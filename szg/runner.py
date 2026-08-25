"""Episode runner and per-environment evaluation (paired hint / no-hint arms).

Information-parity law (hard rule, PREREG H-RULE-1): the solver prompt always
contains the full interface contract and the complete episode transcript;
the hint arm differs ONLY by the presence of the designer's hint block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .env_api import MAX_STEPS_CONTRACT, load_env_class, time_limit, STEP_TIMEOUT_S

MAX_FORMAT_ERRORS = 2

SOLVER_SYSTEM = """You are the Reasoning Agent playing a text-based environment (a seeded, \
verifiable game). You interact turn by turn.

Rules:
- Each turn you receive the latest observation (and possibly a reward signal).
- Think briefly, then end your reply with a single final line of exactly:
  ACTION: <your action string>
- The environment parses only the text after "ACTION:" on that final line.
- You have at most {max_steps} steps total. Some environments give partial rewards \
for progress; maximize total reward.
- Actions are plain strings (e.g. "3", "yes", "guess north", "swap 2 5"). Read the \
observation carefully for the expected format."""

HINT_BLOCK = """
A privileged HINT is available for this episode:
HINT: {hint}
"""

_ACTION_RE = re.compile(r"^ACTION:\s*(.+?)\s*$", re.MULTILINE)


@dataclass
class EpisodeResult:
    return_: float
    steps: int
    ended: str                      # terminated | truncated | step_cap | format_failure | env_error
    format_errors: int
    transcript: list = field(default_factory=list)   # [{role, text}]


def _parse_action(text: str) -> str | None:
    matches = _ACTION_RE.findall(text)
    return matches[-1] if matches else None


def play_episode(env_cls, backend, model: str, seed: int,
                 hint: str | None = None,
                 max_steps: int = MAX_STEPS_CONTRACT) -> EpisodeResult:
    env = env_cls()
    with time_limit(STEP_TIMEOUT_S):
        obs, _info = env.reset(seed=seed)
    system = SOLVER_SYSTEM.format(max_steps=max_steps)
    if hint:
        system += HINT_BLOCK.format(hint=hint)
    transcript = [{"role": "system", "text": system},
                  {"role": "env", "text": str(obs)}]
    total = 0.0
    fmt_errors = 0
    steps = 0
    ended = "step_cap"
    while steps < max_steps:
        prompt_parts = [system, "\n== EPISODE TRANSCRIPT =="]
        for turn in transcript[1:]:
            label = {"env": "OBSERVATION", "agent": "YOU"}.get(turn["role"], turn["role"].upper())
            prompt_parts.append(f"[{label}]\n{turn['text']}")
        prompt_parts.append(
            f"\nStep {steps + 1} of {max_steps}. Reply with brief reasoning and a final 'ACTION:' line.")
        reply = backend.complete("\n\n".join(prompt_parts), model=model)
        transcript.append({"role": "agent", "text": reply})
        action = _parse_action(reply)
        if action is None:
            fmt_errors += 1
            if fmt_errors > MAX_FORMAT_ERRORS:
                ended = "format_failure"
                break
            transcript.append({"role": "env", "text":
                               "FORMAT ERROR: reply must end with a line 'ACTION: <action>'. Try again."})
            continue
        steps += 1
        try:
            with time_limit(STEP_TIMEOUT_S):
                obs, r, term, trunc, _info = env.step(action)
        except Exception as e:  # noqa: BLE001 — env bug surfaces as env_error episode
            ended = "env_error"
            transcript.append({"role": "env", "text": f"ENV ERROR: {e!r}"})
            break
        total += float(r)
        msg = str(obs)
        if r:
            msg += f"\n[reward this step: {float(r):g}]"
        transcript.append({"role": "env", "text": msg})
        if term:
            ended = "terminated"
            break
        if trunc:
            ended = "truncated"
            break
    return EpisodeResult(return_=max(0.0, min(1.0, total)), steps=steps,
                         ended=ended, format_errors=fmt_errors,
                         transcript=transcript)


@dataclass
class EnvEval:
    mean_nohint: float
    mean_hint: float
    floored_regret: float
    win_nohint: float               # fraction of episodes with return >= 0.99
    win_hint: float
    episodes: list = field(default_factory=list)   # [(arm, seed, EpisodeResult)]


def evaluate_env(code: str, hint: str, backend, solver_model: str,
                 G: int = 3, seed0: int = 0,
                 max_steps: int = MAX_STEPS_CONTRACT) -> EnvEval:
    """Paired evaluation: same G seeds for the no-hint and hint arms."""
    env_cls = load_env_class(code)
    episodes = []
    rets = {"nohint": [], "hint": []}
    for arm, h in (("nohint", None), ("hint", hint)):
        for g in range(G):
            res = play_episode(env_cls, backend, solver_model,
                               seed=seed0 + g, hint=h, max_steps=max_steps)
            episodes.append((arm, seed0 + g, res))
            rets[arm].append(res.return_)
    mean_no = sum(rets["nohint"]) / G
    mean_h = sum(rets["hint"]) / G
    return EnvEval(
        mean_nohint=mean_no, mean_hint=mean_h,
        floored_regret=max(0.0, mean_h - mean_no),
        win_nohint=sum(1 for r in rets["nohint"] if r >= 0.99) / G,
        win_hint=sum(1 for r in rets["hint"] if r >= 0.99) / G,
        episodes=episodes,
    )

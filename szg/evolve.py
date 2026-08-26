"""ShinkaEvolve-style strategy evolution over designer strategy TEXTS.

The genotype is the designer's written strategy (a playbook the frozen model
follows). Mutation is LLM-guided rewriting conditioned on FULL measured
feedback (information parity — the mutator sees everything selection saw).
Selection is fitness-weighted parent sampling over the archive, with an
epsilon slot for the least-tried lineage (novelty pressure).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

MUTATION_PROMPT = """You are improving the DESIGN STRATEGY used by a frozen Environment Designer \
model in a SPADE-style loop (SPADE = self-play over executable Gym-style environments; the \
designer's reward is floored hint-based regret: solver return with hint minus without, \
floored at 0, counted only for environments whose no-hint win rate lies in [0.05, 0.95]).

The designer's weights never change. The ONLY lever is this strategy text.

== CURRENT STRATEGY (parent) ==
{parent_text}

== MEASURED RESULTS UNDER THIS STRATEGY ==
{feedback}

== YOUR TASK ==
Write an improved strategy. Diagnose from the measurements what limited fitness \
(too easy? too hard/hint-gated? invalid code? repetitive concepts?) and change the \
strategy to fix precisely that. Keep what worked. Requirements:
- at most 350 words, imperative playbook style;
- concrete, testable design moves (difficulty dials, state-gating patterns, \
partial-reward ladders, hint policy), not platitudes;
- it must instruct designs that satisfy the environment contract (seeded \
determinism, <= 10 steps, verifiable rewards);
- do not mention this mutation process in the strategy.

Output ONLY the new strategy text, starting with the line:
STRATEGY {child_id}:"""


@dataclass
class Strategy:
    id: str
    parent: str | None
    text: str
    fitnesses: list[float]

    @property
    def mean_fitness(self) -> float:
        return sum(self.fitnesses) / len(self.fitnesses) if self.fitnesses else 0.0


class StrategyArchive:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.strategies: dict[str, Strategy] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.strategies[d["id"]] = Strategy(
                        d["id"], d.get("parent"), d["text"], d.get("fitnesses", []))

    def _persist(self) -> None:
        with self.path.open("w") as f:
            for s in self.strategies.values():
                f.write(json.dumps({"id": s.id, "parent": s.parent,
                                    "text": s.text, "fitnesses": s.fitnesses},
                                   ensure_ascii=False) + "\n")

    def add(self, strategy: Strategy) -> None:
        self.strategies[strategy.id] = strategy
        self._persist()

    def record_fitness(self, sid: str, fitness: float) -> None:
        self.strategies[sid].fitnesses.append(fitness)
        self._persist()

    def select_parent(self, rng: random.Random, epsilon: float = 0.25,
                      temperature: float = 0.15) -> Strategy:
        pool = list(self.strategies.values())
        if not pool:
            raise RuntimeError("empty strategy archive")
        untried = [s for s in pool if not s.fitnesses]
        if untried and rng.random() < epsilon:
            return rng.choice(untried)
        scored = [s for s in pool if s.fitnesses] or pool
        weights = [math.exp(s.mean_fitness / temperature) for s in scored]
        return rng.choices(scored, weights=weights, k=1)[0]

    def best(self) -> Strategy:
        return max(self.strategies.values(), key=lambda s: s.mean_fitness)


def _num(rec: dict, key: str, default: float = 0.0) -> float:
    """Numeric field of an env record, tolerating a present-but-None value.

    Same defect A1 fixed in fitness.py (2026-08-26): an env rejected by V1/V2
    never reaches evaluation and its record carries these keys with value
    None, so ``rec.get(key, 0)`` returns None and the ``:.2f`` format below
    raises TypeError. This killed C2 round 0 of the main grid at the mutate
    step, after the round had already been scored and logged. evolve.py is
    not a frozen file, so this is a plain bug fix; the record still shows the
    unevaluated env, now as 0.00 with its gate issues appended.
    """
    v = rec.get(key)
    return default if v is None else v


def render_feedback(round_score, env_records: list[dict], max_envs: int = 8) -> str:
    lines = [
        f"strategy fitness this round: {round_score.fitness:.3f} "
        f"({round_score.n_valid}/{round_score.n_envs} envs valid"
        + (", ROUND INVALID — fitness zeroed" if round_score.invalid_round else "") + ")",
        f"learnable fraction: {round_score.learnable_fraction:.2f} | "
        f"mean no-hint win: {round_score.mean_win_nohint:.2f} | "
        f"diversity proxy: {round_score.diversity:.2f}",
        "per-environment:",
    ]
    for r in env_records[:max_envs]:
        lines.append(
            f"- [{r.get('skill')}] {r.get('concept')} | no-hint {_num(r, 'win_nohint'):.2f} "
            f"hint {_num(r, 'win_hint'):.2f} regret {_num(r, 'floored_regret'):.2f}"
            + ("" if r.get("valid_for_fitness") else
               f" | INVALID: {'; '.join(r.get('gate_issues', ['?'])[:2])}"))
    return "\n".join(lines)


def mutate(backend, model: str, parent: Strategy, feedback: str,
           child_id: str) -> Strategy:
    prompt = MUTATION_PROMPT.format(parent_text=parent.text,
                                    feedback=feedback, child_id=child_id)
    text = backend.complete(prompt, model=model).strip()
    marker = f"STRATEGY {child_id}:"
    if marker in text:
        text = text.split(marker, 1)[1].strip()
    return Strategy(id=child_id, parent=parent.id, text=text, fitnesses=[])
